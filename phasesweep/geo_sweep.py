"""Geometry dimension sweeps with proportional and fixed-gap scaling.

Generates grids of valid Geometry objects for parametric studies.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from dataclasses import dataclass
from itertools import product
from typing import Any, Literal

import numpy as np

from phasesweep.geometry import Geometry


@dataclass(frozen=True)
class SweepAxis:
    field: Literal["r_outer", "L_stk", "r_ag"]
    start: float
    stop: float
    steps: int
    strategy: Literal["proportional", "fixed_gap"] = "proportional"

    def __post_init__(self) -> None:
        if self.steps < 2:
            raise ValueError(f"steps must be >= 2, got {self.steps}")
        if self.start <= 0 or self.stop <= 0:
            raise ValueError("start and stop must be positive")

    def values(self) -> list[float]:
        return np.linspace(self.start, self.stop, self.steps).tolist()


@dataclass(frozen=True)
class SweepPoint:
    geometry: Geometry
    L_stk: float | None = None


def _scale_proportional(geo: Geometry, factor: float) -> Geometry:
    return dataclasses.replace(
        geo,
        r_outer=geo.r_outer * factor,
        r_stator=geo.r_stator * factor,
        r_magnet=geo.r_magnet * factor,
        r_rotor=geo.r_rotor * factor,
        r_inner=geo.r_inner * factor,
        r_ag=geo.r_ag * factor,
        slot_depth=geo.slot_depth * factor,
        slot_opening_width=geo.slot_opening_width * factor,
    )


def _set_airgap(geo: Geometry, new_gap: float) -> Geometry:
    if geo.topology == "inrunner":
        old_gap = geo.r_stator - geo.r_magnet
        delta = new_gap - old_gap
        new_r_magnet = geo.r_magnet - delta / 2
        new_r_stator = geo.r_stator + delta / 2
        new_r_ag = (new_r_stator + new_r_magnet) / 2
        return dataclasses.replace(
            geo,
            r_stator=new_r_stator,
            r_magnet=new_r_magnet,
            r_ag=new_r_ag,
            # opening width scales with bore: slot_opening_ratio invariant
            slot_opening_width=geo.slot_opening_width * (new_r_stator / geo.r_stator),
        )
    else:
        old_gap = geo.r_magnet - geo.r_stator
        delta = new_gap - old_gap
        new_r_magnet = geo.r_magnet + delta / 2
        new_r_stator = geo.r_stator - delta / 2
        new_r_ag = (new_r_stator + new_r_magnet) / 2
        return dataclasses.replace(
            geo,
            r_stator=new_r_stator,
            r_magnet=new_r_magnet,
            r_ag=new_r_ag,
            slot_opening_width=geo.slot_opening_width * (new_r_stator / geo.r_stator),
        )


def apply_axis(
    geo: Geometry,
    axis: SweepAxis,
    value: float,
    base_L_stk: float | None = None,
) -> tuple[Geometry, float | None]:
    if axis.field == "L_stk":
        return geo, value

    if axis.field == "r_outer":
        if axis.strategy == "proportional":
            factor = value / geo.r_outer
            return _scale_proportional(geo, factor), base_L_stk
        else:
            # fixed_gap: scale outer but keep airgap constant
            old_gap = abs(geo.r_stator - geo.r_magnet)
            factor = value / geo.r_outer
            scaled = _scale_proportional(geo, factor)
            return _set_airgap(scaled, old_gap), base_L_stk

    if axis.field == "r_ag":
        return _set_airgap(geo, value), base_L_stk

    raise ValueError(f"unknown sweep field: {axis.field}")


def generate_grid(
    base: Geometry,
    axes: list[SweepAxis],
    base_L_stk: float | None = None,
) -> list[SweepPoint]:
    if not axes:
        return [SweepPoint(geometry=base, L_stk=base_L_stk)]

    value_lists = [a.values() for a in axes]
    points: list[SweepPoint] = []

    for combo in product(*value_lists):
        geo = base
        L_stk = base_L_stk
        valid = True
        for axis, value in zip(axes, combo):
            try:
                geo, L_stk = apply_axis(geo, axis, value, L_stk)
            except ValueError:
                valid = False
                break
        if not valid:
            logging.debug("Skipping invalid combo: %s", combo)
            continue

        points.append(SweepPoint(geometry=geo, L_stk=L_stk))

    return points


# ---------------------------------------------------------------------------
# Parallel sweep runner
# ---------------------------------------------------------------------------


def _sweep_worker(job: dict[str, Any]) -> dict[str, Any]:
    """Module-level worker for execute_generic (must be picklable)."""
    from phasesweep.motor import Motor
    from phasesweep.registry import MODEL_REGISTRY
    from phasesweep.sweep_types import RunConfig

    motor = Motor.from_dict(job["motor_dict"])
    model_key = job["model_key"]
    config_kw = job.get("config_kw", {})

    passthrough = {k: v for k, v in job.items()
                   if k not in {"motor_dict", "config_kw"}}

    info = MODEL_REGISTRY[model_key]
    if info.fn is None:
        return {**passthrough, "status": "ERROR", "metrics": None,
                "error_msg": f"model {model_key!r} has no fn"}

    t0 = time.perf_counter()
    try:
        config = RunConfig(motor=motor, model=model_key, **config_kw)
        metrics = info.fn(config)
        return {**passthrough, "status": "OK", "metrics": metrics,
                "elapsed_s": time.perf_counter() - t0}
    except Exception as e:
        return {**passthrough, "status": "ERROR", "metrics": None,
                "elapsed_s": time.perf_counter() - t0, "error_msg": str(e)}


@dataclass(frozen=True)
class SweepResult:
    point_idx: int
    model: str
    status: str
    metrics: dict[str, Any] | None = None
    elapsed_s: float = 0.0
    error_msg: str | None = None


def run_sweep(
    base_motor: Any,
    points: list[SweepPoint],
    model_keys: list[str],
    *,
    workers: int = 1,
    cache_dir: str | None = None,
    on_complete: Any | None = None,
) -> list[SweepResult]:
    """Run models across sweep points, optionally in parallel.

    Returns a flat list of SweepResult (one per point × model).
    """
    from phasesweep.parallel import execute_generic

    jobs: list[dict[str, Any]] = []
    for i, pt in enumerate(points):
        motor = dataclasses.replace(base_motor, geometry=pt.geometry)
        if pt.L_stk is not None:
            motor = dataclasses.replace(motor, L_stk=pt.L_stk)
        motor_dict = motor.to_dict()

        for mk in model_keys:
            config_kw = {"nonlinear": True} if mk == "fem" else {}
            jobs.append({
                "motor_dict": motor_dict,
                "model_key": mk,
                "config_kw": config_kw,
                "point_idx": i,
            })

    raw = execute_generic(
        _sweep_worker, jobs, workers=workers,
        cache_dir=cache_dir, on_complete=on_complete,
    )

    return [
        SweepResult(
            point_idx=r["point_idx"],
            model=r["model_key"],
            status=r.get("status", "ERROR"),
            metrics=r.get("metrics"),
            elapsed_s=r.get("elapsed_s", 0.0),
            error_msg=r.get("error_msg"),
        )
        for r in raw
    ]
