"""Parallel execution and geometry dimension sweeps.

Parallel execution:
- execute_parallel(): dispatch registry-based model runs (sensitivity sweeps)
- execute_generic(): dispatch arbitrary module-level worker functions

Both fall back to sequential execution when workers=1, preserving debuggability.

Geometry sweeps:
- generate_grid(): generate grids of valid Geometry objects for parametric studies
- run_sweep(): run models across sweep points, optionally in parallel
"""

from __future__ import annotations

import dataclasses
import logging
import multiprocessing
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from phasesweep.machines.motor import Motor

import numpy as np

from phasesweep.machines.geometry import Geometry

_FUTURE_TIMEOUT_S = 600


# ---------------------------------------------------------------------------
# Parallel execution
# ---------------------------------------------------------------------------


def _worker_init(cache_dir: str | None) -> None:
    """ProcessPoolExecutor initializer — runs once per worker process."""
    if cache_dir:
        from phasesweep.solvers.fem_field import set_disk_cache_dir
        set_disk_cache_dir(cache_dir)


_WORKER_COMPUTE_KEYS = {"motor_dict", "config_kw"}


def _passthrough(job: dict[str, Any]) -> dict[str, Any]:
    """Non-compute job keys, forwarded to the result for caller-side routing."""
    return {k: v for k, v in job.items() if k not in _WORKER_COMPUTE_KEYS}


def _failure_result(
    job: dict[str, Any], status: str, elapsed_s: float, msg: str,
) -> dict[str, Any]:
    return {
        **_passthrough(job),
        "status": status,
        "metrics": None,
        "elapsed_s": elapsed_s,
        "error_msg": msg,
    }


def _run_worker(job: dict[str, Any]) -> dict[str, Any]:
    """Top-level worker function (must be module-level for pickling).

    Deserializes Motor, builds RunConfig, calls the registry fn, and returns
    a result dict.  All non-compute keys from the job are forwarded to the
    result for caller-side routing.
    """
    model_key = job["model_key"]

    t0 = time.perf_counter()
    try:
        from phasesweep.machines.motor import Motor
        from phasesweep.registry import MODEL_REGISTRY
        from phasesweep.sweep_types import RunConfig

        motor = Motor.from_dict(job["motor_dict"])
        config_kw = job.get("config_kw", {})
        config = RunConfig(motor=motor, model=model_key, **config_kw)

        info = MODEL_REGISTRY[model_key]
        if info.fn is None:
            raise ValueError(f"model {model_key!r} has no fn")

        metrics = info.fn(config)
        return {
            **_passthrough(job),
            "status": "OK",
            "metrics": metrics,
            "elapsed_s": time.perf_counter() - t0,
            "error_msg": None,
        }
    except Exception as e:
        return _failure_result(job, "ERROR", time.perf_counter() - t0, str(e))


def execute_parallel(
    jobs: list[dict[str, Any]],
    workers: int = 1,
    cache_dir: str | None = None,
    on_complete: Callable[[dict[str, Any], int, int], None] | None = None,
    timeout_s: float | None = None,
    cancel: threading.Event | None = None,
) -> list[dict[str, Any]]:
    """Run registry-based model jobs via _run_worker.

    Thin wrapper around execute_generic — see that function for details.
    """
    return execute_generic(
        _run_worker, jobs, workers=workers,
        cache_dir=cache_dir, on_complete=on_complete, timeout_s=timeout_s,
        cancel=cancel,
    )


def execute_generic(
    worker_fn: Callable[[dict[str, Any]], dict[str, Any]],
    jobs: list[dict[str, Any]],
    workers: int = 1,
    cache_dir: str | None = None,
    on_complete: Callable[[dict[str, Any], int, int], None] | None = None,
    timeout_s: float | None = None,
    cancel: threading.Event | None = None,
) -> list[dict[str, Any]]:
    """Run jobs through an arbitrary module-level worker function.

    Each job dict is passed to worker_fn, which must return a result dict.
    The worker_fn must be defined at module level (picklable).
    A 'elapsed_s' key is injected into each result if not already present.

    timeout_s is a whole-pool deadline (default _FUTURE_TIMEOUT_S), not
    per-job: size it to len(jobs)/workers × the slowest expected job.

    cancel is a cooperative cancellation flag: jobs not yet started when
    it is set return status CANCELLED; already-running jobs complete
    normally (subprocess kills mid-solve are unreliable). In the pool
    path the flag is polled on each completion, so cancellation waits
    for the next job to finish before taking effect.
    """
    total = len(jobs)
    if total == 0:
        return []

    results: list[dict[str, Any]] = []

    def _record(result: dict[str, Any]) -> None:
        results.append(result)
        if on_complete:
            on_complete(result, len(results), total)

    if workers <= 1:
        _worker_init(cache_dir)
        for i, job in enumerate(jobs):
            if cancel is not None and cancel.is_set():
                for skipped in jobs[i:]:
                    _record(_failure_result(
                        skipped, "CANCELLED", 0.0, "Cancelled before start"))
                break
            t0 = time.perf_counter()
            result = worker_fn(job)
            result.setdefault("elapsed_s", time.perf_counter() - t0)
            _record(result)
        return results

    def _harvest(future: Future[dict[str, Any]], idx: int) -> dict[str, Any]:
        try:
            result = future.result()
        except Exception as e:
            return _failure_result(jobs[idx], "ERROR", 0.0, f"Future failed: {e}")
        result.setdefault("elapsed_s", 0.0)
        return result

    ctx = multiprocessing.get_context("forkserver")
    executor = ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=_worker_init,
        initargs=(cache_dir,),
    )
    try:
        future_to_idx = {
            executor.submit(worker_fn, job): i for i, job in enumerate(jobs)
        }
        pending = set(future_to_idx)
        deadline_s = _FUTURE_TIMEOUT_S if timeout_s is None else timeout_s
        try:
            for future in as_completed(future_to_idx, timeout=deadline_s):
                pending.discard(future)
                if future.cancelled():
                    _record(_failure_result(
                        jobs[future_to_idx[future]], "CANCELLED", 0.0,
                        "Cancelled before start"))
                    continue
                _record(_harvest(future, future_to_idx[future]))
                if cancel is not None and cancel.is_set():
                    for f in pending:
                        f.cancel()
        except TimeoutError:
            for future in pending:
                idx = future_to_idx[future]
                if future.done():
                    _record(_harvest(future, idx))
                elif future.cancel():
                    _record(_failure_result(
                        jobs[idx], "TIMEOUT", 0.0,
                        f"Not started before {deadline_s}s pool deadline",
                    ))
                else:
                    _record(_failure_result(
                        jobs[idx], "TIMEOUT", float(deadline_s),
                        f"Exceeded {deadline_s}s pool deadline while running",
                    ))
            for proc in (getattr(executor, "_processes", None) or {}).values():
                proc.kill()
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    return results


# ---------------------------------------------------------------------------
# Geometry sweep
# ---------------------------------------------------------------------------

_AXIS_FIELDS = ("r_outer", "L_stk", "r_ag", "back_iron_thickness")
_STRATEGIES = ("proportional", "fixed_gap")


@dataclass(frozen=True)
class SweepAxis:
    field: Literal["r_outer", "L_stk", "r_ag", "back_iron_thickness"]
    start: float
    stop: float
    steps: int
    strategy: Literal["proportional", "fixed_gap"] = "proportional"

    def __post_init__(self) -> None:
        if self.field not in _AXIS_FIELDS:
            raise ValueError(
                f"unknown sweep field {self.field!r}; "
                f"allowed: {', '.join(_AXIS_FIELDS)}")
        if self.strategy not in _STRATEGIES:
            raise ValueError(
                f"unknown sweep strategy {self.strategy!r}; "
                f"allowed: {', '.join(_STRATEGIES)}")
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
            old_gap = abs(geo.r_stator - geo.r_magnet)
            factor = value / geo.r_outer
            scaled = _scale_proportional(geo, factor)
            return _set_airgap(scaled, old_gap), base_L_stk

    if axis.field == "r_ag":
        return _set_airgap(geo, value), base_L_stk

    if axis.field == "back_iron_thickness":
        return dataclasses.replace(geo, back_iron_thickness=value), base_L_stk

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
    from phasesweep.machines.motor import Motor
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
    base_motor: Motor,
    points: list[SweepPoint],
    model_keys: list[str],
    *,
    workers: int = 1,
    cache_dir: str | None = None,
    on_complete: Callable[[dict[str, Any], int, int], None] | None = None,
) -> list[SweepResult]:
    """Run models across sweep points, optionally in parallel.

    Returns a flat list of SweepResult (one per point × model).
    """
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
