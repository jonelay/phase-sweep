"""Parallel execution and geometry dimension sweeps.

Parallel execution:
- execute_parallel(): dispatch registry-based model runs (sensitivity sweeps)
- execute_generic(): dispatch arbitrary module-level worker functions

Both fall back to in-process sequential execution when workers=1 and no
timeout_s is given, preserving debuggability. A timeout_s always routes
through the process pool: a deadline is only enforceable by killing the
worker process, so the inline path cannot honor one.

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

__all__ = [
    "SweepAxis",
    "SweepPoint",
    "SweepResult",
    "apply_axis",
    "apply_sweep_point",
    "axis_value",
    "execute_generic",
    "execute_parallel",
    "generate_grid",
    "run_sweep",
]

# ---------------------------------------------------------------------------
# Parallel execution
# ---------------------------------------------------------------------------


def _worker_init(cache_dir: str | None) -> None:
    """ProcessPoolExecutor initializer — runs once per worker process."""
    if cache_dir:
        from phasesweep.solvers.fem_mesh import set_disk_cache_dir
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


_DEFAULT_BATCH_SIZE = 128


def _run_batch_worker(job: dict[str, Any]) -> dict[str, Any]:
    """Run one batch job {"model_key": mk, "items": [...]} via batch_fn.

    Returns {"batch": True, "results": [per-item result, ...]} in item
    order. elapsed_s per item is the batch wall time / len(items).
    """
    from phasesweep.machines.motor import Motor
    from phasesweep.registry import MODEL_REGISTRY
    from phasesweep.sweep_types import RunConfig

    model_key = job["model_key"]
    items = job["items"]

    t0 = time.perf_counter()
    construct_errors: dict[int, str] = {}
    configs: list[RunConfig] = []
    config_idx: list[int] = []
    for i, item in enumerate(items):
        try:
            motor = Motor.from_dict(item["motor_dict"])
            configs.append(RunConfig(
                motor=motor, model=model_key, **item.get("config_kw", {})))
            config_idx.append(i)
        except Exception as e:
            construct_errors[i] = str(e)

    batch_fn = MODEL_REGISTRY[model_key].batch_fn
    assert batch_fn is not None  # callers group only batchable models
    try:
        outputs = batch_fn(configs) if configs else []
    except Exception as e:
        outputs = [e] * len(configs)

    per_item_s = (time.perf_counter() - t0) / len(items) if items else 0.0

    results: list[dict[str, Any]] = [
        _failure_result(item, "ERROR", per_item_s,
                        construct_errors.get(i, "no batch output"))
        for i, item in enumerate(items)
    ]
    for i, out in zip(config_idx, outputs):
        if isinstance(out, Exception):
            results[i] = _failure_result(items[i], "ERROR", per_item_s, str(out))
        else:
            results[i] = {
                **_passthrough(items[i]),
                "status": "OK",
                "metrics": out,
                "elapsed_s": per_item_s,
                "error_msg": None,
            }
    return {"batch": True, "results": results}


def _dispatch_worker(job: dict[str, Any]) -> dict[str, Any]:
    if "items" in job:
        return _run_batch_worker(job)
    return _run_worker(job)


def _split_batches(
    jobs: list[dict[str, Any]], batch_size: int, workers: int,
) -> list[dict[str, Any]]:
    """Group batchable jobs by model_key into ≤batch_size chunks.

    Produces at least min(n_batchable, workers) chunks overall so all
    workers stay fed; non-batchable jobs pass through unchanged.
    """
    from phasesweep.registry import MODEL_REGISTRY

    groups: dict[str, list[dict[str, Any]]] = {}
    mixed: list[dict[str, Any]] = []
    for job in jobs:
        info = MODEL_REGISTRY.get(job.get("model_key", ""))
        if info is not None and info.batch_fn is not None:
            groups.setdefault(job["model_key"], []).append(job)
        else:
            mixed.append(job)

    n_batchable = sum(len(g) for g in groups.values())
    if n_batchable:
        target_chunks = min(n_batchable, max(workers, 1))
        for mk, group in groups.items():
            n = len(group)
            k = max(-(-n // batch_size), -(-target_chunks * n // n_batchable))
            for c in range(k):
                lo = c * n // k
                hi = (c + 1) * n // k
                mixed.append({"model_key": mk, "items": group[lo:hi]})
    return mixed


def _expand_batch_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    if result.get("batch"):
        return result["results"]
    if "items" in result:
        # Batch job failed as a whole (future error, cancel, timeout):
        # fan the failure out to every item.
        return [
            {
                **_passthrough(item),
                "status": result.get("status", "ERROR"),
                "metrics": None,
                "elapsed_s": 0.0,
                "error_msg": result.get("error_msg"),
            }
            for item in result["items"]
        ]
    return [result]


def execute_parallel(
    jobs: list[dict[str, Any]],
    workers: int = 1,
    cache_dir: str | None = None,
    on_complete: Callable[[dict[str, Any], int, int], None] | None = None,
    timeout_s: float | None = None,
    cancel: threading.Event | None = None,
    batch_size: int | None = None,
) -> list[dict[str, Any]]:
    """Run registry-based model jobs via _run_worker.

    Thin wrapper around execute_generic — see that function for details.

    batch_size opts in to batched dispatch: jobs whose model has a
    batch_fn are grouped by model_key and run as chunks of at most
    batch_size items; results are flattened back to one dict per input
    job (elapsed_s per item = batch wall time / items). None (default)
    keeps single-job dispatch.
    """
    if batch_size is None:
        return execute_generic(
            _run_worker, jobs, workers=workers,
            cache_dir=cache_dir, on_complete=on_complete, timeout_s=timeout_s,
            cancel=cancel,
        )

    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    mixed = _split_batches(jobs, batch_size, workers)
    total = len(jobs)
    done_count = 0
    count_lock = threading.Lock()

    wrapped: Callable[[dict[str, Any], int, int], None] | None = None
    if on_complete is not None:
        def wrapped(result: dict[str, Any], _done: int, _total: int) -> None:
            nonlocal done_count
            for item in _expand_batch_result(result):
                with count_lock:
                    done_count += 1
                    n = done_count
                on_complete(item, n, total)

    raw = execute_generic(
        _dispatch_worker, mixed, workers=workers,
        cache_dir=cache_dir, on_complete=wrapped, timeout_s=timeout_s,
        cancel=cancel,
    )
    flat: list[dict[str, Any]] = []
    for r in raw:
        flat.extend(_expand_batch_result(r))
    return flat


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

    timeout_s is a whole-pool deadline, not per-job: size it to
    len(jobs)/workers × the slowest expected job. None (the default) sets
    no deadline in either path, so results do not depend on worker count.
    Passing it with workers <= 1 still runs a one-process pool — the
    inline path cannot kill a hung solver, so it never honors a deadline.

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

    if workers <= 1 and timeout_s is None:
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
        max_workers=max(1, workers),
        mp_context=ctx,
        initializer=_worker_init,
        initargs=(cache_dir,),
    )
    try:
        future_to_idx = {
            executor.submit(worker_fn, job): i for i, job in enumerate(jobs)
        }
        pending = set(future_to_idx)
        deadline_s = timeout_s
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
            assert deadline_s is not None  # as_completed only times out on one
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

_AXIS_FIELDS = ("r_outer", "L_stk", "airgap", "back_iron_thickness")
_STRATEGIES = ("proportional", "fixed_gap")


@dataclass(frozen=True)
class SweepAxis:
    """One swept geometry dimension.

    ``airgap`` is the radial gap thickness |r_stator - r_magnet|; the
    stator surface moves and the magnet stays fixed. ``r_ag`` is the
    mid-gap sampling radius, not a sweep axis.
    """

    field: Literal["r_outer", "L_stk", "airgap", "back_iron_thickness"]
    start: float
    stop: float
    steps: int
    strategy: Literal["proportional", "fixed_gap"] = "proportional"

    def __post_init__(self) -> None:
        if self.field == "r_ag":
            raise ValueError(
                "r_ag is the mid-gap sampling radius; the gap-thickness "
                "axis is 'airgap'")
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


def apply_sweep_point(base_motor: Motor, point: SweepPoint) -> Motor:
    """Apply a sweep point's geometry (and optional L_stk) to a motor."""
    motor = dataclasses.replace(base_motor, geometry=point.geometry)
    if point.L_stk is not None:
        motor = dataclasses.replace(motor, L_stk=point.L_stk)
    return motor


# In-plane dimensions that scale with a uniform lamination scale. Axial
# lengths (magnet_overhang, end_ring_axial_length) stay with L_stk, which
# is its own axis.
_SCALED_OPTIONAL = ("back_iron_thickness", "bridge_thickness",
                    "end_ring_radial_height")
_SCALED_BAR = ("bar_width", "bar_height", "bar_bridge_thickness",
               "bar_opening_width")


def _scale_proportional(geo: Geometry, factor: float) -> Geometry:
    kw = {f: getattr(geo, f) * factor for f in _SCALED_OPTIONAL
          if getattr(geo, f) is not None}
    kw.update({f: getattr(geo, f) * factor for f in _SCALED_BAR})
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
        **kw,
    )


def _set_airgap(geo: Geometry, new_gap: float) -> Geometry:
    """Set the gap thickness by moving the stator surface (magnet fixed,
    same rule as perturbation.perturb_motor). r_ag keeps its fractional
    position in the gap."""
    sign = 1.0 if geo.topology == "inrunner" else -1.0
    new_r_stator = geo.r_magnet + sign * new_gap
    frac = (geo.r_ag - geo.r_magnet) / (geo.r_stator - geo.r_magnet)
    return dataclasses.replace(
        geo,
        r_stator=new_r_stator,
        r_ag=geo.r_magnet + frac * (new_r_stator - geo.r_magnet),
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

    if axis.field == "airgap":
        return _set_airgap(geo, value), base_L_stk

    if axis.field == "back_iron_thickness":
        return dataclasses.replace(geo, back_iron_thickness=value), base_L_stk

    raise ValueError(f"unknown sweep field: {axis.field}")


def axis_value(motor: Motor, field: str) -> float | None:
    """The value of sweep-axis `field` as realized on `motor`, in m.

    Read back from the motor, not from the axis grid: a proportional
    r_outer step also moves the airgap, so this is the design's actual
    dimension. None when the motor has no such value."""
    geo = motor.geometry
    if field == "L_stk":
        return motor.L_stk
    if not isinstance(geo, Geometry):
        return None
    if field == "r_outer":
        return geo.r_outer
    if field == "airgap":
        return abs(geo.r_stator - geo.r_magnet)
    if field == "back_iron_thickness":
        return geo.back_iron_thickness
    raise ValueError(f"unknown sweep field: {field}")


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
    batch_size: int | None = _DEFAULT_BATCH_SIZE,
    fem_nonlinear: bool = True,
) -> list[SweepResult]:
    """Run models across sweep points, optionally in parallel.

    Models with a registry batch_fn are dispatched in batches of at most
    batch_size (None disables batching).

    fem_nonlinear sets RunConfig.nonlinear for the "fem" model only (other
    models run at their RunConfig defaults). The default keeps the
    saturating solve that sweeps have always used, so existing sweep run
    IDs are unchanged; False gives the linear-iron solve.

    Returns a flat list of SweepResult (one per point × model).
    """
    jobs: list[dict[str, Any]] = []
    for i, pt in enumerate(points):
        motor_dict = apply_sweep_point(base_motor, pt).to_dict()

        for mk in model_keys:
            config_kw = {"nonlinear": fem_nonlinear} if mk == "fem" else {}
            jobs.append({
                "motor_dict": motor_dict,
                "model_key": mk,
                "config_kw": config_kw,
                "point_idx": i,
            })

    raw = execute_parallel(
        jobs, workers=workers,
        cache_dir=cache_dir, on_complete=on_complete, batch_size=batch_size,
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
