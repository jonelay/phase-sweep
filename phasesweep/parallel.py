"""Parallel execution of independent model runs via ProcessPoolExecutor.

Provides two entry points:
- execute_parallel(): dispatch registry-based model runs (sensitivity sweeps)
- execute_generic(): dispatch arbitrary module-level worker functions

Both fall back to sequential execution when workers=1, preserving debuggability.
"""

from __future__ import annotations

import multiprocessing
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from typing import Any

_FUTURE_TIMEOUT_S = 600


def _worker_init(cache_dir: str | None) -> None:
    """ProcessPoolExecutor initializer — runs once per worker process."""
    if cache_dir:
        from phasesweep.fem_field import set_disk_cache_dir
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
        from phasesweep.motor import Motor
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
            # Stragglers past the pool deadline become TIMEOUT results
            # instead of blocking the pool forever; anything that finished
            # before the deadline is still harvested. Jobs that never
            # started (cancel succeeds) are distinguished from true
            # stragglers — elapsed_s=0 and a "not started" message, so a
            # queued job isn't misreported as having run the full deadline.
            # Best-effort: the executor prefetches workers+1 items into its
            # call queue and marks them RUNNING (uncancellable), so those
            # few report as stragglers even if no worker reached them.
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
            # cancel_futures can't stop already-running workers; kill them
            # so abandoned solves don't keep pinning cores after we return
            for proc in (getattr(executor, "_processes", None) or {}).values():
                proc.kill()
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    return results
