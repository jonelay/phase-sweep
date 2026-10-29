"""Job dataclass and JobManager over parallel.execute_parallel.

A Job fans out to sub-tasks, one RunConfig each. The runner consumes a
FIFO queue one job at a time — a single worker pool at a time is enough
for single-user localhost use. Sub-tasks whose run_id already has an OK
result in the ResultStore are served from cache; completed
sub-tasks persist to the store as they finish, so a restarted server
resumes via the same cache check.

Progress events cross from the executor thread into the event loop via
call_soon_threadsafe onto an asyncio queue; a single dispatcher task
broadcasts them in order.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import math
import threading
import time
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from phasesweep._extras import install_hint

try:
    import structlog
except ImportError as e:
    raise ImportError(f"phasesweep.server needs {e.name}; {install_hint('server')}") from None

from phasesweep._view_sweep_server.protocol import JobStatus, ServerMsg
from phasesweep.geo_parallel import SweepAxis, apply_sweep_point, execute_parallel, generate_grid
from phasesweep.machines.geometry import Geometry
from phasesweep.machines.motor import Motor
from phasesweep.registry import MODEL_REGISTRY, ModelInfo
from phasesweep.result_store import ResultStore
from phasesweep.sweep_types import RunConfig, RunResult, compute_run_id

__all__ = [
    "Job",
    "JobManager",
    "SubTask",
    "SubTaskStatus",
    "derive_model_defaults",
]

log = structlog.get_logger()

# RunConfig fields settable directly through job params; sim_plan,
# duty_profile, and load_mech need structured conversion.
_SCALAR_CONFIG_KEYS = frozenset({
    "maxh_fraction", "n_theta", "nonlinear", "j_s", "i_fault",
    "rotation", "cogging_points", "n_planes",
})
_CONFIG_PARAM_KEYS = _SCALAR_CONFIG_KEYS | {"sim_plan", "duty_profile", "load_mech"}


def server_unsettable(info: ModelInfo) -> frozenset[str]:
    """Runtime fields a model needs that no job param can supply."""
    return info.needs_runtime - _CONFIG_PARAM_KEYS

# hash_fields that imply a per-run param the caller must supply for the
# model to be runnable; gates the default model set for `validate` jobs.
_PARAM_GATED_FIELDS = frozenset({
    "sim_plan", "duty_profile", "i_fault", "load_mech",
    "rotation", "cogging_points",
})

_TERMINAL: frozenset[JobStatus] = frozenset({"completed", "failed", "cancelled"})


def _rated_duty_torque(motor: Motor) -> float:
    """Nameplate torque for the auto-derived S1 duty profile.

    k_T * I_rated for PM machines. For psi_f = 0 (SynRM) that is zero —
    a zero-torque duty profile would pass the thermal screen vacuously —
    so fall back to the full MTPA reluctance torque at I_rated (mirrors
    plan_sim's tau_peak fallback), and raise when there is no saliency
    to produce torque from.
    """
    from phasesweep.models.rated_torque import (
        magnet_torque_constant,
        mtpa_gamma,
        mtpa_torque,
    )
    from phasesweep.solver_params import resolve_psi_f, salient_pair

    assert motor.I_rated is not None  # callers gate on I_rated
    psi_f = resolve_psi_f(motor)
    tau = magnet_torque_constant(motor.n_p, psi_f) * motor.I_rated
    if tau > 0.0:
        return tau
    L_d, L_q = salient_pair(motor)
    if L_d is None or L_q is None or L_d == L_q:
        raise ValueError(
            "psi_f = 0 with no saliency (L_d == L_q or unset): "
            "machine cannot produce torque"
        )
    g = mtpa_gamma(psi_f, L_d, L_q, motor.I_rated)
    return mtpa_torque(motor.n_p, psi_f, L_d, L_q, motor.I_rated, g)


def derive_model_defaults(motor: Motor, model_key: str) -> dict[str, Any]:
    """Derive default param values for *model_key* from *motor* physics.

    Returns a dict of defaults (may be empty). Used by both the
    /defaults endpoint and the job-submission auto-derive path.
    """
    if model_key == "drive_sim":
        from phasesweep.simulation.params import prepare_drive_sim
        from phasesweep.simulation.sim import plan_sim
        return {"sim_plan": plan_sim(prepare_drive_sim(motor)).to_dict()}
    if model_key == "thermal_duty":
        if motor.I_rated is None:
            raise ValueError(
                "thermal_duty defaults need I_rated for duty_profile; "
                "set duty_profile explicitly or add I_rated"
            )
        return {"duty_profile": [[_rated_duty_torque(motor), 3600.0]]}
    if model_key == "demag_screen":
        if motor.drive.MAX_I_S is None:
            raise ValueError(
                "demag_screen defaults need [drive] MAX_I_S for i_fault; "
                "set i_fault explicitly or add MAX_I_S"
            )
        return {"i_fault": motor.drive.MAX_I_S}
    return {}


def _auto_derive_params(
    params: dict[str, Any], model_key: str, motor: Motor,
) -> None:
    """Fill in missing param-gated fields from motor physics."""
    if model_key == "drive_sim_two_mass":
        if params.get("load_mech") is None:
            raise ValueError(
                "drive_sim_two_mass requires params.load_mech"
            )
        if params.get("sim_plan") is None:
            from phasesweep.simulation.params import TwoMassLoad, prepare_drive_sim
            from phasesweep.simulation.sim import plan_two_mass_sim
            try:
                load = TwoMassLoad.from_dict(params["load_mech"])
            except (KeyError, TypeError, ValueError) as e:
                raise ValueError(f"bad load_mech: {e}") from e
            params["sim_plan"] = plan_two_mass_sim(
                prepare_drive_sim(motor), load,
            ).to_dict()
        return
    defaults = derive_model_defaults(motor, model_key)
    for k, v in defaults.items():
        if params.get(k) is None:
            params[k] = v


def _now() -> str:
    return datetime.now().isoformat()


def _dedupe_for_matrix(results: list[RunResult]) -> list[RunResult]:
    """Collapse the validation set to one result per (model, source,
    dataset_id), keeping the newest by timestamp. Callers pass
    current-only results so a physics bump can't leave stale physics in
    the matrix. Measured/published keep dataset_id in the key so
    genuinely distinct datasets stay separate. OK records outrank
    failed/timeout ones regardless of age — a newer TIMEOUT must not
    evict a good run from the matrix."""
    by_key: dict[tuple[str, str, str | None], RunResult] = {}
    for r in results:
        key = (r.model, r.source, r.config.dataset_id)
        prev = by_key.get(key)
        if prev is None or (r.status == "OK", r.timestamp) >= (
                prev.status == "OK", prev.timestamp):
            by_key[key] = r
    return list(by_key.values())


_CONFIG_TYPES: dict[str, type] = {
    "maxh_fraction": float, "n_theta": int, "nonlinear": bool, "j_s": float,
    "i_fault": float, "rotation": float, "cogging_points": int, "n_planes": int,
}
assert set(_CONFIG_TYPES) == _SCALAR_CONFIG_KEYS


def _coerce_param(key: str, v: Any) -> Any:
    """Parse a form/query string into the RunConfig field type.

    Numbers already typed by JSON pass through unchanged (converting
    0 -> 0.0 would re-key existing run IDs); RunConfig validates them.
    "false" must not reach RunConfig as a truthy string.
    """
    if not isinstance(v, str):
        return v
    want = _CONFIG_TYPES[key]
    s = v.strip()
    if want is bool:
        if s.lower() in ("true", "false"):
            return s.lower() == "true"
        raise ValueError(f"{key} must be true or false, got {v!r}")
    try:
        return want(s)
    except ValueError:
        raise ValueError(f"{key} must be {want.__name__}, got {v!r}") from None


def _build_config_kw(params: dict[str, Any]) -> dict[str, Any]:
    kw: dict[str, Any] = {k: _coerce_param(k, params[k])
                          for k in _SCALAR_CONFIG_KEYS if k in params}
    if params.get("duty_profile") is not None:
        try:
            kw["duty_profile"] = tuple(
                (float(t), float(dt)) for t, dt in params["duty_profile"]
            )
        except (TypeError, ValueError) as e:
            raise ValueError(f"bad duty_profile: {e}") from e
    if params.get("sim_plan") is not None:
        from phasesweep.simulation.params import SimPlan
        try:
            kw["sim_plan"] = SimPlan.from_dict(params["sim_plan"])
        except (KeyError, TypeError, ValueError) as e:
            # bare KeyError would masquerade as an unknown-motor 404
            raise ValueError(f"bad sim_plan: {e}") from e
    if params.get("load_mech") is not None:
        from phasesweep.simulation.params import TwoMassLoad
        try:
            kw["load_mech"] = TwoMassLoad.from_dict(params["load_mech"])
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"bad load_mech: {e}") from e
    return kw


def _config_kw_of(rc: RunConfig) -> dict[str, Any]:
    """RunConfig solver params as the config_kw dict _run_worker rebuilds from.

    Every field that reaches compute_run_id must appear here: the run ID
    is computed from the submitted RunConfig, but the worker rebuilds
    from this dict. A field present in one and missing from the other
    stores a result under an ID that describes a configuration the
    worker never ran (rotation and cogging_points were silently dropped
    this way before n_planes was added).
    """
    kw: dict[str, Any] = {
        "maxh_fraction": rc.maxh_fraction,
        "n_theta": rc.n_theta,
        "nonlinear": rc.nonlinear,
        "j_s": rc.j_s,
        "rotation": rc.rotation,
        "cogging_points": rc.cogging_points,
        "n_planes": rc.n_planes,
    }
    if rc.i_fault is not None:
        kw["i_fault"] = rc.i_fault
    if rc.sim_plan is not None:
        kw["sim_plan"] = rc.sim_plan
    if rc.duty_profile is not None:
        kw["duty_profile"] = rc.duty_profile
    if rc.load_mech is not None:
        kw["load_mech"] = rc.load_mech
    if rc.rotation != 0.0:
        kw["rotation"] = rc.rotation
    if rc.cogging_points != 12:
        kw["cogging_points"] = rc.cogging_points
    return kw


def _kw_for_model(base_kw: dict[str, Any], model_key: str) -> dict[str, Any]:
    """Drop param-gated keys that don't belong to *model_key*'s hash_fields."""
    info = MODEL_REGISTRY[model_key]
    drop = {k for k in _PARAM_GATED_FIELDS if k not in info.hash_fields}
    return {k: v for k, v in base_kw.items() if k not in drop}


def _check_cogging_j_s(model_key: str, params: dict[str, Any]) -> None:
    """Reject j_s != 0 at submit time for cogging_torque."""
    if model_key == "cogging_torque" and params.get("j_s", 0.0) != 0.0:
        raise ValueError(
            "cogging_torque requires j_s=0 (locked-rotor ripple "
            "under load is out of scope)"
        )


def _computed_info(model_key: str) -> ModelInfo:
    info = MODEL_REGISTRY.get(model_key)
    if info is None or info.source != "computed" or info.fn is None:
        raise ValueError(
            f"unknown computed model {model_key!r}; expected one of "
            f"{sorted(k for k, v in MODEL_REGISTRY.items() if v.source == 'computed')}"
        )
    return info


SubTaskStatus = Literal[
    "pending", "cached", "OK", "TIMEOUT", "ERROR", "CANCELLED",
]


@dataclass
class SubTask:
    run_id: str
    config: RunConfig
    model: str
    point_idx: int | None = None
    status: SubTaskStatus = "pending"
    error_msg: str | None = None
    elapsed_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "run_id": self.run_id,
            "model": self.model,
            "status": self.status,
            "elapsed_s": self.elapsed_s,
        }
        if self.point_idx is not None:
            d["point_idx"] = self.point_idx
        if self.error_msg is not None:
            d["error_msg"] = self.error_msg
        return d


@dataclass
class Job:
    id: str
    job_type: str
    motor_name: str
    params: dict[str, Any]
    subtasks: list[SubTask]
    status: JobStatus = "pending"
    error: str | None = None
    summary: dict[str, Any] | None = None
    created_at: str = field(default_factory=_now)
    started_at: str | None = None
    finished_at: str | None = None
    cancel_event: threading.Event = field(
        default_factory=threading.Event, repr=False, compare=False)

    @property
    def total(self) -> int:
        return len(self.subtasks)

    @property
    def completed_count(self) -> int:
        return sum(1 for st in self.subtasks if st.status != "pending")

    @property
    def result_ids(self) -> list[str]:
        return [st.run_id for st in self.subtasks if st.status in ("OK", "cached")]

    def to_dict(self, include_subtasks: bool = True) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "job_type": self.job_type,
            "motor_name": self.motor_name,
            "params": self.params,
            "status": self.status,
            "completed": self.completed_count,
            "total": self.total,
            "result_ids": self.result_ids,
            "error": self.error,
            "summary": self.summary,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }
        if include_subtasks:
            d["subtasks"] = [st.to_dict() for st in self.subtasks]
        return d


class JobManager:
    """In-memory job registry + serial async runner."""

    def __init__(
        self,
        motors: dict[str, Motor],
        store: ResultStore,
        *,
        workers: int = 4,
        subtask_timeout_s: float = 600.0,
        mesh_cache_dir: str | None = None,
        batch_size: int = 128,
    ) -> None:
        self.motors = motors
        self.store = store
        self.workers = workers
        self.subtask_timeout_s = subtask_timeout_s
        self.mesh_cache_dir = mesh_cache_dir
        self.batch_size = batch_size
        self.jobs: dict[str, Job] = {}
        self._queue: asyncio.Queue[Job] = asyncio.Queue()
        self._events: asyncio.Queue[ServerMsg] = asyncio.Queue()
        self._save_lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._runner: asyncio.Task[None] | None = None
        self._dispatcher: asyncio.Task[None] | None = None

    # -- lifecycle ----------------------------------------------------------

    async def start(
        self, broadcast: Callable[[ServerMsg], Coroutine[Any, Any, None]],
    ) -> None:
        self._loop = asyncio.get_running_loop()
        self._runner = asyncio.create_task(self._run_loop())
        self._dispatcher = asyncio.create_task(self._dispatch_loop(broadcast))

    async def stop(self) -> None:
        """Signal running jobs to drain, then stop the runner and dispatcher.

        The executor thread of a running job keeps draining in the
        background (already-running sub-tasks complete);
        completed results are already persisted, so a restart resumes
        via the cache check.
        """
        for job in self.jobs.values():
            if job.status in ("pending", "running"):
                job.cancel_event.set()
        for task in (self._runner, self._dispatcher):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    # -- submission ---------------------------------------------------------

    def submit(
        self, job_type: str, motor_name: str, params: dict[str, Any] | None = None,
    ) -> Job:
        """Build and enqueue a job. Raises KeyError for an unknown motor,
        ValueError for a bad job type or params. Full cache hits complete
        immediately without touching the queue."""
        # Typed once up front so the submit-time checks (_check_cogging_j_s,
        # _auto_derive_params) compare numbers, not strings.
        params = {k: (_coerce_param(k, v) if k in _SCALAR_CONFIG_KEYS else v)
                  for k, v in (params or {}).items()}
        motor = self.motors[motor_name]
        unique: dict[str, SubTask] = {}
        for st in self._build_subtasks(job_type, motor, params):
            unique.setdefault(st.run_id, st)
        subtasks = list(unique.values())
        job = Job(
            id=uuid.uuid4().hex[:12], job_type=job_type,
            motor_name=motor_name, params=params, subtasks=subtasks,
        )
        self.jobs[job.id] = job
        log.info("job_submitted", job_id=job.id, job_type=job_type,
                 motor=motor_name, subtasks=len(subtasks))

        self._apply_cache(job)
        if all(st.status == "cached" for st in job.subtasks):
            job.status = "completed"
            job.started_at = job.finished_at = _now()
            if job.job_type == "validate":
                job.summary = self.validation_summary(job.motor_name)
            self._emit({"type": "job_complete", "job_id": job.id,
                        "result_ids": job.result_ids})
            log.info("job_complete", job_id=job.id, cached=True)
        else:
            self._queue.put_nowait(job)
        return job

    def cancel(self, job_id: str) -> Job:
        """Cancel a job. Terminal jobs are a no-op; a pending
        job cancels immediately; a running job stops issuing new sub-tasks
        (in-flight ones complete)."""
        job = self.jobs[job_id]
        if job.status in _TERMINAL:
            return job
        job.cancel_event.set()
        if job.status == "pending":
            self._finish_cancelled(job)
        return job

    # -- sub-task construction ----------------------------------------------

    def _build_subtasks(
        self, job_type: str, motor: Motor, params: dict[str, Any],
    ) -> list[SubTask]:
        if job_type == "sweep":
            return self._sweep_subtasks(motor, params)
        if job_type == "validate":
            return self._validate_subtasks(motor, params)

        info = _computed_info(job_type)
        unsettable = server_unsettable(info)
        if unsettable:
            raise ValueError(
                f"model {job_type!r} needs RunConfig {sorted(unsettable)}, which "
                "the server cannot set; run it through the Python API"
            )
        self._check_params(params, allowed_extra=frozenset())
        foreign = {k for k in _PARAM_GATED_FIELDS
                   if k not in info.hash_fields and params.get(k) is not None}
        if foreign:
            raise ValueError(
                f"{sorted(foreign)} not accepted by model {job_type!r}"
            )
        if info.validate is not None:
            info.validate(motor)
        _check_cogging_j_s(job_type, params)
        _auto_derive_params(params, job_type, motor)
        config = RunConfig(motor=motor, model=job_type, **_build_config_kw(params))
        return [SubTask(run_id=compute_run_id(config), config=config, model=job_type)]

    def _sweep_subtasks(self, motor: Motor, params: dict[str, Any]) -> list[SubTask]:
        self._check_params(params, allowed_extra=frozenset({"model_keys", "axes"}))
        if motor.geometry is None:
            raise ValueError("sweep requires a motor with a [geometry] section")
        if not isinstance(motor.geometry, Geometry):
            raise ValueError("sweep supports radial-flux geometry only")

        model_keys = params.get("model_keys") or ["analytical"]
        for mk in model_keys:
            info = _computed_info(mk)
            if info.validate is not None:
                info.validate(motor)
            _check_cogging_j_s(mk, params)
        raw_axes = params.get("axes")
        if not raw_axes:
            raise ValueError("sweep requires params.axes (list of "
                             "{field, start, stop, steps[, strategy]})")
        try:
            axes = [SweepAxis(**a) for a in raw_axes]
        except TypeError as e:
            raise ValueError(f"bad sweep axis: {e}") from e

        points = generate_grid(motor.geometry, axes, base_L_stk=motor.L_stk)
        if not points:
            raise ValueError("sweep grid is empty (every combination produced "
                             "an invalid geometry)")

        base_kw = _build_config_kw(params)
        subtasks: list[SubTask] = []
        for i, pt in enumerate(points):
            m = apply_sweep_point(motor, pt)
            for mk in model_keys:
                kw = _kw_for_model(base_kw, mk)
                if mk == "fem":
                    kw.setdefault("nonlinear", True)  # mirrors geo_parallel.run_sweep
                config = RunConfig(motor=m, model=mk, **kw)
                subtasks.append(SubTask(
                    run_id=compute_run_id(config), config=config,
                    model=mk, point_idx=i,
                ))
        return subtasks

    def _validate_subtasks(self, motor: Motor, params: dict[str, Any]) -> list[SubTask]:
        self._check_params(params, allowed_extra=frozenset({"models"}))
        requested = params.get("models")
        if requested:
            models = list(requested)
            for mk in models:
                info = _computed_info(mk)
                if info.validate is not None:
                    info.validate(motor)
                _check_cogging_j_s(mk, params)
        else:
            # Default: every fast computed model that is runnable as-is —
            # param-gated models join only when the caller supplies the param.
            models = []
            for key, info in MODEL_REGISTRY.items():
                if info.source != "computed" or info.fn is None or info.cost == "slow":
                    continue
                # needs_runtime covers fields the server cannot set
                # (w_mech, tau_ref, T_coolant): those models never default in.
                gates = (info.hash_fields & _PARAM_GATED_FIELDS) | info.needs_runtime
                if any(params.get(g) is None for g in gates):
                    continue
                if info.validate is not None:
                    try:
                        info.validate(motor)
                    except ValueError:
                        continue
                models.append(key)
            if not models:
                raise ValueError(
                    f"no runnable models for motor {motor.name!r} — "
                    f"pass params.models explicitly"
                )

        base_kw = _build_config_kw(params)
        subtasks = []
        for mk in models:
            kw = _kw_for_model(base_kw, mk)
            config = RunConfig(motor=motor, model=mk, **kw)
            subtasks.append(SubTask(
                run_id=compute_run_id(config), config=config, model=mk))
        return subtasks

    def _check_params(
        self, params: dict[str, Any], allowed_extra: frozenset[str],
    ) -> None:
        unknown = set(params) - _CONFIG_PARAM_KEYS - allowed_extra
        if unknown:
            raise ValueError(
                f"unknown params {sorted(unknown)}; allowed: "
                f"{sorted(_CONFIG_PARAM_KEYS | allowed_extra)}"
            )

    # -- execution ----------------------------------------------------------

    async def _run_loop(self) -> None:
        while True:
            job = await self._queue.get()
            if job.status == "cancelled":
                continue
            try:
                await self._run_job(job)
            except Exception as e:
                job.status = "failed"
                job.error = str(e)
                job.finished_at = _now()
                self._emit({"type": "job_failed", "job_id": job.id, "error": str(e)})
                log.error("job_crashed", job_id=job.id, error=str(e))

    async def _dispatch_loop(
        self, broadcast: Callable[[ServerMsg], Coroutine[Any, Any, None]],
    ) -> None:
        while True:
            msg = await self._events.get()
            try:
                await broadcast(msg)
            except Exception as e:
                log.error("broadcast_failed", error=str(e))

    async def _run_job(self, job: Job) -> None:
        job.status = "running"
        job.started_at = _now()
        t0 = time.perf_counter()

        # Re-check the cache at run time: an earlier queued job may have
        # produced results since submit (partial cache hits).
        self._apply_cache(job)
        pending = [st for st in job.subtasks if st.status == "pending"]
        log.info("job_started", job_id=job.id, subtasks=job.total,
                 cached=job.total - len(pending))

        if pending and not job.cancel_event.is_set():
            by_run_id = {st.run_id: st for st in pending}
            worker_jobs = [
                {
                    "motor_dict": st.config.motor.to_dict(),
                    "model_key": st.model,
                    "config_kw": _config_kw_of(st.config),
                    "run_id": st.run_id,
                }
                for st in pending
            ]

            def on_complete(result: dict[str, Any], done: int, total: int) -> None:
                # Runs on the executor thread — persist, then hand the
                # event to the loop.
                st = by_run_id[result["run_id"]]
                raw_status = result.get("status", "ERROR")
                if raw_status == "CANCELLED":
                    # Never started: no result to persist.
                    st.status = "CANCELLED"
                    return
                if raw_status not in ("OK", "TIMEOUT", "ERROR"):
                    raw_status = "ERROR"
                st.status = raw_status
                st.error_msg = result.get("error_msg")
                st.elapsed_s = result.get("elapsed_s", 0.0)
                latest = None
                if st.status in ("OK", "ERROR", "TIMEOUT"):
                    self.save_result(RunResult(
                        config=st.config, model=st.model,
                        status=st.status,
                        metrics=result.get("metrics"), elapsed_s=st.elapsed_s,
                        error_msg=st.error_msg,
                    ))
                    if st.status == "OK":
                        latest = st.run_id
                self._emit_threadsafe({
                    "type": "job_progress", "job_id": job.id,
                    "completed": job.completed_count, "total": job.total,
                    "latest_result_id": latest,
                })
                log.info("subtask_complete", job_id=job.id, model=st.model,
                         status=st.status,
                         progress=f"{job.completed_count}/{job.total}",
                         elapsed_s=round(st.elapsed_s, 3))

            # Cogging runs N FEM solves per subtask; scale the budget.
            max_multiplier = max(
                (st.config.cogging_points if st.model == "cogging_torque" else 1
                 for st in pending),
                default=1,
            )
            effective_subtask_s = self.subtask_timeout_s * max_multiplier
            timeout_s = max(1, math.ceil(len(pending) / self.workers)) * effective_subtask_s
            # Dedicated daemon thread, not asyncio.to_thread: the default
            # executor's non-daemon threads are joined at loop shutdown, so
            # a draining job would block Ctrl-C until its sub-task finished.
            done = asyncio.Event()
            loop = asyncio.get_running_loop()
            drain_error: list[str] = []

            def _drain() -> None:
                try:
                    execute_parallel(
                        worker_jobs, workers=self.workers,
                        cache_dir=self.mesh_cache_dir, on_complete=on_complete,
                        timeout_s=timeout_s, cancel=job.cancel_event,
                        batch_size=self.batch_size,
                    )
                except Exception as e:
                    drain_error.append(f"{type(e).__name__}: {e}")
                    log.error("job_drain_failed", job_id=job.id, error=drain_error[0])
                finally:
                    with contextlib.suppress(RuntimeError):
                        loop.call_soon_threadsafe(done.set)

            threading.Thread(
                target=_drain, name=f"job-drain-{job.id}", daemon=True).start()
            await done.wait()

            if drain_error:
                # The executor itself failed: sub-tasks it never reported
                # have no result, and a cached sibling must not turn the
                # job into a success.
                for st in pending:
                    if st.status == "pending":
                        st.status = "ERROR"
                        st.error_msg = drain_error[0]
                job.status = "failed"
                job.error = drain_error[0]
                job.finished_at = _now()
                self._emit({"type": "job_failed", "job_id": job.id, "error": job.error})
                return

        if job.cancel_event.is_set():
            self._finish_cancelled(job)
            return

        job.finished_at = _now()
        ok = [st for st in job.subtasks if st.status in ("OK", "cached")]
        failed = [st for st in job.subtasks if st.status in ("ERROR", "TIMEOUT")]
        if not ok:
            job.status = "failed"
            job.error = (failed[0].error_msg if failed else None) or "all sub-tasks failed"
            self._emit({"type": "job_failed", "job_id": job.id, "error": job.error})
            log.info("job_failed", job_id=job.id, error=job.error)
            return

        if job.job_type == "validate":
            job.summary = self.validation_summary(job.motor_name)
        job.status = "completed"
        self._emit({"type": "job_complete", "job_id": job.id,
                    "result_ids": job.result_ids})
        log.info("job_complete", job_id=job.id, ok=len(ok), failed=len(failed),
                 total_s=round(time.perf_counter() - t0, 3))

    def _apply_cache(self, job: Job) -> None:
        slim = self.store.load_slim()
        for st in job.subtasks:
            if st.status != "pending":
                continue
            hit = slim.get(st.run_id)
            if hit is not None and hit.status == "OK":
                st.status = "cached"

    def _finish_cancelled(self, job: Job) -> None:
        for st in job.subtasks:
            if st.status == "pending":
                st.status = "CANCELLED"
        job.status = "cancelled"
        job.finished_at = _now()
        self._emit({"type": "job_cancelled", "job_id": job.id})
        log.info("job_cancelled", job_id=job.id)

    def validation_summary(self, motor_name: str) -> dict[str, Any]:
        """Crossval rows + diagnosis for everything the store holds on a
        motor. Feeds the validate-job summary and GET /api/validation.
        Raises KeyError for an unknown motor.

        Computed results match by exact motor config_id; measured and
        published results match by Motor NAME — the measurement describes
        the hardware, so it stays attached while the user edits parameters.
        derived_params in the payload tags datasets whose
        listed Motor params were derived from that same data: agreement
        there is an echo, not independent validation.

        Duplicate runs of one model would reach compare_all and widen the
        matrix with model↔model self-pairs and doubled columns — deduped
        to latest-per-(model, source, dataset_id) here at the endpoint;
        the CLI compare_all keeps its every-pair semantics."""
        from phasesweep.validation.crossval import compare_all, diagnose

        motor = self.motors[motor_name]
        results = _dedupe_for_matrix(
            self.store.load_results(motor=motor, current_only=True))
        rows = compare_all(results)
        derived: dict[str, list[str]] = {}
        for r in results:
            params = (r.metrics or {}).get("_derived_params")
            if r.source != "computed" and params:
                ds = r.config.dataset_id or "unknown"
                derived[ds] = sorted(set(derived.get(ds, [])) | set(params))
        return {
            "diagnosis": diagnose(results),
            "n_results": len(results),
            "models": sorted({r.model for r in results}),
            "derived_params": derived,
            "rows": [dataclasses.asdict(r) for r in rows],
        }

    # -- persistence + events -----------------------------------------------

    def save_result(self, result: RunResult) -> str:
        """Single choke point for store writes (single-writer discipline);
        also used by the measured-import route. Returns the run_id."""
        with self._save_lock:
            self.store.save(result)
        return compute_run_id(result.config)

    def _emit(self, msg: ServerMsg) -> None:
        self._events.put_nowait(msg)

    def _emit_threadsafe(self, msg: ServerMsg) -> None:
        if self._loop is None:
            return
        with contextlib.suppress(RuntimeError):  # loop already closed at shutdown
            self._loop.call_soon_threadsafe(self._events.put_nowait, msg)
