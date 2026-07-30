"""REST and WebSocket endpoint handlers."""

from __future__ import annotations

import math
import re
import tomllib
from pathlib import Path
from typing import Any

import structlog
import tomli_w
from fastapi import APIRouter, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from phasesweep.configs import motor_from_dict
from phasesweep.measured import MeasuredResult, measured_run_result, validate_measured
from phasesweep.registry import MODEL_REGISTRY
from phasesweep.result_store import version_current
from phasesweep.server.jobs import JobManager
from phasesweep.server.protocol import ServerMsg
from phasesweep.sweep_types import RunConfig, _resolve_model, compute_run_id

log = structlog.get_logger()


def _json_finite(obj: Any) -> Any:
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, list):
        return [_json_finite(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _json_finite(v) for k, v in obj.items()}
    return obj


class FiniteJSONResponse(JSONResponse):
    """NaN/inf → null: starlette's JSONResponse refuses non-finite floats,
    and drive_sim legitimately produces NaN (e.g. t_settle when the speed
    never enters the settle band). Installed as the router default so
    every endpoint is covered without per-handler wrapping."""

    def render(self, content: Any) -> bytes:
        return super().render(_json_finite(content))


api = APIRouter(prefix="/api", default_response_class=FiniteJSONResponse)
ws_router = APIRouter()


def _manager(request: Request) -> JobManager:
    return request.app.state.manager


# -- jobs -----------------------------------------------------------------

@api.post("/jobs", status_code=201)
async def submit_job(request: Request, body: dict[str, Any]) -> dict[str, Any]:
    motor = body.get("motor")
    model = body.get("model")
    if not motor or not model:
        raise HTTPException(422, "body must have 'motor' (config name) and "
                                 "'model' (registry key, 'sweep', or 'validate')")
    params = body.get("params")
    if params is not None and not isinstance(params, dict):
        raise HTTPException(422, "'params' must be an object")
    log.info("job_request", motor=motor, model=model,
             param_keys=sorted(params.keys()) if params else [])
    try:
        job = _manager(request).submit(model, motor, params)
    except KeyError:
        raise HTTPException(404, f"unknown motor config {motor!r}") from None
    except (ValueError, TypeError) as e:
        raise HTTPException(400, str(e)) from None
    return job.to_dict()


@api.get("/jobs")
async def list_jobs(request: Request, status: str | None = None) -> list[dict[str, Any]]:
    jobs = _manager(request).jobs.values()
    return [j.to_dict(include_subtasks=False)
            for j in jobs if status is None or j.status == status]


@api.get("/jobs/{job_id}")
async def get_job(request: Request, job_id: str) -> dict[str, Any]:
    job = _manager(request).jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"unknown job {job_id!r}")
    return job.to_dict()


@api.delete("/jobs/{job_id}")
async def cancel_job(request: Request, job_id: str) -> dict[str, Any]:
    try:
        job = _manager(request).cancel(job_id)
    except KeyError:
        raise HTTPException(404, f"unknown job {job_id!r}") from None
    return job.to_dict()


# -- results ----------------------------------------------------------------

@api.get("/results")
async def list_results(
    request: Request,
    motor: str | None = None,
    motor_config_id: str | None = None,
    model: str | None = None,
    source: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    manager = _manager(request)
    m = None
    if motor is not None:
        m = manager.motors.get(motor)
        if m is None:
            raise HTTPException(404, f"unknown motor config {motor!r}")
    # `motor` applies the attach rule (store-level);
    # `motor_config_id` alone is a plain exact-config filter.
    slim = manager.store.load_slim(
        motor=m, motor_config_id=None if m is not None else motor_config_id,
        model=model, source=source)
    items = [
        {
            "result_id": rid,
            "motor_config_id": s.motor_config_id,
            "motor_name": s.motor_name,
            "model": s.model,
            "source": s.source,
            "status": s.status,
            "timestamp": s.timestamp,
        }
        for rid, s in slim.items()
    ]
    # File order is chronological; keep the most recent `limit`.
    return items[-limit:] if limit > 0 else items


@api.get("/results/{result_id}")
async def get_result(request: Request, result_id: str) -> dict[str, Any]:
    found: dict[str, Any] | None = None
    for d in _manager(request).store.load_all():
        try:
            rc = RunConfig.from_dict(d["config"])
        except (KeyError, ValueError, TypeError):
            continue
        # Same H1 staleness filter as load_slim — this scan bypasses it,
        # and the recomputed run id would otherwise match stale records.
        # Computed only: measured/published can't be regenerated.
        try:
            model = _resolve_model(d)
        except KeyError:
            model = rc.model
        if (d.get("source", "computed") == "computed"
                and not version_current(model, d.get("model_version"))):
            continue
        if compute_run_id(rc) == result_id:
            found = d  # last write wins, matching load_slim
    if found is None:
        raise HTTPException(404, f"no result with id {result_id!r}")
    return {"result_id": result_id, **found}


# -- validation summary ------------------------------------------------------

@api.get("/validation/{motor_name}")
async def get_validation(request: Request, motor_name: str) -> dict[str, Any]:
    """On-demand crossval summary over everything the store holds for a
    motor — same payload as the validate-job summary, so the dashboard
    can show validation state without submitting a job."""
    manager = _manager(request)
    if motor_name not in manager.motors:
        raise HTTPException(404, f"unknown motor config {motor_name!r}")
    return manager.validation_summary(motor_name)


# -- job-type vocabulary ------------------------------------------------------

@api.get("/models")
async def list_models() -> list[dict[str, Any]]:
    """Submittable job types: computed registry keys plus the multi-point
    orchestrations. Feeds the dashboard submit form."""
    from phasesweep.server.jobs import _PARAM_GATED_FIELDS

    items: list[dict[str, Any]] = []
    for key, info in MODEL_REGISTRY.items():
        if info.source != "computed" or info.fn is None:
            continue
        d: dict[str, Any] = {"key": key, "cost": info.cost, "kind": "single"}
        gates = sorted(info.hash_fields & _PARAM_GATED_FIELDS)
        if gates:
            d["needs_params"] = gates
        items.append(d)
    items.append({"key": "sweep", "cost": "varies", "kind": "multi"})
    items.append({"key": "validate", "cost": "varies", "kind": "multi"})
    return items


@api.get("/model-defaults/{motor_name}/{model_key}")
async def model_defaults(
    request: Request, motor_name: str, model_key: str,
) -> dict[str, Any]:
    """Derive sensible default params for a model+motor combination.
    Used by the editor to pre-fill param-gated fields before job submit."""
    manager = _manager(request)
    motor = manager.motors.get(motor_name)
    if motor is None:
        raise HTTPException(404, f"unknown motor config {motor_name!r}")
    try:
        return _derive_model_defaults(motor, model_key)
    except (ValueError, KeyError, TypeError) as e:
        raise HTTPException(400, str(e)) from None


def _derive_model_defaults(motor: Any, model_key: str) -> dict[str, Any]:
    if model_key == "drive_sim":
        from phasesweep.sim import plan_sim
        from phasesweep.solver_params import prepare_drive_sim
        params = prepare_drive_sim(motor)
        return {"sim_plan": plan_sim(params).to_dict()}
    if model_key == "thermal_duty":
        if motor.I_rated is None:
            raise ValueError(
                "thermal_duty defaults need I_rated on the motor")
        from phasesweep.rated_torque import magnet_torque_constant
        from phasesweep.solver_params import _resolve_psi_f
        psi_f = _resolve_psi_f(motor)
        tau = magnet_torque_constant(motor.n_p, psi_f) * motor.I_rated
        return {"duty_profile": [[tau, 3600.0]]}
    if model_key == "demag_screen":
        if motor.drive.MAX_I_S is None:
            raise ValueError(
                "demag_screen defaults need [drive] MAX_I_S for i_fault"
            )
        return {"i_fault": motor.drive.MAX_I_S}
    return {}


# -- configs ------------------------------------------------------------------
# Anchor configs in motors_dir are read-only; writes go to the
# user-configs directory.

_CONFIG_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def _user_names(request: Request) -> set[str]:
    names: set[str] = request.app.state.user_config_names
    return names


def _config_summary(request: Request, name: str) -> dict[str, Any]:
    m = _manager(request).motors[name]
    return {
        "name": name,
        "motor_name": m.name,
        "motor_config_id": m.config_id,
        "n_p": m.n_p,
        "topology": m.geometry.topology if m.geometry is not None else None,
        "has_geometry": m.geometry is not None,
        "editable": name in _user_names(request),
    }


def _config_path(request: Request, name: str) -> Path:
    settings = request.app.state.settings
    base = (settings.user_configs_dir if name in _user_names(request)
            else settings.motors_dir)
    return Path(base) / f"{name}.toml"


@api.get("/configs")
async def list_configs(request: Request) -> list[dict[str, Any]]:
    return [_config_summary(request, name)
            for name in sorted(_manager(request).motors)]


@api.get("/configs/{name}")
async def get_config(request: Request, name: str) -> dict[str, Any]:
    m = _manager(request).motors.get(name)
    if m is None:
        raise HTTPException(404, f"unknown motor config {name!r}")
    return {"name": name, "motor_config_id": m.config_id, "motor": m.to_dict(),
            "editable": name in _user_names(request)}


@api.get("/configs/{name}/raw")
async def get_config_raw(request: Request, name: str) -> dict[str, Any]:
    """The parsed TOML structure, section layout intact — what the config
    editor round-trips through PUT."""
    if name not in _manager(request).motors:
        raise HTTPException(404, f"unknown motor config {name!r}")
    path = _config_path(request, name)
    try:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    except FileNotFoundError:
        raise HTTPException(404, f"config file for {name!r} no longer exists") from None
    return {"name": name, "editable": name in _user_names(request), "raw": raw}


@api.put("/configs/{name}")
async def put_config(
    request: Request, name: str, body: dict[str, Any], response: Response,
) -> dict[str, Any]:
    """Validate a TOML-structure dict, write it to the user-configs
    directory, and hot-load the Motor so jobs see it immediately.
    Anchor names are refused — never silently overwrite."""
    if not _CONFIG_NAME_RE.match(name):
        raise HTTPException(400, f"invalid config name {name!r}: use letters, "
                                 "digits, '_' and '-'")
    manager = _manager(request)
    user_names = _user_names(request)
    if name in manager.motors and name not in user_names:
        raise HTTPException(409, f"{name!r} is a read-only anchor config; "
                                 "save under a different name")
    try:
        motor = motor_from_dict(body, label=f"config {name!r}", default_name=name)
    except (KeyError, ValueError, TypeError) as e:
        raise HTTPException(400, f"invalid config: {e}") from None

    settings = request.app.state.settings
    user_dir = Path(settings.user_configs_dir)
    user_dir.mkdir(parents=True, exist_ok=True)
    try:
        toml_bytes = tomli_w.dumps(body).encode()
    except (TypeError, ValueError) as e:
        raise HTTPException(400, f"not TOML-serializable: {e}") from None
    (user_dir / f"{name}.toml").write_bytes(toml_bytes)

    created = name not in manager.motors
    manager.motors[name] = motor
    user_names.add(name)
    response.status_code = 201 if created else 200
    log.info("config_saved", name=name, created=created,
             motor_config_id=motor.config_id)
    return _config_summary(request, name)


# -- measured import -----------------------------------------------------------

@api.post("/measured/{motor_name}", status_code=201)
async def import_measured(
    request: Request, motor_name: str, body: dict[str, Any],
    dataset_id: str | None = None,
) -> dict[str, Any]:
    manager = _manager(request)
    motor = manager.motors.get(motor_name)
    if motor is None:
        raise HTTPException(404, f"unknown motor config {motor_name!r}")

    body.setdefault("motor_name", motor.name)
    try:
        data = MeasuredResult.from_dict(body)
        validate_measured(data)
    except (KeyError, ValueError, TypeError) as e:
        raise HTTPException(400, f"invalid measured data: {e}") from None

    # Repeat captures of the same test must not collide to one run ID
    dataset_id = dataset_id or (Path(data.source_file).stem if data.source_file else "")
    if not dataset_id:
        raise HTTPException(400, "provide ?dataset_id= or a source_file to "
                                 "identify the dataset")

    result = measured_run_result(data, motor, dataset_id)
    result_id = manager.save_result(result)
    log.info("measured_imported", motor=motor_name, test_type=data.test_type,
             dataset_id=dataset_id, result_id=result_id)
    return {
        "result_id": result_id,
        "model": result.model,
        "source": result.source,
        "status": result.status,
        "dataset_id": dataset_id,
    }


# -- WebSocket ------------------------------------------------------------------

class ConnectionManager:
    """Tracks WS clients and their job subscriptions. Stateless across
    reconnects — clients re-subscribe and re-fetch via REST."""

    def __init__(self) -> None:
        self._subs: dict[WebSocket, set[str]] = {}

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._subs[ws] = set()

    def disconnect(self, ws: WebSocket) -> None:
        self._subs.pop(ws, None)

    def handle(self, ws: WebSocket, msg: Any) -> None:
        if not isinstance(msg, dict):
            return
        job_ids = msg.get("job_ids")
        if not isinstance(job_ids, list):
            return
        ids = {j for j in job_ids if isinstance(j, str)}
        kind = msg.get("type")
        if kind == "subscribe":
            self._subs[ws].update(ids)
        elif kind == "unsubscribe":
            self._subs[ws].difference_update(ids)

    async def broadcast(self, msg: ServerMsg) -> None:
        job_id = msg["job_id"]
        dead = []
        for ws, subs in list(self._subs.items()):
            if job_id not in subs:
                continue
            try:
                await ws.send_json(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


@ws_router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    cm: ConnectionManager = ws.app.state.ws_manager
    await cm.connect(ws)
    try:
        while True:
            try:
                msg = await ws.receive_json()
            except ValueError:
                await ws.close(code=1003)  # unsupported data: not JSON
                return
            cm.handle(ws, msg)
    except WebSocketDisconnect:
        pass
    finally:
        cm.disconnect(ws)
