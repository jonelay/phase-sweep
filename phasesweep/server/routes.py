"""REST endpoint handlers. The WebSocket endpoint and
connection manager come from phasesweep._view_sweep_server.ws."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

from phasesweep._extras import install_hint

try:
    import structlog
    import tomli_w
    from fastapi import APIRouter, HTTPException, Request, Response
except ImportError as e:
    raise ImportError(f"phasesweep.server needs {e.name}; {install_hint('server')}") from None

from phasesweep._view_sweep_server import FiniteJSONResponse
from phasesweep.machines.configs import motor_from_dict
from phasesweep.registry import MODEL_REGISTRY
from phasesweep.result_store import _row_run_id, record_is_current
from phasesweep.server.jobs import JobManager, designs
from phasesweep.sweep_types import resolve_model
from phasesweep.validation.measured import MeasuredResult, measured_run_result, validate_measured

__all__ = [
    "api",
    "cancel_job",
    "config_detail",
    "config_raw",
    "config_summary",
    "find_result",
    "get_config",
    "get_config_raw",
    "get_job",
    "get_result",
    "get_validation",
    "import_measured",
    "list_configs",
    "list_jobs",
    "list_models",
    "list_results",
    "mcp_activity",
    "mcp_designs",
    "model_defaults",
    "put_config",
    "result_index",
    "submit_job",
    "submittable_models",
]

log = structlog.get_logger()


api = APIRouter(prefix="/api", default_response_class=FiniteJSONResponse)


def _manager(request: Request) -> JobManager:
    return request.app.state.manager


# -- jobs ------------------------------------------------------

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


# -- results ---------------------------------------------------

@api.get("/results")
async def list_results(
    request: Request,
    motor: str | None = None,
    motor_config_id: str | None = None,
    model: str | None = None,
    source: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    return result_index(_manager(request), motor=motor,
                        motor_config_id=motor_config_id, model=model,
                        source=source, limit=limit)


def result_index(
    manager: JobManager, *, motor: str | None = None,
    motor_config_id: str | None = None, model: str | None = None,
    source: str | None = None, limit: int = 100,
) -> list[dict[str, Any]]:
    """Slim index of current stored results, most recent `limit` last."""
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
    return find_result(_manager(request), result_id)


def find_result(manager: JobManager, result_id: str) -> dict[str, Any]:
    """The current stored record for a run id; 404 if none."""
    found: dict[str, Any] | None = None
    for d in manager.store.load_all():
        if _row_run_id(d) != result_id:
            continue
        # Same staleness filter as load_slim — this scan bypasses it,
        # and the run id would otherwise match stale records.
        try:
            model = resolve_model(d)
        except KeyError:
            continue
        if not record_is_current(
                d.get("source", "computed"), model, d.get("model_version")):
            continue
        found = d  # last write wins, matching load_slim
    if found is None:
        raise HTTPException(404, f"no result with id {result_id!r}")
    return {"result_id": result_id, **found}


# -- validation summary ----------------------------

@api.get("/validation/{motor_name}")
async def get_validation(request: Request, motor_name: str) -> dict[str, Any]:
    """On-demand crossval summary over everything the store holds for a
    motor — same payload as the validate-job summary, so the dashboard
    can show validation state without submitting a job."""
    manager = _manager(request)
    if motor_name not in manager.motors:
        raise HTTPException(404, f"unknown motor config {motor_name!r}")
    return manager.validation_summary(motor_name)


# -- MCP agent monitor ----------------------------------------------------------

@api.get("/mcp/activity")
async def mcp_activity(request: Request, since: int = 0) -> dict[str, Any]:
    """MCP requests recorded after `since` (the dashboard's agent monitor
    polls this). A changed `boot` means the server restarted and the
    caller's cursor no longer applies."""
    activity = request.app.state.mcp_activity
    out: dict[str, Any] = activity.since(since)
    return out


@api.get("/mcp/designs")
async def mcp_designs(request: Request, version: str | None = None) -> dict[str, Any]:
    """Designs the MCP agent's jobs have produced (jobs.designs), for the
    Agent tab's design-space plot. When `version` matches the current
    one, `designs` is null: the caller's copy is up to date."""
    out = designs(_manager(request).jobs)
    if version == out["version"]:
        return {"version": version, "designs": None}
    return out


# -- job-type vocabulary -------------------------------------------------------

@api.get("/models")
async def list_models(request: Request, motor: str | None = None) -> list[dict[str, Any]]:
    """Submittable job types: computed registry keys plus the multi-point
    orchestrations. Feeds the dashboard submit form.
    `motor` adds per-model applicability to that config."""
    m = None
    if motor is not None:
        m = _manager(request).motors.get(motor)
        if m is None:
            raise HTTPException(404, f"unknown motor config {motor!r}")
    return submittable_models(m)


def _strip_motor_prefix(msg: str, motor: Any) -> str:
    """Validate errors start with "Motor '<name>': "; the caller named it."""
    return msg.removeprefix(f"Motor {motor.name!r}: ")


def _applicability(key: str, info: Any, needs: list[str], motor: Any) -> dict[str, Any]:
    """Whether `key` can run on `motor` as submitted with no params:
    `applicable` (the model's validate passes), `missing_params` (needed
    params the server cannot derive for this motor), and `reason` when
    either blocks a bare submit."""
    from phasesweep.server.jobs import derive_model_defaults

    if info.validate is not None:
        try:
            info.validate(motor)
        except ValueError as e:
            return {"applicable": False, "missing_params": [],
                    "reason": _strip_motor_prefix(str(e), motor)}
    try:
        derived = set(derive_model_defaults(motor, key))
        reason = None
    except ValueError as e:
        derived, reason = set(), _strip_motor_prefix(str(e), motor)
    # sim_plan for the two-mass sim derives from load_mech at submit.
    if key == "drive_sim_two_mass":
        derived.add("sim_plan")
    missing = [p for p in needs if p not in derived]
    if missing and reason is None:
        reason = f"pass params {missing}; the server cannot derive them"
    # fem defaults to nonlinear at submit; the others default linear,
    # which this rotor rejects (jobs._fem_config_kw).
    if key != "fem" and "nonlinear" in info.hash_fields:
        from phasesweep.solver_params import nonlinear_required_reason
        nl_reason = nonlinear_required_reason(motor.geometry)
        if nl_reason is not None:
            missing.append("nonlinear")
            reason = reason or f"pass params.nonlinear=true ({nl_reason})"
    return {"applicable": True, "missing_params": missing, "reason": reason}


def submittable_models(motor: Any = None) -> list[dict[str, Any]]:
    """Each submittable model with `needs_params` (params the run requires;
    the server derives some per motor, see model_defaults) and
    `optional_params` (accepted, with a default). With `motor`, each entry
    also carries `applicable`, `missing_params`, and `reason` for it."""
    from phasesweep.server.jobs import (
        _CONFIG_PARAM_KEYS,
        _PARAM_GATED_FIELDS,
        server_unsettable,
    )

    items: list[dict[str, Any]] = []
    for key, info in MODEL_REGISTRY.items():
        if info.source != "computed" or info.fn is None or server_unsettable(info):
            continue
        d: dict[str, Any] = {"key": key, "cost": info.cost, "kind": "single"}
        needs = sorted(info.needs_runtime & _CONFIG_PARAM_KEYS)
        optional = sorted((info.hash_fields & _PARAM_GATED_FIELDS) - set(needs))
        if needs:
            d["needs_params"] = needs
        if optional:
            d["optional_params"] = optional
        if motor is not None:
            d.update(_applicability(key, info, needs, motor))
        items.append(d)
    sweep: dict[str, Any] = {"key": "sweep", "cost": "varies", "kind": "multi",
                             "needs_params": ["axes"], "optional_params": ["model_keys"]}
    validate: dict[str, Any] = {"key": "validate", "cost": "varies", "kind": "multi",
                                "optional_params": ["models"]}
    if motor is not None:
        from phasesweep.machines.geometry import Geometry

        radial = isinstance(motor.geometry, Geometry)
        sweep.update(
            applicable=radial, missing_params=["axes"] if radial else [],
            reason="pass params.axes" if radial
            else "sweep needs a radial-flux [geometry] section")
        # Mirrors the default set in JobManager._validate_subtasks: fast,
        # applicable, and taking no per-run params (derivable or not).
        default = [d["key"] for d in items
                   if d["applicable"] and d["cost"] != "slow"
                   and not d.get("needs_params") and not d.get("optional_params")]
        validate.update(
            applicable=bool(default), missing_params=[],
            reason=None if default else "no fast model runs on this motor as-is; "
            "pass params.models")
        if default:
            validate["default_models"] = default
    items.append(sweep)
    items.append(validate)
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
    from phasesweep.server.jobs import derive_model_defaults
    return derive_model_defaults(motor, model_key)


# -- configs ----------------------------------------------------
# Anchor configs in motors_dir are read-only; writes go to the
# user-configs directory.

_CONFIG_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def _user_names(request: Request) -> set[str]:
    names: set[str] = request.app.state.user_config_names
    return names


def _geometry_topology(geo: Any) -> str | None:
    if geo is None:
        return None
    if hasattr(geo, "topology"):
        return geo.topology
    if hasattr(geo, "variant"):
        return f"axial:{geo.variant}"
    return None


def _config_summary(request: Request, name: str) -> dict[str, Any]:
    return config_summary(_manager(request), _user_names(request), name)


def config_summary(manager: JobManager, user_names: set[str], name: str) -> dict[str, Any]:
    m = manager.motors[name]
    return {
        "name": name,
        "motor_name": m.name,
        "motor_config_id": m.config_id,
        "n_p": m.n_p,
        "topology": _geometry_topology(m.geometry),
        "has_geometry": m.geometry is not None,
        "editable": name in user_names,
    }


def _config_path(settings: Any, user_names: set[str], name: str) -> Path:
    base = (settings.user_configs_dir if name in user_names
            else settings.motors_dir)
    return Path(base) / f"{name}.toml"


@api.get("/configs")
async def list_configs(request: Request) -> list[dict[str, Any]]:
    return [_config_summary(request, name)
            for name in sorted(_manager(request).motors)]


@api.get("/configs/{name}")
async def get_config(request: Request, name: str) -> dict[str, Any]:
    return config_detail(_manager(request), _user_names(request), name)


def config_detail(manager: JobManager, user_names: set[str], name: str) -> dict[str, Any]:
    m = manager.motors.get(name)
    if m is None:
        raise HTTPException(404, f"unknown motor config {name!r}")
    return {"name": name, "motor_config_id": m.config_id, "motor": m.to_dict(),
            "editable": name in user_names}


@api.get("/configs/{name}/raw")
async def get_config_raw(request: Request, name: str) -> dict[str, Any]:
    """The parsed TOML structure, section layout intact — what the config
    editor round-trips through PUT."""
    return config_raw(_manager(request), request.app.state.settings,
                      _user_names(request), name)


def config_raw(
    manager: JobManager, settings: Any, user_names: set[str], name: str,
) -> dict[str, Any]:
    if name not in manager.motors:
        raise HTTPException(404, f"unknown motor config {name!r}")
    path = _config_path(settings, user_names, name)
    try:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    except FileNotFoundError:
        raise HTTPException(404, f"config file for {name!r} no longer exists") from None
    return {"name": name, "editable": name in user_names, "raw": raw}


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


# -- measured import -------------------------------------------

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
