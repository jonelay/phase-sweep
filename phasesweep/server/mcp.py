"""MCP tools over the running server's JobManager and result store.

Mounted at ``/mcp`` (streamable HTTP, stateless, JSON responses) on the
same app as the dashboard, so MCP-submitted jobs share the dashboard's
worker pool, appear in its job list, and outlive the client connection.
Slow models run as jobs: submit, poll ``get_job``, then read results.

Host/Origin checking is owned by the view-sweep LocalOnlyMiddleware that
wraps the whole app; the SDK's own DNS-rebinding check is disabled so the
two allowlists cannot disagree (``--host`` widens only the outer one).
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from collections import deque
from datetime import UTC, datetime
from typing import Any

from phasesweep._extras import MissingExtra

try:
    from fastapi import FastAPI, HTTPException
    from mcp.server.context import CallNext, HandlerResult, ServerRequestContext
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
    from mcp.server.transport_security import TransportSecuritySettings
    from pydantic import BaseModel
    from starlette.routing import BaseRoute
except ImportError as e:
    raise MissingExtra(f"phasesweep.server needs {e.name}", "server") from None

from phasesweep.server import jobs as jobs_mod
from phasesweep.server import routes
from phasesweep.server.jobs import _TERMINAL, Job, JobManager
from phasesweep.trust import assess_record, headline_record

__all__ = ["MCP_PATH", "McpActivity", "build_mcp", "mcp_routes", "summarize_arrays"]

MCP_PATH = "/mcp"

# Activity entries kept for the dashboard's agent monitor, and the cap on
# the JSON length of the arguments echoed into each entry.
_ACTIVITY_CAP = 500
_ARGS_CHARS = 400
_SDK_ERROR_PREFIX = re.compile(r"^Error executing tool \S+: ")

# Lists longer than this are replaced by a summary in get_result unless
# full=True: a FEM record's waveforms and field grid run to ~130 kB.
_ARRAY_LIMIT = 32

# Per-result outcomes (headline + caveat codes) returned with a terminal
# job; sweeps beyond this are left to result queries. Results whose
# caveats are copied into a job's activity entry.
_JOB_RESULTS = 20
_JOB_CAVEAT_RESULTS = 5
# get_job long-poll: cap per call (below common MCP client tool timeouts)
# and the status re-check interval while holding.
GET_JOB_MAX_WAIT_S = 60.0
_WAIT_POLL_S = 0.25

_INSTRUCTIONS = """\
phase-sweep motor design models. Typical flow: list_motors -> list_models
-> (model_defaults for param-gated models) -> submit_job -> get_job(wait_s=60)
until status is completed/failed/cancelled -> get_result for each result_id.
Sweeps: submit_job(dry_run=True) first to see sub-task count and time
estimate; read a running or finished sweep with sweep_table(job_id).
Costs: fast models finish in about a second; 'slow' FEM models can take
minutes per sub-task. Identical runs are served from the result-store
cache. Units are SI unless a field name says otherwise. PM-machine
currents, voltages and flux linkages are peak per phase unless the key
ends in _rms; induction-machine currents and voltages are rms per phase
and their power and torque are machine totals; speeds are
mechanical rad/s.
Every OK result carries `headline` (the model's key metrics with units)
and `caveats` (severity info|warn, derived from the result's own
provenance keys, plus the model's inherent limits). When you report a
number, report its warn-level caveats with it: e.g. a torque_speed peak
torque with envelope_source=analytical is a constant-L over-prediction."""


def _job_view(job: Job) -> dict[str, Any]:
    """Job dict; once terminal, plus per-result headline + caveat codes."""
    d = job.to_dict(include_subtasks=False)
    if job.status in _TERMINAL and job.result_ids:
        d["results"] = job.outcomes(_JOB_RESULTS)
        if len(job.result_ids) > _JOB_RESULTS:
            d["results_truncated"] = True
    return d


def summarize_arrays(value: Any, limit: int = _ARRAY_LIMIT) -> Any:
    """Replace lists longer than `limit` with {len, min, max, first, last}.

    Numeric lists get min/max; nested or mixed lists report only length
    and endpoints. Dicts and short lists are walked recursively."""
    if isinstance(value, dict):
        return {k: summarize_arrays(v, limit) for k, v in value.items()}
    if isinstance(value, list):
        if len(value) <= limit:
            return [summarize_arrays(v, limit) for v in value]
        out: dict[str, Any] = {"_summarized": True, "len": len(value)}
        nums = [v for v in value if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if len(nums) == len(value):
            finite = [v for v in nums if v == v and abs(v) != float("inf")]
            if finite:
                out["min"], out["max"] = min(finite), max(finite)
            out["first"], out["last"] = value[0], value[-1]
        return out
    return value


def _http_error(e: HTTPException) -> ToolError:
    """Route helpers raise HTTPException. Only ToolError text reaches the
    model (other exceptions surface as a bare 'Error executing tool'), so
    the status code is folded into its message."""
    return ToolError(f"{e.status_code}: {e.detail}")


class McpActivity:
    """In-memory log of MCP requests for the dashboard's agent monitor.

    Entries carry a monotonically increasing `seq`; readers poll with the
    last seq they saw. `boot` changes per process, so a client that sees
    a new boot id knows its seq cursor is stale and starts over. Not
    persisted: the result store and job list are the durable record."""

    def __init__(self, cap: int = _ACTIVITY_CAP) -> None:
        self.boot = uuid.uuid4().hex[:8]
        self._seq = 0
        self._entries: deque[dict[str, Any]] = deque(maxlen=cap)

    def record(self, entry: dict[str, Any]) -> None:
        self._seq += 1
        self._entries.append({"seq": self._seq, **entry})

    def record_job(self, job: Job) -> None:
        """JobManager terminal listener: log how an MCP-submitted job
        ended, with the first result's headline and every result's
        provenance caveats (model limits omitted)."""
        if job.origin != "mcp":
            return
        entry: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "kind": "job", "job_id": job.id, "job_type": job.job_type,
            "motor": job.motor_name, "status": job.status,
            "ok": job.status == "completed",
            "n_results": len(job.result_ids), "total": job.total,
            "n_cached": sum(st.status == "cached" for st in job.subtasks),
        }
        if job.started_at and job.finished_at:
            entry["s"] = round((datetime.fromisoformat(job.finished_at)
                                - datetime.fromisoformat(job.started_at)).total_seconds(), 2)
        if job.error:
            entry["error"] = job.error
        done = [st for st in job.subtasks if st.status in ("OK", "cached")]
        if done:
            entry["headline"] = done[0].headline or []
            seen: dict[tuple[str, str, str], dict[str, str]] = {}
            for st in done[:_JOB_CAVEAT_RESULTS]:
                for c in st.caveats or []:
                    if c["code"] != "limits":
                        seen.setdefault((st.model, c["code"], c["text"]),
                                        {**c, "model": st.model})
            entry["caveats"] = list(seen.values())
        self.record(entry)

    def since(self, seq: int) -> dict[str, Any]:
        return {"boot": self.boot, "seq": self._seq,
                "entries": [e for e in self._entries if e["seq"] > seq]}


def _short_json(value: Any, limit: int = _ARGS_CHARS) -> str:
    s = json.dumps(value, default=str, separators=(",", ":"))
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _result_fields(result: HandlerResult) -> dict[str, Any]:
    """ok/error/summary/job_id from a tools/call result, without keeping
    the payload itself."""
    d = (result.model_dump(mode="json", by_alias=True, exclude_none=True)
         if isinstance(result, BaseModel) else dict(result or {}))
    if d.get("isError"):
        text = next((c.get("text", "") for c in d.get("content", [])
                     if c.get("type") == "text"), "")
        # the entry already names the tool; drop the SDK's echo of it
        return {"ok": False, "error": _SDK_ERROR_PREFIX.sub("", text)}
    sc = d.get("structuredContent")
    out: dict[str, Any] = {"ok": True}
    if isinstance(sc, dict) and set(sc) == {"result"} and isinstance(sc["result"], list):
        out["summary"] = f"{len(sc['result'])} items"
    elif isinstance(sc, dict):
        if sc.get("dry_run"):
            if sc["estimate_complete"]:
                est = f"~{sc['estimate_s']:g} s"
            elif sc["estimate_s"]:
                est = f"≥{sc['estimate_s']:g} s"
            else:
                est = "no run-time history"
            out["summary"] = (f"dry run: {sc['subtasks']} sub-tasks, "
                              f"{sc['cached']} cached, {est}")
        elif "columns" in sc and "rows" in sc:  # sweep_table
            out["summary"] = (f"sweep table: {len(sc['rows'])}/{sc['n_rows']} rows, "
                              f"{sc['completed']}/{sc['total']} done")
        elif "id" in sc and "status" in sc:  # a job
            out["job_id"] = sc["id"]
            out["summary"] = f"job {sc['id']} {sc['status']}"
        elif "result_id" in sc:
            out["summary"] = f"result {sc['result_id']} {sc.get('status', '')}".rstrip()
        else:
            out["summary"] = f"{len(sc)} fields"
    return out


def _observer(activity: McpActivity):
    """ServerMiddleware that records initialize and tools/call requests."""
    async def observe(ctx: ServerRequestContext[Any, Any], call_next: CallNext) -> HandlerResult:
        if ctx.method not in ("tools/call", "initialize"):
            return await call_next(ctx)
        params = dict(ctx.params or {})
        entry: dict[str, Any] = {"ts": datetime.now(UTC).isoformat(timespec="milliseconds")}
        if ctx.method == "initialize":
            info = params.get("clientInfo") or {}
            entry |= {"kind": "connect",
                      "client": f"{info.get('name', '?')} {info.get('version', '')}".strip()}
        else:
            entry |= {"kind": "tool", "tool": params.get("name", "?"),
                      "args": _short_json(params.get("arguments") or {})}
        t0 = time.monotonic()
        try:
            result = await call_next(ctx)
        except Exception as e:
            entry |= {"ok": False, "error": f"{type(e).__name__}: {e}",
                      "ms": round(1000 * (time.monotonic() - t0), 1)}
            activity.record(entry)
            raise
        entry["ms"] = round(1000 * (time.monotonic() - t0), 1)
        entry |= _result_fields(result) if entry["kind"] == "tool" else {"ok": True}
        activity.record(entry)
        return result
    return observe


def build_mcp(app: FastAPI, activity: McpActivity | None = None) -> MCPServer:
    """MCP server whose tools read `app.state` at call time (the
    JobManager exists only once the app lifespan has started). Requests
    are recorded to `activity` when one is given."""
    server = MCPServer(
        name="phase-sweep", instructions=_INSTRUCTIONS,
        middleware=[_observer(activity)] if activity is not None else None,
    )

    def manager() -> JobManager:
        return app.state.manager

    def user_names() -> set[str]:
        names: set[str] = app.state.user_config_names
        return names

    @server.tool()
    def list_motors() -> list[dict[str, Any]]:
        """Motor configs the server loaded: name, pole pairs, topology,
        whether geometry is present (needed by field solvers)."""
        m = manager()
        return [routes.config_summary(m, user_names(), name) for name in sorted(m.motors)]

    @server.tool()
    def get_motor(name: str, raw_toml: bool = False) -> dict[str, Any]:
        """One motor config. Default: the resolved Motor as a dict (with
        defaults filled). raw_toml=True: the TOML file's section layout."""
        m = manager()
        try:
            if raw_toml:
                return routes.config_raw(m, app.state.settings, user_names(), name)
            return routes.config_detail(m, user_names(), name)
        except HTTPException as e:
            raise _http_error(e) from None

    @server.tool()
    def list_models(motor: str | None = None) -> list[dict[str, Any]]:
        """Call with motor=<config name> before submit_job: each model
        then carries `applicable` (false when the motor lacks what the
        model needs), `missing_params` (params you must pass yourself),
        and `reason` (why, when either blocks a plain submit_job). Without
        motor it lists every submittable job type and cannot say what
        runs. Entries: registry model keys with cost (fast/medium/slow),
        plus the multi-point 'sweep' and 'validate' jobs; `needs_params`
        are params a run requires (the server derives some per motor; see
        model_defaults), `optional_params` have defaults."""
        if motor is None:
            return routes.submittable_models()
        m = manager().motors.get(motor)
        if m is None:
            raise ToolError(f"404: unknown motor config {motor!r}")
        return routes.submittable_models(m)

    @server.tool()
    def model_defaults(motor: str, model: str) -> dict[str, Any]:
        """Default per-run params the server derives for a model on a
        motor (e.g. duty profile, fault current). Pass them, edited if
        needed, as submit_job params."""
        m = manager().motors.get(motor)
        if m is None:
            raise ToolError(f"404: unknown motor config {motor!r}")
        try:
            return routes._derive_model_defaults(m, model)
        except (ValueError, KeyError, TypeError) as e:
            raise ToolError(f"400: {e}") from None
        except MissingExtra as e:
            raise ToolError(f"501: {e}") from None

    @server.tool()
    def submit_job(
        motor: str, model: str, params: dict[str, Any] | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Queue a job: `model` is a registry key, 'sweep', or 'validate'.
        Returns the job with its id and status; a full cache hit returns
        status 'completed' immediately, with `results` as in get_job.
        Wait on slow models with get_job(job_id, wait_s=60).

        params by model:
        - registry key: per-run params from list_models / model_defaults,
          e.g. {"rotation": 0.1} for fem. fem defaults to
          nonlinear=true; nonlinear=false is rejected on inset or bridged
          rotors.
        - 'sweep': {"model_keys": ["fem"], "axes": [{"field": "airgap",
          "start": 0.001, "stop": 0.002, "steps": 3}]}. field is r_outer,
          L_stk, airgap, or back_iron_thickness; start/stop are absolute
          lengths in m; steps >= 2; optional "strategy": "proportional"
          (default) or "fixed_gap". Several axes form a grid. Radial
          geometry only.
        - 'validate': {"models": [...]} (optional; default is every fast
          model that runs on the motor as-is).

        dry_run=True queues nothing and returns the plan: sub-task count,
        how many are already cached, and estimate_s (wall time from the
        median stored run time per model; estimate_complete is false when
        a model has no stored runs, making it a lower bound). Dry-run any
        sweep with a slow model first. Sweeps are capped at
        max_sweep_points grid points."""
        try:
            if dry_run:
                return manager().plan(model, motor, params)
            job = manager().submit(model, motor, params, origin="mcp")
        except KeyError:
            raise ToolError(f"404: unknown motor config {motor!r}") from None
        except (ValueError, TypeError) as e:
            raise ToolError(f"400: {e}") from None
        except MissingExtra as e:
            raise ToolError(f"501: {e}") from None
        return _job_view(job)

    @server.tool()
    def sweep_table(
        job_id: str, sort_by: str | None = None, descending: bool = True,
        offset: int = 0, limit: int = 200,
    ) -> dict[str, Any]:
        """A sweep job as a table, available while it runs: one row per
        grid point with the swept dimensions (m, as realized on that
        point's motor), each model's headline metrics (`model.key`
        columns, units in `units`), and `notes` (failed or pending
        sub-tasks, caveat codes). sort_by names a column; rows missing
        it sort last."""
        job = manager().jobs.get(job_id)
        if job is None:
            raise ToolError(f"404: unknown job {job_id!r}")
        try:
            return jobs_mod.sweep_table(job, sort_by=sort_by, descending=descending,
                                        offset=offset, limit=limit)
        except ValueError as e:
            raise ToolError(f"400: {e}") from None

    @server.tool()
    async def get_job(
        job_id: str, include_subtasks: bool = False, wait_s: float = 0.0,
    ) -> dict[str, Any]:
        """Job status, progress counts, and result_ids once sub-tasks
        finish. Status is pending/running/completed/failed/cancelled.
        A finished job adds `results`: per result, its headline metrics
        (value + unit) and caveat codes (first 20; read get_result for
        caveat text). `wait_s` (up to 60) holds the call until the job
        finishes or that many seconds pass, so one call replaces a
        polling loop; a running job comes back as running."""
        job = manager().jobs.get(job_id)
        if job is None:
            raise ToolError(f"404: unknown job {job_id!r}")
        if not 0.0 <= wait_s <= GET_JOB_MAX_WAIT_S:
            raise ToolError(
                f"400: wait_s must be between 0 and {GET_JOB_MAX_WAIT_S:g}, got {wait_s!r}")
        deadline = time.monotonic() + wait_s
        while job.status not in _TERMINAL and time.monotonic() < deadline:
            await asyncio.sleep(min(_WAIT_POLL_S, max(0.0, deadline - time.monotonic())))
        d = _job_view(job)
        if include_subtasks:
            d["subtasks"] = [st.to_dict() for st in job.subtasks]
        return d

    @server.tool()
    def list_jobs(status: str | None = None) -> list[dict[str, Any]]:
        """Jobs this server process has seen, optionally filtered by status."""
        return [j.to_dict(include_subtasks=False) for j in manager().jobs.values()
                if status is None or j.status == status]

    @server.tool()
    def cancel_job(job_id: str) -> dict[str, Any]:
        """Cancel a job. Pending jobs stop at once; running jobs stop
        issuing sub-tasks (in-flight ones complete)."""
        try:
            return manager().cancel(job_id).to_dict(include_subtasks=False)
        except KeyError:
            raise ToolError(f"404: unknown job {job_id!r}") from None

    @server.tool()
    def query_results(
        motor: str | None = None, model: str | None = None,
        source: str | None = None, limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Current stored results (stale model versions are hidden), most
        recent last. source is 'computed', 'measured', or 'published'."""
        try:
            return routes.result_index(manager(), motor=motor, model=model,
                                       source=source, limit=limit)
        except HTTPException as e:
            raise _http_error(e) from None

    @server.tool()
    def get_result(
        result_id: str, fields: list[str] | None = None, full: bool = False,
    ) -> dict[str, Any]:
        """One stored result: headline metrics with units, caveats
        (code, severity, text, basis), model_version, motor_config_id,
        then status, config, and metrics. Lists longer than 32 entries
        (waveforms, field grids) are summarized as
        {len, min, max, first, last} unless full=True. fields=[...]
        returns the headline block, status, and only those metric keys,
        unsummarized, with no config (missing keys are
        listed under `missing_fields`)."""
        try:
            rec = routes.find_result(manager(), result_id)
        except HTTPException as e:
            raise _http_error(e) from None
        caveats = assess_record(rec)
        if rec.get("source", "computed") == "computed" and rec.get("status") == "OK":
            name = ((rec.get("config") or {}).get("motor") or {}).get("name")
            extra = (manager().measured_caveat(name, rec.get("motor_config_id", ""))
                     if name else None)
            if extra is not None:
                caveats.append(extra)
        head = {
            "result_id": rec["result_id"],
            "model": rec.get("model"),
            "headline": headline_record(rec),
            "caveats": [c.to_dict() for c in caveats],
            "model_version": rec.get("model_version"),
            "motor_config_id": rec.get("motor_config_id"),
        }
        if fields is not None:
            metrics = rec.get("metrics") or {}
            out = head | {"status": rec.get("status"),
                          "metrics": {k: metrics[k] for k in fields if k in metrics}}
            missing = [k for k in fields if k not in metrics]
            if missing:
                out["missing_fields"] = missing
            return out
        rest = {k: v for k, v in rec.items() if k not in head}
        return head | (rest if full else summarize_arrays(rest))

    @server.tool()
    def validation_summary(motor: str) -> dict[str, Any]:
        """Computed-vs-measured/published comparison over everything the
        store holds for a motor."""
        m = manager()
        if motor not in m.motors:
            raise ToolError(f"404: unknown motor config {motor!r}")
        return m.validation_summary(motor)

    return server


def mcp_routes(server: MCPServer) -> list[BaseRoute]:
    """The streamable-HTTP route(s) for `server`, to splice into the app
    ahead of the static '/' mount. Calling this creates the session
    manager, which the app lifespan must then run."""
    sub = server.streamable_http_app(
        streamable_http_path=MCP_PATH,
        stateless_http=True,
        json_response=True,
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )
    return list(sub.routes)
