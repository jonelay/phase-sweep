"""MCP endpoint on phasesweep-server: tools over the shared JobManager."""

from __future__ import annotations

import json
import time

import pytest

from tests.conftest import SERVER_SKIP

pytest.importorskip("fastapi", reason=SERVER_SKIP)
pytest.importorskip("mcp", reason=SERVER_SKIP)

from phasesweep.server.mcp import MCP_PATH, summarize_arrays
from tests.test_server import make_client, workspace  # noqa: F401  (fixture)

HEADERS = {"Accept": "application/json, text/event-stream",
           "Content-Type": "application/json"}


class Mcp:
    def __init__(self, client):
        self.client = client
        self.n = 0
        r = self._rpc("initialize", {
            "protocolVersion": "2025-06-18", "capabilities": {},
            "clientInfo": {"name": "test", "version": "0"}})
        assert "result" in r, r

    def _rpc(self, method, params=None):
        self.n += 1
        body = {"jsonrpc": "2.0", "id": self.n, "method": method}
        if params is not None:
            body["params"] = params
        r = self.client.post(MCP_PATH, headers=HEADERS, json=body)
        assert r.status_code == 200, r.text
        return r.json()

    def tools(self):
        return {t["name"] for t in self._rpc("tools/list")["result"]["tools"]}

    def call(self, tool, **args):
        res = self._rpc("tools/call", {"name": tool, "arguments": args})["result"]
        if res.get("isError"):
            raise RuntimeError(res["content"][0]["text"])
        if "structuredContent" in res:
            sc = res["structuredContent"]
            return sc.get("result", sc) if set(sc) == {"result"} else sc
        return json.loads(res["content"][0]["text"])


@pytest.fixture
def mcp(workspace):  # noqa: F811
    with make_client(workspace) as c:
        yield Mcp(c)


def test_tools_listed(mcp):
    assert {"list_motors", "get_motor", "list_models", "model_defaults",
            "submit_job", "get_job", "list_jobs", "cancel_job",
            "query_results", "get_result", "validation_summary",
            "sweep_table"} <= mcp.tools()


def test_motor_and_model_listing(mcp):
    names = [m["name"] for m in mcp.call("list_motors")]
    assert names == ["test_inrunner"]
    assert mcp.call("get_motor", name="test_inrunner")["motor"]["n_p"] > 0
    assert "raw" in mcp.call("get_motor", name="test_inrunner", raw_toml=True)
    keys = {m["key"] for m in mcp.call("list_models")}
    assert {"analytical", "sweep", "validate"} <= keys
    by_key = {m["key"]: m for m in mcp.call("list_models", motor="test_inrunner")}
    assert by_key["analytical"]["applicable"] is True
    assert by_key["analytical_axial"]["applicable"] is False
    with pytest.raises(RuntimeError, match="404"):
        mcp.call("list_models", motor="nope")


def test_job_lifecycle_and_result(mcp):
    job = mcp.call("submit_job", motor="test_inrunner", model="analytical")
    deadline = time.monotonic() + 30
    while job["status"] not in ("completed", "failed", "cancelled"):
        assert time.monotonic() < deadline, job
        time.sleep(0.1)
        job = mcp.call("get_job", job_id=job["id"])
    assert job["status"] == "completed", job
    # The dashboard REST view sees the MCP-submitted job (shared manager).
    rest = mcp.client.get(f"/api/jobs/{job['id']}").json()
    assert rest["status"] == "completed"

    rid = job["result_ids"][0]
    idx = mcp.call("query_results", motor="test_inrunner", model="analytical")
    assert rid in [r["result_id"] for r in idx]
    full = mcp.call("get_result", result_id=rid, full=True)
    slim = mcp.call("get_result", result_id=rid)
    assert full["status"] == "OK"
    assert len(json.dumps(slim)) < len(json.dumps(full))


def _wait(mcp, job):
    deadline = time.monotonic() + 30
    while job["status"] not in ("completed", "failed", "cancelled"):
        assert time.monotonic() < deadline, job
        time.sleep(0.1)
        job = mcp.call("get_job", job_id=job["id"])
    return job


def test_result_headline_caveats_and_fields(mcp):
    job = _wait(mcp, mcp.call("submit_job", motor="test_inrunner", model="analytical"))
    assert job["status"] == "completed"
    (out,) = job["results"]
    rid = out["result_id"]
    keys = [h["key"] for h in out["headline"]]
    assert keys and keys[0] == "fundamental"
    assert out["headline"][0]["unit"] == "T"
    assert "limits" not in out["caveat_codes"]

    res = mcp.call("get_result", result_id=rid)
    assert list(res)[:6] == ["result_id", "model", "headline", "caveats",
                             "model_version", "motor_config_id"]
    assert res["headline"] == out["headline"]
    assert any(c["code"] == "limits" and c["severity"] == "info" for c in res["caveats"])
    assert {c["code"] for c in res["caveats"]} - {"limits"} == set(out["caveat_codes"])

    proj = mcp.call("get_result", result_id=rid, fields=["B_r_list", "nope"])
    assert set(proj["metrics"]) == {"B_r_list"}
    assert isinstance(proj["metrics"]["B_r_list"], list)  # unsummarized
    assert proj["missing_fields"] == ["nope"]
    assert "config" not in proj and proj["headline"] == res["headline"]

    # full cache hit: submit_job itself returns the outcomes
    again = mcp.call("submit_job", motor="test_inrunner", model="analytical")
    assert again["status"] == "completed"
    assert [r["result_id"] for r in again["results"]] == [rid]
    assert out["cached"] is False and again["results"][0]["cached"] is True


def test_activity_records_mcp_job_outcome(mcp):
    job = _wait(mcp, mcp.call("submit_job", motor="test_inrunner", model="analytical"))
    # a REST-submitted job is not an agent job
    r = mcp.client.post("/api/jobs", json={"model": "analytical",
                                           "motor": "test_inrunner"})
    assert r.status_code == 201, r.text
    deadline = time.monotonic() + 10
    while True:
        entries = mcp.client.get("/api/mcp/activity").json()["entries"]
        jobs = [e for e in entries if e["kind"] == "job"]
        if jobs or time.monotonic() > deadline:
            break
        time.sleep(0.05)
    (entry,) = jobs
    assert entry["job_id"] == job["id"] and entry["status"] == "completed"
    assert entry["ok"] and entry["n_results"] == 1
    assert entry["headline"][0]["key"] == "fundamental"
    assert all(c["code"] != "limits" for c in entry["caveats"])


_SWEEP = {"model_keys": ["analytical"], "axes": [
    {"field": "r_outer", "start": 0.9, "stop": 1.1, "steps": 2},
    {"field": "L_stk", "start": 0.08, "stop": 0.12, "steps": 2},
]}


def test_dry_run_plans_without_queueing(mcp):
    plan = mcp.call("submit_job", motor="test_inrunner", model="sweep",
                    params=_SWEEP, dry_run=True)
    assert plan["dry_run"] and plan["subtasks"] == 4 and plan["points"] == 4
    assert plan["cached"] == 0 and plan["pending"] == 4
    # empty store: no run-time history, so no complete estimate
    assert plan["models"]["analytical"]["samples"] == 0
    assert plan["estimate_complete"] is False
    assert mcp.call("list_jobs") == []

    _wait(mcp, mcp.call("submit_job", motor="test_inrunner", model="sweep", params=_SWEEP))
    again = mcp.call("submit_job", motor="test_inrunner", model="sweep",
                     params=_SWEEP, dry_run=True)
    assert again["cached"] == 4 and again["pending"] == 0
    assert again["models"]["analytical"]["samples"] == 4
    assert again["estimate_complete"] is True
    summary = mcp.client.get("/api/mcp/activity").json()["entries"][-1]["summary"]
    assert summary.startswith("dry run: 4 sub-tasks, 4 cached")


def test_dry_run_estimate_never_below_one_subtask(workspace):  # noqa: F811
    """With more workers than sub-tasks the estimate is the run time of
    one sub-task, not that time divided by the worker count."""
    from phasesweep.result_store import ResultStore
    from phasesweep.server.app import load_motor_configs
    from phasesweep.server.jobs import JobManager
    from phasesweep.sweep_types import RunConfig, RunResult

    motors = load_motor_configs(workspace["motors"])
    store = ResultStore(workspace["output"])
    motor = motors["test_inrunner"]
    store.save(RunResult(config=RunConfig(motor=motor, model="analytical", n_theta=720),
                         model="analytical", status="OK", metrics={}, elapsed_s=8.0))
    plan = JobManager(motors, store, workers=8).plan("analytical", "test_inrunner")
    assert plan["pending"] == 1 and plan["models"]["analytical"]["median_s"] == 8.0
    assert plan["estimate_s"] == 8.0 and plan["estimate_complete"]


def test_sweep_point_cap(mcp):
    big = {"axes": [{"field": "r_outer", "start": 0.9, "stop": 1.1, "steps": 30},
                    {"field": "L_stk", "start": 0.08, "stop": 0.12, "steps": 30}]}
    for dry in (False, True):
        with pytest.raises(RuntimeError, match="900 points; this server allows 500"):
            mcp.call("submit_job", motor="test_inrunner", model="sweep",
                     params=big, dry_run=dry)


def test_sweep_table(mcp):
    job = _wait(mcp, mcp.call("submit_job", motor="test_inrunner", model="sweep",
                              params=_SWEEP))
    t = mcp.call("sweep_table", job_id=job["id"])
    assert t["columns"][:4] == ["point", "r_outer", "L_stk", "analytical.fundamental"]
    assert t["columns"][-1] == "notes" and t["units"]["L_stk"] == "m"
    # caveats shared by every row are reported once, with severities
    assert all(r[-1] == [] for r in t["rows"])
    for n in t["common_notes"]:
        assert t["note_severity"][n] in ("info", "warn")
    # axis values carry no linspace float noise
    assert {r[1] for r in t["rows"]} == {0.9, 1.1}
    assert t["units"]["analytical.fundamental"] == "T"
    assert t["n_rows"] == 4 and not t["truncated"]
    assert sorted(r[2] for r in t["rows"]) == pytest.approx([0.08, 0.08, 0.12, 0.12])
    assert all(isinstance(r[3], float) for r in t["rows"])

    i = t["columns"].index("analytical.flux_linkage_peak")
    top = mcp.call("sweep_table", job_id=job["id"], sort_by="analytical.flux_linkage_peak",
                   limit=1)
    assert top["truncated"] and top["rows"][0][i] == max(r[i] for r in t["rows"])

    with pytest.raises(RuntimeError, match="unknown sort_by"):
        mcp.call("sweep_table", job_id=job["id"], sort_by="nope")
    single = _wait(mcp, mcp.call("submit_job", motor="test_inrunner", model="analytical"))
    with pytest.raises(RuntimeError, match="not a sweep"):
        mcp.call("sweep_table", job_id=single["id"])


def test_designs_endpoint(mcp):
    _wait(mcp, mcp.call("submit_job", motor="test_inrunner", model="sweep", params=_SWEEP))
    d = mcp.client.get("/api/mcp/designs").json()
    assert len(d["designs"]) == 4
    assert [x["order"] for x in d["designs"]] == [0, 1, 2, 3]
    first = d["designs"][0]
    assert first["values"]["L_stk"] == pytest.approx(0.08)
    assert "analytical.fundamental" in first["values"]
    assert d["units"]["analytical.fundamental"] == "T"
    assert set(first["result_ids"]) == {"analytical"}
    same = mcp.client.get(f"/api/mcp/designs?version={d['version']}").json()
    assert same["designs"] is None
    # dashboard (non-MCP) jobs are not the agent's designs
    r = mcp.client.post("/api/jobs", json={"model": "analytical", "motor": "test_inrunner"})
    assert r.status_code == 201
    assert len(mcp.client.get("/api/mcp/designs").json()["designs"]) == 4


def test_get_job_wait_returns_when_finished(mcp):
    """One get_job(wait_s=) call replaces the polling loop."""
    job = mcp.call("submit_job", motor="test_inrunner", model="analytical")
    t0 = time.monotonic()
    job = mcp.call("get_job", job_id=job["id"], wait_s=30)
    assert job["status"] == "completed", job
    assert job["results"]
    assert time.monotonic() - t0 < 25   # returned on finish, not on the deadline


def test_get_job_wait_times_out_on_running_job(mcp):
    from phasesweep.server.jobs import Job
    manager = mcp.client.app.state.manager
    manager.jobs["held"] = Job(id="held", job_type="analytical",
                               motor_name="test_inrunner", params={},
                               subtasks=[], status="running")
    t0 = time.monotonic()
    job = mcp.call("get_job", job_id="held", wait_s=0.6)
    assert 0.5 <= time.monotonic() - t0 < 5
    assert job["status"] == "running"
    with pytest.raises(RuntimeError, match="400: wait_s"):
        mcp.call("get_job", job_id="held", wait_s=61)
    with pytest.raises(RuntimeError, match="400: wait_s"):
        mcp.call("get_job", job_id="held", wait_s=-1)


def test_errors_are_tool_errors(mcp):
    with pytest.raises(RuntimeError, match="unknown motor"):
        mcp.call("submit_job", motor="nope", model="analytical")
    with pytest.raises(RuntimeError, match="unknown job"):
        mcp.call("get_job", job_id="nope")
    with pytest.raises(RuntimeError, match="400: unknown computed model"):
        mcp.call("submit_job", motor="test_inrunner", model="no_such_model")
    with pytest.raises(RuntimeError, match="no result"):
        mcp.call("get_result", result_id="nope")


def test_foreign_host_rejected(mcp):
    r = mcp.client.post(MCP_PATH, headers={**HEADERS, "Host": "evil.example"},
                        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert r.status_code in (400, 403, 421), r.status_code


def test_dashboard_still_served(mcp):
    assert mcp.client.get("/").status_code == 200


def test_summarize_arrays():
    rec = {"a": list(range(100)), "b": [1, 2], "c": {"d": [[0, 1]] * 40}}
    s = summarize_arrays(rec)
    assert s["a"] == {"_summarized": True, "len": 100, "min": 0, "max": 99,
                      "first": 0, "last": 99}
    assert s["b"] == [1, 2]
    assert s["c"]["d"] == {"_summarized": True, "len": 40}


def test_activity_log_records_tool_calls(mcp):
    job = mcp.call("submit_job", motor="test_inrunner", model="analytical")
    with pytest.raises(RuntimeError):
        mcp.call("get_job", job_id="nope")
    act = mcp.client.get("/api/mcp/activity").json()
    kinds = [(e["kind"], e.get("tool")) for e in act["entries"]]
    assert kinds == [("connect", None), ("tool", "submit_job"), ("tool", "get_job")]
    connect, submit, bad = act["entries"]
    assert connect["client"] == "test 0"
    assert submit["ok"] and submit["job_id"] == job["id"]
    assert '"model":"analytical"' in submit["args"]
    assert not bad["ok"] and bad["error"] == "404: unknown job 'nope'"
    # cursor: nothing new after the latest seq
    again = mcp.client.get(f"/api/mcp/activity?since={act['seq']}").json()
    assert again["entries"] == [] and again["boot"] == act["boot"]
