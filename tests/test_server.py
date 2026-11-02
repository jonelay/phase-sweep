"""Server core tests: REST endpoints, job lifecycle, cache check,
sweep fan-out, WebSocket message sequence, measured import.

All jobs run the fast `analytical` model — no FEM (spec scoping).
"""

from __future__ import annotations

import asyncio
import queue
import re
import threading
import time
from importlib.util import find_spec

import pytest

if not find_spec("fastapi"):
    pytest.skip("requires phasesweep[server] (fastapi not installed)", allow_module_level=True)

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from phasesweep._view_sweep_server.ws import ConnectionManager
from phasesweep.result_store import ResultStore
from phasesweep.server.app import ServerSettings, create_app, load_motor_configs
from phasesweep.server.jobs import JobManager

MOTOR_TOML = """
[motor]
name = "Server Test Inrunner"
topology = "inrunner"

[circuit]
n_p = 2
R_s = 0.2
L_d = 4e-3
L_q = 4e-3
psi_f = 0.1
J = 0.002
I_rated = 5.0

[winding]
N = 50
k_w = 0.966
coils_series = 1

[geometry]
r_outer = 1.0
r_stator = 0.70
r_magnet = 0.64
r_rotor = 0.30
r_inner = 0.0
L_stk = 0.10

[materials]
# B_rem omitted — derived from psi_f via winding params
mu_r_fe = 1000.0
mu_r_pm = 1.05

[drive]
U_DC = 48.0
MAX_I_S = 10.0
W_REF = 314.159
"""

MEASURED_JSON = {
    "test_type": "resistance_test",
    "conditions": {
        "speed_rpm": 0.0,
        "temperature_C": 22.0,
        "load_torque_Nm": 0.0,
        "date": "2026-07-09",
        "instrument": "bench DMM",
    },
    "quantities": {"R_s": 0.21},
    "waveforms": {},
    "uncertainty": {"R_s": 0.01},
    "source_file": "bench_r_test.json",
}


@pytest.fixture
def workspace(tmp_path):
    motors = tmp_path / "motors"
    motors.mkdir()
    (motors / "test_inrunner.toml").write_text(MOTOR_TOML)
    return {"motors": motors, "user": tmp_path / "user_configs",
            "output": tmp_path / "output"}


def make_client(workspace):
    app = create_app(ServerSettings(
        motors_dir=workspace["motors"], user_configs_dir=workspace["user"],
        output_dir=workspace["output"], workers=1, subtask_timeout_s=60.0,
    ))
    return TestClient(app)


@pytest.fixture
def client(workspace):
    with make_client(workspace) as c:
        yield c


def wait_job(client, job_id, deadline_s=30.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < deadline_s:
        r = client.get(f"/api/jobs/{job_id}")
        assert r.status_code == 200
        d = r.json()
        if d["status"] in ("completed", "failed", "cancelled"):
            return d
        time.sleep(0.02)
    pytest.fail(f"job {job_id} did not reach a terminal state in {deadline_s}s")


def submit(client, model, params=None, motor="test_inrunner", expect=201):
    r = client.post("/api/jobs", json={
        "motor": motor, "model": model, "params": params or {}})
    assert r.status_code == expect, r.text
    return r.json()


def ws_receive_json(ws, timeout_s=30.0):
    """WebSocketTestSession.receive_json blocks forever; bound it so a
    server that never sends fails the test instead of hanging it."""
    out: queue.Queue = queue.Queue()

    def _recv():
        try:
            out.put(("ok", ws.receive_json()))
        except BaseException as e:  # relayed to the test thread
            out.put(("err", e))

    threading.Thread(target=_recv, daemon=True).start()
    try:
        kind, val = out.get(timeout=timeout_s)
    except queue.Empty:
        pytest.fail(f"no WebSocket message within {timeout_s}s")
    if kind == "err":
        raise val
    return val


# -- configs ---------------------------------------------------

def test_configs_list_and_detail(client):
    r = client.get("/api/configs")
    assert r.status_code == 200
    configs = r.json()
    assert len(configs) == 1
    assert configs[0]["name"] == "test_inrunner"  # filename stem, not display name
    assert configs[0]["has_geometry"] is True
    assert configs[0]["topology"] == "inrunner"

    r = client.get("/api/configs/test_inrunner")
    assert r.status_code == 200
    detail = r.json()
    assert detail["motor"]["name"] == "Server Test Inrunner"
    assert detail["motor"]["n_p"] == 2
    assert detail["motor_config_id"] == configs[0]["motor_config_id"]

    assert client.get("/api/configs/nope").status_code == 404


# -- config editor write path -----------------------------------

def put_config(client, name, mutate=None, expect=201):
    raw = client.get("/api/configs/test_inrunner/raw").json()["raw"]
    if mutate:
        mutate(raw)
    r = client.put(f"/api/configs/{name}", json=raw)
    assert r.status_code == expect, r.text
    return r.json()


def test_config_editable_flags_and_raw(client):
    cfg = client.get("/api/configs").json()[0]
    assert cfg["editable"] is False  # anchor fleet is read-only

    r = client.get("/api/configs/test_inrunner/raw")
    assert r.status_code == 200
    d = r.json()
    assert d["editable"] is False
    assert d["raw"]["circuit"]["n_p"] == 2
    assert d["raw"]["geometry"]["r_stator"] == 0.70

    assert client.get("/api/configs/nope/raw").status_code == 404


def test_put_config_create_update_and_job(client, workspace):
    saved = put_config(client, "variant_a",
                       mutate=lambda raw: raw["circuit"].__setitem__("psi_f", 0.12))
    assert saved["editable"] is True
    assert (workspace["user"] / "variant_a.toml").exists()

    names = {c["name"]: c for c in client.get("/api/configs").json()}
    assert set(names) == {"test_inrunner", "variant_a"}
    assert names["variant_a"]["editable"] is True
    assert (names["variant_a"]["motor_config_id"]
            != names["test_inrunner"]["motor_config_id"])

    # round-trip: raw reflects the edit
    raw = client.get("/api/configs/variant_a/raw").json()
    assert raw["editable"] is True
    assert raw["raw"]["circuit"]["psi_f"] == 0.12

    # update in place → 200, config_id changes with content
    updated = put_config(client, "variant_a", expect=200,
                         mutate=lambda raw: raw["circuit"].__setitem__("psi_f", 0.15))
    assert updated["motor_config_id"] != saved["motor_config_id"]

    # hot-loaded: jobs run against the saved config immediately
    job = wait_job(client, submit(client, "analytical", motor="variant_a")["id"])
    assert job["status"] == "completed"


def test_put_config_refuses_anchor_and_bad_input(client, workspace):
    # anchor name → 409, file untouched
    before = (workspace["motors"] / "test_inrunner.toml").read_text()
    put_config(client, "test_inrunner", expect=409)
    assert (workspace["motors"] / "test_inrunner.toml").read_text() == before

    # invalid motor payload → 400 naming the problem
    r = client.put("/api/configs/bad_one", json={"motor": {"name": "x"}})
    assert r.status_code == 400
    assert "circuit" in r.json()["detail"]
    assert not (workspace["user"] / "bad_one.toml").exists()

    # unsafe names rejected (405 = router normalized '..' away before us)
    assert client.put("/api/configs/..", json={}).status_code in (400, 404, 405)
    put_config(client, "-leading-dash", expect=400)
    assert client.put("/api/configs/nul%2Fx", json={}).status_code in (400, 404, 405)


def test_put_config_null_section_is_rejected_not_crashed(client):
    """A JSON null for any optional section must be a 4xx, not a 500.

    TOML has no null, but the editor PUTs JSON: `raw.get(sec, {})` returned
    None for a present-but-null key and every downstream `.get` blew up
    (found by API fuzzing).
    """
    base = {"motor": {"name": "x", "topology": "inrunner"}, "circuit": {"n_p": 2}}
    for sec in ("motor", "circuit", "geometry", "winding",
                "materials", "drive", "thermal", "iron"):
        r = client.put(f"/api/configs/null_{sec}", json={**base, sec: None})
        assert r.status_code < 500, f"[{sec}] = null gave {r.status_code}"


def test_user_configs_persist_and_never_shadow_anchor(workspace):
    with make_client(workspace) as c:
        put_config(c, "variant_b",
                   mutate=lambda raw: raw["geometry"].__setitem__("L_stk", 0.12))

    # a user file whose stem collides with an anchor is skipped at startup
    (workspace["user"] / "test_inrunner.toml").write_text(MOTOR_TOML)

    with make_client(workspace) as c:
        names = {cfg["name"]: cfg for cfg in c.get("/api/configs").json()}
        assert names["variant_b"]["editable"] is True
        assert names["test_inrunner"]["editable"] is False  # anchor won
        raw = c.get("/api/configs/variant_b/raw").json()["raw"]
        assert raw["geometry"]["L_stk"] == 0.12


# -- job lifecycle ---------------------------------------

def test_single_point_job_lifecycle(client):
    job = submit(client, "analytical")
    assert job["status"] in ("pending", "running")
    assert job["total"] == 1

    done = wait_job(client, job["id"])
    assert done["status"] == "completed"
    assert len(done["result_ids"]) == 1
    assert done["subtasks"][0]["status"] == "OK"

    rid = done["result_ids"][0]
    r = client.get(f"/api/results/{rid}")
    assert r.status_code == 200
    metrics = r.json()["metrics"]
    assert metrics["fundamental"] > 0
    assert len(metrics["B_r_list"]) == 360

    r = client.get("/api/results", params={"model": "analytical"})
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 1
    assert rows[0]["result_id"] == rid
    assert rows[0]["status"] == "OK"
    assert rows[0]["motor_name"]
    assert rows[0]["timestamp"]  # run history needs when, not just what


def test_result_with_nan_metric_served_as_null(client):
    """drive_sim legitimately emits NaN (t_settle when the speed never
    settles); starlette refuses non-finite floats, so the API must serve
    them as null instead of 500ing."""
    from phasesweep.sweep_types import RunConfig, RunResult, compute_run_id

    manager = client.app.state.manager
    motor = manager.motors["test_inrunner"]
    config = RunConfig(motor=motor, model="analytical")
    manager.save_result(RunResult(
        config=config, model="analytical", status="OK",
        metrics={"fundamental": float("nan"), "peak_Br": 1.0}, elapsed_s=0.1,
    ))
    r = client.get(f"/api/results/{compute_run_id(config)}")
    assert r.status_code == 200
    metrics = r.json()["metrics"]
    assert metrics["fundamental"] is None
    assert metrics["peak_Br"] == 1.0


def test_get_result_by_stored_run_id(client, workspace):
    """GET /results/{id} keys on the stamped run_id, not a recomputed hash."""
    import json

    done = wait_job(client, submit(client, "analytical")["id"])
    rid = done["result_ids"][0]
    results_file = workspace["output"] / "results.jsonl"
    d = json.loads(results_file.read_text())
    assert d["run_id"] == rid
    d["run_id"] = "sentinel00001"
    results_file.write_text(json.dumps(d) + "\n")

    assert client.get("/api/results/sentinel00001").status_code == 200
    assert client.get(f"/api/results/{rid}").status_code == 404


def test_job_list_filter(client):
    job = submit(client, "analytical")
    wait_job(client, job["id"])
    assert len(client.get("/api/jobs", params={"status": "completed"}).json()) == 1
    assert client.get("/api/jobs", params={"status": "running"}).json() == []


def test_cache_hit_completes_immediately(client, workspace):
    first = wait_job(client, submit(client, "analytical")["id"])
    results_file = workspace["output"] / "results.jsonl"
    n_lines = len(results_file.read_text().splitlines())

    again = submit(client, "analytical")
    # Full cache hit: completed in the POST response, no queue round-trip
    assert again["status"] == "completed"
    assert again["subtasks"][0]["status"] == "cached"
    assert again["result_ids"] == first["result_ids"]
    assert len(results_file.read_text().splitlines()) == n_lines


def test_stale_store_records_never_serve(client, workspace):
    """Records with a missing or superseded model-version
    stamp are invisible to the API (no run-history row, no full fetch)
    and never serve as cache hits — the job recomputes."""
    import json as jsonlib

    first = wait_job(client, submit(client, "analytical")["id"])
    rid = first["result_ids"][0]
    results_file = workspace["output"] / "results.jsonl"

    for tamper in ("missing", "superseded"):
        d = jsonlib.loads(results_file.read_text().splitlines()[-1])
        if tamper == "missing":
            del d["model_version"]
        else:
            d["model_version"] -= 1
        results_file.write_text(jsonlib.dumps(d) + "\n")

        assert client.get("/api/results").json() == []
        assert client.get(f"/api/results/{rid}").status_code == 404

        again = submit(client, "analytical")
        assert again["subtasks"][0]["status"] != "cached"
        done = wait_job(client, again["id"])
        assert done["result_ids"] == [rid]

    # Recomputed record is stamped current and serves again
    rows = client.get("/api/results").json()
    assert [r["result_id"] for r in rows] == [rid]
    assert client.get(f"/api/results/{rid}").status_code == 200


def test_submit_validation_errors(client):
    submit(client, "analytical", motor="nope", expect=404)
    submit(client, "flux_map", expect=400)          # not a job type (standing decision)
    submit(client, "backemf_capture", expect=400)   # measured key, not computed
    submit(client, "analytical", params={"bogus": 1}, expect=400)
    submit(client, "sweep", params={"axes": [
        {"field": "r_outer", "start": 0.9, "stop": 1.1, "steps": 1}]}, expect=400)
    submit(client, "sweep", expect=400)             # no axes
    r = client.post("/api/jobs", json={"motor": "test_inrunner"})
    assert r.status_code == 422
    r = client.post("/api/jobs", json={
        "motor": "test_inrunner", "model": "analytical", "params": "oops"})
    assert r.status_code == 422                     # params must be an object


def test_submit_bad_param_shapes_are_400(client):
    """Malformed param structures must map to 400, not 500, and a
    KeyError from sim_plan parsing must not surface as an unknown-motor 404."""
    r = client.post("/api/jobs", json={
        "motor": "test_inrunner", "model": "drive_sim", "params": {"sim_plan": {}}})
    assert r.status_code == 400, r.text
    assert "sim_plan" in r.json()["detail"]
    submit(client, "drive_sim", params={"sim_plan": "oops"}, expect=400)
    submit(client, "analytical", params={"duty_profile": 42}, expect=400)
    submit(client, "sweep", params={"model_keys": 42, "axes": [
        {"field": "r_outer", "start": 0.9, "stop": 1.1, "steps": 2}]}, expect=400)


@pytest.mark.parametrize("axis", [
    {"name": "airgap", "start": 0.001, "stop": 0.002, "steps": 3},
    "airgap",
])
def test_bad_sweep_axis_error_states_whole_format(client, axis):
    """One error must carry the full axis shape. Naming only the first bad
    keyword cost a live agent 14 guesses at the field names."""
    r = client.post("/api/jobs", json={"motor": "test_inrunner", "model": "sweep",
                                       "params": {"axes": [axis]}})
    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    for word in ("field", "start", "stop", "steps", "airgap", "in m"):
        assert word in detail, detail


def test_submit_string_params_are_typed(client):
    """Query/form strings are parsed into the RunConfig type: "false"
    must not hash as a truthy string, garbage must be a 400."""
    typed = submit(client, "analytical", params={"nonlinear": False, "n_theta": 180})
    parsed = submit(client, "analytical", params={"nonlinear": "false", "n_theta": "180"})
    assert parsed["subtasks"][0]["run_id"] == typed["subtasks"][0]["run_id"]
    submit(client, "analytical", params={"nonlinear": "nope"}, expect=400)
    submit(client, "analytical", params={"n_theta": "18.5"}, expect=400)
    submit(client, "analytical", params={"maxh_fraction": "nan"}, expect=400)


def test_drive_sim_two_mass_submit_and_auto_derive(client):
    """drive_sim_two_mass auto-derives sim_plan from motor, same as
    drive_sim, and accepts load_mech as structured params."""
    load_mech = {"J_L": 0.01, "K_S": 500.0, "C_S": 0.1, "B_L": 0.0}
    job = submit(client, "drive_sim_two_mass", params={"load_mech": load_mech})
    assert job["total"] == 1
    done = wait_job(client, job["id"], deadline_s=90.0)
    assert done["status"] == "completed"
    assert len(done["result_ids"]) == 1


def test_drive_sim_two_mass_bad_load_mech_is_400(client):
    submit(client, "drive_sim_two_mass",
           params={"load_mech": "oops"}, expect=400)
    submit(client, "drive_sim_two_mass",
           params={"load_mech": {"J_L": -1, "K_S": 500, "C_S": 0}}, expect=400)
    submit(client, "drive_sim_two_mass",
           params={"load_mech": {}}, expect=400)


def test_drive_sim_two_mass_without_load_mech_is_400(client):
    """Missing load_mech must fail at submit (400), not at runtime."""
    r = client.post("/api/jobs", json={
        "motor": "test_inrunner", "model": "drive_sim_two_mass", "params": {}})
    assert r.status_code == 400
    assert "load_mech" in r.json()["detail"]


def test_drive_sim_rejects_load_mech(client):
    """load_mech on plain drive_sim is a caller error, not silently ignored."""
    load_mech = {"J_L": 0.01, "K_S": 500.0, "C_S": 0.1}
    r = client.post("/api/jobs", json={
        "motor": "test_inrunner", "model": "drive_sim",
        "params": {"load_mech": load_mech}})
    assert r.status_code == 400


def test_validate_filters_load_mech_per_model(client):
    """Validate with sim_plan + load_mech runs both drive_sim and
    drive_sim_two_mass — load_mech is filtered out for drive_sim."""
    from phasesweep.server.app import load_motor_configs
    from phasesweep.simulation.params import prepare_drive_sim
    from phasesweep.simulation.sim import plan_sim

    motor = load_motor_configs(client.app.state.settings.motors_dir)["test_inrunner"]
    sim_plan = plan_sim(prepare_drive_sim(motor)).to_dict()
    load_mech = {"J_L": 0.01, "K_S": 500.0, "C_S": 0.1}
    job = submit(client, "validate", params={
        "models": ["drive_sim", "drive_sim_two_mass"],
        "sim_plan": sim_plan,
        "load_mech": load_mech,
    })
    assert job["total"] == 2
    done = wait_job(client, job["id"], deadline_s=90.0)
    assert done["status"] == "completed"
    statuses = {st["model"]: st["status"] for st in done["subtasks"]}
    assert statuses["drive_sim"] in ("OK", "cached")
    assert statuses["drive_sim_two_mass"] in ("OK", "cached")


def test_thermal_duty_auto_derive_rejects_at_submit(client):
    """An r_th-budget motor validates for thermal_duty without I_rated, but
    the duty_profile auto-derive needs it — fail at submit (400) like
    demag_screen, not with a runtime job failure."""
    def rth_no_irated(raw):
        del raw["circuit"]["I_rated"]
        raw["thermal"] = {"r_th": 0.5, "winding_temp_limit": 120.0,
                          "ambient_temp": 25.0}
    put_config(client, "rth_motor", mutate=rth_no_irated)
    r = client.post("/api/jobs", json={
        "motor": "rth_motor", "model": "thermal_duty", "params": {}})
    assert r.status_code == 400, r.text
    assert "I_rated" in r.json()["detail"]


@pytest.mark.parametrize("route", ["single", "validate", "sweep"])
def test_cogging_rejects_nonzero_j_s(client, route):
    """cogging_torque with j_s != 0 must be rejected at submit time on all
    three server routes — otherwise cache aliasing serves the j_s=0 result."""
    if route == "single":
        r = client.post("/api/jobs", json={
            "motor": "test_inrunner", "model": "cogging_torque",
            "params": {"j_s": 100.0}})
    elif route == "validate":
        r = client.post("/api/jobs", json={
            "motor": "test_inrunner", "model": "validate",
            "params": {"models": ["cogging_torque"], "j_s": 100.0}})
    else:
        r = client.post("/api/jobs", json={
            "motor": "test_inrunner", "model": "sweep",
            "params": {
                "model_keys": ["cogging_torque"], "j_s": 100.0,
                "axes": [{"field": "r_outer", "start": 0.9,
                          "stop": 1.1, "steps": 2}],
            }})
    assert r.status_code == 400, f"{route}: {r.text}"
    assert "j_s" in r.json()["detail"].lower()


# -- sweep fan-out ---------------------------------------------

def test_sweep_fan_out(client):
    job = submit(client, "sweep", params={
        "model_keys": ["analytical"],
        "axes": [{"field": "r_outer", "start": 0.9, "stop": 1.1, "steps": 3}],
    })
    assert job["total"] == 3

    done = wait_job(client, job["id"])
    assert done["status"] == "completed"
    assert len(done["result_ids"]) == 3
    assert sorted(st["point_idx"] for st in done["subtasks"]) == [0, 1, 2]

    rows = client.get("/api/results").json()
    assert len(rows) == 3
    # Distinct geometries hash to distinct run ids
    assert len({r["result_id"] for r in rows}) == 3


def test_sweep_partial_cache(client):
    small = wait_job(client, submit(client, "sweep", params={
        "model_keys": ["analytical"],
        "axes": [{"field": "r_outer", "start": 0.9, "stop": 1.1, "steps": 2}],
    })["id"])
    assert small["status"] == "completed"

    # steps=3 shares its endpoints with steps=2 — only the midpoint runs
    big = wait_job(client, submit(client, "sweep", params={
        "model_keys": ["analytical"],
        "axes": [{"field": "r_outer", "start": 0.9, "stop": 1.1, "steps": 3}],
    })["id"])
    statuses = sorted(st["status"] for st in big["subtasks"])
    assert statuses == ["OK", "cached", "cached"]


# -- validate job ------------------------------------------------

def test_validate_job(client):
    job = submit(client, "validate", params={
        "models": ["analytical", "rated_torque"]})
    assert job["total"] == 2

    done = wait_job(client, job["id"])
    assert done["status"] == "completed"
    summary = done["summary"]
    assert isinstance(summary["diagnosis"], str)
    assert set(summary["models"]) == {"analytical", "rated_torque"}
    assert summary["n_results"] == 2
    assert isinstance(summary["rows"], list)


def test_validate_default_models_exclude_gated_and_slow(client):
    job = submit(client, "validate")
    models = {st["model"] for st in job["subtasks"]}
    assert "analytical" in models
    assert "fem" not in models          # slow-cost excluded from default
    assert "demag_screen" not in models  # i_fault-gated
    assert "drive_sim" not in models           # sim_plan-gated
    assert "drive_sim_two_mass" not in models  # load_mech-gated
    assert "thermal_duty" not in models        # duty_profile-gated
    assert "operating_point" not in models     # w_mech/tau_ref not settable
    assert "motor_thermal" not in models
    done = wait_job(client, job["id"])
    assert done["status"] == "completed"


def test_duplicate_run_ids_collapse_to_one_subtask(client):
    """Duplicate model keys hash to one run_id; without dedupe the
    twin sub-task stayed 'pending' forever and skewed completed/total."""
    job = submit(client, "validate", params={"models": ["analytical", "analytical"]})
    assert job["total"] == 1
    done = wait_job(client, job["id"])
    assert done["status"] == "completed"
    assert done["completed"] == done["total"] == 1


# -- cancellation ------------------------------------------------

def test_cancel_terminal_job_is_noop(client):
    job = submit(client, "analytical")
    wait_job(client, job["id"])
    r = client.delete(f"/api/jobs/{job['id']}")
    assert r.status_code == 200
    assert r.json()["status"] == "completed"
    assert client.delete("/api/jobs/nope").status_code == 404


def test_cancel_pending_job(workspace):
    async def scenario():
        motors = load_motor_configs(workspace["motors"])
        manager = JobManager(motors, ResultStore(workspace["output"]), workers=1)
        events = []

        async def broadcast(msg):
            events.append(msg)

        # Runner not started: the job stays pending, cancellation is
        # deterministic.
        manager._loop = asyncio.get_running_loop()
        job = manager.submit("analytical", "test_inrunner")
        assert job.status == "pending"
        cancelled = manager.cancel(job.id)
        assert cancelled.status == "cancelled"
        assert cancelled.subtasks[0].status == "CANCELLED"

        # A cancelled pending job is skipped when the runner starts
        await manager.start(broadcast)
        await asyncio.sleep(0.05)
        assert job.status == "cancelled"
        await manager.stop()
        return events

    events = asyncio.run(scenario())
    assert [e["type"] for e in events] == ["job_cancelled"]


def test_stop_returns_while_job_draining(workspace, monkeypatch):
    """Ctrl-C: the executor drain runs on a daemon thread, so
    manager.stop() (and interpreter exit) never block on a running
    sub-task — the drain finishes in the background."""
    import phasesweep.server.jobs as jobs_mod

    started = threading.Event()
    release = threading.Event()
    drain: dict = {}

    def fake_execute(worker_jobs, **kw):
        drain["thread"] = threading.current_thread()
        started.set()
        release.wait(10.0)
        return []

    monkeypatch.setattr(jobs_mod, "execute_parallel", fake_execute)

    async def scenario():
        motors = load_motor_configs(workspace["motors"])
        manager = JobManager(motors, ResultStore(workspace["output"]), workers=1)

        async def broadcast(msg):
            pass

        await manager.start(broadcast)
        manager.submit("analytical", "test_inrunner")
        assert await asyncio.to_thread(started.wait, 10.0)
        await asyncio.wait_for(manager.stop(), timeout=5.0)

    asyncio.run(scenario())
    assert drain["thread"].daemon  # non-daemon would block interpreter exit
    release.set()


def _run_manager_jobs(workspace, submissions, monkeypatch=None, fake=None):
    """Run *submissions* (model, params) in order on a fresh manager,
    swapping execute_parallel for *fake* after the first job completes."""
    import phasesweep.server.jobs as jobs_mod

    async def scenario():
        motors = load_motor_configs(workspace["motors"])
        store = ResultStore(workspace["output"])
        manager = JobManager(motors, store, workers=1)

        async def broadcast(msg):
            pass

        await manager.start(broadcast)
        done = []
        for i, (model, params) in enumerate(submissions):
            if i == 1 and fake is not None:
                monkeypatch.setattr(jobs_mod, "execute_parallel", fake)
            job = manager.submit(model, "test_inrunner", params)
            t0 = time.monotonic()
            while job.status not in ("completed", "failed", "cancelled"):
                assert time.monotonic() - t0 < 30.0
                await asyncio.sleep(0.02)
            done.append(job)
        await manager.stop()
        return done, store

    return asyncio.run(scenario())


def test_executor_exception_fails_job_despite_cached_sibling(workspace, monkeypatch):
    """An exception out of execute_parallel marks the unreported sub-tasks
    ERROR with its message and fails the job — a cached sibling used to
    turn it into 'completed' with a sub-task stuck 'pending'."""
    def boom(worker_jobs, **kw):
        raise RuntimeError("pool broke")

    (_, job), store = _run_manager_jobs(workspace, [
        ("analytical", None),
        ("validate", {"models": ["analytical", "rated_torque"]}),
    ], monkeypatch, boom)
    assert job.status == "failed"
    assert job.error == "RuntimeError: pool broke"
    by_model = {st.model: st for st in job.subtasks}
    assert by_model["analytical"].status == "cached"
    assert by_model["rated_torque"].status == "ERROR"
    assert by_model["rated_torque"].error_msg == "RuntimeError: pool broke"
    assert {r.model for r in store.load_results()} == {"analytical"}


def test_cancelled_subtasks_persist_no_rows(workspace, monkeypatch):
    """Sub-tasks the executor reports CANCELLED never ran: status only,
    no ERROR row in the store."""
    def cancel_all(worker_jobs, *, cancel, on_complete, **kw):
        cancel.set()
        for i, wj in enumerate(worker_jobs, 1):
            on_complete({"run_id": wj["run_id"], "status": "CANCELLED",
                         "error_msg": "Cancelled before start"},
                        i, len(worker_jobs))
        return []

    (_, job), store = _run_manager_jobs(workspace, [
        ("analytical", None),
        ("validate", {"models": ["analytical", "rated_torque"]}),
    ], monkeypatch, cancel_all)
    assert job.status == "cancelled"
    by_model = {st.model: st for st in job.subtasks}
    assert by_model["rated_torque"].status == "CANCELLED"
    assert {r.model for r in store.load_results()} == {"analytical"}


def test_sweep_forces_nonlinear_for_fem_only(workspace):
    """Server sweeps mirror geo_parallel.run_sweep: nonlinear defaults on
    for fem only; cogging_torque hashes nonlinear, so forcing it there
    split one grid into two run-id sets. On creator's inset rotor that
    unforced linear default is rejected at submit, not at run time."""
    motors = load_motor_configs(workspace["motors"])
    manager = JobManager(motors, ResultStore(workspace["output"]), workers=1)
    axes = [{"field": "r_outer", "start": 0.9, "stop": 1.1, "steps": 2}]
    motor = motors["test_inrunner"]
    fem = manager._sweep_subtasks(motor, {"model_keys": ["fem"], "axes": axes})
    assert all(st.config.nonlinear for st in fem)
    cog_motor = _repo_motor("creator_case_pmsm")
    cog_params = {
        "model_keys": ["cogging_torque"],
        "axes": [{"field": "r_outer", "start": cog_motor.geometry.r_outer,
                  "stop": cog_motor.geometry.r_outer * 1.02, "steps": 2}]}
    with pytest.raises(ValueError, match="cogging_torque: magnet_mount='inset'"):
        manager._sweep_subtasks(cog_motor, cog_params)
    cog = manager._sweep_subtasks(cog_motor, {**cog_params, "nonlinear": True})
    assert cog and all(st.config.nonlinear for st in cog)


def _repo_motor(name: str):
    from phasesweep import load_motor
    from tests.conftest import REPO_ROOT
    return load_motor(REPO_ROOT / f"motors/{name}.toml")


def test_single_fem_defaults_nonlinear(workspace):
    """Single and validate fem jobs default to nonlinear like sweeps; an
    explicit nonlinear=false still runs linear where the rotor allows it."""
    motors = load_motor_configs(workspace["motors"])
    manager = JobManager(motors, ResultStore(workspace["output"]), workers=1)
    motor = motors["test_inrunner"]
    [st] = manager._build_subtasks("fem", motor, {})
    assert st.config.nonlinear is True
    [st] = manager._build_subtasks("fem", motor, {"nonlinear": False})
    assert st.config.nonlinear is False
    [st] = manager._build_subtasks("validate", motor, {"models": ["fem"]})
    assert st.config.nonlinear is True


def test_linear_fem_on_inset_rotor_rejected_at_submit(workspace):
    """bmw_i3 is inset: bare fem now runs nonlinear, and a linear request
    is a submit-time error instead of a run-time failure in fem_field."""
    manager = JobManager({}, ResultStore(workspace["output"]), workers=1)
    motor = _repo_motor("bmw_i3")
    [st] = manager._build_subtasks("fem", motor, {})
    assert st.config.nonlinear is True
    with pytest.raises(ValueError, match="pass params.nonlinear=true"):
        manager._build_subtasks("fem", motor, {"nonlinear": False})


def test_applicability_flags_nonlinear_on_inset_rotor():
    from phasesweep.server.routes import submittable_models
    by_key = {m["key"]: m for m in submittable_models(_repo_motor("creator_case_pmsm"))}
    assert "nonlinear" not in by_key["fem"]["missing_params"]
    cog = by_key["cogging_torque"]
    assert cog["applicable"] is True
    assert "nonlinear" in cog["missing_params"]
    assert "nonlinear=true" in cog["reason"]


# -- WebSocket ------------------------------------------------------------------

def test_ws_message_sequence_manager_level(workspace):
    """Deterministic full-sequence assertion: progress 1..N, then complete."""
    async def scenario():
        motors = load_motor_configs(workspace["motors"])
        manager = JobManager(motors, ResultStore(workspace["output"]), workers=1)
        events = []

        async def broadcast(msg):
            events.append(msg)

        await manager.start(broadcast)
        job = manager.submit("sweep", "test_inrunner", {
            "model_keys": ["analytical"],
            "axes": [{"field": "r_outer", "start": 0.9, "stop": 1.1, "steps": 3}],
        })
        t0 = time.monotonic()
        while job.status not in ("completed", "failed", "cancelled"):
            assert time.monotonic() - t0 < 30.0
            await asyncio.sleep(0.02)
        while not manager._events.empty():
            await asyncio.sleep(0.02)
        await manager.stop()
        return job, events

    job, events = asyncio.run(scenario())
    assert job.status == "completed"
    progress = [e for e in events if e["type"] == "job_progress"]
    assert [e["completed"] for e in progress] == [1, 2, 3]
    assert all(e["total"] == 3 for e in progress)
    assert all(e["latest_result_id"] for e in progress)
    assert events[-1]["type"] == "job_complete"
    assert sorted(events[-1]["result_ids"]) == sorted(job.result_ids)


def test_ws_endpoint_subscribe_and_filter(client):
    """End-to-end over the wire: subscribe while the runner is busy with an
    earlier job, receive only the subscribed job's events."""
    with client.websocket_connect("/ws") as ws:
        blocker = submit(client, "sweep", params={
            "model_keys": ["analytical"],
            "axes": [{"field": "r_outer", "start": 0.85, "stop": 1.15, "steps": 30}],
        })
        watched = submit(client, "sweep", params={
            "model_keys": ["analytical"],
            "axes": [{"field": "airgap", "start": 0.05, "stop": 0.07, "steps": 3}],
        })
        # Serial FIFO runner: `watched` cannot start until `blocker`'s 30
        # sub-tasks finish, so this subscribe always lands first.
        ws.send_json({"type": "subscribe", "job_ids": [watched["id"]]})

        messages = []
        while True:
            msg = ws_receive_json(ws)
            messages.append(msg)
            if msg["type"] == "job_complete":
                break

    assert all(m["job_id"] == watched["id"] for m in messages)
    progress = [m for m in messages if m["type"] == "job_progress"]
    assert [m["completed"] for m in progress] == [1, 2, 3]
    assert sorted(messages[-1]["result_ids"]) == \
        sorted(wait_job(client, watched["id"])["result_ids"])
    wait_job(client, blocker["id"])


def test_ws_handle_ignores_malformed_frames():
    """Non-dict frames, missing/non-list job_ids, and lists with non-string
    ids are dropped whole instead of raising in the endpoint loop. The
    manager is view-sweep's; this pins the protocol behaviour the server
    relies on (full coverage lives in view-sweep's test_ws.py)."""
    cm = ConnectionManager()
    ws = object()
    cm._subs[ws] = set()
    cm.handle(ws, ["subscribe"])
    cm.handle(ws, {"type": "subscribe"})
    cm.handle(ws, {"type": "subscribe", "job_ids": "abc"})
    cm.handle(ws, {"type": "subscribe", "job_ids": [1, "ok", None]})
    cm.handle(ws, {"type": "subscribe", "job_ids": ["ok"]})
    cm.handle(ws, {"type": "unsubscribe", "job_ids": ["missing"]})
    assert cm.subscriptions(ws) == {"ok"}


def test_ws_cancel_frame_cancels_job(client):
    """A `cancel` client message reaches JobManager.cancel."""
    blocker = submit(client, "sweep", params={
        "model_keys": ["analytical"],
        "axes": [{"field": "r_outer", "start": 0.85, "stop": 1.15, "steps": 30}],
    })
    target = submit(client, "analytical")
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "cancel", "job_id": target["id"]})
        ws.send_json({"type": "cancel", "job_id": "no-such-job"})  # logged, not fatal
        ws.send_json({"type": "subscribe", "job_ids": [target["id"]]})
        assert wait_job(client, target["id"])["status"] == "cancelled"
    wait_job(client, blocker["id"])


def test_ws_malformed_json_closes_connection(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_text("not json")
        with pytest.raises(WebSocketDisconnect):
            ws_receive_json(ws, timeout_s=10.0)


def test_view_sweep_assets_mounted_but_nothing_else(client):
    """The dashboard needs /view-sweep/{css,js,spec}; the rest of the
    view-sweep checkout must not be reachable."""
    assert client.get("/view-sweep/css/viewsweep.css").status_code == 200
    assert client.get("/view-sweep/spec/palettes/engineering.css").status_code == 200
    assert client.get("/view-sweep/js/theme.js").status_code == 200
    assert client.get("/view-sweep/.git/HEAD").status_code == 404
    assert client.get("/view-sweep/uv.lock").status_code == 404
    assert client.get("/").status_code == 200  # dashboard index


# -- host/origin validation ------------------------------------------------------

def test_host_and_origin_validation(client):
    """DNS-rebinding / CSRF guard: non-local Host is rejected outright,
    a cross-site Origin is rejected, local and absent Origins pass."""
    assert client.get("/api/configs", headers={"Host": "evil.example"}).status_code == 400
    assert client.get("/api/configs",
                      headers={"Host": "evil.example:8000"}).status_code == 400
    r = client.post("/api/jobs",
                    json={"motor": "test_inrunner", "model": "analytical"},
                    headers={"Origin": "http://evil.example"})
    assert r.status_code == 403
    assert client.get("/api/configs",
                      headers={"Origin": "null"}).status_code == 403
    # same-origin dashboard and localhost variants keep working
    assert client.get("/api/configs",
                      headers={"Origin": "http://localhost:8000"}).status_code == 200
    assert client.get("/api/configs",
                      headers={"Host": "127.0.0.1:8000"}).status_code == 200
    assert client.get("/api/configs", headers={"Host": "[::1]:8000"}).status_code == 200


def test_ws_rejects_cross_origin(client):
    with pytest.raises(WebSocketDisconnect), client.websocket_connect(
            "/ws", headers={"Origin": "http://evil.example"}):
        pass


# -- measured import ----------------------------------------------

def test_measured_import(client):
    r = client.post("/api/measured/test_inrunner", json=dict(MEASURED_JSON))
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["source"] == "measured"
    assert body["model"] == "resistance_test"
    assert body["dataset_id"] == "bench_r_test"

    rows = client.get("/api/results", params={"source": "measured"}).json()
    assert len(rows) == 1
    assert rows[0]["result_id"] == body["result_id"]

    detail = client.get(f"/api/results/{body['result_id']}").json()
    assert detail["metrics"]["R_s"] == 0.21


def test_measured_import_errors(client):
    bad = dict(MEASURED_JSON)
    bad["test_type"] = "not_a_test"
    assert client.post("/api/measured/test_inrunner", json=bad).status_code == 400

    no_dataset = dict(MEASURED_JSON)
    no_dataset["source_file"] = ""
    assert client.post("/api/measured/test_inrunner", json=no_dataset).status_code == 400

    assert client.post("/api/measured/nope", json=dict(MEASURED_JSON)).status_code == 404


def test_measured_import_string_value_rejected_before_persist(client):
    bad = dict(MEASURED_JSON)
    bad["quantities"] = {k: str(v) for k, v in MEASURED_JSON["quantities"].items()}
    r = client.post("/api/measured/test_inrunner", json=bad)
    assert r.status_code == 400, r.text
    assert client.get("/api/results", params={"source": "measured"}).json() == []


def test_measured_caveat_on_config_results_not_sweep_points(client):
    """A computed result on a config whose motor has measured records points
    at validation_summary; a sweep point keeps the motor name but is a
    different design, so it must not claim those measurements."""
    assert client.post("/api/measured/test_inrunner",
                       json=dict(MEASURED_JSON)).status_code == 201

    single = wait_job(client, submit(client, "analytical")["id"])
    (st,) = single["subtasks"]
    codes = {c["code"]: c for c in st["caveats"]}
    assert "motor='test_inrunner'" in codes["measured_data_available"]["text"]

    sweep = wait_job(client, submit(client, "sweep", params={
        "model_keys": ["analytical"],
        "axes": [{"field": "r_outer", "start": 0.9, "stop": 1.1, "steps": 2}]})["id"])
    assert sweep["subtasks"]
    for st in sweep["subtasks"]:
        assert "measured_data_available" not in {c["code"] for c in st["caveats"] or []}


def test_validation_summary_bound_only_is_not_validated(workspace):
    """Prius 2004: constant-L stall torque (~905 N·m) passes the ORNL
    340 N·m >= bound only because it over-predicts. The summary must carry
    the anchor's note and must not say 'validated'."""
    import json
    import shutil
    from pathlib import Path
    repo = Path(__file__).resolve().parent.parent
    shutil.copy(repo / "motors" / "prius_2004.toml", workspace["motors"])
    ref = json.loads((repo / "data/prius_2004/torque_rated.json").read_text())
    with make_client(workspace) as c:
        assert c.post("/api/measured/prius_2004", json=ref).status_code == 201
        wait_job(c, submit(c, "stall_torque", motor="prius_2004")["id"])
        s = c.get("/api/validation/prius_2004").json()
    (row,) = [r for r in s["rows"] if r["comparison_type"] == "bound"]
    assert row["passed"] and "over-predicts" in row["note"]
    assert s["diagnosis"].startswith("within bounds only")


# tau_mtpa = 1.5 * n_p * psi_f * I_rated = 1.5*2*0.1*5 = 1.5 N·m
TORQUE_JSON = {
    **MEASURED_JSON,
    "test_type": "torque_test",
    "quantities": {"tau_mtpa": 1.5},
    "uncertainty": {"tau_mtpa": 0.05},
    "source_file": "bench_torque.json",
}


def test_validate_sees_measured_data(client):
    r = client.post("/api/measured/test_inrunner", json=dict(TORQUE_JSON))
    assert r.status_code == 201, r.text

    done = wait_job(client, submit(client, "validate", params={
        "models": ["rated_torque"]})["id"])
    summary = done["summary"]
    assert "torque_test" in summary["models"]
    assert summary["n_results"] == 2
    # tau_mtpa is produced by rated_torque and measured by torque_test
    tau_rows = [row for row in summary["rows"] if row["quantity"] == "tau_mtpa"]
    assert len(tau_rows) == 1
    assert tau_rows[0]["passed"] is True
    # One computed model matching measured within tolerance → validated
    assert summary["diagnosis"] == "validated"


def test_validation_endpoint(client):
    # Empty store: summary exists but has nothing to compare
    r = client.get("/api/validation/test_inrunner")
    assert r.status_code == 200
    assert r.json()["n_results"] == 0
    assert r.json()["diagnosis"] == "insufficient data for diagnosis"

    assert client.get("/api/validation/nope").status_code == 404

    # Same payload as the validate-job summary, without submitting one
    client.post("/api/measured/test_inrunner", json=dict(TORQUE_JSON))
    wait_job(client, submit(client, "analytical")["id"])
    wait_job(client, submit(client, "rated_torque")["id"])
    summary = client.get("/api/validation/test_inrunner").json()
    assert summary["n_results"] == 3
    assert set(summary["models"]) == {"analytical", "rated_torque", "torque_test"}
    assert summary["rows"]
    row = summary["rows"][0]
    assert {"quantity", "rel_pct", "tol_pct", "passed", "comparison_type"} <= set(row)


# -- calibration workflow ---------------------------------------

def test_measured_attaches_across_config_edits(client):
    # Measured data describes the hardware: it follows the Motor NAME
    # through parameter edits, while computed results stay pinned to
    # their exact config_id (the attach rule).
    client.post("/api/measured/test_inrunner", json=dict(TORQUE_JSON))
    wait_job(client, submit(client, "rated_torque")["id"])

    # edited copy: new config name, same [motor] name, psi_f detuned +30%
    put_config(client, "variant_cal",
               mutate=lambda raw: raw["circuit"].__setitem__("psi_f", 0.13))

    summary = client.get("/api/validation/variant_cal").json()
    assert summary["models"] == ["torque_test"]  # anchor's computed run stays behind
    rows = client.get("/api/results", params={"motor": "variant_cal"}).json()
    assert [r["source"] for r in rows] == ["measured"]

    # re-run on the edited config: 1.95 vs 1.5 measured → red
    wait_job(client, submit(client, "rated_torque", motor="variant_cal")["id"])
    summary = client.get("/api/validation/variant_cal").json()
    tau = [r for r in summary["rows"] if r["quantity"] == "tau_mtpa"]
    assert len(tau) == 1 and tau[0]["passed"] is False

    # calibrate back: psi_f 0.1 reproduces the measurement → green
    put_config(client, "variant_cal", expect=200,
               mutate=lambda raw: raw["circuit"].__setitem__("psi_f", 0.1))
    wait_job(client, submit(client, "rated_torque", motor="variant_cal")["id"])
    summary = client.get("/api/validation/variant_cal").json()
    tau = [r for r in summary["rows"] if r["quantity"] == "tau_mtpa"]
    assert len(tau) == 1 and tau[0]["passed"] is True
    assert summary["diagnosis"] == "validated"


def test_measured_does_not_attach_to_different_hardware(client):
    # A config that is neither the same parameter set (config_id) nor the
    # same hardware (Motor name) sees none of the measured data. A pure
    # rename WOULD still match — identical params share a config_id by
    # the three-tier hashing design.
    client.post("/api/measured/test_inrunner", json=dict(TORQUE_JSON))

    def other_hardware(raw):
        raw["motor"]["name"] = "Different Hardware"
        raw["circuit"]["psi_f"] = 0.2
    put_config(client, "other_motor", mutate=other_hardware)
    assert client.get("/api/validation/other_motor").json()["n_results"] == 0
    assert client.get("/api/results", params={"motor": "other_motor"}).json() == []


def test_validation_surfaces_derived_params(client):
    # A dataset tagged derived_params=["psi_f"] is an echo source for
    # psi_f fits — the summary carries the tag so the dashboard can warn.
    tagged = {**TORQUE_JSON, "derived_params": ["psi_f"]}
    client.post("/api/measured/test_inrunner", json=tagged)
    summary = client.get("/api/validation/test_inrunner").json()
    assert summary["derived_params"] == {"bench_torque": ["psi_f"]}

    # untagged datasets add no entry
    plain = {**TORQUE_JSON, "source_file": "bench_torque_2.json"}
    client.post("/api/measured/test_inrunner", json=plain)
    summary = client.get("/api/validation/test_inrunner").json()
    assert summary["derived_params"] == {"bench_torque": ["psi_f"]}


def test_validation_dedupes_duplicate_and_stale_runs(client):
    """Duplicate or superseded runs of one model would reach compare_all
    and produce model↔model self-pair columns plus doubled cross columns.
    The summary loads current-only and dedupes to
    latest-per-(model, source, dataset_id) — the newest current run wins,
    superseded-version records drop out."""
    from phasesweep.sweep_types import RunConfig, RunResult

    manager = client.app.state.manager
    motor = manager.motors["test_inrunner"]
    client.post("/api/measured/test_inrunner", json=dict(TORQUE_JSON))
    wait_job(client, submit(client, "analytical")["id"])
    wait_job(client, submit(client, "rated_torque")["id"])

    # an older current copy and a stale copy (superseded physics) linger
    # beside the job's fresh one — the stale copy is NEWEST by timestamp
    # with a poison metric, so only the version check can drop it
    # (timestamp dedupe alone would let it win)
    cfg = RunConfig(motor=motor, model="analytical")
    manager.save_result(RunResult(
        config=cfg, model="analytical", status="OK",
        metrics={"fundamental": 1.0}, elapsed_s=0.1,
        timestamp="2000-01-01T00:00:00"))
    manager.save_result(RunResult(
        config=cfg, model="analytical", status="OK",
        metrics={"fundamental": 999.0}, elapsed_s=0.1, model_version=-1,
        timestamp="2999-01-01T00:00:00"))

    summary = client.get("/api/validation/test_inrunner").json()
    # one result per (model, source): analytical, rated_torque, torque_test
    assert summary["n_results"] == 3
    pairs = {(r["model_a"], r["model_b"]) for r in summary["rows"]}
    assert not any(a == b for a, b in pairs), f"self-pair leaked: {pairs}"
    # the stale record's poison metric must not have won the dedupe
    vals = [v for r in summary["rows"] for v in (r["val_a"], r["val_b"])]
    assert 999.0 not in vals, "superseded-physics record survived dedupe"


def test_validation_dedupe_prefers_ok_over_failed(client):
    """A newer TIMEOUT/ERROR record for the same (model, source,
    dataset_id) must not evict the OK record from the matrix."""
    from phasesweep.sweep_types import RunConfig, RunResult

    manager = client.app.state.manager
    motor = manager.motors["test_inrunner"]
    client.post("/api/measured/test_inrunner", json=dict(TORQUE_JSON))
    wait_job(client, submit(client, "rated_torque")["id"])

    cfg = RunConfig(motor=motor, model="rated_torque")
    manager.save_result(RunResult(
        config=cfg, model="rated_torque", status="TIMEOUT",
        metrics=None, elapsed_s=0.1, error_msg="pool deadline",
        timestamp="2999-01-01T00:00:00"))

    summary = client.get("/api/validation/test_inrunner").json()
    tau = [r for r in summary["rows"] if r["quantity"] == "tau_mtpa"]
    assert len(tau) == 1 and tau[0]["passed"] is True


# -- job-type vocabulary ----------------------------------------

def test_models_endpoint(client):
    r = client.get("/api/models")
    assert r.status_code == 200
    by_key = {m["key"]: m for m in r.json()}
    assert by_key["analytical"] == {"key": "analytical", "cost": "fast", "kind": "single"}
    assert by_key["fem"]["kind"] == "single"
    assert by_key["sweep"]["kind"] == "multi"
    assert by_key["validate"]["kind"] == "multi"
    # measured registry entries are not submittable job types
    assert "torque_test" not in by_key
    # nor models needing RunConfig fields no job param can set
    assert "operating_point" not in by_key
    assert "motor_thermal" not in by_key
    # needs_params is required params only; defaulted ones are optional
    assert "needs_params" not in by_key["fem"]
    assert by_key["fem"]["optional_params"] == ["rotation"]
    assert by_key["drive_sim"]["needs_params"] == ["sim_plan"]
    assert by_key["sweep"]["needs_params"] == ["axes"]


def test_models_endpoint_motor_applicability(client):
    by_key = {m["key"]: m for m in
              client.get("/api/models", params={"motor": "test_inrunner"}).json()}
    for m in by_key.values():
        assert {"applicable", "missing_params", "reason"} <= set(m)
        blocked = not m["applicable"] or m["missing_params"]
        assert (m["reason"] is not None) == bool(blocked), m
    assert by_key["analytical"]["applicable"] is True
    assert by_key["analytical"]["reason"] is None
    # radial motor: axial model rejected, with the validate message
    # minus the redundant "Motor '<name>': " prefix
    axial = by_key["analytical_axial"]
    assert axial["applicable"] is False
    assert "AxialFluxGeometry" in axial["reason"]
    assert not axial["reason"].startswith("Motor ")
    # load_mech has no motor-derived default; sim_plan derives from it
    two_mass = by_key["drive_sim_two_mass"]
    if two_mass["applicable"]:
        assert two_mass["missing_params"] == ["load_mech"]
    assert by_key["sweep"]["missing_params"] == ["axes"]


def test_models_endpoint_validate_default_matches_job(client):
    by_key = {m["key"]: m for m in
              client.get("/api/models", params={"motor": "test_inrunner"}).json()}
    job = submit(client, "validate")
    assert set(by_key["validate"]["default_models"]) == {
        st["model"] for st in job["subtasks"]}
    wait_job(client, job["id"])


def test_models_endpoint_unknown_motor(client):
    assert client.get("/api/models", params={"motor": "nope"}).status_code == 404


@pytest.mark.parametrize("model", ["operating_point", "motor_thermal"])
def test_submit_rejects_model_needing_unsettable_fields(client, workspace, model):
    """A model needing w_mech/tau_ref is a 400 at submit, not a failed
    job with a stored ERROR row."""
    r = client.post("/api/jobs", json={
        "motor": "test_inrunner", "model": model, "params": {}})
    assert r.status_code == 400
    assert "Python API" in r.json()["detail"]
    assert not list(ResultStore(workspace["output"]).load_slim())


# -- dashboard static files ------------------

def test_dashboard_served_at_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert 'id="main-tabs"' in r.text

    for path in ("/style.css", "/app.js", "/panels/jobs.js", "/panels/br_waveform.js"):
        assert client.get(path).status_code == 200, path


def test_vendored_plotly_served(client):
    # Pin check via index.html so a vendor bump can't silently 404
    index = client.get("/").text
    m = re.search(r'src="(/vendor/plotly-[^"]+\.min\.js)"', index)
    assert m, "index.html must reference a vendored plotly build"
    r = client.get(m.group(1))
    assert r.status_code == 200
    assert "Plotly" in r.text[:3000]


def test_api_routes_win_over_static_mount(client):
    # The "/" static mount must not shadow the API routers
    assert client.get("/api/configs").status_code == 200


def test_rated_duty_torque_pm_unchanged():
    from phasesweep.server.jobs import _rated_duty_torque
    from tests.conftest import make_motor
    tau = _rated_duty_torque(make_motor(I_rated=10.0))
    assert tau == pytest.approx(1.5 * 2 * 0.1 * 10.0, rel=1e-12)


def test_rated_duty_torque_synrm_uses_mtpa():
    from phasesweep.server.jobs import _rated_duty_torque
    from tests.conftest import make_motor
    m = make_motor(psi_f=0.0, L_d=8e-3, L_q=4e-3, I_rated=10.0)
    tau = _rated_duty_torque(m)
    assert tau == pytest.approx(0.75 * 2 * 4e-3 * 10.0**2, rel=1e-9)


def test_rated_duty_torque_synrm_no_saliency_raises():
    from phasesweep.server.jobs import _rated_duty_torque
    from tests.conftest import make_motor
    m = make_motor(psi_f=0.0, I_rated=10.0)
    with pytest.raises(ValueError, match="cannot produce torque"):
        _rated_duty_torque(m)


def test_model_defaults_duty_profile_synrm():
    from phasesweep.server.routes import _derive_model_defaults
    from tests.conftest import make_motor
    m = make_motor(psi_f=0.0, L_d=8e-3, L_q=4e-3, I_rated=10.0)
    d = _derive_model_defaults(m, "thermal_duty")
    ((tau, dt),) = d["duty_profile"]
    assert tau == pytest.approx(0.75 * 2 * 4e-3 * 10.0**2, rel=1e-9)
    assert dt == 3600.0


def test_n_planes_survives_the_worker_boundary():
    from phasesweep.machines.configs import load_motor
    from phasesweep.server.jobs import (
        _SCALAR_CONFIG_KEYS,
        _build_config_kw,
        _config_kw_of,
    )
    from phasesweep.sweep_types import RunConfig, compute_run_id
    from tests.conftest import REPO_ROOT

    motor = load_motor(REPO_ROOT / "data/parviainen_afpm/parviainen_5kw_afpm.toml")
    rc = RunConfig(motor=motor, model="fem_axial", n_planes=5,
                   rotation=0.25, cogging_points=24)

    kw = _config_kw_of(rc)
    assert kw["n_planes"] == 5
    assert kw["rotation"] == 0.25
    assert kw["cogging_points"] == 24

    assert "n_planes" in _SCALAR_CONFIG_KEYS
    rebuilt = RunConfig(motor=motor, model="fem_axial",
                        **_build_config_kw({"n_planes": 5, "rotation": 0.25,
                                            "cogging_points": 24}))
    assert compute_run_id(rebuilt) == compute_run_id(rc)

    assert compute_run_id(RunConfig(motor=motor, model="fem_axial")) != \
        compute_run_id(rc)


# ---------------------------------------------------------------------------
# __main__ argv -> settings smoke
# ---------------------------------------------------------------------------

def test_main_argv_reaches_settings_and_uvicorn(monkeypatch, tmp_path):
    """CLI flags must land in ServerSettings and uvicorn.run — a renamed
    argparse dest or dropped plumbing otherwise only fails at launch."""
    import uvicorn

    import phasesweep.server.__main__ as srv_main
    import phasesweep.server.app as srv_app

    seen: dict[str, object] = {}

    def fake_create_app(settings):
        seen["settings"] = settings
        return "SENTINEL_APP"

    def fake_run(app, **kw):
        seen["app"] = app
        seen["run_kw"] = kw

    monkeypatch.setattr(srv_app, "create_app", fake_create_app)
    monkeypatch.setattr(uvicorn, "run", fake_run)
    monkeypatch.setattr("sys.argv", [
        "phasesweep-server", "--host", "0.0.0.0", "--port", "9999",
        "--motors-dir", str(tmp_path / "m"),
        "--output-dir", str(tmp_path / "o"),
        "--workers", "2", "--subtask-timeout", "30.5", "--log-json",
    ])
    srv_main.main()

    s = seen["settings"]
    assert s.motors_dir == tmp_path / "m"
    assert s.output_dir == tmp_path / "o"
    assert s.workers == 2
    assert s.subtask_timeout_s == 30.5
    assert s.log_json is True
    assert "0.0.0.0" in s.extra_allowed_hosts
    assert seen["app"] == "SENTINEL_APP"
    assert seen["run_kw"]["host"] == "0.0.0.0"
    assert seen["run_kw"]["port"] == 9999
