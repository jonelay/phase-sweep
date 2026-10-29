"""Infrastructure tests for sweep robustness.

Tests focus on three failure modes:
1. Config validation — reject invalid params before expensive computation
2. Incremental saving — crash recovery via JSONL append
3. Timeout — subprocess isolation prevents infinite hangs
"""

import dataclasses
import json

import pytest

from phasesweep.sweep_types import RunConfig, RunResult, compute_run_id
from tests.conftest import make_motor, requires_fem, requires_sim


def _make_rc(model="fem", **kw):
    return RunConfig(motor=make_motor(), model=model, **kw)


# ---------------------------------------------------------------------------
# RunConfig
# ---------------------------------------------------------------------------

class TestRunConfig:

    def test_construction(self):
        rc = _make_rc()
        assert rc.model == "fem"
        assert rc.maxh_fraction == 0.05

    def test_to_dict_roundtrip(self):
        rc1 = _make_rc()
        rc2 = RunConfig.from_dict(rc1.to_dict())
        assert rc2.model == rc1.model
        assert rc2.motor.config_id == rc1.motor.config_id
        assert rc2.maxh_fraction == rc1.maxh_fraction

    def test_json_roundtrip(self):
        """RunConfig survives JSON serialization (subprocess transport)."""
        rc1 = _make_rc()
        d = rc1.to_dict()
        d_json = json.loads(json.dumps(d))
        rc2 = RunConfig.from_dict(d_json)
        assert rc2.motor.config_id == rc1.motor.config_id

    @requires_sim
    def test_sim_plan_roundtrip(self):
        from phasesweep.simulation.params import prepare_drive_sim
        from phasesweep.simulation.sim import plan_sim
        plan = plan_sim(prepare_drive_sim(make_motor()))
        rc1 = RunConfig(motor=make_motor(), model="drive_sim", sim_plan=plan)
        d = json.loads(json.dumps(rc1.to_dict()))
        rc2 = RunConfig.from_dict(d)
        assert rc2.sim_plan is not None
        assert rc2.sim_plan.load_torque == rc1.sim_plan.load_torque
        assert rc2.sim_plan.t_stop == rc1.sim_plan.t_stop
        assert rc2.sim_plan.accel_window == rc1.sim_plan.accel_window
        assert rc2.sim_plan.alpha_s == rc1.sim_plan.alpha_s
        assert rc2.sim_plan.alpha_c == rc1.sim_plan.alpha_c
        assert rc2.sim_plan.T_s == rc1.sim_plan.T_s

    def test_sim_plan_none_by_default(self):
        rc = _make_rc()
        assert rc.sim_plan is None
        d = rc.to_dict()
        assert "sim_plan" not in d


# ---------------------------------------------------------------------------
# RunResult
# ---------------------------------------------------------------------------

class TestRunResult:

    def test_construction(self):
        rc = _make_rc()
        rr = RunResult(config=rc, model="fem", status="OK",
                       metrics={"peak_Br": 0.12}, elapsed_s=5.0)
        assert rr.status == "OK"
        assert rr.schema_version == "v2.1"
        assert rr.model_version is None  # stamped at save, not construction

    def test_from_dict_pre_stamp_record_defaults(self):
        """Pre-v2.1 records deserialize with no stamp and the old schema tag."""
        rc = _make_rc()
        d = RunResult(config=rc, model="fem", status="OK",
                      metrics={}, elapsed_s=1.0).to_dict()
        del d["model_version"]
        del d["schema_version"]
        rr = RunResult.from_dict(d)
        assert rr.model_version is None
        assert rr.schema_version == "v2.0"

    def test_to_dict_roundtrip(self):
        rc = _make_rc()
        rr1 = RunResult(config=rc, model="fem", status="OK",
                        metrics={"peak_Br": 0.12}, elapsed_s=5.0)
        rr2 = RunResult.from_dict(rr1.to_dict())
        assert rr2.status == rr1.status
        assert rr2.model == rr1.model
        assert rr2.metrics == rr1.metrics

    def test_legacy_run_type_mapped_to_model(self):
        rc = _make_rc()
        d = rc.to_dict()
        raw = {
            "config": d, "run_type": "sim", "status": "OK",
            "metrics": {"t_settle": 0.3}, "elapsed_s": 1.0,
        }
        rr = RunResult.from_dict(raw)
        assert rr.model == "drive_sim"

    def test_legacy_run_type_fem_unchanged(self):
        rc = _make_rc()
        d = rc.to_dict()
        raw = {
            "config": d, "run_type": "fem", "status": "OK",
            "metrics": {"peak_Br": 0.1}, "elapsed_s": 2.0,
        }
        rr = RunResult.from_dict(raw)
        assert rr.model == "fem"

    def test_missing_model_and_run_type_raises(self):
        rc = _make_rc()
        d = rc.to_dict()
        raw = {
            "config": d, "status": "OK",
            "metrics": {}, "elapsed_s": 1.0,
        }
        with pytest.raises(KeyError):
            RunResult.from_dict(raw)

    def test_legacy_descriptive_source_normalized(self):
        d = RunResult(config=_make_rc(), model="fem", status="OK",
                      metrics={}, elapsed_s=1.0, source="published").to_dict()
        d["source"] = "published (ANSYS FEM)"
        assert RunResult.from_dict(d).source == "published"

    @pytest.mark.parametrize("field,value", [
        ("source", "simulated"), ("status", "ok"), ("status", "FAILED"),
    ])
    def test_unknown_literal_rejected(self, field, value):
        d = RunResult(config=_make_rc(), model="fem", status="OK",
                      metrics={}, elapsed_s=1.0).to_dict()
        d[field] = value
        with pytest.raises(ValueError, match=field):
            RunResult.from_dict(d)


# ---------------------------------------------------------------------------
# compute_run_id
# ---------------------------------------------------------------------------

class TestComputeRunId:

    def test_determinism(self):
        rc1 = _make_rc()
        rc2 = _make_rc()
        assert compute_run_id(rc1) == compute_run_id(rc2)

    def test_different_models(self):
        rc1 = _make_rc(model="fem_linear")
        rc2 = _make_rc(model="analytical")
        assert compute_run_id(rc1) != compute_run_id(rc2)

    def test_run_id_is_12_chars(self):
        rc = _make_rc()
        assert len(compute_run_id(rc)) == 12

    def test_model_aware_filtering(self):
        """FEM model ignores sim_plan fields in hash."""
        from phasesweep.simulation.params import SimPlan
        plan1 = SimPlan(
            load_torque=3.0, load_time=0.5, t_stop=1.0, speed_step_time=0.05,
            settle_threshold=0.05, ss_window=0.1, droop_window=0.1,
            accel_window=(0.05, 0.3), alpha_s=25.0, alpha_c=1257.0,
            T_s=125e-6, tau_m=0.1,
        )
        plan2 = dataclasses.replace(plan1, load_torque=6.0)
        rc1 = _make_rc(sim_plan=plan1)
        rc2 = _make_rc(sim_plan=plan2)
        assert compute_run_id(rc1) == compute_run_id(rc2)

    def test_sim_plan_rows_load_without_motulator(self):
        """SimPlan lives beside TwoMassLoad, not in sim.py: a stored
        drive_sim row must load on a host without the [sim] extra."""
        import subprocess
        import sys

        from phasesweep.simulation.params import SimPlan
        plan = SimPlan(
            load_torque=3.0, load_time=0.5, t_stop=1.0, speed_step_time=0.05,
            settle_threshold=0.05, ss_window=0.1, droop_window=0.1,
            accel_window=(0.05, 0.3), alpha_s=25.0, alpha_c=1257.0,
            T_s=125e-6, tau_m=0.1,
        )
        d = _make_rc(model="drive_sim", sim_plan=plan).to_dict()
        code = (
            "import json, sys\n"
            "sys.modules['motulator'] = None\n"
            "from phasesweep.sweep_types import RunConfig\n"
            "rc = RunConfig.from_dict(json.loads(sys.stdin.read()))\n"
            "assert rc.sim_plan.load_torque == 3.0\n"
            "assert 'phasesweep.simulation.sim' not in sys.modules\n"
        )
        r = subprocess.run([sys.executable, "-c", code], input=json.dumps(d),
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr

    def test_model_aware_auto_lookup_differs_by_model(self):
        """Different models with same motor produce different run IDs."""
        rc_fem = _make_rc(model="fem")
        rc_sim = _make_rc(model="drive_sim")
        assert compute_run_id(rc_fem) != compute_run_id(rc_sim)

    def test_unknown_model_hashes_all_params(self):
        """Unknown models hash all solver params (no registry entry)."""
        rc1 = _make_rc(model="unknown_model", j_s=0.0)
        rc2 = _make_rc(model="unknown_model", j_s=1.0)
        assert compute_run_id(rc1) != compute_run_id(rc2)


# ---------------------------------------------------------------------------
# ResultStore
# ---------------------------------------------------------------------------

class TestResultStore:

    def _make_result(self, n_p=2):
        motor = make_motor(n_p=n_p)
        rc = RunConfig(motor=motor, model="drive_sim")
        return RunResult(config=rc, model="drive_sim", status="OK",
                         metrics={"t_settle": 0.3}, elapsed_s=1.0)

    def test_save_creates_jsonl(self, tmp_path):
        from phasesweep.result_store import ResultStore
        store = ResultStore(tmp_path)
        store.save(self._make_result())
        assert (tmp_path / "results.jsonl").exists()

    def test_load_all_returns_saved(self, tmp_path):
        from phasesweep.result_store import ResultStore
        store = ResultStore(tmp_path)
        store.save(self._make_result())
        results = store.load_all()
        assert len(results) == 1
        assert results[0]["status"] == "OK"

    def test_multiple_saves_append(self, tmp_path):
        from phasesweep.result_store import ResultStore
        store = ResultStore(tmp_path)
        for n_p in [2, 4, 6]:
            store.save(self._make_result(n_p=n_p))
        assert len(store.load_all()) == 3

    def test_get_known_ids_for_resume(self, tmp_path):
        from phasesweep.result_store import ResultStore
        store = ResultStore(tmp_path)
        r = self._make_result()
        store.save(r)
        cid = compute_run_id(r.config)
        assert cid in store.get_known_ids()

    def test_timeout_tracked_in_index(self, tmp_path):
        from phasesweep.result_store import ResultStore
        store = ResultStore(tmp_path)
        rc = RunConfig(motor=make_motor(), model="drive_sim")
        r = RunResult(config=rc, model="drive_sim", status="TIMEOUT",
                      metrics=None, elapsed_s=60.0, error_msg="timeout")
        store.save(r)
        assert compute_run_id(rc) in store.get_known_ids()

    def test_empty_store(self, tmp_path):
        from phasesweep.result_store import ResultStore
        store = ResultStore(tmp_path)
        assert store.load_all() == []
        assert store.get_known_ids() == set()



    def test_load_results_returns_run_results(self, tmp_path):
        from phasesweep.result_store import ResultStore
        store = ResultStore(tmp_path)
        store.save(self._make_result())
        results = store.load_results()
        assert len(results) == 1
        assert isinstance(results[0], RunResult)

    def test_malformed_line_skipped(self, tmp_path):
        from phasesweep.result_store import ResultStore
        store = ResultStore(tmp_path)
        store.save(self._make_result())
        with open(store.results_file, "a") as f:
            f.write("NOT_JSON\n")
        store.save(self._make_result(n_p=4))
        results = store.load_all()
        assert len(results) == 2

    def test_load_slim_returns_slim_results(self, tmp_path):
        from phasesweep.result_store import ResultStore, SlimResult
        store = ResultStore(tmp_path)
        r = self._make_result()
        store.save(r)
        slim = store.load_slim()
        cid = compute_run_id(r.config)
        assert cid in slim
        s = slim[cid]
        assert isinstance(s, SlimResult)
        assert s.status == "OK"
        assert s.metrics == {"t_settle": 0.3}
        assert s.motor_config_id == r.config.motor.config_id
        assert s.model == "drive_sim"
        assert s.source == "computed"
        assert s.timestamp == r.timestamp
        assert s.timestamp != ""

    def test_result_carries_motor_config_id(self, tmp_path):
        from phasesweep.result_store import ResultStore
        store = ResultStore(tmp_path)
        r = self._make_result()
        store.save(r)
        raw = store.load_all()
        assert raw[0]["motor_config_id"] == r.config.motor.config_id
        assert raw[0]["source"] == "computed"

    def test_load_slim_legacy_run_type(self, tmp_path):
        """load_slim maps legacy run_type to model."""
        from phasesweep.result_store import ResultStore
        store = ResultStore(tmp_path)
        r = self._make_result()
        store.save(r)
        raw = store.results_file.read_text().strip()
        d = json.loads(raw)
        d.pop("model")
        d["run_type"] = "sim"
        store.results_file.write_text(json.dumps(d) + "\n")
        slim = store.load_slim()
        assert len(slim) == 1
        s = list(slim.values())[0]
        assert s.model == "drive_sim"

    def test_load_slim_filter_by_model(self, tmp_path):
        from phasesweep.result_store import ResultStore
        store = ResultStore(tmp_path)
        store.save(self._make_result(n_p=2))
        motor4 = make_motor(n_p=4)
        rc4 = RunConfig(motor=motor4, model="fem")
        store.save(RunResult(config=rc4, model="fem", status="OK",
                             metrics={"peak_Br": 0.1}, elapsed_s=2.0))
        assert len(store.load_slim()) == 2
        assert len(store.load_slim(model="drive_sim")) == 1
        assert len(store.load_slim(model="fem")) == 1
        assert len(store.load_slim(model="analytical")) == 0

    def test_load_slim_filter_by_motor_config_id(self, tmp_path):
        from phasesweep.result_store import ResultStore
        store = ResultStore(tmp_path)
        r2 = self._make_result(n_p=2)
        r4 = self._make_result(n_p=4)
        store.save(r2)
        store.save(r4)
        mcid = r2.config.motor.config_id
        slim = store.load_slim(motor_config_id=mcid)
        assert len(slim) == 1
        assert list(slim.values())[0].motor_config_id == mcid

    def test_load_results_filter_by_model(self, tmp_path):
        from phasesweep.result_store import ResultStore
        store = ResultStore(tmp_path)
        store.save(self._make_result(n_p=2))
        motor4 = make_motor(n_p=4)
        rc4 = RunConfig(motor=motor4, model="fem")
        store.save(RunResult(config=rc4, model="fem", status="OK",
                             metrics={"peak_Br": 0.1}, elapsed_s=2.0))
        assert len(store.load_results(model="drive_sim")) == 1
        assert len(store.load_results(model="fem")) == 1


    def test_corrupt_index_recovers_on_save(self, tmp_path):
        from phasesweep.result_store import ResultStore
        store = ResultStore(tmp_path)
        store.save(self._make_result())
        store.index_file.write_text("{corrupt")
        # Simulate process restart after crash-corrupted index.
        store2 = ResultStore(tmp_path)
        assert store2.get_known_ids() == set()
        store2.save(self._make_result(n_p=4))
        assert len(store2.get_known_ids()) == 1

    def _save_and_edit_row(self, tmp_path, edit):
        """Save one result, apply `edit` to its JSONL dict, rewrite the file."""
        from phasesweep.result_store import ResultStore
        store = ResultStore(tmp_path)
        r = self._make_result()
        store.save(r)
        d = json.loads(store.results_file.read_text())
        edit(d)
        store.results_file.write_text(json.dumps(d) + "\n")
        return store, r

    def test_load_slim_keys_by_stored_run_id(self, tmp_path):
        store, _ = self._save_and_edit_row(
            tmp_path, lambda d: d.__setitem__("run_id", "sentinel00001"))
        assert list(store.load_slim()) == ["sentinel00001"]

    def test_load_slim_recomputes_when_run_id_absent(self, tmp_path):
        store, r = self._save_and_edit_row(tmp_path, lambda d: d.pop("run_id"))
        assert list(store.load_slim()) == [compute_run_id(r.config)]

    def test_load_slim_skips_hash_for_current_rows(self, tmp_path, monkeypatch):
        import phasesweep.result_store as rs
        store = rs.ResultStore(tmp_path)
        store.save(self._make_result())
        calls = []
        monkeypatch.setattr(rs, "compute_run_id",
                            lambda *a, **k: calls.append(1) or "x")
        assert len(store.load_slim()) == 1
        assert calls == []

    def test_load_slim_stale_row_never_deserialized(self, tmp_path):
        def edit(d):
            d["model_version"] = -1
            d["config"] = "not a config"
        store, _ = self._save_and_edit_row(tmp_path, edit)
        assert store.load_slim() == {}

    def test_load_slim_tolerates_sim_plan_without_sim_extra(self, tmp_path, monkeypatch):
        import sys

        from phasesweep.result_store import ResultStore

        store = ResultStore(tmp_path)
        store.save(self._make_result(n_p=4))
        rows = [json.loads(line) for line in store.results_file.read_text().splitlines()]
        legacy = dict(rows[0])
        legacy.pop("run_id")
        legacy["config"] = dict(legacy["config"], sim_plan={"load_torque": 1.0})
        legacy["config"]["motor"] = dict(legacy["config"]["motor"], name="legacy-sim")
        with open(store.results_file, "a") as f:
            f.write(json.dumps(legacy) + "\n")
        monkeypatch.setitem(sys.modules, "phasesweep.simulation.sim", None)
        slim = store.load_slim()
        assert [s.motor_name for s in slim.values()] == ["test"]

    def test_measured_models_do_not_bump(self):
        from phasesweep.registry import MODEL_REGISTRY
        measured = {n: m.version for n, m in MODEL_REGISTRY.items()
                    if m.source == "measured"}
        assert measured
        assert set(measured.values()) == {1}, measured


# ---------------------------------------------------------------------------
# Model version stamp + staleness filter
# ---------------------------------------------------------------------------

class TestModelVersionStamp:
    """Stored results carry the registry model version at save time; the
    read side serves only current-version records (missing stamp = stale)."""

    def _save_one(self, tmp_path, model="drive_sim"):
        from phasesweep.result_store import ResultStore
        store = ResultStore(tmp_path)
        rc = RunConfig(motor=make_motor(), model=model)
        store.save(RunResult(config=rc, model=model, status="OK",
                             metrics={"t_settle": 0.3}, elapsed_s=1.0))
        return store, rc

    def test_save_stamps_registry_version(self, tmp_path):
        from phasesweep.registry import MODEL_REGISTRY
        store, _ = self._save_one(tmp_path, model="fem")
        raw = json.loads(store.results_file.read_text())
        assert raw["model_version"] == MODEL_REGISTRY["fem"].version

    def test_save_stamps_v1_models_too(self, tmp_path):
        """v1 is omitted from run-ID hashing for backcompat, but MUST be
        stamped: an unstamped record must not be confusable with a v1 one.

        Any v1 model proves the point, so pick one out of the registry
        rather than naming it — a named one only stays v1 until it is
        bumped, which is how this test previously rotted.
        """
        from phasesweep.registry import MODEL_REGISTRY
        model = next(n for n, m in sorted(MODEL_REGISTRY.items())
                     if m.version == 1)
        store, _ = self._save_one(tmp_path, model=model)
        raw = json.loads(store.results_file.read_text())
        assert raw["model_version"] == 1

    def test_load_slim_serves_current_stamp(self, tmp_path):
        store, rc = self._save_one(tmp_path)
        assert compute_run_id(rc) in store.load_slim()

    def test_load_slim_drops_missing_stamp(self, tmp_path):
        store, _ = self._save_one(tmp_path)
        d = json.loads(store.results_file.read_text())
        del d["model_version"]
        store.results_file.write_text(json.dumps(d) + "\n")
        assert store.load_slim() == {}

    def test_load_slim_drops_stale_stamp(self, tmp_path):
        store, _ = self._save_one(tmp_path)
        d = json.loads(store.results_file.read_text())
        d["model_version"] -= 1
        store.results_file.write_text(json.dumps(d) + "\n")
        assert store.load_slim() == {}

    def test_load_results_current_only_drops_stale_stamp(self, tmp_path):
        store, _ = self._save_one(tmp_path)
        d = json.loads(store.results_file.read_text())
        d["model_version"] -= 1
        store.results_file.write_text(json.dumps(d) + "\n")
        assert len(store.load_results()) == 1
        assert store.load_results(current_only=True) == []

    def test_measured_records_stamp_uniformly(self, tmp_path):
        from phasesweep.result_store import ResultStore
        store = ResultStore(tmp_path)
        rc = RunConfig(motor=make_motor(), model="backemf_capture",
                       dataset_id="ds1")
        store.save(RunResult(config=rc, model="backemf_capture", status="OK",
                             metrics={}, elapsed_s=0.0, source="measured"))
        raw = json.loads(store.results_file.read_text())
        assert raw["model_version"] == 1
        assert compute_run_id(rc) in store.load_slim()

    def test_unstamped_measured_records_still_serve(self, tmp_path):
        """Measured/published records describe hardware, not physics code,
        and can't be regenerated by a re-run (fn=None) — a pre-stamp store
        must keep serving them."""
        from phasesweep.result_store import ResultStore
        store = ResultStore(tmp_path)
        rc = RunConfig(motor=make_motor(), model="backemf_capture",
                       dataset_id="ds1")
        store.save(RunResult(config=rc, model="backemf_capture", status="OK",
                             metrics={}, elapsed_s=0.0, source="measured"))
        d = json.loads(store.results_file.read_text())
        del d["model_version"]
        store.results_file.write_text(json.dumps(d) + "\n")
        assert compute_run_id(rc) in store.load_slim()

    def test_load_results_roundtrip_preserves_stamp(self, tmp_path):
        store, _ = self._save_one(tmp_path)
        rr = store.load_results()[0]
        assert rr.model_version is not None

    def test_save_does_not_mutate_caller(self, tmp_path):
        """The stamp goes into the serialized record only — in-place
        stamping changed the caller's RunResult behind its back."""
        from phasesweep.result_store import ResultStore
        store = ResultStore(tmp_path)
        rc = RunConfig(motor=make_motor(), model="drive_sim")
        rr = RunResult(config=rc, model="drive_sim", status="OK",
                       metrics={"t_settle": 0.3}, elapsed_s=1.0)
        assert rr.model_version is None
        store.save(rr)
        assert rr.model_version is None
        raw = json.loads(store.results_file.read_text())
        assert raw["model_version"] is not None

    def test_index_updates_survive_second_writer(self, tmp_path):
        """save() does a read-modify-write of the index — a second store
        instance's entries must not be clobbered by a concurrent writer."""
        from phasesweep.result_store import ResultStore
        s1 = ResultStore(tmp_path)
        s2 = ResultStore(tmp_path)
        rc1 = RunConfig(motor=make_motor(n_p=2), model="drive_sim")
        rc2 = RunConfig(motor=make_motor(n_p=4), model="drive_sim")
        rc3 = RunConfig(motor=make_motor(n_p=6), model="drive_sim")
        def mk(rc):
            return RunResult(config=rc, model="drive_sim", status="OK",
                             metrics={}, elapsed_s=1.0)
        s1.save(mk(rc1))          # populates s1's cache
        s2.save(mk(rc2))          # s1's cache no longer matches disk
        s1.save(mk(rc3))          # must not write back without rc2
        assert compute_run_id(rc2) in s1.get_known_ids()
        assert s1.get_known_ids() == s2.get_known_ids()


# ---------------------------------------------------------------------------
# fem_runner
# ---------------------------------------------------------------------------

@requires_fem
class TestFemRunner:

    @pytest.mark.timeout(120)
    def test_alpha_p_reduces_fem_fundamental(self):
        from phasesweep.solvers.fem_runner import run_fem
        m_full = make_motor(B_rem=1.2, psi_f=None, alpha_p=1.0)
        m_part = make_motor(B_rem=1.2, psi_f=None, alpha_p=0.75)
        rc_full = RunConfig(motor=m_full, model="fem", n_theta=60, maxh_fraction=0.08)
        rc_part = RunConfig(motor=m_part, model="fem", n_theta=60, maxh_fraction=0.08)
        B1_full = run_fem(rc_full)["fundamental"]
        B1_part = run_fem(rc_part)["fundamental"]
        assert B1_part < B1_full
        # Square-wave source: the FEM arc fundamental scales exactly
        # as the analytical pole-arc factor sin(π·α_p/2)
        from math import pi, sin
        assert B1_part / B1_full == pytest.approx(sin(pi * 0.75 / 2), rel=0.005)


def test_n_planes_default_does_not_change_archived_run_ids():
    """Default-suppressed emission: adding the field must not re-key any
    existing record, including on the unregistered-model fallback path
    where compute_run_id hashes every solver param."""
    from phasesweep.machines.configs import load_motor
    from phasesweep.sweep_types import RunConfig, compute_run_id
    from tests.conftest import REPO_ROOT

    motor = load_motor(REPO_ROOT / "motors/lrk_outrunner.toml")
    rc = RunConfig(motor=motor, model="not_a_registered_model")
    assert "n_planes" not in rc.to_dict()
    # pinned before n_planes existed
    assert compute_run_id(rc) == compute_run_id(RunConfig.from_dict(rc.to_dict()))


def test_demand_fields_default_does_not_change_archived_run_ids():
    """w_mech / tau_ref follow the n_planes pattern: absent from to_dict
    and the hash string when None, so no archived ID re-keys — including
    the unregistered-model path that hashes every solver param."""
    from phasesweep.machines.configs import load_motor
    from phasesweep.sweep_types import RunConfig, _run_id_key, compute_run_id
    from tests.conftest import REPO_ROOT

    motor = load_motor(REPO_ROOT / "motors/lrk_outrunner.toml")
    rc = RunConfig(motor=motor, model="not_a_registered_model")
    assert "w_mech" not in rc.to_dict() and "tau_ref" not in rc.to_dict()
    assert "w_mech" not in _run_id_key(rc) and "tau_ref" not in _run_id_key(rc)
    assert compute_run_id(rc) == compute_run_id(RunConfig.from_dict(rc.to_dict()))
    with_demand = RunConfig(motor=motor, model="not_a_registered_model",
                            w_mech=100.0, tau_ref=1.0)
    assert compute_run_id(with_demand) != compute_run_id(rc)
    assert RunConfig.from_dict(with_demand.to_dict()) == with_demand


def test_nonlinear_solver_default_suppressed_and_hashed():
    """nonlinear_solver follows the rotation/n_planes pattern: the
    "newton" default is suppressed from to_dict and the run-ID hash
    (archived IDs unchanged), while a non-default "picard" round-trips
    through to_dict/from_dict and re-keys fem-family runs."""
    from phasesweep.machines.configs import load_motor
    from phasesweep.sweep_types import RunConfig, compute_run_id
    from tests.conftest import REPO_ROOT

    motor = load_motor(REPO_ROOT / "motors/lrk_outrunner.toml")
    rc_default = RunConfig(motor=motor, model="fem", nonlinear=True)
    assert "nonlinear_solver" not in rc_default.to_dict()
    assert compute_run_id(rc_default) == compute_run_id(
        RunConfig.from_dict(rc_default.to_dict()))

    rc_picard = RunConfig(motor=motor, model="fem", nonlinear=True,
                          nonlinear_solver="picard")
    d = rc_picard.to_dict()
    assert d["nonlinear_solver"] == "picard"
    assert RunConfig.from_dict(d).nonlinear_solver == "picard"
    assert compute_run_id(rc_picard) != compute_run_id(rc_default)
    # unknown-model fallback path also stays stable at the default
    rc_unknown = RunConfig(motor=motor, model="not_a_registered_model")
    assert "nonlinear_solver" not in rc_unknown.to_dict()


_NAN = float("nan")


@pytest.mark.parametrize("kw", [
    {"nonlinear": "false"}, {"nonlinear": 1},
    {"n_theta": "180"}, {"n_theta": 180.5}, {"n_theta": True}, {"n_theta": 0},
    {"cogging_points": 2.5}, {"n_planes": "3"},
    {"nonlinear_solver": "Newton"},
    {"maxh_fraction": 0.0}, {"maxh_fraction": _NAN},
    {"j_s": _NAN}, {"rotation": float("inf")}, {"j_s": "0"},
    {"i_fault": _NAN}, {"slip": _NAN},
    {"duty_profile": ((1.0, _NAN),)}, {"duty_profile": ((1.0, 0.0),)},
], ids=repr)
def test_run_config_rejects_mistyped_inputs(kw):
    with pytest.raises(ValueError):
        RunConfig(motor=make_motor(), model="analytical", **kw)


def test_run_config_normalizes_integral_counts():
    import numpy as np
    a = RunConfig(motor=make_motor(), model="analytical", n_theta=180)
    b = RunConfig(motor=make_motor(), model="analytical", n_theta=180.0)
    c = RunConfig(motor=make_motor(), model="analytical", n_theta=np.int64(180))
    assert type(b.n_theta) is int and type(c.n_theta) is int
    assert compute_run_id(a) == compute_run_id(b) == compute_run_id(c)


def test_model_params_reject_non_finite():
    from phasesweep.models.rated_torque import RatedTorqueParams, StallTorqueParams
    from phasesweep.models.thermal_duty import ThermalDutyParams
    with pytest.raises(ValueError, match="non-finite"):
        RatedTorqueParams(n_p=4, psi_f=_NAN, I_rated=10.0)
    with pytest.raises(ValueError, match="non-finite"):
        StallTorqueParams(n_p=4, psi_f=0.1, I_stall=10.0, L_d=_NAN)
    with pytest.raises(ValueError, match="non-finite"):
        ThermalDutyParams(n_p=4, psi_f=0.1, R_s=0.5, p_s1_budget=100.0,
                          budget_source="rated_current",
                          duty_profile=((1.0, 1.0),), p_fe=_NAN)
    with pytest.raises(ValueError, match="finite"):
        ThermalDutyParams(n_p=4, psi_f=0.1, R_s=0.5, p_s1_budget=100.0,
                          budget_source="rated_current",
                          duty_profile=((_NAN, 1.0),))


@pytest.mark.parametrize("field", ["ambient_temp", "B_knee", "alpha_B_knee"])
def test_motor_rejects_non_finite_thermal_and_knee(field):
    with pytest.raises(ValueError, match=field):
        dataclasses.replace(make_motor(), **{field: _NAN})


def test_bh_curve_rejects_swapped_columns():
    from phasesweep.solvers.fem_materials import BHCurve
    with pytest.raises(ValueError, match="columns swapped"):
        BHCurve.from_rows([(0.5, 100.0), (1.0, 500.0), (1.5, 5000.0)])
