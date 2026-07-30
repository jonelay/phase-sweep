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
from tests.conftest import make_motor


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

    def test_sim_plan_roundtrip(self):
        from phasesweep.sim import plan_sim
        from phasesweep.solver_params import prepare_drive_sim
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
        from phasesweep.sim import SimPlan
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

    def test_mark_pending_removes_from_index(self, tmp_path):
        from phasesweep.result_store import ResultStore
        store = ResultStore(tmp_path)
        r = self._make_result()
        store.save(r)
        cid = compute_run_id(r.config)
        assert cid in store.get_known_ids()
        store.mark_pending({cid})
        assert cid not in store.get_known_ids()

    def test_get_stats(self, tmp_path):
        from phasesweep.result_store import ResultStore
        store = ResultStore(tmp_path)
        store.save(self._make_result(n_p=2))
        rc = RunConfig(motor=make_motor(n_p=4), model="drive_sim")
        store.save(RunResult(config=rc, model="drive_sim", status="TIMEOUT",
                             metrics=None, elapsed_s=60.0, error_msg="t"))
        stats = store.get_stats()
        assert stats["total"] == 2
        assert stats["ok"] == 1
        assert stats["timeout"] == 1

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

    def test_mark_pending_atomic_write(self, tmp_path):
        """mark_pending uses atomic tempfile+os.replace, not bare write_text."""
        from phasesweep.result_store import ResultStore
        store = ResultStore(tmp_path)
        r = self._make_result()
        store.save(r)
        cid = compute_run_id(r.config)
        store.mark_pending({cid})
        index = json.loads(store.index_file.read_text())
        assert cid not in index

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
        stamped: an unstamped record must not be confusable with a v1 one."""
        store, _ = self._save_one(tmp_path, model="torque_speed")
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
        """_update_index rewrites the whole index dict — a stale in-memory
        cache must not clobber entries another store instance added since
        our last read."""
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
# sim_runner subprocess
# ---------------------------------------------------------------------------

class TestSimRunner:

    @pytest.mark.timeout(30)
    def test_returns_status_not_hang(self):
        from phasesweep.sim import plan_sim
        from phasesweep.sim_runner import run_sim_safe
        from phasesweep.solver_params import prepare_drive_sim
        motor = make_motor()
        plan = plan_sim(prepare_drive_sim(motor))
        short_plan = dataclasses.replace(plan, t_stop=0.1)
        rc = RunConfig(motor=motor, model="drive_sim", sim_plan=short_plan)
        result = run_sim_safe(rc, timeout_s=2)
        assert result.status in ("OK", "TIMEOUT", "ERROR")

    @pytest.mark.timeout(60)
    def test_successful_run_returns_metrics(self):
        from phasesweep.sim import plan_sim
        from phasesweep.sim_runner import run_sim_safe
        from phasesweep.solver_params import prepare_drive_sim
        motor = make_motor()
        plan = plan_sim(prepare_drive_sim(motor))
        rc = RunConfig(motor=motor, model="drive_sim", sim_plan=plan)
        result = run_sim_safe(rc, timeout_s=30)
        assert result.status == "OK", f"Expected OK, got {result.status}: {result.error_msg}"
        assert result.metrics is not None
        assert "t_settle" in result.metrics

    @pytest.mark.timeout(30)
    def test_missing_sim_plan_raises(self):
        from phasesweep.sim_runner import run_sim_safe
        rc = _make_rc(model="drive_sim")
        result = run_sim_safe(rc, timeout_s=15)
        assert result.status == "ERROR"
        assert result.error_msg is not None


# ---------------------------------------------------------------------------
# fem_runner subprocess
# ---------------------------------------------------------------------------

class TestFemRunner:

    @pytest.mark.timeout(60)
    def test_returns_status_not_hang(self):
        from phasesweep.fem_runner import run_fem_safe
        rc = _make_rc(model="fem", n_theta=60, maxh_fraction=0.08)
        result = run_fem_safe(rc, timeout_s=30)
        assert result.status in ("OK", "TIMEOUT", "ERROR")
        assert result.model == "fem"

    @pytest.mark.timeout(60)
    def test_ok_result_has_fem_metrics(self):
        from phasesweep.fem_runner import run_fem_safe
        rc = _make_rc(model="fem", n_theta=60, maxh_fraction=0.08)
        result = run_fem_safe(rc, timeout_s=30)
        assert result.status == "OK", f"Expected OK, got {result.status}: {result.error_msg}"
        assert result.metrics is not None
        assert "peak_Br" in result.metrics
        assert "thd_pct" in result.metrics
        assert result.metrics["peak_Br"] > 0

    @pytest.mark.timeout(120)
    def test_alpha_p_reduces_fem_fundamental(self):
        from phasesweep.fem_runner import _run_fem_impl
        m_full = make_motor(B_rem=1.2, psi_f=None, alpha_p=1.0)
        m_part = make_motor(B_rem=1.2, psi_f=None, alpha_p=0.75)
        rc_full = RunConfig(motor=m_full, model="fem", n_theta=60, maxh_fraction=0.08)
        rc_part = RunConfig(motor=m_part, model="fem", n_theta=60, maxh_fraction=0.08)
        B1_full = _run_fem_impl(rc_full)["fundamental"]
        B1_part = _run_fem_impl(rc_part)["fundamental"]
        assert B1_part < B1_full
        # Square-wave source: the FEM arc fundamental scales exactly
        # as the analytical pole-arc factor sin(π·α_p/2)
        from math import pi, sin
        assert B1_part / B1_full == pytest.approx(sin(pi * 0.75 / 2), rel=0.005)

