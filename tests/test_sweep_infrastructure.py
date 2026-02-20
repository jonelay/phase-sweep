"""Infrastructure tests for sweep robustness.

Tests focus on three failure modes:
1. Config validation — reject invalid params before expensive computation
2. Incremental saving — crash recovery via JSONL append
3. Timeout — subprocess isolation prevents infinite hangs
"""

import json

import pytest

from phasesweep.sweep_types import MotorSweepConfig, SweepResult


# ---------------------------------------------------------------------------
# MotorSweepConfig
# ---------------------------------------------------------------------------

class TestMotorSweepConfig:

    def test_rejects_n_p_below_minimum(self):
        with pytest.raises(ValueError, match="n_p="):
            MotorSweepConfig(n_p=0, R_s=0.2, L_d=4e-3, L_q=4e-3, psi_f=0.1, J=0.002)

    def test_rejects_n_p_above_maximum(self):
        with pytest.raises(ValueError, match="n_p="):
            MotorSweepConfig(n_p=21, R_s=0.2, L_d=4e-3, L_q=4e-3, psi_f=0.1, J=0.002)

    def test_rejects_psi_f_too_small(self):
        with pytest.raises(ValueError, match="psi_f="):
            MotorSweepConfig(n_p=2, R_s=0.2, L_d=4e-3, L_q=4e-3, psi_f=1e-5, J=0.002)

    def test_rejects_L_d_too_large(self):
        with pytest.raises(ValueError, match="L_d="):
            MotorSweepConfig(n_p=2, R_s=0.2, L_d=2.0, L_q=4e-3, psi_f=0.1, J=0.002)

    def test_rejects_n_slots_negative(self):
        with pytest.raises(ValueError, match="n_slots="):
            MotorSweepConfig(n_p=2, R_s=0.2, L_d=4e-3, L_q=4e-3, psi_f=0.1, J=0.002,
                             n_slots=-1)

    def test_valid_config_at_boundaries(self):
        MotorSweepConfig(n_p=1, R_s=1e-4, L_d=1e-6, L_q=1e-6, psi_f=1e-4, J=1e-6)
        MotorSweepConfig(n_p=20, R_s=100.0, L_d=1.0, L_q=1.0, psi_f=10.0, J=100.0)

    def test_config_id_deterministic(self):
        c1 = MotorSweepConfig(n_p=2, R_s=0.2, L_d=4e-3, L_q=4e-3, psi_f=0.1, J=0.002)
        c2 = MotorSweepConfig(n_p=2, R_s=0.2, L_d=4e-3, L_q=4e-3, psi_f=0.1, J=0.002)
        assert c1.config_id == c2.config_id

    def test_config_id_unique_for_different_params(self):
        c1 = MotorSweepConfig(n_p=2, R_s=0.2, L_d=4e-3, L_q=4e-3, psi_f=0.1, J=0.002)
        c2 = MotorSweepConfig(n_p=4, R_s=0.2, L_d=4e-3, L_q=4e-3, psi_f=0.1, J=0.002)
        assert c1.config_id != c2.config_id

    def test_config_id_differs_by_psi_f(self):
        c1 = MotorSweepConfig(n_p=2, R_s=0.2, L_d=4e-3, L_q=4e-3, psi_f=0.1, J=0.002)
        c2 = MotorSweepConfig(n_p=2, R_s=0.2, L_d=4e-3, L_q=4e-3, psi_f=0.2, J=0.002)
        assert c1.config_id != c2.config_id

    def test_config_id_differs_by_n_slots(self):
        c1 = MotorSweepConfig(n_p=2, R_s=0.2, L_d=4e-3, L_q=4e-3, psi_f=0.1, J=0.002,
                              n_slots=0)
        c2 = MotorSweepConfig(n_p=2, R_s=0.2, L_d=4e-3, L_q=4e-3, psi_f=0.1, J=0.002,
                              n_slots=12)
        assert c1.config_id != c2.config_id

    def test_to_dict_roundtrip(self):
        c1 = MotorSweepConfig(n_p=2, R_s=0.2, L_d=4e-3, L_q=6e-3, psi_f=0.1, J=0.002,
                              n_slots=12, j_s=0.1)
        c2 = MotorSweepConfig.from_dict(c1.to_dict())
        assert c1 == c2

    def test_from_dict_backward_compat_defaults(self):
        """Old dicts without new fields should use sensible defaults."""
        d = {"n_p": 2, "R_s": 0.2, "L_d": 4e-3, "L_q": 4e-3,
             "psi_f": 0.1, "J": 0.002, "config_id": "ignored"}
        cfg = MotorSweepConfig.from_dict(d)
        assert cfg.n_slots == 0
        assert cfg.j_s == 0.0
        assert cfg.load_torque == 3.0

    def test_config_id_is_12_chars(self):
        cfg = MotorSweepConfig(n_p=2, R_s=0.2, L_d=4e-3, L_q=4e-3, psi_f=0.1, J=0.002)
        assert len(cfg.config_id) == 12


# ---------------------------------------------------------------------------
# SweepResult
# ---------------------------------------------------------------------------

class TestSweepResult:

    def _make_cfg(self):
        return MotorSweepConfig(n_p=2, R_s=0.2, L_d=4e-3, L_q=4e-3, psi_f=0.1, J=0.002)

    def test_ok_result(self):
        r = SweepResult(
            config=self._make_cfg(), run_type="sim", status="OK",
            metrics={"t_settle": 0.3, "i_ss": 5.1},
            elapsed_s=1.5,
        )
        assert r.status == "OK"
        assert r.metrics is not None
        assert r.error_msg is None

    def test_timeout_result(self):
        r = SweepResult(
            config=self._make_cfg(), run_type="sim", status="TIMEOUT",
            metrics=None, elapsed_s=60.0,
            error_msg="Exceeded 60s timeout",
        )
        assert r.status == "TIMEOUT"
        assert r.error_msg is not None

    def test_to_dict_roundtrip(self):
        cfg = self._make_cfg()
        r1 = SweepResult(
            config=cfg, run_type="fem", status="OK",
            metrics={"peak_Br": 0.12, "thd_pct": 8.5},
            elapsed_s=45.2,
        )
        r2 = SweepResult.from_dict(r1.to_dict())
        assert r1.status == r2.status
        assert r1.run_type == r2.run_type
        assert r1.config.config_id == r2.config.config_id
        assert r1.metrics == r2.metrics

    def test_schema_version_present(self):
        r = SweepResult(
            config=self._make_cfg(), run_type="sim", status="OK",
            metrics={}, elapsed_s=1.0,
        )
        assert r.schema_version == "v1.0"
        assert r.to_dict()["schema_version"] == "v1.0"

    def test_backward_compat_missing_run_type(self):
        d = {
            "config": {"n_p": 2, "R_s": 0.2, "L_d": 4e-3, "L_q": 4e-3,
                       "psi_f": 0.1, "J": 0.002, "config_id": "abc"},
            "status": "OK", "metrics": {}, "elapsed_s": 1.0,
        }
        r = SweepResult.from_dict(d)
        assert r.run_type == "sim"


# ---------------------------------------------------------------------------
# ResultStore
# ---------------------------------------------------------------------------

class TestResultStore:

    def _make_result(self, n_p=2):
        cfg = MotorSweepConfig(n_p=n_p, R_s=0.2, L_d=4e-3, L_q=4e-3, psi_f=0.1, J=0.002)
        return SweepResult(config=cfg, run_type="sim", status="OK",
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
        assert r.config.config_id in store.get_known_ids()

    def test_timeout_tracked_in_index(self, tmp_path):
        from phasesweep.result_store import ResultStore
        store = ResultStore(tmp_path)
        cfg = MotorSweepConfig(n_p=2, R_s=0.2, L_d=4e-3, L_q=4e-3, psi_f=0.1, J=0.002)
        r = SweepResult(config=cfg, run_type="sim", status="TIMEOUT",
                        metrics=None, elapsed_s=60.0, error_msg="timeout")
        store.save(r)
        assert cfg.config_id in store.get_known_ids()

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
        cid = r.config.config_id
        assert cid in store.get_known_ids()
        store.mark_pending({cid})
        assert cid not in store.get_known_ids()

    def test_get_stats(self, tmp_path):
        from phasesweep.result_store import ResultStore
        store = ResultStore(tmp_path)
        store.save(self._make_result(n_p=2))
        cfg = MotorSweepConfig(n_p=4, R_s=0.2, L_d=4e-3, L_q=4e-3, psi_f=0.1, J=0.002)
        store.save(SweepResult(config=cfg, run_type="sim", status="TIMEOUT",
                               metrics=None, elapsed_s=60.0, error_msg="t"))
        stats = store.get_stats()
        assert stats["total"] == 2
        assert stats["ok"] == 1
        assert stats["timeout"] == 1

    def test_load_results_returns_sweep_results(self, tmp_path):
        from phasesweep.result_store import ResultStore
        store = ResultStore(tmp_path)
        store.save(self._make_result())
        results = store.load_results()
        assert len(results) == 1
        assert isinstance(results[0], SweepResult)

    def test_malformed_line_skipped(self, tmp_path):
        from phasesweep.result_store import ResultStore
        store = ResultStore(tmp_path)
        store.save(self._make_result())
        # Inject a bad line
        with open(store.results_file, "a") as f:
            f.write("NOT_JSON\n")
        store.save(self._make_result(n_p=4))
        results = store.load_all()
        assert len(results) == 2  # bad line skipped

    def test_load_slim_returns_slim_results(self, tmp_path):
        from phasesweep.result_store import ResultStore, SlimResult
        store = ResultStore(tmp_path)
        r = self._make_result()
        store.save(r)
        slim = store.load_slim()
        assert r.config.config_id in slim
        s = slim[r.config.config_id]
        assert isinstance(s, SlimResult)
        assert s.status == "OK"
        assert s.metrics == {"t_settle": 0.3}

    def test_load_slim_deduplicates_after_rerun(self, tmp_path):
        from phasesweep.result_store import ResultStore
        store = ResultStore(tmp_path)
        cfg = MotorSweepConfig(n_p=2, R_s=0.2, L_d=4e-3, L_q=4e-3, psi_f=0.1, J=0.002)
        r1 = SweepResult(config=cfg, run_type="sim", status="ERROR",
                         metrics=None, elapsed_s=1.0, error_msg="fail")
        store.save(r1)
        r2 = SweepResult(config=cfg, run_type="sim", status="OK",
                         metrics={"t_settle": 0.5}, elapsed_s=2.0)
        store.save(r2)
        slim = store.load_slim()
        assert slim[cfg.config_id].status == "OK"
        assert slim[cfg.config_id].metrics == {"t_settle": 0.5}

    def test_load_slim_skips_malformed(self, tmp_path):
        from phasesweep.result_store import ResultStore
        store = ResultStore(tmp_path)
        store.save(self._make_result())
        with open(store.results_file, "a") as f:
            f.write("NOT_JSON\n")
        store.save(self._make_result(n_p=4))
        slim = store.load_slim()
        assert len(slim) == 2

    def test_corrupt_index_recovers_on_save(self, tmp_path):
        from phasesweep.result_store import ResultStore
        store = ResultStore(tmp_path)
        store.save(self._make_result())
        store.index_file.write_text("{corrupt")
        assert store.get_known_ids() == set()
        store.save(self._make_result(n_p=4))
        assert len(store.get_known_ids()) == 1


# ---------------------------------------------------------------------------
# sim_runner subprocess
# ---------------------------------------------------------------------------

class TestSimRunner:

    @pytest.mark.timeout(30)
    def test_returns_status_not_hang(self):
        """With a very short timeout, runner must return, never hang."""
        from phasesweep.sim_runner import run_sim_safe
        cfg = MotorSweepConfig(n_p=2, R_s=0.2, L_d=4e-3, L_q=4e-3, psi_f=0.1, J=0.002,
                               t_stop=0.1)
        result = run_sim_safe(cfg, timeout_s=2)
        assert result.status in ("OK", "TIMEOUT", "ERROR")

    @pytest.mark.timeout(60)
    def test_successful_run_returns_metrics(self):
        """Short simulation should complete and return metric keys."""
        from phasesweep.sim_runner import run_sim_safe
        cfg = MotorSweepConfig(n_p=2, R_s=0.2, L_d=4e-3, L_q=4e-3, psi_f=0.1, J=0.002,
                               t_stop=0.5)
        result = run_sim_safe(cfg, timeout_s=30)
        assert result.status == "OK", f"Expected OK, got {result.status}: {result.error_msg}"
        assert result.metrics is not None
        assert "t_settle" in result.metrics

    @pytest.mark.timeout(30)
    def test_error_captured_not_raised(self):
        """Errors in subprocess must return ERROR status, not propagate."""
        from phasesweep.sim_runner import run_sim_safe
        # t_stop=0 should cause simulation error
        cfg = MotorSweepConfig(n_p=2, R_s=0.2, L_d=4e-3, L_q=4e-3, psi_f=0.1, J=0.002,
                               t_stop=0.0)
        result = run_sim_safe(cfg, timeout_s=15)
        assert result.status in ("OK", "ERROR")
        if result.status == "ERROR":
            assert result.error_msg is not None


# ---------------------------------------------------------------------------
# fem_runner subprocess
# ---------------------------------------------------------------------------

class TestFemRunner:

    @pytest.mark.timeout(60)
    def test_returns_status_not_hang(self):
        """FEM runner must return a status, never hang."""
        from phasesweep.fem_runner import run_fem_safe
        cfg = MotorSweepConfig(n_p=2, R_s=0.2, L_d=4e-3, L_q=4e-3, psi_f=0.1, J=0.002,
                               n_theta=60, maxh=0.08)
        result = run_fem_safe(cfg, timeout_s=30)
        assert result.status in ("OK", "TIMEOUT", "ERROR")
        assert result.run_type == "fem"

    @pytest.mark.timeout(60)
    def test_ok_result_has_fem_metrics(self):
        """Successful FEM run should return peak_Br, fundamental, thd_pct."""
        from phasesweep.fem_runner import run_fem_safe
        cfg = MotorSweepConfig(n_p=2, R_s=0.2, L_d=4e-3, L_q=4e-3, psi_f=0.1, J=0.002,
                               n_theta=60, maxh=0.08)
        result = run_fem_safe(cfg, timeout_s=30)
        assert result.status == "OK", f"Expected OK, got {result.status}: {result.error_msg}"
        assert result.metrics is not None
        assert "peak_Br" in result.metrics
        assert "thd_pct" in result.metrics
        assert result.metrics["peak_Br"] > 0


# ---------------------------------------------------------------------------
# _collect_results
# ---------------------------------------------------------------------------

class TestCollectResults:

    def _make_cfg(self, n_p=2):
        return MotorSweepConfig(n_p=n_p, R_s=0.2, L_d=4e-3, L_q=4e-3, psi_f=0.1, J=0.002)

    def _make_result(self, n_p=2, status="OK", metrics=None):
        cfg = self._make_cfg(n_p)
        if metrics is None and status == "OK":
            metrics = {"t_settle": 0.3}
        return SweepResult(
            config=cfg, run_type="sim", status=status,
            metrics=metrics, elapsed_s=1.0,
            error_msg=None if status == "OK" else "fail",
        )

    def test_collect_results_memory_only(self, tmp_path):
        from phasesweep.result_store import ResultStore
        from phasesweep.sim import _collect_results
        store = ResultStore(tmp_path)
        r = self._make_result()
        in_memory = {r.config.config_id: r}
        merged = _collect_results(store, in_memory, completed=set())
        assert len(merged) == 1
        slim = merged[r.config.config_id]
        assert slim.status == "OK"
        assert slim.metrics == {"t_settle": 0.3}

    def test_collect_results_resume_merges_and_overrides(self, tmp_path):
        from phasesweep.result_store import ResultStore
        from phasesweep.sim import _collect_results
        store = ResultStore(tmp_path)
        r_a_disk = self._make_result(n_p=2, status="ERROR", metrics=None)
        r_b_disk = self._make_result(n_p=4, status="OK", metrics={"t_settle": 0.5})
        store.save(r_a_disk)
        store.save(r_b_disk)
        a_id = r_a_disk.config.config_id
        b_id = r_b_disk.config.config_id
        r_a_mem = self._make_result(n_p=2, status="OK", metrics={"t_settle": 0.4})
        in_memory = {a_id: r_a_mem}
        merged = _collect_results(store, in_memory, completed={a_id, b_id})
        assert len(merged) == 2
        assert merged[a_id].status == "OK"
        assert merged[a_id].metrics == {"t_settle": 0.4}
        assert merged[b_id].status == "OK"
