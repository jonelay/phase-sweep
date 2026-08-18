"""Tests for phasesweep.parallel — parallel job execution."""

from __future__ import annotations

import time
from unittest.mock import patch

from phasesweep.geo_parallel import _run_worker, execute_generic, execute_parallel
from phasesweep.machines.geometry import default_inrunner
from phasesweep.machines.motor import Motor


def _sleepy_worker(job: dict) -> dict:
    time.sleep(job.get("sleep_s", 0))
    return {"tag": job.get("tag", ""), "status": "OK", "metrics": {}}


def _analytical_motor() -> Motor:
    return Motor(
        name="test-parallel",
        geometry=default_inrunner(),
        n_p=4,
        B_rem=1.2,
        mu_r_pm=1.05,
    )


def _make_job(motor: Motor, model_key: str = "analytical", **kw) -> dict:
    return {
        "motor_dict": motor.to_dict(),
        "model_key": model_key,
        "config_kw": kw.get("config_kw", {}),
        "tag": kw.get("tag", "test"),
        "param": kw.get("param", "B_rem"),
        "delta_idx": kw.get("delta_idx", 0),
    }


class TestRunWorker:
    def test_round_trip(self):
        job = _make_job(_analytical_motor())
        result = _run_worker(job)
        assert result["status"] == "OK"
        assert result["model_key"] == "analytical"
        assert result["metrics"] is not None
        assert "fundamental" in result["metrics"]
        assert result["elapsed_s"] > 0
        assert result["error_msg"] is None
        assert result["tag"] == "test"
        assert result["param"] == "B_rem"
        assert result["delta_idx"] == 0

    def test_error_handling(self):
        job = {
            "motor_dict": {"name": "bad", "geometry": {}, "n_p": 2},
            "model_key": "analytical",
            "config_kw": {},
            "tag": "err",
            "param": "x",
            "delta_idx": 0,
        }
        result = _run_worker(job)
        assert result["status"] == "ERROR"
        assert result["error_msg"] is not None
        assert result["metrics"] is None

    def test_no_fn_model(self):
        job = _make_job(_analytical_motor(), model_key="backemf_capture")
        result = _run_worker(job)
        assert result["status"] == "ERROR"
        assert "no fn" in result["error_msg"]

    def test_tag_fields_echoed(self):
        job = _make_job(_analytical_motor(), tag="OD:3:fem", param="OD", delta_idx=7)
        result = _run_worker(job)
        assert result["tag"] == "OD:3:fem"
        assert result["param"] == "OD"
        assert result["delta_idx"] == 7


class TestExecuteParallel:
    def test_empty_jobs(self):
        assert execute_parallel([]) == []

    def test_sequential(self):
        motor = _analytical_motor()
        jobs = [_make_job(motor, tag=f"j{i}", delta_idx=i) for i in range(3)]
        results = execute_parallel(jobs, workers=1)
        assert len(results) == 3
        assert all(r["status"] == "OK" for r in results)
        assert {r["delta_idx"] for r in results} == {0, 1, 2}

    def test_sequential_on_complete_callback(self):
        motor = _analytical_motor()
        jobs = [_make_job(motor, delta_idx=i) for i in range(3)]
        progress = []

        def cb(result, done, total):
            progress.append((done, total, result["status"]))

        execute_parallel(jobs, workers=1, on_complete=cb)
        assert len(progress) == 3
        assert progress[-1] == (3, 3, "OK")

    def test_parallel_two_workers(self):
        motor = _analytical_motor()
        jobs = [_make_job(motor, tag=f"j{i}", delta_idx=i) for i in range(4)]
        results = execute_parallel(jobs, workers=2)
        assert len(results) == 4
        assert all(r["status"] == "OK" for r in results)
        assert {r["delta_idx"] for r in results} == {0, 1, 2, 3}

    def test_parallel_on_complete_callback(self):
        motor = _analytical_motor()
        jobs = [_make_job(motor, delta_idx=i) for i in range(4)]
        progress = []

        def cb(result, done, total):
            progress.append((done, total))

        execute_parallel(jobs, workers=2, on_complete=cb)
        assert len(progress) == 4
        assert progress[-1][1] == 4

    def test_disk_cache_init_called(self):
        motor = _analytical_motor()
        jobs = [_make_job(motor)]
        with patch("phasesweep.geo_parallel._worker_init") as mock_init:
            # Sequential path must honor cache_dir like the pool path does
            execute_parallel(jobs, workers=1, cache_dir="/tmp/test_cache")
            mock_init.assert_called_once_with("/tmp/test_cache")

    def test_mixed_ok_and_error(self):
        good = _make_job(_analytical_motor(), tag="good", delta_idx=0)
        bad = {
            "motor_dict": {"name": "bad", "geometry": {}, "n_p": 2},
            "model_key": "analytical",
            "config_kw": {},
            "tag": "bad",
            "param": "x",
            "delta_idx": 1,
        }
        results = execute_parallel([good, bad], workers=1)
        assert len(results) == 2
        statuses = {r["tag"]: r["status"] for r in results}
        assert statuses["good"] == "OK"
        assert statuses["bad"] == "ERROR"

    def test_failed_future_forwards_job_keys(self):
        """Future-level failure (unpicklable job) keeps routing keys."""
        import threading
        good = _make_job(_analytical_motor(), tag="good")
        bad = _make_job(_analytical_motor(), tag="bad")
        bad["point_idx"] = 7
        bad["unpicklable"] = threading.Lock()
        results = execute_parallel([good, bad], workers=2)
        failed = [r for r in results if r["status"] == "ERROR"]
        assert len(failed) == 1
        assert failed[0]["tag"] == "bad"
        assert failed[0]["point_idx"] == 7
        assert failed[0]["metrics"] is None

    def test_straggler_becomes_timeout(self, monkeypatch):
        """A worker outliving the pool timeout yields TIMEOUT, not a hang."""
        import phasesweep.geo_parallel as par
        monkeypatch.setattr(par, "_FUTURE_TIMEOUT_S", 1)
        jobs = [{"tag": "slow", "sleep_s": 30, "point_idx": 3}]
        t0 = time.monotonic()
        results = execute_generic(_sleepy_worker, jobs, workers=2)
        assert time.monotonic() - t0 < 20
        assert len(results) == 1
        assert results[0]["status"] == "TIMEOUT"
        assert results[0]["point_idx"] == 3

    def test_pool_deadline_distinguishes_not_started(self):
        """A queued job hit by the pool deadline reports elapsed 0 and a
        'not started' message — not the full deadline as if it had run."""
        jobs = [
            {"tag": t, "sleep_s": 30, "point_idx": i}
            for i, t in enumerate("abcd")
        ]
        results = execute_generic(_sleepy_worker, jobs, workers=2, timeout_s=2)
        assert len(results) == 4
        assert all(r["status"] == "TIMEOUT" for r in results)
        queued = [r for r in results if "Not started" in r["error_msg"]]
        running = [r for r in results if "deadline while" in r["error_msg"]]
        # 2 workers prefetch workers+1 jobs into the call queue (those are
        # uncancellable); the 4th sits beyond the buffer and must be
        # reported as never started, not as having run the full deadline
        assert len(queued) >= 1
        assert len(queued) + len(running) == 4
        assert all(r["elapsed_s"] == 0.0 for r in queued)
        assert all(r["elapsed_s"] == 2.0 for r in running)
