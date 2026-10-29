"""Tests for phasesweep.geo_parallel — parallel job execution."""

from __future__ import annotations

import time

import pytest

from phasesweep.geo_parallel import _run_worker, execute_generic, execute_parallel
from phasesweep.machines.geometry import default_inrunner
from phasesweep.machines.motor import DriveParams, Motor


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


def _ts_motor(**overrides) -> Motor:
    kw = dict(name="test-ts", geometry=None, n_p=4, psi_f=0.1, R_s=0.5,
              L_d=1e-3, L_q=2e-3,
              drive=DriveParams(U_DC=540.0, MAX_I_S=20.0, W_REF=100.0))
    kw.update(overrides)
    return Motor(**kw)


def _strip_elapsed(r: dict) -> dict:
    return {k: v for k, v in r.items() if k != "elapsed_s"}


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

    def test_disk_cache_init_called(self, tmp_path):
        from phasesweep.solvers.fem_mesh import disk_cache_dir, set_disk_cache_dir
        set_disk_cache_dir(None)
        try:
            motor = _analytical_motor()
            jobs = [_make_job(motor)]
            execute_parallel(jobs, workers=1, cache_dir=str(tmp_path))
            assert disk_cache_dir() == tmp_path
        finally:
            set_disk_cache_dir(None)

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

    def test_straggler_becomes_timeout(self):
        """A worker outliving the pool timeout yields TIMEOUT, not a hang."""
        jobs = [{"tag": "slow", "sleep_s": 30, "point_idx": 3}]
        t0 = time.monotonic()
        results = execute_generic(_sleepy_worker, jobs, workers=2, timeout_s=1)
        assert time.monotonic() - t0 < 20
        assert len(results) == 1
        assert results[0]["status"] == "TIMEOUT"
        assert results[0]["point_idx"] == 3

    def test_pool_without_timeout_has_no_deadline(self, monkeypatch):
        """No timeout_s means no deadline with workers > 1, as with one
        worker: a sweep's results must not depend on its worker count."""
        import concurrent.futures as cf

        import phasesweep.geo_parallel as par
        seen = []
        real = cf.as_completed

        def spy(fs, timeout=None):
            seen.append(timeout)
            return real(fs, timeout=timeout)

        monkeypatch.setattr(par, "as_completed", spy)
        jobs = [{"tag": "fast", "sleep_s": 0, "point_idx": i} for i in range(2)]
        results = execute_generic(_sleepy_worker, jobs, workers=2)
        assert seen == [None]
        assert all(r["status"] != "TIMEOUT" for r in results)

    def test_single_worker_with_timeout_uses_pool_deadline(self):
        """workers=1 plus a timeout_s must not fall back to the inline
        path: a hung solver there has no deadline at all."""
        jobs = [{"tag": "slow", "sleep_s": 30, "point_idx": 0}]
        t0 = time.monotonic()
        results = execute_generic(_sleepy_worker, jobs, workers=1, timeout_s=1)
        assert time.monotonic() - t0 < 20
        assert results[0]["status"] == "TIMEOUT"

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


class TestBatchDispatch:
    def test_batch_groups_torque_speed(self):
        """Batchable model grouped, results match serial."""
        motor = _ts_motor()
        jobs = [_make_job(motor, model_key="torque_speed", tag=f"t{i}",
                          delta_idx=i) for i in range(10)]
        batched = execute_parallel([dict(j) for j in jobs], workers=1,
                                   batch_size=4)
        serial = execute_parallel([dict(j) for j in jobs], workers=1)
        assert len(batched) == 10
        b = {r["delta_idx"]: r for r in batched}
        s = {r["delta_idx"]: r for r in serial}
        for i in range(10):
            assert b[i]["status"] == "OK"
            assert _strip_elapsed(b[i]) == _strip_elapsed(s[i])

    def test_mixed_batch_and_single(self):
        """Mix of torque_speed (batchable) and analytical (not batchable)."""
        ts, an = _ts_motor(), _analytical_motor()
        jobs = (
            [_make_job(ts, model_key="torque_speed", tag=f"ts{i}",
                       delta_idx=i) for i in range(3)]
            + [_make_job(an, model_key="analytical", tag=f"an{i}",
                         delta_idx=10 + i) for i in range(2)]
        )
        batched = execute_parallel([dict(j) for j in jobs], workers=1,
                                   batch_size=4)
        serial = execute_parallel([dict(j) for j in jobs], workers=1)
        assert len(batched) == 5
        b = {r["delta_idx"]: r for r in batched}
        s = {r["delta_idx"]: r for r in serial}
        assert set(b) == set(s) == {0, 1, 2, 10, 11}
        for i in b:
            assert b[i]["status"] == "OK"
            assert _strip_elapsed(b[i]) == _strip_elapsed(s[i])

    def test_on_complete_per_item(self):
        """on_complete called once per original item, monotone done count."""
        ts, an = _ts_motor(), _analytical_motor()
        jobs = (
            [_make_job(ts, model_key="torque_speed", delta_idx=i)
             for i in range(7)]
            + [_make_job(an, model_key="analytical", delta_idx=10 + i)
               for i in range(2)]
        )
        progress = []

        def cb(result, done, total):
            progress.append((done, total, result["delta_idx"]))

        execute_parallel(jobs, workers=1, on_complete=cb, batch_size=3)
        assert len(progress) == 9
        assert [p[0] for p in progress] == list(range(1, 10))
        assert all(p[1] == 9 for p in progress)
        assert len({p[2] for p in progress}) == 9

    def test_per_item_error_in_batch(self):
        """One bad motor mid-batch → that item ERROR, siblings OK."""
        good = _ts_motor()
        bad = _ts_motor(name="test-ts-bad", L_d=None)
        jobs = [
            _make_job(good, model_key="torque_speed", tag="g0", delta_idx=0),
            _make_job(bad, model_key="torque_speed", tag="bad", delta_idx=1),
            _make_job(good, model_key="torque_speed", tag="g2", delta_idx=2),
        ]
        results = execute_parallel(jobs, workers=1, batch_size=8)
        assert len(results) == 3
        by_idx = {r["delta_idx"]: r for r in results}
        assert by_idx[0]["status"] == "OK"
        assert by_idx[2]["status"] == "OK"
        assert by_idx[1]["status"] == "ERROR"
        assert by_idx[1]["error_msg"]
        assert by_idx[1]["metrics"] is None
        serial = _run_worker(_make_job(good, model_key="torque_speed",
                                       tag="g0", delta_idx=0))
        assert by_idx[0]["metrics"] == serial["metrics"]

    def test_batch_size_none_is_current_behavior(self):
        """batch_size=None → identical results to today."""
        motor = _ts_motor()
        jobs = [_make_job(motor, model_key="torque_speed", tag=f"t{i}",
                          delta_idx=i) for i in range(3)]
        default = execute_parallel([dict(j) for j in jobs], workers=1)
        explicit = execute_parallel([dict(j) for j in jobs], workers=1,
                                    batch_size=None)
        assert [_strip_elapsed(r) for r in default] == \
               [_strip_elapsed(r) for r in explicit]
        assert all(r["status"] == "OK" for r in default)

    def test_pool_batched_matches_serial(self):
        """Pool path (workers=2) with batch_size produces same metrics as serial."""
        motor = _ts_motor()
        jobs = [_make_job(motor, model_key="torque_speed", tag=f"t{i}",
                          delta_idx=i) for i in range(10)]
        batched = execute_parallel([dict(j) for j in jobs], workers=2,
                                   batch_size=4)
        serial = execute_parallel([dict(j) for j in jobs], workers=1)
        assert len(batched) == 10
        b = {r["delta_idx"]: r for r in batched}
        s = {r["delta_idx"]: r for r in serial}
        for i in range(10):
            assert b[i]["status"] == "OK"
            assert b[i]["metrics"] == s[i]["metrics"]

    def test_batch_future_failure_fans_out(self):
        """A batch-job-level failure fans out to every item in the batch."""
        import threading
        motor = _ts_motor()
        jobs = [_make_job(motor, model_key="torque_speed", tag=f"t{i}",
                          delta_idx=i) for i in range(4)]
        # With workers=2 the 4 jobs split into 2 batches of 2 (items 0-1
        # and 2-3). The Lock makes the second batch unpicklable, so that
        # whole batch fails at submit and fans out to both its items.
        jobs[2]["unpicklable"] = threading.Lock()
        results = execute_parallel(jobs, workers=2, batch_size=8)
        assert len(results) == 4
        by_idx = {r["delta_idx"]: r for r in results}
        # Routing keys preserved on each
        assert set(by_idx) == {0, 1, 2, 3}
        assert by_idx[0]["status"] == "OK"
        assert by_idx[1]["status"] == "OK"
        for i in (2, 3):
            assert by_idx[i]["status"] == "ERROR"
            assert by_idx[i]["metrics"] is None
            assert by_idx[i]["error_msg"]

    def test_empty_jobs_with_batch_size(self):
        """Empty jobs with batch_size returns [] without error."""
        assert execute_parallel([], batch_size=128) == []

    def test_cancel_pre_set_with_batching(self):
        """Pre-set cancel event with batch_size → all items CANCELLED."""
        import threading
        motor = _ts_motor()
        jobs = [_make_job(motor, model_key="torque_speed", tag=f"t{i}",
                          delta_idx=i) for i in range(10)]
        cancel = threading.Event()
        cancel.set()
        results = execute_parallel(jobs, workers=1, batch_size=4,
                                   cancel=cancel)
        assert len(results) == 10
        assert all(r["status"] == "CANCELLED" for r in results)
        assert {r["delta_idx"] for r in results} == set(range(10))

    def test_batch_size_zero_raises(self):
        """batch_size=0 raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            execute_parallel([], batch_size=0)

    def test_batch_equivalence_full_round_trip(self, tmp_path):
        """Store contents after batched sweep == serial sweep, exact metrics."""
        from phasesweep.result_store import ResultStore
        from phasesweep.sweep_types import RunConfig, RunResult, compute_run_id

        motors = [_ts_motor(name=f"ts-{i}", psi_f=0.1 + 0.01 * i)
                  for i in range(5)]
        configs = [RunConfig(motor=m, model="torque_speed") for m in motors]
        jobs = [
            {
                "motor_dict": c.motor.to_dict(),
                "model_key": "torque_speed",
                "config_kw": {},
                "run_id": compute_run_id(c),
            }
            for c in configs
        ]
        by_run_id = {compute_run_id(c): c for c in configs}

        def sweep(store_dir, batch_size):
            store = ResultStore(store_dir)

            def save(result, done, total):
                c = by_run_id[result["run_id"]]
                store.save(RunResult(
                    config=c, model="torque_speed",
                    status=result["status"], metrics=result["metrics"],
                    elapsed_s=0.0, error_msg=result["error_msg"],
                ))

            execute_parallel([dict(j) for j in jobs], workers=1,
                             on_complete=save, batch_size=batch_size)
            return store

        serial = sweep(tmp_path / "serial", None)
        batched = sweep(tmp_path / "batched", 2)

        s_recs = {r["config"]["motor"]["name"]: r for r in serial.load_all()}
        b_recs = {r["config"]["motor"]["name"]: r for r in batched.load_all()}
        assert set(s_recs) == set(b_recs)
        assert len(s_recs) == 5
        for name in s_recs:
            assert s_recs[name]["status"] == "OK"
            assert b_recs[name]["metrics"] == s_recs[name]["metrics"]
