"""Tests for populate_cache.py: metric comparison and the --verify pass.

The verify pass exists to catch staleness the model-version stamp cannot
see — records written by code that has since changed, or by a bad batch
(the 2026-07-20 records carried another model's metrics).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from phasesweep.result_store import ResultStore
from phasesweep.sweep_types import RunConfig, RunResult

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import populate_cache as pc

# --- _metric_diff -----------------------------------------------------------

def test_metric_diff_identical_scalars():
    assert pc._metric_diff(1.5, 1.5, 1e-9) is None


def test_metric_diff_within_tolerance():
    assert pc._metric_diff(1.0, 1.0 + 1e-12, 1e-9) is None


def test_metric_diff_beyond_tolerance_reports_relative():
    reason = pc._metric_diff(1.0, 1.1, 1e-9)
    assert reason is not None
    assert "1e-01" in reason or "9.09e-02" in reason


def test_metric_diff_catches_scale_error_in_list():
    """The failure mode the wedge k_end band missed: a uniform scale."""
    stored = [1.0, 2.0, 3.0]
    assert pc._metric_diff(stored, [x * 1.1 for x in stored], 1e-9) is not None


def test_metric_diff_length_change():
    assert "length" in pc._metric_diff([1.0, 2.0], [1.0], 1e-9)


def test_metric_diff_nested_lists():
    assert pc._metric_diff([[1.0, 2.0]], [[1.0, 2.0]], 1e-9) is None
    assert pc._metric_diff([[1.0, 2.0]], [[1.0, 2.5]], 1e-9) is not None


def test_metric_diff_nan_matches_nan():
    assert pc._metric_diff(float("nan"), float("nan"), 1e-9) is None


def test_metric_diff_zero_pair_is_equal():
    assert pc._metric_diff(0.0, 0.0, 1e-9) is None


def test_metric_diff_strings_compare_exactly():
    assert pc._metric_diff("rated", "rated", 1e-9) is None
    assert pc._metric_diff("rated", "stall", 1e-9) is not None


def test_metric_diff_bool_not_treated_as_number():
    """bool is an int subclass — True vs 1.0 must not pass as a numeric match
    when the stored type changed."""
    assert pc._metric_diff(True, True, 1e-9) is None
    assert pc._metric_diff(True, False, 1e-9) is not None


# --- _verify ----------------------------------------------------------------

@pytest.fixture
def store_dir(tmp_path, paper_motor_8pole):
    """A store holding one genuine analytical record."""
    out = tmp_path / "out"
    store = ResultStore(out)
    rc = RunConfig(motor=paper_motor_8pole, model="analytical")
    from phasesweep.registry import MODEL_REGISTRY
    metrics = MODEL_REGISTRY["analytical"].fn(rc)
    store.save(RunResult(config=rc, model="analytical", status="OK",
                         metrics=metrics, elapsed_s=0.0))
    return out


def _rewrite(out: Path, rows):
    (out / "results.jsonl").write_text(
        "".join(json.dumps(r, default=str) + "\n" for r in rows))


def test_verify_passes_on_a_fresh_record(store_dir):
    assert pc._verify(store_dir, 1e-9, include_fem=False)[0] == set()


def test_verify_catches_perturbed_metric(store_dir):
    rows = ResultStore(store_dir).load_all()
    rows[0]["metrics"]["fundamental"] *= 1.05
    _rewrite(store_dir, rows)
    assert pc._verify(store_dir, 1e-9, include_fem=False)[0] == {0}


def test_verify_catches_cross_assigned_metrics(store_dir):
    """The bad batch: a record whose metrics belong to another model."""
    rows = ResultStore(store_dir).load_all()
    rows[0]["metrics"] = {"tau_mtpa": 1.0, "k_T": 2.0}
    _rewrite(store_dir, rows)
    assert pc._verify(store_dir, 1e-9, include_fem=False)[0] == {0}


def test_verify_skips_superseded_versions(store_dir):
    """Version-stale records are _purge_stale's job, not verify's."""
    rows = ResultStore(store_dir).load_all()
    rows[0]["model_version"] = 1
    rows[0]["metrics"]["fundamental"] *= 1.05
    _rewrite(store_dir, rows)
    assert pc._verify(store_dir, 1e-9, include_fem=False)[0] == set()


def test_verify_skips_measured_records(store_dir):
    rows = ResultStore(store_dir).load_all()
    rows[0]["source"] = "measured"
    rows[0]["metrics"]["fundamental"] *= 1.05
    _rewrite(store_dir, rows)
    assert pc._verify(store_dir, 1e-9, include_fem=False)[0] == set()


def test_verify_skips_fem_unless_asked(store_dir):
    """FEM is slow; it stays out of the default pass."""
    rows = ResultStore(store_dir).load_all()
    rows[0]["model"] = "fem"
    _rewrite(store_dir, rows)
    assert pc._verify(store_dir, 1e-9, include_fem=False)[0] == set()


def test_verify_flags_a_record_that_cannot_recompute(store_dir):
    rows = ResultStore(store_dir).load_all()
    rows[0]["config"]["motor"]["n_p"] = -1
    _rewrite(store_dir, rows)
    assert pc._verify(store_dir, 1e-9, include_fem=False)[0] == {0}


# --- coverage counts and the --require-model gate ---------------------------

def test_verify_counts_records_per_model(store_dir):
    assert pc._verify(store_dir, 1e-9, include_fem=False)[1] == {"analytical": 1}


def test_verify_counts_exclude_models_it_skipped(store_dir):
    """A model verify never checked must not appear as covered."""
    rows = ResultStore(store_dir).load_all()
    rows[0]["model"] = "fem"
    _rewrite(store_dir, rows)
    assert pc._verify(store_dir, 1e-9, include_fem=False)[1] == {}


def _run_main(monkeypatch, store_dir, *args):
    monkeypatch.setattr(sys, "argv",
                        ["populate_cache.py", "--verify",
                         "--output-dir", str(store_dir), *args])
    pc.main()


def test_gate_passes_when_required_model_is_covered(monkeypatch, store_dir):
    _run_main(monkeypatch, store_dir, "--require-model", "analytical")


def test_gate_fails_when_required_model_has_no_records(monkeypatch, store_dir):
    """The failure shape: an empty check reading as green."""
    with pytest.raises(SystemExit) as e:
        _run_main(monkeypatch, store_dir, "--require-model", "rated_torque")
    assert e.value.code == 1


def test_gate_splits_comma_separated_models(monkeypatch, store_dir):
    with pytest.raises(SystemExit):
        _run_main(monkeypatch, store_dir,
                  "--require-model", "analytical,rated_torque")


def test_gate_rejects_a_model_verify_cannot_check(monkeypatch, store_dir, caplog):
    """demag_screen is real but outside verify's model set — silently
    passing would be worse than refusing.

    Asserting on the refusal message, not just the exit code: a model
    that later joins ALL_MODELS would still exit 1 here, via the
    zero-coverage path, and this test would go on passing for the wrong
    reason. iron_loss did exactly that when it was added to FAST_MODELS.
    """
    with pytest.raises(SystemExit) as e:
        _run_main(monkeypatch, store_dir, "--require-model", "demag_screen")
    assert e.value.code == 1
    assert "not checked by --verify" in caplog.text


def test_gate_fails_on_mismatch_without_require_model(monkeypatch, store_dir):
    rows = ResultStore(store_dir).load_all()
    rows[0]["metrics"]["fundamental"] *= 1.05
    _rewrite(store_dir, rows)
    with pytest.raises(SystemExit) as e:
        _run_main(monkeypatch, store_dir)
    assert e.value.code == 1


def test_prune_mismatched_repairs_and_exits_zero(monkeypatch, store_dir):
    rows = ResultStore(store_dir).load_all()
    rows[0]["metrics"]["fundamental"] *= 1.05
    _rewrite(store_dir, rows)
    _run_main(monkeypatch, store_dir, "--prune-mismatched")
    assert ResultStore(store_dir).load_all() == []


# --- numpy scalars survive the store round-trip -----------------------------

def test_numpy_bool_round_trips_as_bool(tmp_path, paper_motor_8pole):
    """str(np.False_) stores "False", and bool("False") is True — the stored
    value has to stay a JSON bool."""
    store = ResultStore(tmp_path)
    rc = RunConfig(motor=paper_motor_8pole, model="analytical")
    store.save(RunResult(config=rc, model="analytical", status="OK",
                         metrics={"flag": np.False_, "n": np.float64(1.5)},
                         elapsed_s=0.0))
    metrics = store.load_all()[0]["metrics"]
    assert metrics["flag"] is False
    assert metrics["n"] == pytest.approx(1.5)


def test_numpy_array_round_trips_as_list(tmp_path, paper_motor_8pole):
    store = ResultStore(tmp_path)
    rc = RunConfig(motor=paper_motor_8pole, model="analytical")
    store.save(RunResult(config=rc, model="analytical", status="OK",
                         metrics={"curve": np.array([1.0, 2.0])}, elapsed_s=0.0))
    assert store.load_all()[0]["metrics"]["curve"] == [1.0, 2.0]
