"""Tests for cross-validation framework."""

import pytest

from phasesweep.crossval import (
    DEFAULT_TOLERANCE_PCT,
    TOLERANCES_MODEL_TO_MEASURED,
    TOLERANCES_MODEL_TO_MODEL,
    TOLERANCES_MODEL_TO_PUBLISHED,
    ComparisonRow,
    DiagnosisSummary,
    compare_all,
    compare_results,
    diagnose,
    diagnose_detailed,
    format_diagnosis,
    format_table,
)
from phasesweep.sweep_types import RunConfig, RunResult
from tests.conftest import make_motor


def _result(model="analytical", source="computed", metrics=None, **kw):
    m = kw.pop("motor", make_motor())
    rc = RunConfig(motor=m, model=model)
    defaults = dict(
        config=rc, model=model, status="OK",
        metrics=metrics or {}, elapsed_s=0.0, source=source,
    )
    defaults.update(kw)
    return RunResult(**defaults)


# ---------------------------------------------------------------------------
# compare_results
# ---------------------------------------------------------------------------

class TestCompareResults:

    def test_shared_quantities(self):
        a = _result(model="analytical", metrics={"fundamental": 0.5, "thd_pct": 0.0})
        b = _result(model="fem", metrics={"fundamental": 0.51, "peak_Br": 0.8})
        rows = compare_results(a, b)
        assert len(rows) == 1
        assert rows[0].quantity == "fundamental"

    def test_delta_computation(self):
        a = _result(metrics={"backemf_fundamental": 100.0})
        b = _result(model="fem", metrics={"backemf_fundamental": 105.0})
        rows = compare_results(a, b)
        assert rows[0].delta == pytest.approx(5.0)
        assert rows[0].rel_pct == pytest.approx(5.0)

    def test_pass_within_tolerance(self):
        a = _result(metrics={"fundamental": 1.0})
        b = _result(model="fem", metrics={"fundamental": 1.005})
        rows = compare_results(a, b)
        assert rows[0].passed is True

    def test_fail_outside_tolerance(self):
        a = _result(metrics={"fundamental": 1.0})
        b = _result(model="fem", metrics={"fundamental": 1.05})
        rows = compare_results(a, b)
        assert rows[0].rel_pct == pytest.approx(5.0)
        assert rows[0].passed is False

    def test_r_s_scaled_to_measurement_temperature(self):
        # 0.5 Ω copper at 20 °C reads 0.6179 Ω at 80 °C. Without scaling
        # the 20 °C reference, temperature alone eats the 15% tolerance.
        from phasesweep.thermal_duty import COPPER_TEMP_COEFF
        r_s_hot = 0.5 * (1 + COPPER_TEMP_COEFF * (80.0 - 20.0))
        measured = _result(
            model="resistance_test", source="measured",
            metrics={"R_s": r_s_hot, "_conditions": {"temperature_C": 80.0}},
        )
        reference = _result(metrics={"R_s": 0.5})
        rows = compare_results(measured, reference)
        assert rows[0].quantity == "R_s"
        assert rows[0].rel_pct == pytest.approx(0.0, abs=1e-9)
        assert rows[0].passed is True

    def test_r_s_unscaled_without_conditions(self):
        measured = _result(
            model="resistance_test", source="measured",
            metrics={"R_s": 0.6},
        )
        reference = _result(metrics={"R_s": 0.5})
        rows = compare_results(measured, reference)
        assert rows[0].rel_pct == pytest.approx(0.1 / 0.6 * 100)
        assert rows[0].passed is False

    def test_model_to_measured_wider_tolerance(self):
        a = _result(model="analytical", source="computed",
                    metrics={"backemf_fundamental": 100.0})
        b = _result(model="backemf_capture", source="measured",
                    metrics={"backemf_fundamental": 104.0})
        rows = compare_results(a, b)
        # 4% delta, model-to-measured tolerance is 5%
        assert rows[0].tol_pct == 5.0
        assert rows[0].passed is True

    def test_model_to_model_tighter_tolerance(self):
        a = _result(model="analytical", metrics={"backemf_fundamental": 100.0})
        b = _result(model="fem", metrics={"backemf_fundamental": 104.0})
        rows = compare_results(a, b)
        # 4% delta, model-to-model tolerance is 1%
        assert rows[0].tol_pct == 1.0
        assert rows[0].passed is False

    def test_custom_tolerance_override(self):
        a = _result(metrics={"fundamental": 1.0})
        b = _result(model="fem", metrics={"fundamental": 1.2})
        rows = compare_results(a, b, tolerances={"fundamental": 25.0})
        assert rows[0].tol_pct == 25.0
        assert rows[0].passed is True

    def test_no_shared_quantities(self):
        a = _result(metrics={"fundamental": 1.0})
        b = _result(model="fem", metrics={"peak_Br": 0.8})
        rows = compare_results(a, b)
        assert rows == []

    def test_skips_non_scalar(self):
        a = _result(metrics={"theta_list": [1, 2, 3], "fundamental": 0.5})
        b = _result(model="fem", metrics={"theta_list": [1, 2, 3], "fundamental": 0.5})
        rows = compare_results(a, b)
        assert len(rows) == 1
        assert rows[0].quantity == "fundamental"

    def test_none_metrics(self):
        a = _result(metrics=None)
        b = _result(model="fem", metrics={"fundamental": 0.5})
        assert compare_results(a, b) == []


# ---------------------------------------------------------------------------
# Shape scalars vs fundamental-only models
# ---------------------------------------------------------------------------

class TestShapeScalarSkip:
    """Shape scalars (thd_pct, peak_Br, ...) against a fundamental-only
    model must surface as skipped rows, not guaranteed-FAIL deltas.

    No registry model sets fundamental_only any more (the analytical
    odd-harmonic series landed), so the machinery is exercised through a
    monkeypatched fundamental-only variant; the live-registry test at the
    bottom pins that analytical THD now compares for real."""

    @pytest.fixture
    def fundamental_only_analytical(self, monkeypatch):
        import dataclasses

        from phasesweep.registry import MODEL_REGISTRY
        info = MODEL_REGISTRY["analytical"]
        monkeypatch.setitem(MODEL_REGISTRY, "analytical",
                            dataclasses.replace(info, fundamental_only=True))

    def test_thd_skipped_against_fundamental_only_model(
            self, fundamental_only_analytical):
        a = _result(model="analytical", metrics={"fundamental": 0.5, "thd_pct": 0.0})
        b = _result(model="fem", metrics={"fundamental": 0.502, "thd_pct": 4.2})
        rows = compare_results(a, b)
        thd = next(r for r in rows if r.quantity == "thd_pct")
        assert thd.comparison_type == "skipped"
        assert thd.passed is True
        fund = next(r for r in rows if r.quantity == "fundamental")
        assert fund.comparison_type == "delta"

    def test_thd_skipped_vs_measured(self, fundamental_only_analytical):
        a = _result(model="analytical", metrics={"fundamental": 1.0, "thd_pct": 0.0})
        m = _result(model="backemf_capture", source="measured",
                    metrics={"fundamental": 1.01, "thd_pct": 8.0})
        rows = compare_results(a, m)
        thd = next(r for r in rows if r.quantity == "thd_pct")
        assert thd.comparison_type == "skipped"

    def test_thd_still_compared_between_waveform_models(self):
        b = _result(model="fem", metrics={"thd_pct": 4.0})
        m = _result(model="backemf_capture", source="measured",
                    metrics={"thd_pct": 4.5})
        rows = compare_results(b, m)
        assert rows[0].comparison_type == "delta"
        assert rows[0].passed is True

    def test_thd_skip_restores_models_agree(self, fundamental_only_analytical):
        # The guaranteed-FAIL thd row used to flip models_agree and yield
        # spurious "models disagree" even with B₁ in perfect agreement.
        a = _result(model="analytical", metrics={"fundamental": 1.0, "thd_pct": 0.0})
        b = _result(model="fem", metrics={"fundamental": 1.005, "thd_pct": 5.0})
        assert diagnose([a, b]) == "models agree"

    def test_skipped_row_in_summary_and_diagnosis(
            self, fundamental_only_analytical):
        a = _result(model="analytical", metrics={"fundamental": 1.0, "thd_pct": 0.0})
        b = _result(model="fem", metrics={"fundamental": 1.005, "thd_pct": 5.0})
        summary = diagnose_detailed([a, b])
        assert len(summary.skipped_rows) == 1
        assert summary.all_pass
        assert "skipped" in format_diagnosis(summary)

    def test_analytical_thd_compares_for_real_since_odd_harmonics(self):
        # Live registry: the odd-harmonic series gives analytical a real
        # waveform, so THD rows are genuine delta comparisons now.
        a = _result(model="analytical", metrics={"thd_pct": 30.0})
        b = _result(model="fem", metrics={"thd_pct": 28.0})
        rows = compare_results(a, b)
        thd = next(r for r in rows if r.quantity == "thd_pct")
        assert thd.comparison_type == "delta"


# ---------------------------------------------------------------------------
# compare_all
# ---------------------------------------------------------------------------

class TestCompareAll:

    def test_pairwise(self):
        r1 = _result(model="analytical", metrics={"fundamental": 0.5})
        r2 = _result(model="fem", metrics={"fundamental": 0.51})
        r3 = _result(model="backemf_capture", source="measured",
                      metrics={"fundamental": 0.49})
        rows = compare_all([r1, r2, r3])
        # 3 pairs: (r1,r2), (r1,r3), (r2,r3)
        assert len(rows) == 3


# ---------------------------------------------------------------------------
# diagnose
# ---------------------------------------------------------------------------

class TestDiagnose:

    def test_insufficient_data(self):
        assert diagnose([_result()]) == "insufficient data for diagnosis"

    def test_measured_only_is_insufficient(self):
        m = _result(model="backemf_capture", source="measured",
                    metrics={"fundamental": 0.5})
        assert diagnose([m]) == "insufficient data for diagnosis"

    def test_models_agree_no_measured(self):
        a = _result(model="analytical", metrics={"fundamental": 1.0})
        b = _result(model="fem", metrics={"fundamental": 1.005})
        assert diagnose([a, b]) == "models agree"

    def test_models_disagree_no_measured(self):
        a = _result(model="analytical", metrics={"fundamental": 1.0})
        b = _result(model="fem", metrics={"fundamental": 1.5})
        assert diagnose([a, b]) == "models disagree (no measured data)"

    def test_validated(self):
        a = _result(model="analytical", metrics={"backemf_fundamental": 100.0})
        b = _result(model="fem", metrics={"backemf_fundamental": 100.5})
        m = _result(model="backemf_capture", source="measured",
                    metrics={"backemf_fundamental": 101.0})
        assert diagnose([a, b, m]) == "validated"

    def test_models_agree_measured_disagrees(self):
        a = _result(model="analytical", metrics={"backemf_fundamental": 100.0})
        b = _result(model="fem", metrics={"backemf_fundamental": 100.5})
        m = _result(model="backemf_capture", source="measured",
                    metrics={"backemf_fundamental": 150.0})
        result = diagnose([a, b, m])
        assert "motor config" in result or "measurement" in result


# ---------------------------------------------------------------------------
# format_table
# ---------------------------------------------------------------------------

class TestFormatTable:

    def test_empty(self):
        assert "no shared" in format_table([])

    def test_has_header_and_rows(self):
        rows = [ComparisonRow("fundamental", "analytical", 0.5, "fem", 0.51,
                              0.01, 2.0, 1.0, False)]
        table = format_table(rows)
        assert "Quantity" in table
        assert "fundamental" in table
        assert "FAIL" in table

    def test_pass_tag(self):
        rows = [ComparisonRow("fundamental", "analytical", 0.5, "fem", 0.501,
                              0.001, 0.2, 1.0, True)]
        table = format_table(rows)
        assert "PASS" in table


# ---------------------------------------------------------------------------
# Tolerance tables
# ---------------------------------------------------------------------------

class TestToleranceTables:

    def test_model_to_model_values(self):
        assert TOLERANCES_MODEL_TO_MODEL["fundamental"] == 1.0
        assert TOLERANCES_MODEL_TO_MODEL["thd_pct"] == 10.0
        assert TOLERANCES_MODEL_TO_MODEL["backemf_fundamental"] == 1.0

    def test_model_to_measured_wider(self):
        for q in TOLERANCES_MODEL_TO_MODEL:
            if q in TOLERANCES_MODEL_TO_MEASURED:
                assert TOLERANCES_MODEL_TO_MEASURED[q] >= TOLERANCES_MODEL_TO_MODEL[q]

    def test_default_fallback(self):
        assert DEFAULT_TOLERANCE_PCT == 10.0


# ---------------------------------------------------------------------------
# Per-dataset tolerances
# ---------------------------------------------------------------------------

class TestPerDatasetTolerances:

    def test_dataset_tolerance_wins(self):
        a = _result(model="analytical", metrics={"backemf_fundamental": 100.0})
        b = _result(model="backemf_capture", source="measured",
                    metrics={"backemf_fundamental": 108.0},
                    tolerances={"backemf_fundamental": 10.0})
        rows = compare_results(a, b)
        r = [r for r in rows if r.quantity == "backemf_fundamental"][0]
        assert r.tol_pct == 10.0  # dataset tolerance, not source-tier 5%
        assert r.passed is True

    def test_max_of_both_tolerances(self):
        a = _result(model="analytical", metrics={"fundamental": 100.0},
                    tolerances={"fundamental": 3.0})
        b = _result(model="fem", metrics={"fundamental": 105.0},
                    tolerances={"fundamental": 7.0})
        rows = compare_results(a, b)
        assert rows[0].tol_pct == 7.0  # max(3, 7)

    def test_published_tier_selection(self):
        a = _result(model="analytical", metrics={"backemf_fundamental": 100.0})
        b = _result(model="airgap_flux_test", source="published",
                    metrics={"backemf_fundamental": 104.0})
        rows = compare_results(a, b)
        r = [r for r in rows if r.quantity == "backemf_fundamental"][0]
        assert r.tol_pct == TOLERANCES_MODEL_TO_PUBLISHED["backemf_fundamental"]


# ---------------------------------------------------------------------------
# Bound comparison
# ---------------------------------------------------------------------------

class TestBoundCompare:

    def test_bound_gte_passes(self):
        measured = _result(
            model="torque_test", source="measured",
            metrics={
                "tau_rated": 0.10,
                "_bound_compare": {"tau_rated": {"computed_key": "tau_mtpa", "relation": "gte"}},
            },
        )
        computed = _result(
            model="rated_torque",
            metrics={"tau_mtpa": 0.12, "k_T": 0.3, "k_T_rms": 0.42, "gamma_opt_deg": 5.0},
        )
        rows = compare_results(measured, computed)
        bound_rows = [r for r in rows if r.comparison_type == "bound"]
        assert len(bound_rows) == 1
        assert bound_rows[0].passed is True
        assert bound_rows[0].rel_pct > 0  # positive margin

    def test_bound_gte_fails(self):
        measured = _result(
            model="torque_test", source="measured",
            metrics={
                "tau_rated": 0.10,
                "_bound_compare": {"tau_rated": {"computed_key": "tau_mtpa", "relation": "gte"}},
            },
        )
        computed = _result(
            model="rated_torque",
            metrics={"tau_mtpa": 0.08},
        )
        rows = compare_results(measured, computed)
        bound_rows = [r for r in rows if r.comparison_type == "bound"]
        assert len(bound_rows) == 1
        assert bound_rows[0].passed is False

    def test_bound_lte_passes(self):
        measured = _result(
            model="torque_test", source="measured",
            metrics={
                "limit": 100.0,
                "_bound_compare": {"limit": {"computed_key": "val", "relation": "lte"}},
            },
        )
        computed = _result(metrics={"val": 90.0})
        rows = compare_results(measured, computed)
        bound_rows = [r for r in rows if r.comparison_type == "bound"]
        assert len(bound_rows) == 1
        assert bound_rows[0].passed is True


# ---------------------------------------------------------------------------
# Curve comparison
# ---------------------------------------------------------------------------

class TestCurveCompare:

    def test_curve_interp(self):
        measured = _result(
            model="torque_test", source="measured",
            metrics={
                "gamma_at_5A": 10.0,
                "_curve_compare": {"gamma_at_5A": {
                    "curve_x": "I_curve", "curve_y": "gamma_curve_deg",
                    "at_x": 5.0, "extract": "interp",
                }},
            },
            tolerances={"gamma_at_5A": 20.0},
        )
        computed = _result(
            model="rated_torque",
            metrics={
                "I_curve": [0, 5, 10],
                "gamma_curve_deg": [0, 10.5, 20],
            },
        )
        rows = compare_results(measured, computed)
        curve_rows = [r for r in rows if r.comparison_type == "curve"]
        assert len(curve_rows) == 1
        assert curve_rows[0].val_b == pytest.approx(10.5)
        assert curve_rows[0].passed is True

    @pytest.mark.parametrize("model,n_curve,n_skipped", [
        ("fem", 1, 0),
        ("analytical", 1, 0),  # real waveform since analytical v4
        ("unregistered-model", 1, 0),
    ])
    def test_curve_extract_max(self, model, n_curve, n_skipped):
        """Waveform models get a real extract=max comparison. (The
        fundamental-only skip path this used to pin against analytical is
        covered by TestShapeScalarSkip's monkeypatched variant — since
        analytical v4 every registry waveform model has a true peak.)"""
        measured = _result(
            model="airgap_flux_test", source="published",
            metrics={
                "B_ag_peak": 0.8,
                "_curve_compare": {"B_ag_peak": {
                    "curve_x": "theta_list", "curve_y": "B_r_list", "extract": "max",
                }},
            },
            tolerances={"B_ag_peak": 5.0},
        )
        computed = _result(
            model=model,
            metrics={
                "theta_list": [0, 1, 2, 3],
                "B_r_list": [0.5, 0.82, 0.7, 0.3],
            },
        )
        rows = compare_results(measured, computed)
        curve_rows = [r for r in rows if r.comparison_type == "curve"]
        skipped = [r for r in rows if r.comparison_type == "skipped"]
        assert len(curve_rows) == n_curve
        assert len(skipped) == n_skipped
        if curve_rows:
            assert curve_rows[0].val_b == pytest.approx(0.82)
            assert curve_rows[0].passed is True
        if skipped:
            assert skipped[0].passed is True
            summary = diagnose_detailed([measured, computed])
            assert len(summary.skipped_rows) == 1
            assert "skipped" in format_diagnosis(summary)

    def test_extract_interp_allowed_for_fundamental_only_model(self):
        measured = _result(
            model="airgap_flux_test", source="published",
            metrics={
                "B_at_1": 0.8,
                "_curve_compare": {"B_at_1": {
                    "curve_x": "theta_list", "curve_y": "B_r_list",
                    "at_x": 1.0, "extract": "interp",
                }},
            },
            tolerances={"B_at_1": 5.0},
        )
        computed = _result(
            model="analytical",
            metrics={
                "theta_list": [0, 1, 2, 3],
                "B_r_list": [0.5, 0.82, 0.7, 0.3],
            },
        )
        rows = compare_results(measured, computed)
        curve_rows = [r for r in rows if r.comparison_type == "curve"]
        assert len(curve_rows) == 1
        assert curve_rows[0].val_b == pytest.approx(0.82)

    def test_curve_extrapolated_flag(self):
        measured = _result(
            model="torque_test", source="measured",
            metrics={
                "gamma_far": 10.0,
                "_curve_compare": {"gamma_far": {
                    "curve_x": "I_curve", "curve_y": "gamma_curve_deg",
                    "at_x": 100.0, "extract": "interp",
                }},
            },
            tolerances={"gamma_far": 99.0},
        )
        computed = _result(
            model="rated_torque",
            metrics={"I_curve": [0, 5, 10], "gamma_curve_deg": [0, 10, 20]},
        )
        rows = compare_results(measured, computed)
        curve_rows = [r for r in rows if r.comparison_type == "curve"]
        assert len(curve_rows) == 1
        assert curve_rows[0].extrapolated is True


# ---------------------------------------------------------------------------
# Key mapping comparison
# ---------------------------------------------------------------------------

class TestKeyMapping:

    def test_key_mapping_renames_and_compares(self):
        measured = _result(
            model="airgap_flux_test", source="published",
            metrics={
                "backemf_peak": 67.5,
                "_key_mapping": {"backemf_peak": {
                    "computed_key": "backemf_fundamental",
                    "semantic_note": "peak ≈ fundamental",
                }},
            },
            tolerances={"backemf_peak": 10.0},
        )
        computed = _result(
            model="analytical",
            metrics={"backemf_fundamental": 65.0},
        )
        rows = compare_results(measured, computed)
        mapped = [r for r in rows if r.quantity == "backemf_peak"]
        assert len(mapped) == 1
        assert mapped[0].val_b == pytest.approx(65.0)
        assert mapped[0].passed is True  # ~3.7% < 10%


# ---------------------------------------------------------------------------
# Precedence
# ---------------------------------------------------------------------------

class TestPrecedence:

    def test_bound_takes_precedence_over_direct(self):
        """If a key has both a bound_compare and a direct scalar match, bound wins."""
        measured = _result(
            model="torque_test", source="measured",
            metrics={
                "tau_rated": 0.10,
                "_bound_compare": {"tau_rated": {"computed_key": "tau_mtpa", "relation": "gte"}},
            },
        )
        computed = _result(
            model="rated_torque",
            # tau_rated present for direct match, tau_mtpa for bound lookup
            metrics={"tau_rated": 0.11, "tau_mtpa": 0.12},
        )
        rows = compare_results(measured, computed)
        tau_rows = [r for r in rows if r.quantity == "tau_rated"]
        assert len(tau_rows) == 1
        assert tau_rows[0].comparison_type == "bound"


# ---------------------------------------------------------------------------
# DiagnosisSummary
# ---------------------------------------------------------------------------

class TestDiagnosisSummary:

    def test_decompose_by_type(self):
        measured = _result(
            model="torque_test", source="measured",
            metrics={
                "tau_rated": 0.10,
                "gamma_at_5A": 10.0,
                "_bound_compare": {"tau_rated": {"computed_key": "tau_mtpa", "relation": "gte"}},
                "_curve_compare": {"gamma_at_5A": {
                    "curve_x": "I_curve", "curve_y": "gamma_curve_deg",
                    "at_x": 5.0, "extract": "interp",
                }},
            },
            tolerances={"tau_rated": 15.0, "gamma_at_5A": 20.0},
        )
        computed = _result(
            model="rated_torque",
            metrics={
                "tau_mtpa": 0.12, "k_T": 0.3,
                "I_curve": [0, 5, 10], "gamma_curve_deg": [0, 10.5, 20],
            },
        )
        summary = diagnose_detailed([measured, computed])
        assert len(summary.bound_rows) == 1
        assert len(summary.curve_rows) == 1
        assert summary.all_pass

    def test_format_diagnosis_all_pass(self):
        s = DiagnosisSummary(
            delta_rows=[ComparisonRow("q", "a", 1.0, "b", 1.0, 0, 0, 5, True)],
            bound_rows=[ComparisonRow("q2", "a", 1.0, "b", 1.1, 0.1, 10, 0, True, "bound")],
            curve_rows=[],
        )
        msg = format_diagnosis(s)
        assert "validated" in msg
        assert "1 delta" in msg
        assert "1 bound" in msg

    def test_format_diagnosis_bound_failure(self):
        s = DiagnosisSummary(
            delta_rows=[],
            bound_rows=[ComparisonRow("tau_rated", "torque_test", 0.10,
                                      "rated_torque", 0.08, -0.02, -20, 0, False, "bound")],
            curve_rows=[],
        )
        msg = format_diagnosis(s)
        assert "BOUND FAILURE" in msg

    def test_format_diagnosis_delta_and_curve_fail(self):
        s = DiagnosisSummary(
            delta_rows=[ComparisonRow("q", "a", 1.0, "b", 2.0, 1.0, 100, 5, False)],
            bound_rows=[],
            curve_rows=[ComparisonRow("g", "a", 10, "b", 20, 10, 100, 5, False, "curve")],
        )
        msg = format_diagnosis(s)
        assert "partial" in msg
        assert "delta" in msg
        assert "curve" in msg

    def test_format_diagnosis_extrapolation_warning(self):
        s = DiagnosisSummary(
            delta_rows=[],
            bound_rows=[],
            curve_rows=[ComparisonRow("g", "a", 10, "b", 10.5, 0.5, 5, 20, True,
                                      "curve", extrapolated=True)],
        )
        msg = format_diagnosis(s)
        assert "extrapolated" in msg.lower()


# ---------------------------------------------------------------------------
# Underscore-prefix filtering
# ---------------------------------------------------------------------------

class TestUnderscorePrefixFilter:

    def test_underscore_keys_excluded_from_delta(self):
        a = _result(metrics={"fundamental": 0.5, "_metadata": 99.0})
        b = _result(model="fem", metrics={"fundamental": 0.5, "_metadata": 99.0})
        rows = compare_results(a, b)
        quantities = {r.quantity for r in rows}
        assert "_metadata" not in quantities
        assert "fundamental" in quantities


# ---------------------------------------------------------------------------
# Convention-mismatch detector (R1)
# ---------------------------------------------------------------------------

class TestConventionHint:
    def test_hint_flags_sqrt2(self):
        from phasesweep.crossval import _convention_hint
        assert "√2" in (_convention_hint(0.1, 0.1 * 2 ** 0.5) or "")
        # symmetric in argument order
        assert "√2" in (_convention_hint(0.1 * 2 ** 0.5, 0.1) or "")

    def test_hint_flags_sqrt3(self):
        from phasesweep.crossval import _convention_hint
        assert "√3" in (_convention_hint(10.0, 10.0 * 3 ** 0.5) or "")

    def test_hint_none_for_unrelated_ratio(self):
        from phasesweep.crossval import _convention_hint
        assert _convention_hint(1.0, 1.25) is None
        assert _convention_hint(0.0, 1.0) is None

    def test_failed_scalar_gets_convention_note(self):
        # measured k_T is √2 above computed (peak-vs-RMS slip) → fails the 10%
        # measured tolerance and is annotated rather than read as a physics error.
        comp = _result(model="rated_torque", metrics={"k_T": 0.10})
        meas = _result(model="rated_torque", source="measured",
                       metrics={"k_T": 0.10 * 2 ** 0.5})
        rows = compare_results(comp, meas)
        row = next(r for r in rows if r.quantity == "k_T")
        assert not row.passed
        assert "convention mismatch" in row.note and "√2" in row.note

    def test_passing_scalar_has_no_note(self):
        comp = _result(model="rated_torque", metrics={"k_T": 0.10})
        meas = _result(model="rated_torque", source="measured",
                       metrics={"k_T": 0.101})
        rows = compare_results(comp, meas)
        row = next(r for r in rows if r.quantity == "k_T")
        assert row.passed and row.note == ""


class TestToleranceSelection:
    """_pick_tolerance tiers + _scalar_quantities filtering (the
    default-tolerance warning and bool exclusion were untested)."""

    def test_unknown_quantity_warns_and_uses_default(self):
        from phasesweep.crossval import _pick_tolerance
        with pytest.warns(UserWarning, match="No explicit tolerance"):
            tol = _pick_tolerance("no_such_quantity", "computed", "measured")
        assert tol == DEFAULT_TOLERANCE_PCT

    def test_known_quantity_does_not_warn(self):
        import warnings

        from phasesweep.crossval import (
            TOLERANCES_MODEL_TO_MODEL,
            _pick_tolerance,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            tol = _pick_tolerance("fundamental", "computed", "computed")
        assert tol == TOLERANCES_MODEL_TO_MODEL["fundamental"]

    def test_dataset_tolerance_beats_default_without_warning(self):
        import warnings

        from phasesweep.crossval import _pick_tolerance
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            tol = _pick_tolerance("no_such_quantity", "computed", "measured",
                                  tol_a={"no_such_quantity": 3.5})
        assert tol == 3.5

    def test_scalar_quantities_excludes_bools_and_meta(self):
        from phasesweep.crossval import _scalar_quantities
        out = _scalar_quantities(
            {"a": 1.0, "flag": True, "_meta": 2.0, "n": 3})
        assert out == {"a": 1.0, "n": 3}
        assert _scalar_quantities(None) == {}
