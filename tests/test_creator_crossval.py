"""End-to-end cross-validation: TU Graz CREATOR Case PMSM.

Runs analytical model, imports measured back-EMF data, and validates
via the cross-validation framework.

Source: arXiv:2501.15921, DOI 10.3217/sns1d-77m43
"""


import pytest

from phasesweep.configs import load_motor
from phasesweep.crossval import compare_results, diagnose, format_table
from phasesweep.measured import import_measured
from phasesweep.registry import MODEL_REGISTRY
from phasesweep.sweep_types import RunConfig, RunResult
from tests.conftest import CREATOR_CSV_SKIP, REPO_ROOT

MOTOR_TOML = REPO_ROOT / "motors/creator_case_pmsm.toml"
MEASURED_JSON = REPO_ROOT / "data/creator_case_pmsm/backemf_measured.json"

pytestmark = pytest.mark.skipif(
    not MEASURED_JSON.exists(), reason=CREATOR_CSV_SKIP,
)


@pytest.fixture(scope="module")
def creator():
    return load_motor(MOTOR_TOML)


@pytest.fixture(scope="module")
def analytical_result(creator):
    rc = RunConfig(motor=creator, model="analytical")
    metrics = MODEL_REGISTRY["analytical"].fn(rc)
    return RunResult(
        config=rc, model="analytical", status="OK",
        metrics=metrics, elapsed_s=0.0, source="computed",
    )


@pytest.fixture(scope="module")
def measured_result(creator, tmp_path_factory):
    out = tmp_path_factory.mktemp("crossval")
    return import_measured(MEASURED_JSON, creator, out)


@pytest.fixture(scope="module")
def rated_result(creator):
    rc = RunConfig(motor=creator, model="rated_torque")
    metrics = MODEL_REGISTRY["rated_torque"].fn(rc)
    return RunResult(
        config=rc, model="rated_torque", status="OK",
        metrics=metrics, elapsed_s=0.0, source="computed",
    )


class TestCreatorCrossValidation:

    def test_analytical_backemf_within_tolerance(self, analytical_result, measured_result):
        """Analytical back-EMF vs measured should be within 5% (model-to-measured)."""
        rows = compare_results(analytical_result, measured_result)
        backemf_rows = [r for r in rows if r.quantity == "backemf_fundamental"]
        assert len(backemf_rows) == 1
        row = backemf_rows[0]
        assert row.rel_pct < 5.0, f"backemf_fundamental delta {row.rel_pct:.1f}% > 5%"
        assert row.passed

    def test_analytical_backemf_value(self, analytical_result):
        """Analytical back-EMF should be ~47.9 V (psi_f * w_e)."""
        backemf = analytical_result.metrics["backemf_fundamental"]
        assert backemf == pytest.approx(47.92, rel=0.01)

    def test_measured_backemf_value(self, measured_result):
        """Measured back-EMF from JSON is 47.37 V."""
        assert measured_result.metrics["backemf_fundamental"] == 47.37

    def test_diagnose_with_analytical_and_measured(self, analytical_result, measured_result):
        """With one model + measured, diagnose should not report failure."""
        result = diagnose([analytical_result, measured_result])
        assert result != "models disagree and none match measured — multiple problems likely"

    def test_format_table_produces_output(self, analytical_result, measured_result):
        rows = compare_results(analytical_result, measured_result)
        table = format_table(rows)
        assert "backemf_fundamental" in table
        assert "PASS" in table

    def test_rated_torque_physical(self, rated_result):
        """Rated torque should be physically reasonable for a 70W motor."""
        tau = rated_result.metrics["tau_mtpa"]
        k_T = rated_result.metrics["k_T"]
        assert 0.05 < tau < 0.5, f"tau_rated={tau:.3f} outside [0.05, 0.5] Nm"
        assert k_T > 0


# ---------------------------------------------------------------------------
# Bound comparison: torque rated
# ---------------------------------------------------------------------------

class TestCreatorTorqueBound:

    def test_import_torque_rated(self, creator, tmp_path):
        result = import_measured(
            REPO_ROOT / "data/creator_case_pmsm/torque_rated.json",
            creator, tmp_path,
        )
        assert result.source == "published"
        assert "_bound_compare" in result.metrics

    def test_bound_check_rated_gte_published(self, creator, rated_result, tmp_path):
        measured = import_measured(
            REPO_ROOT / "data/creator_case_pmsm/torque_rated.json",
            creator, tmp_path,
        )
        rows = compare_results(measured, rated_result)
        bound_rows = [r for r in rows if r.comparison_type == "bound"]
        assert len(bound_rows) == 1
        assert bound_rows[0].passed, (
            f"tau_rated={bound_rows[0].val_b:.4f} should be >= "
            f"published={bound_rows[0].val_a:.4f}"
        )


# ---------------------------------------------------------------------------
# Curve comparison: MTPA angles
# ---------------------------------------------------------------------------

class TestCreatorMTPACurve:

    def test_import_mtpa_angles(self, creator, tmp_path):
        result = import_measured(
            REPO_ROOT / "data/creator_case_pmsm/mtpa_angles.json",
            creator, tmp_path,
        )
        assert "_curve_compare" in result.metrics

    def test_curve_compare_mtpa_angles(self, creator, rated_result, tmp_path):
        measured = import_measured(
            REPO_ROOT / "data/creator_case_pmsm/mtpa_angles.json",
            creator, tmp_path,
        )
        rows = compare_results(measured, rated_result)
        curve_rows = [r for r in rows if r.comparison_type == "curve"]
        assert len(curve_rows) == 2
        for r in curve_rows:
            assert r.passed, f"{r.quantity}: {r.rel_pct:.1f}% > {r.tol_pct}%"
