"""End-to-end cross-validation: TU Graz CREATOR Case PMSM.

Imports measured back-EMF data and validates rated-torque outputs via
the cross-validation framework. The inset rotor
is FEM-only — the analytical solver rejects it, so no analytical
comparisons run here.

Source: arXiv:2501.15921, DOI 10.3217/sns1d-77m43
"""


import pytest

from phasesweep.machines.configs import load_motor
from phasesweep.registry import MODEL_REGISTRY
from phasesweep.sweep_types import RunConfig, RunResult
from phasesweep.validation.crossval import compare_results
from phasesweep.validation.measured import import_measured
from tests.conftest import CREATOR_CSV_SKIP, REPO_ROOT, creator_file

MOTOR_TOML = REPO_ROOT / "motors/creator_case_pmsm.toml"
MEASURED_JSON = REPO_ROOT / "data/creator_case_pmsm/backemf_measured.json"

_needs_backemf = pytest.mark.skipif(
    not MEASURED_JSON.exists(), reason=CREATOR_CSV_SKIP,
)


@pytest.fixture(scope="module")
def creator():
    return load_motor(MOTOR_TOML)


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


@_needs_backemf
class TestCreatorCrossValidation:

    def test_measured_backemf_value(self, measured_result):
        """Measured back-EMF from JSON is ~47.38 V (FFT of scope capture)."""
        assert measured_result.metrics["backemf_fundamental"] == pytest.approx(
            47.38, abs=0.01)

    def test_rated_torque_physical(self, rated_result):
        """Rated torque physically reasonable for a 70W motor, plus a
        tight regression pin — the 10x sanity window alone would pass a
        2x model error. The pin is the current model value (rated_torque
        v4; the published continuous rating is 0.10 Nm, Muetze et al.
        Table 6); re-freeze deliberately on any MTPA physics change."""
        tau = rated_result.metrics["tau_mtpa"]
        k_T = rated_result.metrics["k_T"]
        assert 0.05 < tau < 0.5, f"tau_rated={tau:.3f} outside [0.05, 0.5] Nm"
        assert k_T > 0
        assert tau == pytest.approx(0.106846, rel=1e-3)
        assert k_T == pytest.approx(0.3432, rel=1e-3)


# ---------------------------------------------------------------------------
# Bound comparison: torque rated
# ---------------------------------------------------------------------------

class TestCreatorTorqueBound:

    def test_import_torque_rated(self, creator, tmp_path):
        result = import_measured(
            creator_file("torque_rated.json"),
            creator, tmp_path,
        )
        assert result.source == "published"
        assert "_bound_compare" in result.metrics

    def test_bound_check_rated_gte_published(self, creator, rated_result, tmp_path):
        measured = import_measured(
            creator_file("torque_rated.json"),
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
            creator_file("mtpa_angles.json"),
            creator, tmp_path,
        )
        assert "_curve_compare" in result.metrics

    def test_curve_compare_mtpa_angles(self, creator, rated_result, tmp_path):
        measured = import_measured(
            creator_file("mtpa_angles.json"),
            creator, tmp_path,
        )
        rows = compare_results(measured, rated_result)
        curve_rows = [r for r in rows if r.comparison_type == "curve"]
        assert len(curve_rows) == 2
        for r in curve_rows:
            assert r.passed, f"{r.quantity}: {r.rel_pct:.1f}% > {r.tol_pct}%"
