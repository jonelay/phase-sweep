"""Cross-validation: Belkhadir 22p/24s external rotor PMSM.

Validates analytical and FEM air-gap flux density against published data
from Belkhadir et al., IECON 2023 (DOI 10.1109/IECON51785.2023.10312419).

Source: HAL hal-04296882 (open access)
"""


import numpy as np
import pytest

from phasesweep.machines.configs import load_motor
from phasesweep.registry import MODEL_REGISTRY
from phasesweep.sweep_types import RunConfig, RunResult
from phasesweep.validation.crossval import compare_results
from phasesweep.validation.measured import import_measured
from tests.conftest import REPO_ROOT, requires_fem

MOTOR_TOML = REPO_ROOT / "motors/belkhadir_outrunner.toml"


@pytest.fixture(scope="module")
def motor():
    return load_motor(MOTOR_TOML)


@pytest.fixture(scope="module")
def analytical_result(motor):
    rc = RunConfig(motor=motor, model="analytical", n_theta=3600)
    return MODEL_REGISTRY["analytical"].fn(rc)


@pytest.fixture(scope="module")
def fem_result(motor):
    rc = RunConfig(motor=motor, model="fem", n_theta=360, maxh_fraction=0.05)
    return MODEL_REGISTRY["fem"].fn(rc)


class TestBelkhadirOutrunner:
    """Validate against Belkhadir et al. IECON 2023 published results."""

    def test_motor_loads(self, motor):
        assert motor.name == "Belkhadir 22p/24s ER-PMSM"
        assert motor.n_p == 11
        assert motor.geometry.topology == "outrunner"

    def test_geometry_consistency(self, motor):
        """Radii match paper Table I."""
        g = motor.geometry
        air_gap = (g.r_magnet - g.r_stator) * 1000  # mm
        magnet_t = (g.r_rotor - g.r_magnet) * 1000
        yoke_t = (g.r_outer - g.r_rotor) * 1000
        assert air_gap == pytest.approx(1.365, abs=0.01)
        assert magnet_t == pytest.approx(3.0, abs=0.01)
        assert yoke_t == pytest.approx(7.0, abs=0.01)

    def test_analytical_fundamental(self, analytical_result):
        """B₁ regression anchor under the square-wave convention.

        The published 0.806/0.818 T values are waveform *peaks* — under
        the sinusoidal convention the fundamental happened to sit near
        them, which masked the missing 4/π. Post-fix the fundamental is
        ≈ 1.026 T (= peak × ~4/π relationship for a flat-top wave);
        published-peak comparisons live on the FEM waveform peak below.
        """
        fundamental = analytical_result["fundamental"]
        assert fundamental == pytest.approx(1.026, rel=0.03)

    @requires_fem
    def test_fem_peak_vs_paper_eq2(self, fem_result):
        """FEM waveform peak vs paper's simplified eq (2) flat-top 0.776 T.

        Eq (2) assumes infinite mu_r and no slots; our slotted FEM peak
        sits above it (+17% currently). 20% tolerance matches the
        published-data JSON. Slot dimensions estimated (not published).
        """
        B_peak = float(np.max(np.abs(fem_result["B_r_list"])))
        assert B_peak == pytest.approx(0.776, rel=0.20), (
            f"FEM peak {B_peak:.4f} vs eq (2) 0.776"
        )

    @requires_fem
    def test_fem_peak_vs_paper_fem(self, fem_result):
        """FEM waveform peak vs paper's published FEM peak 0.818 T.

        Currently +11% — estimated slot dimensions and 2D linear iron.
        """
        B_peak = float(np.max(np.abs(fem_result["B_r_list"])))
        assert B_peak == pytest.approx(0.818, rel=0.15), (
            f"FEM peak {B_peak:.4f} vs published FEM 0.818"
        )

    @requires_fem
    def test_analytical_vs_fem_agreement(self, analytical_result, fem_result):
        """Analytical vs FEM fundamental within 3% (shared source
        convention; residual is slotting/mesh, currently < 1%)."""
        a = analytical_result["fundamental"]
        f = fem_result["fundamental"]
        rel_diff = abs(a - f) / a * 100
        assert rel_diff < 3.0, f"Analytical vs FEM: {rel_diff:.1f}% > 3%"

    def test_backemf_from_psi_f(self, analytical_result):
        """Back-EMF should be ~67.5 V at 600 rpm (from Figs 14-15)."""
        backemf = analytical_result["backemf_fundamental"]
        assert backemf == pytest.approx(67.5, rel=0.05)

    def test_paper_simplified_formula(self, motor):
        """Verify paper's eq (2): B_PM = B_r * h_m / (g + h_m / alpha_p).

        This is the simplified flat-top flux density, not the Zhu & Howe
        boundary solution. We check it as a sanity cross-reference.
        """
        B_r = motor.B_rem  # 1.26 T
        g = motor.geometry
        h_m = g.r_rotor - g.r_magnet  # magnet thickness
        gap = g.r_magnet - g.r_stator  # air gap
        alpha_p = 0.855  # from paper: theta_PM / tau_p

        B_PM = B_r * h_m / (gap + h_m / alpha_p)
        assert B_PM == pytest.approx(0.776, rel=0.01)


# ---------------------------------------------------------------------------
# Published data cross-validation
# ---------------------------------------------------------------------------

PUBLISHED_JSON = REPO_ROOT / "data/belkhadir_outrunner/airgap_flux_published.json"


@pytest.fixture(scope="module")
def analytical_run_result(motor):
    rc = RunConfig(motor=motor, model="analytical", n_theta=3600)
    metrics = MODEL_REGISTRY["analytical"].fn(rc)
    return RunResult(
        config=rc, model="analytical", status="OK",
        metrics=metrics, elapsed_s=0.0, source="computed",
    )


@pytest.fixture(scope="module")
def published_import(motor, tmp_path_factory):
    out = tmp_path_factory.mktemp("belkhadir_pub")
    return import_measured(PUBLISHED_JSON, motor, out)


@pytest.fixture(scope="module")
def fem_run_result(motor, fem_result):
    rc = RunConfig(motor=motor, model="fem", n_theta=360, maxh_fraction=0.05)
    return RunResult(
        config=rc, model="fem", status="OK",
        metrics=fem_result, elapsed_s=0.0, source="computed",
    )


class TestBelkhadirPublishedCrossval:

    def test_published_source_tier(self, published_import):
        assert published_import.source == "published"

    @pytest.mark.parametrize("computed_fixture", [
        "analytical_run_result",
        pytest.param("fem_run_result", marks=requires_fem),
    ])
    def test_b_ag_fundamental(self, published_import, computed_fixture, request):
        # Eq (2) flat-top 0.776 T converted to commensurate B₁ = 0.9625 T
        # via (4/π)·sin(π·α_p/2) — compared fundamental-to-fundamental,
        # not extract=max.
        computed = request.getfixturevalue(computed_fixture)
        rows = compare_results(published_import, computed)
        b_rows = [r for r in rows if r.quantity == "B_ag_fundamental"]
        assert len(b_rows) == 1
        assert b_rows[0].passed, (
            f"B_ag_fundamental: {b_rows[0].rel_pct:.1f}% > {b_rows[0].tol_pct}%"
        )

    def test_backemf_key_mapping(self, published_import, analytical_run_result):
        rows = compare_results(published_import, analytical_run_result)
        mapped = [r for r in rows if r.quantity == "backemf_peak"]
        assert len(mapped) == 1
        assert mapped[0].passed, f"backemf_peak: {mapped[0].rel_pct:.1f}% > {mapped[0].tol_pct}%"


# ---------------------------------------------------------------------------
# Rated torque cross-validation
# ---------------------------------------------------------------------------

TORQUE_JSON = REPO_ROOT / "data/belkhadir_outrunner/torque_rated.json"


@pytest.fixture(scope="module")
def rated_result(motor):
    rc = RunConfig(motor=motor, model="rated_torque")
    metrics = MODEL_REGISTRY["rated_torque"].fn(rc)
    return RunResult(
        config=rc, model="rated_torque", status="OK",
        metrics=metrics, elapsed_s=0.0, source="computed",
    )


@pytest.fixture(scope="module")
def torque_published(motor, tmp_path_factory):
    out = tmp_path_factory.mktemp("belkhadir_torque")
    return import_measured(TORQUE_JSON, motor, out)


class TestBelkhadirRatedTorque:

    def test_bound_rated_gte_published(self, torque_published, rated_result):
        """Computed non-salient MTPA at I_rated >= published 24 Nm."""
        rows = compare_results(torque_published, rated_result)
        bound_rows = [r for r in rows if r.comparison_type == "bound"]
        assert len(bound_rows) == 1
        assert bound_rows[0].passed, (
            f"tau_rated={bound_rows[0].val_b:.2f} should be >= "
            f"published={bound_rows[0].val_a:.2f}"
        )

    def test_rated_torque_physical(self, rated_result):
        """Rated torque should be physically reasonable for 1.5 kW / 600 rpm."""
        tau = rated_result.metrics["tau_mtpa"]
        assert 20.0 < tau < 35.0, f"tau_rated={tau:.2f} outside [20, 35] Nm"
