"""Cross-validation: Deylami 8p/12s outer-rotor cooling fan (ferrite SPMSM).

Published data from ANSYS Maxwell 2D FEM:
- B_ag_peak = 0.49 T (peak air-gap flux density)

Source: Deylami et al., TechRxiv preprint, 2024
DOI: 10.36227/techrxiv.171439839.97768997/v1
"""


import numpy as np
import pytest

from phasesweep.machines.configs import load_motor
from phasesweep.registry import MODEL_REGISTRY
from phasesweep.solvers.harmonics import harmonics_1sided
from phasesweep.sweep_types import RunConfig, RunResult
from phasesweep.validation.crossval import compare_results
from phasesweep.validation.measured import import_measured
from tests.conftest import REPO_ROOT, requires_fem

MOTOR_TOML = REPO_ROOT / "motors/deylami_fan.toml"
PUBLISHED_JSON = REPO_ROOT / "data/deylami_fan/fem_flux_published.json"


@pytest.fixture(scope="module")
def deylami():
    return load_motor(MOTOR_TOML)


@pytest.fixture(scope="module")
def analytical_result(deylami):
    rc = RunConfig(motor=deylami, model="analytical")
    metrics = MODEL_REGISTRY["analytical"].fn(rc)
    return RunResult(
        config=rc, model="analytical", status="OK",
        metrics=metrics, elapsed_s=0.0, source="computed",
    )


@pytest.fixture(scope="module")
def fem_result(deylami):
    rc = RunConfig(motor=deylami, model="fem", maxh_fraction=0.08)
    metrics = MODEL_REGISTRY["fem"].fn(rc)
    return RunResult(
        config=rc, model="fem", status="OK",
        metrics=metrics, elapsed_s=0.0, source="computed",
    )


@pytest.fixture(scope="module")
def published_result(deylami, tmp_path_factory):
    out = tmp_path_factory.mktemp("deylami_crossval")
    return import_measured(PUBLISHED_JSON, deylami, out)


class TestDeylamiCrossValidation:

    def test_motor_loads(self, deylami):
        assert deylami.name == "Deylami 8p/12s Cooling Fan"
        assert deylami.n_p == 4
        assert deylami.B_rem == pytest.approx(0.41)
        assert deylami.geometry.topology == "outrunner"

    def test_analytical_fundamental_range(self, analytical_result):
        """Analytical B₁ for ferrite outrunner with α_p=0.95 (square-wave
        convention: currently ≈ 0.45 T)."""
        fund = analytical_result.metrics["fundamental"]
        assert 0.35 < fund < 0.55, f"analytical fundamental {fund:.3f} outside [0.35, 0.55] T"

    @requires_fem
    @pytest.mark.timeout(30)
    def test_fem_peak_vs_published(self, published_result, fem_result):
        """FEM peak air-gap flux vs ANSYS Maxwell published 0.49 T (25% tol).

        Currently −16%: the square-wave flat-top is rounded by airgap
        harmonic decay at the sampling radius, so the midgap peak stays
        below the magnet-surface value ANSYS reports.
        """
        rows = compare_results(published_result, fem_result)
        curve_rows = [r for r in rows if r.comparison_type == "curve"]
        assert len(curve_rows) == 1
        row = curve_rows[0]
        assert row.rel_pct < 25.0, (
            f"B_ag_peak delta {row.rel_pct:.1f}% > 25% "
            f"(published={row.val_a:.3f}, FEM={row.val_b:.3f})"
        )

    def test_analytical_peak_vs_published(self, published_result, analytical_result):
        """Analytical waveform peak vs the published ANSYS 0.49 T.

        The odd-harmonic series gives the analytical model a true waveform
        peak (0.365 T here, BELOW its own 0.452 T fundamental — the
        square-wave flat-top sits under the Gibbs-overshooting
        fundamental), so the row that used to be skipped is a real
        comparison now. It reads −25.5%: same direction as FEM's −16%
        (midgap harmonic attenuation rounds the flat-top; ANSYS reports
        nearer the magnet surface), deeper because the smooth-stator
        Carter-corrected field lacks the slotted FEM's local peaking.
        Pinned so drift is visible; direction is physics, magnitude is an
        observation."""
        rows = compare_results(published_result, analytical_result)
        assert [r for r in rows if r.comparison_type == "skipped"] == []
        curve_rows = [r for r in rows if r.comparison_type == "curve"]
        assert len(curve_rows) == 1
        row = curve_rows[0]
        assert row.quantity == "B_ag_peak"
        assert row.val_b < row.val_a  # model peak below published (direction)
        assert row.rel_pct == pytest.approx(25.5, abs=1.0)  # unsigned delta

    @requires_fem
    @pytest.mark.timeout(30)
    def test_analytical_vs_fem_agreement(self, analytical_result, fem_result):
        """Analytical and FEM fundamentals within 5% (shared source
        convention; currently < 1%)."""
        anal_fund = analytical_result.metrics["fundamental"]
        B_r = np.array(fem_result.metrics["B_r_list"])
        fem_amps = harmonics_1sided(B_r)
        n_p = 4
        fem_fund = float(fem_amps[n_p]) if n_p < len(fem_amps) else 0.0
        rel = abs(anal_fund - fem_fund) / max(anal_fund, fem_fund, 1e-12) * 100
        assert rel < 5.0, (
            f"analytical={anal_fund:.4f} vs FEM={fem_fund:.4f}: {rel:.1f}% > 5%"
        )

    @requires_fem
    @pytest.mark.timeout(30)
    def test_fem_peak_to_peak(self, fem_result):
        """FEM radial peak-to-peak vs published 0.9 T (ANSYS Maxwell, 25% tol)."""
        B_r = np.array(fem_result.metrics["B_r_list"])
        B_pp = float(np.max(B_r) - np.min(B_r))
        assert B_pp == pytest.approx(0.9, rel=0.25), (
            f"B_pp={B_pp:.3f} vs published 0.9 T"
        )
