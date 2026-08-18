"""Model consistency checks: FEMM LRK 14p/12s miniature outrunner.

No experimental measurements exist for this motor. Reference values are from
D. Meeker's analytical worksheet (1-D magnetic circuit, trapezoidal waveform
assumption). Tests verify model self-consistency and plausible agreement with
worksheet predictions — not experimental validation.

Reference: D. Meeker, "LRK Motor Analysis Worksheet", femm.info, 2004
URL: https://www.femm.info/examples/lrk40/lrk-bldc.pdf
"""


import numpy as np
import pytest

from phasesweep.machines.configs import load_motor
from phasesweep.registry import MODEL_REGISTRY
from phasesweep.solvers.harmonics import harmonics_1sided
from phasesweep.sweep_types import RunConfig, RunResult
from tests.conftest import REPO_ROOT, requires_fem

MOTOR_TOML = REPO_ROOT / "motors/lrk_outrunner.toml"


@pytest.fixture(scope="module")
def lrk():
    return load_motor(MOTOR_TOML)


@pytest.fixture(scope="module")
def analytical_result(lrk):
    rc = RunConfig(motor=lrk, model="analytical")
    metrics = MODEL_REGISTRY["analytical"].fn(rc)
    return RunResult(
        config=rc, model="analytical", status="OK",
        metrics=metrics, elapsed_s=0.0, source="computed",
    )


@pytest.fixture(scope="module")
def fem_result(lrk):
    rc = RunConfig(motor=lrk, model="fem", maxh_fraction=0.08)
    metrics = MODEL_REGISTRY["fem"].fn(rc)
    return RunResult(
        config=rc, model="fem", status="OK",
        metrics=metrics, elapsed_s=0.0, source="computed",
    )


class TestLrkCrossValidation:

    def test_motor_loads(self, lrk):
        assert lrk.name == "FEMM LRK 14p/12s Outrunner"
        assert lrk.n_p == 7
        assert lrk.B_rem == pytest.approx(1.265)
        assert lrk.geometry.topology == "outrunner"

    def test_analytical_fundamental_positive(self, analytical_result):
        """Analytical B₁ should be positive and in reasonable range for NdFeB."""
        fund = analytical_result.metrics["fundamental"]
        assert fund > 0, f"fundamental {fund:.4f} should be positive"
        assert fund < 2.0, f"fundamental {fund:.4f} unreasonably high"

    def test_backemf_vs_worksheet_kp(self, lrk, analytical_result):
        """Back-EMF from analytical model vs worksheet K_p = 3.433e-3 Wb.

        K_p is the trapezoidal back-EMF plateau per phase, while ours is
        fundamental-based — the conventions differ, so exact agreement is
        not expected. Square-wave magnetization plus the corrected
        single-layer k_w = 0.966 (0.933 was the double-layer value)
        gives a current delta of ≈ +3.3%. Tolerance 5% guards the
        Carter/winding-formula approximations.
        """
        backemf = analytical_result.metrics.get("backemf_fundamental")
        if backemf is None:
            pytest.skip("backemf_fundamental not available (psi_f not derivable)")

        # Worksheet K_p = peak phase back-EMF / omega_mech
        # Our backemf_fundamental = omega_elec * psi_f
        # K_p_ours = backemf_fundamental / omega_mech = n_p * psi_f
        omega_mech = lrk.drive.W_REF  # 523.6 rad/s (5000 RPM)
        K_p_ours = backemf / omega_mech

        K_p_published = 3.433e-3  # Wb
        rel = abs(K_p_ours - K_p_published) / K_p_published * 100
        assert rel < 5.0, (
            f"K_p: ours={K_p_ours:.4e} vs published={K_p_published:.4e}, "
            f"delta={rel:.1f}% > 5%"
        )

    @requires_fem
    @pytest.mark.timeout(30)
    def test_analytical_vs_fem_fundamental(self, analytical_result, fem_result):
        """Analytical and FEM fundamental harmonics should agree within 5%.

        Both solvers share the square-wave source fundamental, so
        the residual is slotting/mesh only (currently < 1%).
        """
        anal_fund = analytical_result.metrics["fundamental"]
        B_r = np.array(fem_result.metrics["B_r_list"])
        fem_amps = harmonics_1sided(B_r)
        n_p = 7
        fem_fund = float(fem_amps[n_p]) if n_p < len(fem_amps) else 0.0
        rel = abs(anal_fund - fem_fund) / max(anal_fund, fem_fund, 1e-12) * 100
        assert rel < 5.0, (
            f"analytical={anal_fund:.4f} vs FEM={fem_fund:.4f}: {rel:.1f}% > 5%"
        )


@pytest.fixture(scope="module")
def rated_result(lrk):
    rc = RunConfig(motor=lrk, model="rated_torque")
    metrics = MODEL_REGISTRY["rated_torque"].fn(rc)
    return RunResult(
        config=rc, model="rated_torque", status="OK",
        metrics=metrics, elapsed_s=0.0, source="computed",
    )


class TestLrkRatedTorque:

    def test_rated_torque_positive(self, rated_result):
        """rated_torque model produces positive torque (psi_f derived from B_rem)."""
        tau = rated_result.metrics["tau_mtpa"]
        assert tau > 0, f"tau_rated={tau} should be positive"

    def test_torque_constant_positive(self, rated_result):
        """Torque constant k_T should be positive and finite."""
        k_T = rated_result.metrics["k_T"]
        assert k_T > 0, f"k_T={k_T} should be positive"
        assert np.isfinite(k_T), f"k_T={k_T} should be finite"
