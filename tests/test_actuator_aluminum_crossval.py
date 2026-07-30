"""Torque-constant validation: Actuator Aluminum Rotor.

Second actuator validation point (aluminum-shell rotor variant, same stator
as the steel rotor, tighter 0.262 mm air gap). The clean 3-channel back-EMF
sweep used for the steel rotor was not repeated here — not for lack of signal
(this motor's back-EMF exceeds steel's) but because it was measured
self-powered, so 60 Hz pickup (disconnect coast-down) and winding I*Z drop
(powered no-load) swamp it. So psi_f is derived from the torque constant via
power-balance on reliable ODrive digital telemetry:

    T_em = (P_elec - 1.5*R_s*Iq^2) / omega_mech
    Kt   = slope(T_em vs Iq, through origin)
    psi_f = Kt / (1.5 * n_p)

Source: no-load drag sweeps 2026-04-14, 10-80 rps, CW+CCW, two runs.
See data/actuator_aluminum_rotor/README.md.
"""

from __future__ import annotations

import json

import pytest

from phasesweep.configs import load_motor
from phasesweep.solver_params import derive_psi_f_smooth
from tests.conftest import REPO_ROOT

MOTOR_TOML = REPO_ROOT / "motors/actuator_aluminum_rotor.toml"
SWEEP_JSON = REPO_ROOT / "data/actuator_aluminum_rotor/torque_constant_sweep.json"

PSI_F_MEAS = 0.000312      # Wb, power-balance Kt-derived (through-origin, as-recorded)
PSI_F_BIAS_CORRECTED = 0.000299  # Wb, physical loss fit (Kt*Iq + P_inv/omega); 0.312 is the upper bound
KE_MEASURED = 0.001873     # V/(rad/s) mechanical (n_p * psi_f)
KT_MEASURED = 0.002808     # Nm/A (1.5 * n_p * psi_f)
PSI_F_STEEL = 0.000268     # Wb, steel rotor measured (back-EMF)


@pytest.fixture(scope="module")
def motor():
    return load_motor(MOTOR_TOML)


@pytest.fixture(scope="module")
def sweep_data():
    with open(SWEEP_JSON) as f:
        return json.load(f)


class TestAluminumTorqueConstant:

    def test_model_overpredicts(self, motor):
        """1-D smooth-bore model over-predicts measured psi_f (tighter gap)."""
        psi_f_model = derive_psi_f_smooth(motor)
        assert psi_f_model is not None
        assert psi_f_model > PSI_F_MEAS, (
            f"Expected over-prediction but model psi_f={psi_f_model*1e3:.3f} "
            f"< measured {PSI_F_MEAS*1e3:.3f} mWb"
        )

    def test_overprediction_magnitude(self, motor):
        """Model over-prediction is ~44%; pin the band so drift is visible.

        0.448 mWb (smooth) / 0.438 (Carter) model vs 0.312 recorded / 0.299
        bias-corrected. ~2/3 of the gap is the shared raw-2D bias (steel +27%);
        ~1/3 is an aluminum-specific gap over-credit (model gap-sensitivity
        ~2.5x too high). Band kept wide because the measured value is a
        Kt-derived upper bound.
        """
        psi_f_model = derive_psi_f_smooth(motor)
        delta_pct = (psi_f_model - PSI_F_MEAS) / PSI_F_MEAS * 100
        assert 35.0 < delta_pct < 55.0, (
            f"Model over-prediction {delta_pct:+.1f}% outside expected [35%, 55%]"
        )

    def test_gap_larger_than_steel(self, motor):
        """Aluminum model gap should EXCEED the steel rotor's ~30%.

        The larger over-prediction is the model over-crediting the tighter gap
        (predicts +29% flux, the rotors deliver only +12%), not the gap
        "amplifying" a correct bias. Geometry/model error, not magnet grade.
        """
        psi_f_model = derive_psi_f_smooth(motor)
        alum_gap_pct = (psi_f_model - PSI_F_MEAS) / PSI_F_MEAS * 100
        steel_gap_pct = 30.0  # steel rotor measured (test_actuator_crossval)
        assert alum_gap_pct > steel_gap_pct, (
            f"Aluminum gap {alum_gap_pct:.1f}% not larger than steel {steel_gap_pct:.1f}%"
        )

    def test_measured_above_steel(self):
        """Measured aluminum psi_f exceeds steel's — tighter gap, more flux."""
        assert PSI_F_MEAS > PSI_F_STEEL, (
            f"Aluminum {PSI_F_MEAS*1e3:.3f} should exceed steel "
            f"{PSI_F_STEEL*1e3:.3f} mWb (tighter gap)"
        )
        ratio = PSI_F_MEAS / PSI_F_STEEL
        assert 1.05 < ratio < 1.30, (
            f"Aluminum/steel psi_f ratio {ratio:.3f} outside plausible [1.05, 1.30]"
        )

    def test_kt_psi_relation(self):
        """Kt = 1.5 * n_p * psi_f and Ke = n_p * psi_f must be self-consistent."""
        n_p = 6
        assert KT_MEASURED == pytest.approx(1.5 * n_p * PSI_F_MEAS, rel=1e-3)
        assert KE_MEASURED == pytest.approx(n_p * PSI_F_MEAS, rel=1e-3)

    def test_sweep_json_consistent(self, sweep_data):
        """Distilled JSON quantities match the test constants."""
        q = sweep_data["quantities"]
        assert q["psi_f"] == pytest.approx(PSI_F_MEAS, rel=1e-3)
        assert q["Ke_mech"] == pytest.approx(KE_MEASURED, rel=1e-3)
        assert q["Kt"] == pytest.approx(KT_MEASURED, rel=1e-3)

    def test_two_runs_present(self, sweep_data):
        """Both independent runs recorded, bracketing the canonical value."""
        runs = sweep_data["runs"]
        assert len(runs) == 2
        psis = sorted(r["psi_f_mWb"] for r in runs)
        assert psis[0] <= PSI_F_MEAS * 1e3 <= psis[1], (
            f"Canonical {PSI_F_MEAS*1e3:.3f} not bracketed by runs {psis}"
        )

    def test_run_internal_consistency(self, sweep_data):
        """Each run's Kt = 1.5 * n_p * psi_f within rounding."""
        n_p = 6
        for run in sweep_data["runs"]:
            psi = run["psi_f_mWb"] * 1e-3
            kt = run["Kt_mNm_per_A"] * 1e-3
            assert kt == pytest.approx(1.5 * n_p * psi, rel=0.02), (
                f"{run['label']}: Kt {kt*1e3:.3f} vs 1.5*n_p*psi "
                f"{1.5*n_p*psi*1e3:.3f} mNm/A"
            )

    def test_recorded_is_upper_bound(self):
        """Through-origin 0.312 is an upper bound; bias-corrected is below it.

        The physical loss fit (T_em = Kt*Iq + P_inv/omega) removes ~20 mW of
        inverter loss carried by the through-origin Kt, dropping psi_f ~5%.
        Documents the direction so the two constants don't drift together.
        """
        assert PSI_F_BIAS_CORRECTED < PSI_F_MEAS
        assert PSI_F_BIAS_CORRECTED == pytest.approx(PSI_F_MEAS, rel=0.07)
