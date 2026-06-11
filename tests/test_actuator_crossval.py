"""Back-EMF speed sweep validation: Actuator Steel Rotor.

Validates model-predicted back-EMF voltage against 3-channel measured data
at 8 speeds (20-120 rps). Tests linearity (Ke constant across speeds) and
absolute Ke magnitude.

Source: speed sweep measurement (2026-03-18), RTB2004 oscilloscope,
3-channel Hann-windowed FFT, linear fit R² = 0.99997.
"""

from __future__ import annotations

import json
import sys
from math import pi

import numpy as np
import pytest

from phasesweep.configs import load_motor
from phasesweep.solver_params import _derive_psi_f, prepare_drive_sim
from tests.conftest import REPO_ROOT

MOTOR_TOML = REPO_ROOT / "motors/actuator_steel_rotor.toml"
SWEEP_JSON = REPO_ROOT / "data/actuator_steel_rotor/backemf_speed_sweep.json"
CAPTURES_DIR = REPO_ROOT / "data/actuator_steel_rotor/captures"

KE_MEASURED = 1.607e-3  # V/(rad/s) mechanical, combined linear fit
PSI_F_MEAS = 0.000268   # Wb


@pytest.fixture(scope="module")
def motor():
    return load_motor(MOTOR_TOML)


@pytest.fixture(scope="module")
def sweep_data():
    with open(SWEEP_JSON) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def speed_points(sweep_data):
    """Return list of (omega_mech, V_peak_avg) tuples from measured data."""
    points = []
    for pt in sweep_data["speed_sweep"]:
        omega = pt["actual_rps"] * 2 * pi
        v_avg = sum(pt["V_pk_mV"]) / len(pt["V_pk_mV"]) * 1e-3  # V
        points.append((omega, v_avg))
    return points


class TestActuatorSpeedSweep:

    def test_predicted_vpeak_at_each_speed(self, motor, speed_points):
        """Model-predicted V_peak over-predicts by 30-50% at every speed.

        The raw 2D model gap is +42% (square-wave convention, S110): no
        end effects (k_end ≈ 0.84 for this 7 mm stack), datasheet
        B_rem = 1.45 T (N52 nominal, grade unmeasured), nominal alpha_p
        and k_w. This test pins the raw gap so convention/regression
        drift is visible; the k_end-corrected expectation is Tier-0
        below. Will tighten after gaussmeter measurement provides true
        B_rem.
        """
        psi_f = _derive_psi_f(motor)
        assert psi_f is not None
        Ke_pred = psi_f * motor.n_p

        for omega, v_meas in speed_points:
            v_pred = Ke_pred * omega
            err_pct = (v_pred - v_meas) / v_meas * 100
            assert 30.0 < err_pct < 50.0, (
                f"At ω={omega:.1f} rad/s: predicted {v_pred*1e3:.1f} mV "
                f"vs measured {v_meas*1e3:.1f} mV ({err_pct:+.1f}% "
                f"outside [30%, 50%])"
            )

    def test_predicted_ke_direction(self, motor):
        """Model Ke should over-predict (2D model, no end effects)."""
        psi_f = _derive_psi_f(motor)
        Ke_pred = psi_f * motor.n_p
        assert Ke_pred > KE_MEASURED, (
            f"Expected over-prediction but Ke_pred={Ke_pred*1e3:.3f} < "
            f"Ke_meas={KE_MEASURED*1e3:.3f} mV/(rad/s)"
        )

    def test_predicted_ke_not_wildly_off(self, motor):
        """Model Ke within 50% of measured (sanity bound, raw 2D model)."""
        psi_f = _derive_psi_f(motor)
        Ke_pred = psi_f * motor.n_p
        err_pct = abs(Ke_pred - KE_MEASURED) / KE_MEASURED * 100
        assert err_pct < 50.0, f"Ke delta {err_pct:.1f}% exceeds 50% sanity bound"

    def test_ke_with_end_effect_correction(self, motor):
        """Tier-0: k_end-corrected Ke lands within [8%, 25%] of measured.

        Russell-Norsworthy k_end ≈ 0.841 for L_stk = 7 mm brings the raw
        +42% gap to ≈ +19%. Informational tier — k_end is uncalibrated
        (no 3D validation yet) and is NOT applied to production psi_f;
        the residual absorbs magnet grade, effective alpha_p, and k_w.
        """
        from phasesweep.fem_field import end_effect_factor
        psi_f = _derive_psi_f(motor)
        g = motor.geometry
        g_eff = abs(g.r_stator - g.r_magnet) + abs(g.r_magnet - g.r_rotor) / motor.mu_r_pm
        k_end = end_effect_factor(motor.L_stk, g_eff)
        Ke_corr = psi_f * k_end * motor.n_p
        delta_pct = (Ke_corr - KE_MEASURED) / KE_MEASURED * 100
        assert 8.0 < delta_pct < 25.0, (
            f"k_end-corrected Ke delta {delta_pct:+.1f}% outside [8%, 25%]"
        )

    def test_measured_linearity(self, speed_points):
        """Ke should be nearly constant across all speeds (R² > 0.999).

        Validates that V_peak is linear in ω — any deviation would indicate
        saturation onset or frequency-dependent losses.
        """
        omegas = np.array([w for w, _ in speed_points])
        voltages = np.array([v for _, v in speed_points])

        # Linear fit forced through origin: V = Ke * omega
        Ke_fit = np.sum(omegas * voltages) / np.sum(omegas**2)
        residuals = voltages - Ke_fit * omegas
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((voltages - np.mean(voltages))**2)
        r_squared = 1.0 - ss_res / ss_tot

        assert r_squared > 0.999, f"R² = {r_squared:.6f}, linearity violated"

    def test_measured_ke_value(self, speed_points):
        """Measured Ke from linear fit matches the canonical 1.607 mV/(rad/s)."""
        omegas = np.array([w for w, _ in speed_points])
        voltages = np.array([v for _, v in speed_points])
        Ke_fit = np.sum(omegas * voltages) / np.sum(omegas**2)
        assert Ke_fit == pytest.approx(KE_MEASURED, rel=0.01)

    def test_phase_balance(self, sweep_data):
        """Phase imbalance < 3% at all speeds (symmetric winding check)."""
        for pt in sweep_data["speed_sweep"]:
            assert pt["imbalance_pct"] < 3.0, (
                f"Imbalance {pt['imbalance_pct']}% at {pt['cmd_rps']} rps"
            )

    def test_eight_speed_points(self, sweep_data):
        """All 8 canonical speed points present."""
        assert len(sweep_data["speed_sweep"]) == 8
        cmd_speeds = [pt["cmd_rps"] for pt in sweep_data["speed_sweep"]]
        assert cmd_speeds == [20, 30, 40, 50, 60, 80, 100, 120]

    def test_model_discrepancy_documented(self, motor):
        """Quantify the B_rem calibration gap for the record.

        Not a pass/fail — just records the current model-vs-measured delta
        so regressions are visible in test output.
        """
        psi_f = _derive_psi_f(motor)
        Ke_pred = psi_f * motor.n_p
        delta_pct = (Ke_pred - KE_MEASURED) / KE_MEASURED * 100
        # Current expected: +41.9% (raw 2D, square-wave convention, S110).
        # Alert if it changes significantly.
        assert 30.0 < delta_pct < 48.0, (
            f"Model over-prediction {delta_pct:.1f}% outside expected [30%, 48%] band"
        )


# Import script functions — optional, tests skip if unavailable. The path
# entry is removed after import so it can't leak into later test modules.
_scripts_dir = str(REPO_ROOT / "scripts")
sys.path.insert(0, _scripts_dir)
try:
    from backemf_validation import (
        analyze_phase_balance,
        compute_effective_B_rem,
        extract_measured_harmonics,
        find_matching_value,
        run_calibrated_drive_sim,
    )
    _HAS_BACKEMF = True
except ImportError:
    _HAS_BACKEMF = False
finally:
    sys.path.remove(_scripts_dir)

_skip_no_script = pytest.mark.skipif(
    not _HAS_BACKEMF, reason="backemf_validation script not available"
)


@_skip_no_script
class TestEffectiveBrem:

    def test_effective_B_rem_range(self, motor):
        """Effective B_rem under the square-wave convention is diagnostic, not a
        grade: it sits below any sintered NdFeB grade (N30 ~ 1.08 T), absorbing
        the real magnetization profile, 3D leakage, and winding-parameter error.
        Smooth-bore inversion gives ~1.02 T (Carter-consistent: ~1.04 T)."""
        result = compute_effective_B_rem(motor, PSI_F_MEAS)
        assert 0.95 < result["B_rem_eff"] < 1.08, (
            f"Effective B_rem {result['B_rem_eff']:.3f} T outside [0.95, 1.08]"
        )

    def test_effective_B_rem_matches_sweep(self, motor):
        """Effective B_rem should match the sensitivity sweep interpolation."""
        result = compute_effective_B_rem(motor, PSI_F_MEAS)
        b_vals = np.linspace(0.9, 1.45, 50)
        b_match = find_matching_value(motor, "B_rem", b_vals, PSI_F_MEAS)
        assert b_match is not None
        assert abs(result["B_rem_eff"] - b_match) < 0.002, (
            f"B_rem_eff={result['B_rem_eff']:.4f} vs sweep={b_match:.4f}"
        )


@_skip_no_script
class TestMeasuredHarmonics:

    @pytest.fixture(scope="class")
    def harmonics_80(self):
        if not (CAPTURES_DIR / "backemf_080rps.csv").exists():
            pytest.skip("80 rps capture not available")
        return extract_measured_harmonics(80, 6, actual_rps=79.86)

    def test_thd_below_bound(self, harmonics_80):
        """Measured THD should be below 2.5% (clean sinusoidal back-EMF)."""
        assert harmonics_80 is not None
        assert harmonics_80["thd_pct"] < 2.5, (
            f"THD {harmonics_80['thd_pct']:.2f}% exceeds 2.5% bound"
        )

    def test_3rd_harmonic_range(self, harmonics_80):
        """3rd harmonic should be in [0.1%, 2.0%]."""
        assert harmonics_80 is not None
        h3 = harmonics_80["harmonics_pct"].get(3, 0)
        assert 0.1 < h3 < 2.0, f"3rd harmonic {h3:.2f}% outside [0.1%, 2.0%]"

    def test_thd_consistent_across_speeds(self):
        """THD should be consistent (< 0.5% absolute variation) across 3+ speeds."""
        speeds = [(40, 39.58), (60, 59.72), (80, 79.86), (100, 100.0), (120, 120.14)]
        thds = []
        for cmd, actual in speeds:
            h = extract_measured_harmonics(cmd, 6, actual)
            if h is not None:
                thds.append(h["thd_pct"])
        if len(thds) < 3:
            pytest.skip("Fewer than 3 captures available")
        spread = max(thds) - min(thds)
        assert spread < 0.5, f"THD spread {spread:.2f}% across {len(thds)} speeds"


@_skip_no_script
class TestPhaseBalance:

    @pytest.fixture(scope="class")
    def balance_80(self):
        if not (CAPTURES_DIR / "backemf_080rps.csv").exists():
            pytest.skip("80 rps capture not available")
        return analyze_phase_balance(80, 6, actual_rps=79.86)

    def test_angles_near_120(self, balance_80):
        """All pairwise angles should be within 5 degrees of 120."""
        assert balance_80 is not None
        for pair, angle in balance_80["angles"].items():
            assert abs(angle - 120.0) < 5.0, (
                f"{pair}: {angle:.1f} deg, delta {angle - 120:+.1f}"
            )

    def test_angles_speed_independent(self):
        """Phase angles should vary < 2 degrees across speeds."""
        speeds = [(40, 39.58), (60, 59.72), (80, 79.86), (100, 100.0), (120, 120.14)]
        all_angles = {}
        for cmd, actual in speeds:
            bal = analyze_phase_balance(cmd, 6, actual)
            if bal is None:
                continue
            for pair, angle in bal["angles"].items():
                all_angles.setdefault(pair, []).append(angle)
        if not all_angles:
            pytest.skip("No captures available")
        for pair, angles in all_angles.items():
            if len(angles) < 3:
                continue
            spread = max(angles) - min(angles)
            assert spread < 2.0, (
                f"{pair}: spread {spread:.1f} deg across speeds"
            )


@_skip_no_script
class TestCalibratedDriveSim:

    def test_calibrated_psi_f_used(self, motor):
        """Calibrated motor should use 268 uWb, not B_rem-derived."""
        from dataclasses import replace
        motor_cal = replace(motor, psi_f=PSI_F_MEAS)
        params = prepare_drive_sim(motor_cal)
        assert params.psi_f == pytest.approx(PSI_F_MEAS, rel=1e-6)

    def test_calibrated_tau_peak_lower(self, motor):
        """Calibrated psi_f (268 uWb) should give lower tau_peak than uncalibrated."""

        psi_f_uncal = _derive_psi_f(motor)
        assert psi_f_uncal is not None
        assert PSI_F_MEAS < psi_f_uncal  # calibrated is lower

        # tau_peak = 1.5 * n_p * psi_f * I_limit
        kt_cal = 1.5 * motor.n_p * PSI_F_MEAS
        kt_uncal = 1.5 * motor.n_p * psi_f_uncal
        assert kt_cal < kt_uncal

    @pytest.mark.slow
    def test_graceful_crash(self, motor):
        """run_calibrated_drive_sim should not raise even if motulator crashes."""
        result = run_calibrated_drive_sim(motor, PSI_F_MEAS)
        assert "uncalibrated" in result
        assert "calibrated" in result
        # Each entry should have either "metrics" or "error" — not an unhandled crash
        for label in ("uncalibrated", "calibrated"):
            entry = result[label]
            assert "metrics" in entry or "error" in entry
