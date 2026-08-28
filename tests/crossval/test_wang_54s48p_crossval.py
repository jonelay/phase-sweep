"""Cross-validation: Wang 54s/48p outer-rotor SPM in-wheel motor.

Source: Wang, Q., WEVJ 16(5):247, 2025. DOI: 10.3390/wevj16050247
Southeast University, Nanjing, China.

First outrunner SPM in the repo with full geometry AND LCR-measured
circuit parameters. 13 kW prototype manufactured and tested.
Measured: R_s, L_d, L_q (HIOKI LCR), back-EMF at multiple speeds,
torque vs current, torque-speed curve, efficiency map.

Eligible for: rated_torque (d-q model), analytical/FEM (full geometry).
"""

from math import pi, sqrt

import pytest

from phasesweep.machines.configs import load_motor
from phasesweep.models.rated_torque import mtpa_gamma, mtpa_torque

TOML = "data/wang_54s48p/wang_54s48p_outrunner.toml"

N_P = 24
R_S = 0.044
L_D = 0.000431
L_Q = 0.000653
PSI_F = 0.10476
I_PEAK = 83.0

# Published measured values
KE_MEASURED = 0.3225       # V_line_rms / rpm (Table 5, measured)
BACKEMF_LL_RMS_400RPM = 129.0  # V (Section 6.1, measured at 400 rpm)
TORQUE_FEM_PEAK = 396.5    # Nm (Table 4, FEM at peak current)


class TestWang54s48pLoading:

    def test_toml_loads(self):
        m = load_motor(TOML)
        assert m.name == "Wang 54s/48p Outrunner SPM"
        assert m.n_p == N_P
        assert m.R_s == R_S
        assert m.L_d == L_D
        assert m.L_q == L_Q
        assert m.psi_f == PSI_F

    def test_outrunner_topology(self):
        m = load_motor(TOML)
        assert m.geometry is not None
        assert m.geometry.topology == "outrunner"

    def test_geometry_ordering(self):
        """Outrunner: r_outer > r_rotor > r_magnet > r_stator > r_inner > 0."""
        m = load_motor(TOML)
        g = m.geometry
        assert g.r_outer > g.r_rotor > g.r_magnet > g.r_stator > g.r_inner > 0

    def test_airgap_1mm(self):
        """Airgap = r_magnet - r_stator = 1.0 mm (Table 1)."""
        m = load_motor(TOML)
        gap = m.geometry.r_magnet - m.geometry.r_stator
        assert gap == pytest.approx(0.001, abs=1e-6)

    def test_pm_thickness_4mm(self):
        """PM thickness = r_rotor - r_magnet = 4.0 mm (Table 1)."""
        m = load_motor(TOML)
        pm = m.geometry.r_rotor - m.geometry.r_magnet
        assert pm == pytest.approx(0.004, abs=1e-6)

    def test_r_ag_in_airgap(self):
        """Sampling radius must be in the airgap."""
        m = load_motor(TOML)
        g = m.geometry
        assert g.r_magnet > g.r_ag > g.r_stator


class TestWang54s48pBackEMF:

    def test_backemf_recovery_at_400rpm(self):
        """psi_f must recover the measured 129 V L-L rms at 400 rpm."""
        omega_e = N_P * 2 * pi * 400 / 60
        E_phase_rms = omega_e * PSI_F / sqrt(2)
        E_LL_rms = E_phase_rms * sqrt(3)
        assert E_LL_rms == pytest.approx(BACKEMF_LL_RMS_400RPM, rel=0.02), (
            f"E_LL_rms={E_LL_rms:.1f} V, measured {BACKEMF_LL_RMS_400RPM} V"
        )

    def test_backemf_coefficient(self):
        """ke = E_LL_rms / rpm must match measured 0.3225 V/rpm (±5%)."""
        omega_e = N_P * 2 * pi * 1.0 / 60  # per rpm
        E_phase_rms_per_rpm = omega_e * PSI_F / sqrt(2)
        ke = E_phase_rms_per_rpm * sqrt(3)
        assert ke == pytest.approx(KE_MEASURED, rel=0.05), (
            f"ke={ke:.4f} V/rpm, measured {KE_MEASURED}"
        )

    def test_backemf_linear_with_speed(self):
        """Back-EMF is linear with speed (confirmed by Fig 14)."""
        for rpm in [200, 400, 800, 1200]:
            omega_e = N_P * 2 * pi * rpm / 60
            E_LL_rms = omega_e * PSI_F / sqrt(2) * sqrt(3)
            ke_check = E_LL_rms / rpm
            assert ke_check == pytest.approx(KE_MEASURED, rel=0.02)


class TestWang54s48pSaliency:

    def test_saliency_ratio(self):
        """L_q/L_d = 1.51 — slot-induced saliency for an SPM (Table 5)."""
        ratio = L_Q / L_D
        assert ratio == pytest.approx(1.515, rel=0.01)

    def test_measured_vs_fem_ld(self):
        """Measured L_d = 0.431 mH vs FEM 0.43 mH — <1% error."""
        assert L_D == pytest.approx(0.00043, rel=0.01)

    def test_measured_vs_fem_lq(self):
        """Measured L_q = 0.653 mH vs FEM 0.65 mH — <1% error."""
        assert L_Q == pytest.approx(0.00065, rel=0.01)


class TestWang54s48pMTPA:

    def test_mtpa_torque_physically_reasonable(self):
        """13 kW in-wheel: MTPA at 83 A peak should be in (200, 500) Nm."""
        gamma = mtpa_gamma(PSI_F, L_D, L_Q, I_PEAK)
        tau = mtpa_torque(N_P, PSI_F, L_D, L_Q, I_PEAK, gamma)
        assert 200.0 < tau < 500.0, f"tau={tau:.1f} outside expected range"

    def test_mtpa_exceeds_pure_magnet_torque(self):
        """With L_q/L_d = 1.51, MTPA should exceed pure q-axis torque."""
        gamma = mtpa_gamma(PSI_F, L_D, L_Q, I_PEAK)
        tau_mtpa = mtpa_torque(N_P, PSI_F, L_D, L_Q, I_PEAK, gamma)
        tau_q_only = 1.5 * N_P * PSI_F * I_PEAK
        assert tau_mtpa > tau_q_only, (
            f"MTPA={tau_mtpa:.1f} should exceed q-only={tau_q_only:.1f}"
        )

    def test_mtpa_in_fem_torque_range(self):
        """MTPA at peak current should be in the range of the FEM average
        torque (396.5 Nm) — electromagnetic optimum vs FEM may differ by
        up to ~20% due to saturation, slotting, winding factors."""
        gamma = mtpa_gamma(PSI_F, L_D, L_Q, I_PEAK)
        tau_mtpa = mtpa_torque(N_P, PSI_F, L_D, L_Q, I_PEAK, gamma)
        assert tau_mtpa == pytest.approx(TORQUE_FEM_PEAK, rel=0.25), (
            f"MTPA={tau_mtpa:.1f} vs FEM={TORQUE_FEM_PEAK} Nm"
        )

    def test_mtpa_angle_off_q_axis(self):
        """Mild saliency: MTPA angle should be small but nonzero (not gamma=0
        like a pure non-salient SPM)."""
        gamma = mtpa_gamma(PSI_F, L_D, L_Q, I_PEAK)
        assert abs(gamma) > 0.01, "gamma should be nonzero for L_q > L_d"
        assert abs(gamma) < pi / 4, "gamma should be moderate, not 45 deg"
