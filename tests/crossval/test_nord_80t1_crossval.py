"""Cross-validation: NORD 80T1/4 IPM (published electrical datasheet).

Source: NORD / Watt-Drive motor datasheet for 80T1/4 (Nr. 43931265).
Published: R_ww 5.20 Ohm, L_d 63 mH, L_q 133 mH, phase back-EMF 187 V rms
at 2100 rpm. Nameplate: 400 V Y, 70 Hz, 2100 rpm, 5.0 Nm, 1.1 kW.

Strong saliency (L_q/L_d = 2.11) confirms IPM topology. No geometry published.
Eligible for: rated_torque (electrical params), stall_torque, torque_speed.
Not eligible for: analytical, FEM (no geometry).
"""

from math import pi, sqrt

import pytest

from phasesweep.machines.configs import load_motor
from phasesweep.models.rated_torque import mtpa_gamma, mtpa_torque

TOML = "data/nord_80t1/nord_80t1_4.toml"

N_P = 2
PSI_F = 0.6014
L_D = 0.063
L_Q = 0.133
I_RATED_RMS = 5.8
I_RATED_PEAK = I_RATED_RMS * sqrt(2)
SPEED_RPM = 2100
BACKEMF_RMS_PUBLISHED = 187.0  # V, phase back-EMF at 2100 rpm
TORQUE_NAMEPLATE = 5.0         # Nm, S1 continuous
POWER_NAMEPLATE = 1.1          # kW


class TestNord80T1Crossval:

    def test_toml_loads(self):
        m = load_motor(TOML)
        assert m.name == "NORD 80T1/4 IPM"
        assert m.n_p == N_P
        assert m.L_d == L_D
        assert m.L_q == L_Q
        assert m.psi_f == PSI_F
        assert m.geometry is None

    def test_saliency_ratio(self):
        """L_q/L_d = 2.11 from datasheet — confirms IPM topology."""
        assert L_Q / L_D == pytest.approx(2.111, rel=0.01)

    def test_backemf_recovery(self):
        """Phase back-EMF at 2100 rpm must recover the published 187 V rms."""
        omega_e = N_P * 2 * pi * SPEED_RPM / 60
        E0_rms = omega_e * PSI_F / sqrt(2)
        assert E0_rms == pytest.approx(BACKEMF_RMS_PUBLISHED, rel=0.005), (
            f"E0_rms={E0_rms:.1f} V, published {BACKEMF_RMS_PUBLISHED} V"
        )

    def test_nameplate_power_consistency(self):
        """T * omega = 5.0 * 2*pi*2100/60 = 1100 W ~ 1.1 kW."""
        omega = 2 * pi * SPEED_RPM / 60
        P = TORQUE_NAMEPLATE * omega
        assert P == pytest.approx(POWER_NAMEPLATE * 1000, rel=0.01)

    def test_mtpa_gte_nameplate(self):
        """MTPA at I_rated must exceed the S1 continuous nameplate torque."""
        gamma = mtpa_gamma(PSI_F, L_D, L_Q, I_RATED_PEAK)
        tau = mtpa_torque(N_P, PSI_F, L_D, L_Q, I_RATED_PEAK, gamma)
        assert tau >= TORQUE_NAMEPLATE, (
            f"MTPA tau={tau:.2f} Nm should be >= nameplate {TORQUE_NAMEPLATE} Nm"
        )

    def test_mtpa_physically_reasonable(self):
        """1.1 kW IPM: MTPA at I_rated should be in (5, 30) Nm range.
        Expect ~19 Nm (well above 5.0 Nm S1 rating — the nameplate is
        thermally derated, not current-limited at the rated point)."""
        gamma = mtpa_gamma(PSI_F, L_D, L_Q, I_RATED_PEAK)
        tau = mtpa_torque(N_P, PSI_F, L_D, L_Q, I_RATED_PEAK, gamma)
        assert 5.0 < tau < 30.0, f"tau={tau:.2f} outside expected range"

    def test_mtpa_exceeds_pure_magnet_torque(self):
        """With L_q/L_d = 2.11, MTPA should exceed pure q-axis (magnet-only)
        torque by >10%, confirming reluctance contribution."""
        gamma = mtpa_gamma(PSI_F, L_D, L_Q, I_RATED_PEAK)
        tau_mtpa = mtpa_torque(N_P, PSI_F, L_D, L_Q, I_RATED_PEAK, gamma)
        tau_q_only = 1.5 * N_P * PSI_F * I_RATED_PEAK
        assert tau_mtpa > tau_q_only * 1.10, (
            f"MTPA={tau_mtpa:.2f} Nm should be >10% above "
            f"pure-magnet {tau_q_only:.2f} Nm for this saliency"
        )
