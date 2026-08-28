"""Cross-validation: Uppsala 5 kW Marine Current Generator (120-pole SPM).

Sources:
  [1] Thomas, Grabbe, Yuen, Leijon, ISRN Renewable Energy 2012, Art. 489379.
  [2] Grabbe, Eriksson, Leijon, Scientific World Journal 2015, PMC4386604.

120-pole (60 pole pairs) cable-wound surface-magnet PMSG, 2-meter stator,
10 rpm nominal, 10.5 mm airgap. Published R = 0.475 Ohm, L = 11.5 mH.
psi_f derived from circuit analysis at 4.44 Ohm/phase nominal load.
"""

from math import pi, sqrt

import pytest

from phasesweep.machines.configs import load_motor
from phasesweep.models.rated_torque import mtpa_gamma, mtpa_torque

TOML = "data/uppsala_marine/uppsala_5kw_marine.toml"

N_P = 60
PSI_F = 2.181
L_D = 0.0115
L_Q = 0.0115
R_S = 0.475
SPEED_RPM = 10
POWER_KW = 5.0
EFFICIENCY = 0.864
V_RATED_LL = 150.0
R_LOAD = 4.44
OMEGA_MECH = 2 * pi * SPEED_RPM / 60
OMEGA_E = N_P * OMEGA_MECH

V_PHASE_LOAD = V_RATED_LL / sqrt(3)
I_RATED_RMS = V_PHASE_LOAD / R_LOAD
I_RATED_PEAK = I_RATED_RMS * sqrt(2)


class TestUppsalaMarineCrossval:

    def test_toml_loads(self):
        m = load_motor(TOML)
        assert m.name == "Uppsala 5 kW Marine Current Generator"
        assert m.n_p == N_P
        assert m.L_d == L_D
        assert m.L_q == L_Q
        assert m.R_s == R_S
        assert m.psi_f == PSI_F
        assert m.geometry is None

    def test_spm_non_salient(self):
        """L_d = L_q confirms surface-magnet (no saliency)."""
        assert L_D == L_Q

    def test_psi_f_circuit_derivation(self):
        """psi_f derived from circuit: E_phase / omega_e. Cross-check
        that the derived E_phase is consistent with rated load."""
        Z_total = complex(R_S + R_LOAD, OMEGA_E * L_D)
        E_phase_rms = I_RATED_RMS * abs(Z_total)
        psi_f_check = E_phase_rms * sqrt(2) / OMEGA_E
        assert psi_f_check == pytest.approx(PSI_F, rel=0.005), (
            f"psi_f from circuit={psi_f_check:.4f}, TOML={PSI_F}"
        )

    def test_power_at_load(self):
        """3-phase power at 4.44 Ohm/phase load should be ~5 kW."""
        P_load = 3 * I_RATED_RMS**2 * R_LOAD
        assert P_load == pytest.approx(POWER_KW * 1000, rel=0.02), (
            f"P_load={P_load:.0f} W, expected ~{POWER_KW*1000:.0f} W"
        )

    def test_loss_budget(self):
        """Total losses ~ 785 W (published [2])."""
        P_load = 3 * I_RATED_RMS**2 * R_LOAD
        P_mech = P_load / EFFICIENCY
        losses = P_mech - P_load
        assert losses == pytest.approx(785, rel=0.05), (
            f"losses={losses:.0f} W, published ~785 W"
        )

    def test_mtpa_torque_physically_reasonable(self):
        """5 kW at 10 rpm: torque in (4000, 7000) Nm range, plus a tight
        regression pin — the loose bound alone would pass a 2x model
        error. The pin is the current model value (rated_torque v4), not
        a published number; re-freeze deliberately on any MTPA physics
        change."""
        gamma = mtpa_gamma(PSI_F, L_D, L_Q, I_RATED_PEAK)
        tau = mtpa_torque(N_P, PSI_F, L_D, L_Q, I_RATED_PEAK, gamma)
        assert 4000 < tau < 7000, f"tau={tau:.0f} outside expected range"
        assert tau == pytest.approx(5414.53, rel=1e-3)

    def test_mtpa_gamma_zero_for_spm(self):
        """SPM machine: MTPA angle should be zero (pure q-axis current)."""
        gamma = mtpa_gamma(PSI_F, L_D, L_Q, I_RATED_PEAK)
        assert abs(gamma) < 0.01, f"gamma={gamma:.4f}, expected ~0 for SPM"

    def test_electromagnetic_power_exceeds_output(self):
        """Electromagnetic power > rated output (losses are positive)."""
        gamma = mtpa_gamma(PSI_F, L_D, L_Q, I_RATED_PEAK)
        tau = mtpa_torque(N_P, PSI_F, L_D, L_Q, I_RATED_PEAK, gamma)
        P_em = tau * OMEGA_MECH
        assert P_em > POWER_KW * 1000, (
            f"P_em={P_em:.0f} W should exceed {POWER_KW*1000:.0f} W"
        )
