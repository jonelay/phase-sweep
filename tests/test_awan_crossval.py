"""Cross-validation: Awan 2.2-kW IPM (published torque data).

Source: Awan (2019) Table 6.2, Aalto University doctoral dissertation 187/2019.
DOI: 10.1109/TIA.2018.2862410

This motor is from a controls dissertation -- only electrical parameters published.
No geometry, materials, or winding data available. Cannot be loaded as a Motor
(Motor requires Geometry). Validated directly using the published electrical
params and the MTPA computation functions.

Eligible for: rated_torque (electrical params only).
Not eligible for: analytical, FEM (no geometry), drive_sim (no J).
"""

from math import sqrt

import pytest

from phasesweep.rated_torque import mtpa_gamma, mtpa_torque

# Published values: Awan (2019) Table 6.2
N_P = 3
PSI_F = 0.545        # Wb
L_D = 0.036          # H
L_Q = 0.051          # H
I_RATED = 4.3 * sqrt(2)  # A peak (4.3 A_rms from Table 6.2)

# Published target
TAU_RATED_PUBLISHED = 14.0  # Nm, Table 6.2 (0.80 p.u.)


class TestAwanRatedTorque:

    def test_saliency_ratio(self):
        """L_q/L_d = 1.42 from Table 6.2."""
        assert L_Q / L_D == pytest.approx(1.42, rel=0.01)

    def test_backemf_from_psi_f(self):
        """Back-EMF E0 = W_REF * n_p * psi_f ~ 257 V (parameter sanity)."""
        W_REF = 157.08  # rad/s, 1500 rpm
        E0 = W_REF * N_P * PSI_F
        assert E0 == pytest.approx(256.8, rel=0.01), f"E0={E0:.1f} V, expected ~256.8 V"

    def test_mtpa_gte_published_rating(self):
        """MTPA at I_rated must be >= 14 Nm published continuous rating."""
        gamma = mtpa_gamma(PSI_F, L_D, L_Q, I_RATED)
        tau = mtpa_torque(N_P, PSI_F, L_D, L_Q, I_RATED, gamma)
        assert tau >= TAU_RATED_PUBLISHED, (
            f"MTPA tau={tau:.2f} Nm should be >= published {TAU_RATED_PUBLISHED} Nm"
        )

    def test_mtpa_physically_reasonable(self):
        """2.2 kW motor: tau in (14, 30) Nm range."""
        gamma = mtpa_gamma(PSI_F, L_D, L_Q, I_RATED)
        tau = mtpa_torque(N_P, PSI_F, L_D, L_Q, I_RATED, gamma)
        assert 14.0 < tau < 30.0, f"tau={tau:.2f} outside expected range"
