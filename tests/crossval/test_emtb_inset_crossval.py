"""Cross-validation: NTU e-MTB Surface-Inset PMSM (measured torque/power).

Source: Goh, Xie, Wang, Zhu, Yu, Lee (2025), Energies 18(14):3709.
DOI: 10.3390/en18143709

First surface-inset PMSM in the repo. Prototype fabricated and
dynamometer-tested (HBM torque transducer, LeCroy MDA 8108HD).
Peak torque 4.62 Nm at 29.3 A peak, 500 r/min. No R_s/L_d/L_q/psi_f
published — topology reference only.
"""

from math import pi

import pytest

from phasesweep.machines.configs import load_motor

TOML = "data/emtb_inset/emtb_inset_spm.toml"

TORQUE_MEASURED = 4.62       # Nm, Table 3 experimental
POWER_MEASURED = 726.0       # W, Table 3 experimental
SPEED_RATED = 1500           # rpm, Table 3
TORQUE_DENSITY = 28.41       # kNm/m3, Table 3 experimental
EFFICIENCY_MEASURED = 0.80   # Table 3 experimental


class TestEmtbInsetCrossval:

    def test_toml_loads(self):
        m = load_motor(TOML)
        assert m.name == "NTU e-MTB Surface-Inset PMSM"
        assert m.n_p == 7

    def test_no_geometry(self):
        """Rotor dimensions not published — no geometry section."""
        m = load_motor(TOML)
        assert m.geometry is None

    def test_no_dq_model(self):
        """R_s, L_d, L_q, psi_f all unpublished."""
        m = load_motor(TOML)
        assert m.R_s is None
        assert m.L_d is None
        assert m.L_q is None
        assert m.psi_f is None

    def test_pole_count(self):
        """12 slots / 14 poles (Table 2)."""
        m = load_motor(TOML)
        assert m.n_p == 7, f"n_p={m.n_p}, expected 7 (14 poles)"

    def test_power_torque_consistency(self):
        """P = T * omega: 4.62 Nm at 1500 rpm = 726 W (Table 3)."""
        omega = 2 * pi * SPEED_RATED / 60
        P = TORQUE_MEASURED * omega
        assert P == pytest.approx(POWER_MEASURED, rel=0.02), (
            f"P={P:.1f} W, expected {POWER_MEASURED} W"
        )

    def test_winding_factor(self):
        """k_w = 0.933 (Table 2)."""
        m = load_motor(TOML)
        assert m.k_w == pytest.approx(0.933, rel=0.01)

    def test_b_rem_n48(self):
        """N48 NdFeB: B_rem in typical range 1.36-1.43 T."""
        m = load_motor(TOML)
        assert m.B_rem is not None
        assert 1.36 <= m.B_rem <= 1.43, (
            f"B_rem={m.B_rem}, expected N48 range 1.36-1.43 T"
        )
