"""Tests for solver parameter types and factory functions."""

import pytest

from phasesweep.geometry import default_inrunner
from phasesweep.motor import DriveParams
from phasesweep.solver_params import (
    AnalyticalParams,
    DriveSimParams,
    FemParams,
    prepare_analytical,
    prepare_drive_sim,
    prepare_fem,
    winding_transfer,
)
from tests.conftest import make_motor


def test_winding_transfer_uses_n_eff():
    # N=50, coils_series=2 -> N_eff=100; bare N would be off by 2x
    m = make_motor(coils_series=2)
    expected = 2 * 100 * 0.966 * 0.70 * 0.10 / 2
    assert winding_transfer(m) == pytest.approx(expected, rel=1e-12)


def _geo():
    return default_inrunner()


# ---------------------------------------------------------------------------
# AnalyticalParams construction and validation
# ---------------------------------------------------------------------------

class TestAnalyticalParams:

    def test_valid_construction(self):
        p = AnalyticalParams(geometry=_geo(), n_p=4, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=1000)
        assert p.n_p == 4
        assert p.B_rem == 1.2

    def test_frozen(self):
        p = AnalyticalParams(geometry=_geo(), n_p=4, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=1000)
        with pytest.raises(AttributeError):
            p.B_rem = 0.5  # type: ignore[misc]

    def test_n_p_below_2_raises(self):
        with pytest.raises(ValueError, match="n_p"):
            AnalyticalParams(geometry=_geo(), n_p=1, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=1000)

    def test_B_rem_zero_raises(self):
        with pytest.raises(ValueError, match="B_rem"):
            AnalyticalParams(geometry=_geo(), n_p=4, B_rem=0.0, mu_r_pm=1.05, mu_r_fe=1000)

    def test_B_rem_negative_raises(self):
        with pytest.raises(ValueError, match="B_rem"):
            AnalyticalParams(geometry=_geo(), n_p=4, B_rem=-0.5, mu_r_pm=1.05, mu_r_fe=1000)

    def test_mu_r_pm_zero_raises(self):
        with pytest.raises(ValueError, match="mu_r_pm"):
            AnalyticalParams(geometry=_geo(), n_p=4, B_rem=1.2, mu_r_pm=0.0, mu_r_fe=1000)

    def test_mu_r_fe_zero_raises(self):
        with pytest.raises(ValueError, match="mu_r_fe"):
            AnalyticalParams(geometry=_geo(), n_p=4, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=0.0)


# ---------------------------------------------------------------------------
# FemParams construction and validation
# ---------------------------------------------------------------------------

class TestFemParams:

    def test_valid_construction(self):
        p = FemParams(geometry=_geo(), n_p=4, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=1000)
        assert p.n_p == 4

    def test_B_rem_negative_raises(self):
        with pytest.raises(ValueError, match="B_rem"):
            FemParams(geometry=_geo(), n_p=4, B_rem=-0.5, mu_r_pm=1.05, mu_r_fe=1000)

    def test_n_p_below_2_raises(self):
        with pytest.raises(ValueError, match="n_p"):
            FemParams(geometry=_geo(), n_p=1, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=1000)


# ---------------------------------------------------------------------------
# DriveSimParams construction and validation
# ---------------------------------------------------------------------------

class TestDriveSimParams:

    def test_valid_construction(self):
        p = DriveSimParams(
            geometry=_geo(), n_p=2,
            R_s=0.2, L_d=4e-3, L_q=4e-3, psi_f=0.1, J=0.002,
            drive=DriveParams(),
        )
        assert p.R_s == 0.2

    def test_R_s_negative_raises(self):
        with pytest.raises(ValueError, match="R_s"):
            DriveSimParams(
                geometry=_geo(), n_p=2,
                R_s=-1.0, L_d=4e-3, L_q=4e-3, psi_f=0.1, J=0.002,
                drive=DriveParams(),
            )

    def test_R_s_zero_ok(self):
        p = DriveSimParams(
            geometry=_geo(), n_p=2,
            R_s=0.0, L_d=4e-3, L_q=4e-3, psi_f=0.1, J=0.002,
            drive=DriveParams(),
        )
        assert p.R_s == 0.0

    def test_L_d_zero_raises(self):
        with pytest.raises(ValueError, match="L_d"):
            DriveSimParams(
                geometry=_geo(), n_p=2,
                R_s=0.2, L_d=0.0, L_q=4e-3, psi_f=0.1, J=0.002,
                drive=DriveParams(),
            )

    def test_L_q_negative_raises(self):
        with pytest.raises(ValueError, match="L_q"):
            DriveSimParams(
                geometry=_geo(), n_p=2,
                R_s=0.2, L_d=4e-3, L_q=-1e-3, psi_f=0.1, J=0.002,
                drive=DriveParams(),
            )

    def test_psi_f_zero_raises(self):
        with pytest.raises(ValueError, match="psi_f"):
            DriveSimParams(
                geometry=_geo(), n_p=2,
                R_s=0.2, L_d=4e-3, L_q=4e-3, psi_f=0.0, J=0.002,
                drive=DriveParams(),
            )

    def test_J_zero_raises(self):
        with pytest.raises(ValueError, match="J"):
            DriveSimParams(
                geometry=_geo(), n_p=2,
                R_s=0.2, L_d=4e-3, L_q=4e-3, psi_f=0.1, J=0.0,
                drive=DriveParams(),
            )

    def test_geometry_optional(self):
        p = DriveSimParams(
            n_p=2,
            R_s=0.2, L_d=4e-3, L_q=4e-3, psi_f=0.1, J=0.002,
            drive=DriveParams(),
        )
        assert p.geometry is None


# ---------------------------------------------------------------------------
# prepare_analytical factory
# ---------------------------------------------------------------------------

class TestPrepareAnalytical:

    def test_motor_with_b_rem(self):
        motor = make_motor(B_rem=1.2, psi_f=None)
        p = prepare_analytical(motor)
        assert p.B_rem == motor.B_rem
        assert p.geometry is motor.geometry

    def test_derives_b_rem_from_psi_f(self):
        motor = make_motor()
        p = prepare_analytical(motor)
        assert p.B_rem > 0

    def test_missing_b_rem_and_derivation_raises(self):
        motor = make_motor(psi_f=None)
        with pytest.raises(ValueError, match="cannot determine B_rem"):
            prepare_analytical(motor)

    def test_missing_winding_for_derivation_raises(self):
        motor = make_motor(N=None)
        with pytest.raises(ValueError, match="cannot determine B_rem"):
            prepare_analytical(motor)

    def test_consistency_warning(self):
        motor = make_motor(B_rem=5.0)  # inconsistent with psi_f
        with pytest.warns(UserWarning, match="disagrees"):
            prepare_analytical(motor)

    def test_consistent_no_warning(self):
        # Derive B_rem from psi_f, then supply both — should be consistent
        p1 = prepare_analytical(make_motor())
        motor_consistent = make_motor(B_rem=p1.B_rem)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            prepare_analytical(motor_consistent)


# ---------------------------------------------------------------------------
# prepare_fem factory
# ---------------------------------------------------------------------------

class TestPrepareFem:

    def test_motor_with_b_rem(self):
        p = prepare_fem(make_motor(B_rem=1.2, psi_f=None))
        assert isinstance(p, FemParams)
        assert p.B_rem == 1.2

    def test_derives_b_rem(self):
        p = prepare_fem(make_motor())
        assert p.B_rem > 0

    def test_missing_raises(self):
        with pytest.raises(ValueError, match="cannot determine B_rem"):
            prepare_fem(make_motor(psi_f=None))


# ---------------------------------------------------------------------------
# prepare_drive_sim factory
# ---------------------------------------------------------------------------

class TestPrepareDriveSim:

    def test_full_motor(self):
        p = prepare_drive_sim(make_motor())
        assert isinstance(p, DriveSimParams)
        assert p.R_s == 0.2
        assert p.psi_f == 0.1

    def test_derives_psi_f_from_b_rem(self):
        motor = make_motor(B_rem=1.2, psi_f=None)
        p = prepare_drive_sim(motor)
        assert p.psi_f > 0

    def test_missing_psi_f_and_derivation_raises(self):
        motor = make_motor(psi_f=None)
        with pytest.raises(ValueError, match="cannot determine psi_f"):
            prepare_drive_sim(motor)

    def test_missing_R_s_raises(self):
        motor = make_motor(R_s=None)
        with pytest.raises(ValueError, match="R_s"):
            prepare_drive_sim(motor)

    def test_missing_multiple_raises(self):
        motor = make_motor(R_s=None, L_d=None)
        with pytest.raises(ValueError, match="R_s, L_d"):
            prepare_drive_sim(motor)

    def test_missing_J_raises(self):
        motor = make_motor(J=None)
        with pytest.raises(ValueError, match="J"):
            prepare_drive_sim(motor)
