"""Tests for solver parameter types and factory functions."""

import pytest

from phasesweep.geometry import default_inrunner, inrunner, outrunner
from phasesweep.motor import DriveParams
from phasesweep.solver_params import (
    AnalyticalParams,
    DriveSimParams,
    FemParams,
    j_s_from_phase_current,
    phase_current_from_j_s,
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
            drive=DriveParams(U_DC=540.0, MAX_I_S=20.0, W_REF=314.16),
        )
        assert p.R_s == 0.2

    def test_R_s_negative_raises(self):
        with pytest.raises(ValueError, match="R_s"):
            DriveSimParams(
                geometry=_geo(), n_p=2,
                R_s=-1.0, L_d=4e-3, L_q=4e-3, psi_f=0.1, J=0.002,
                drive=DriveParams(U_DC=540.0, MAX_I_S=20.0, W_REF=314.16),
            )

    def test_R_s_zero_ok(self):
        p = DriveSimParams(
            geometry=_geo(), n_p=2,
            R_s=0.0, L_d=4e-3, L_q=4e-3, psi_f=0.1, J=0.002,
            drive=DriveParams(U_DC=540.0, MAX_I_S=20.0, W_REF=314.16),
        )
        assert p.R_s == 0.0

    def test_L_d_zero_raises(self):
        with pytest.raises(ValueError, match="L_d"):
            DriveSimParams(
                geometry=_geo(), n_p=2,
                R_s=0.2, L_d=0.0, L_q=4e-3, psi_f=0.1, J=0.002,
                drive=DriveParams(U_DC=540.0, MAX_I_S=20.0, W_REF=314.16),
            )

    def test_L_q_negative_raises(self):
        with pytest.raises(ValueError, match="L_q"):
            DriveSimParams(
                geometry=_geo(), n_p=2,
                R_s=0.2, L_d=4e-3, L_q=-1e-3, psi_f=0.1, J=0.002,
                drive=DriveParams(U_DC=540.0, MAX_I_S=20.0, W_REF=314.16),
            )

    def test_psi_f_zero_raises(self):
        with pytest.raises(ValueError, match="psi_f"):
            DriveSimParams(
                geometry=_geo(), n_p=2,
                R_s=0.2, L_d=4e-3, L_q=4e-3, psi_f=0.0, J=0.002,
                drive=DriveParams(U_DC=540.0, MAX_I_S=20.0, W_REF=314.16),
            )

    def test_J_zero_raises(self):
        with pytest.raises(ValueError, match="J"):
            DriveSimParams(
                geometry=_geo(), n_p=2,
                R_s=0.2, L_d=4e-3, L_q=4e-3, psi_f=0.1, J=0.0,
                drive=DriveParams(U_DC=540.0, MAX_I_S=20.0, W_REF=314.16),
            )

    def test_geometry_optional(self):
        p = DriveSimParams(
            n_p=2,
            R_s=0.2, L_d=4e-3, L_q=4e-3, psi_f=0.1, J=0.002,
            drive=DriveParams(U_DC=540.0, MAX_I_S=20.0, W_REF=314.16),
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

    def test_missing_drive_fields_raises(self):
        motor = make_motor(drive=DriveParams())
        with pytest.raises(ValueError, match=r"\[drive\].*U_DC.*MAX_I_S.*W_REF"):
            prepare_drive_sim(motor)

    def test_partial_drive_raises(self):
        motor = make_motor(drive=DriveParams(U_DC=540.0))
        with pytest.raises(ValueError, match=r"\[drive\].*MAX_I_S.*W_REF"):
            prepare_drive_sim(motor)


# ---------------------------------------------------------------------------
# j_s <-> phase current mapping
# ---------------------------------------------------------------------------

def _slotted_geo(**overrides):
    kw = dict(r_outer=1.0, r_stator=0.70, r_magnet=0.64, r_rotor=0.30,
              r_inner=0.0, n_slots=12, slot_depth=0.05)
    kw.update(overrides)
    return inrunner(**kw)


class TestJsPhaseCurrentMapping:

    def test_moment_untapered_closed_form(self):
        from phasesweep.fem_field import slot_source_moment
        geo = _slotted_geo()  # slot_width_ratio default 0.6, no opening taper
        expected = 0.6 * (0.75**2 - 0.70**2) / 2
        assert slot_source_moment(geo) == pytest.approx(expected, rel=1e-12)

    def test_moment_tapered_sums_opening_and_body(self):
        from phasesweep.defaults import SLOT_OPENING_FRACTION
        from phasesweep.fem_field import slot_source_moment
        geo = _slotted_geo(slot_opening_width=0.055)  # opening ratio ~0.15
        w_open = geo.slot_opening_ratio
        r_step = 0.70 + SLOT_OPENING_FRACTION * 0.05
        expected = (w_open * (r_step**2 - 0.70**2) / 2
                    + 0.6 * (0.75**2 - r_step**2) / 2)
        assert slot_source_moment(geo) == pytest.approx(expected, rel=1e-12)

    def test_moment_outrunner_slots_go_inward(self):
        from phasesweep.fem_field import slot_source_moment
        geo = outrunner(r_outer=0.10, r_rotor=0.08, r_magnet=0.06,
                        r_stator=0.04, r_inner=0.02, n_slots=6,
                        slot_depth=0.005)
        expected = 0.6 * (0.04**2 - 0.035**2) / 2
        assert slot_source_moment(geo) == pytest.approx(expected, rel=1e-12)

    def test_moment_no_slot_faces_raises(self):
        from phasesweep.fem_field import slot_source_moment
        with pytest.raises(ValueError, match="slot faces"):
            slot_source_moment(default_inrunner())  # n_slots = 0
        with pytest.raises(ValueError, match="slot faces"):
            slot_source_moment(_slotted_geo(slot_depth=0.0))

    def test_moment_degenerate_slot_pole_combo_raises(self):
        # When 2*n_p ≡ 0 mod n_slots the comb cross terms
        # do not cancel (+63.7% at duty 0.5) — combos are 3-phase-
        # degenerate, so the guard is garbage-in insurance
        from phasesweep.fem_field import slot_source_moment
        geo = _slotted_geo()  # Q = 12
        with pytest.raises(ValueError, match="multiple of n_slots"):
            slot_source_moment(geo, n_p=6)  # 2*6 = 12 ≡ 0 mod 12
        # healthy combo (Q=12, n_p=7) unchanged, with or without n_p
        assert (slot_source_moment(geo, n_p=7)
                == pytest.approx(slot_source_moment(geo), rel=1e-15))

    def test_mapping_matches_winding_formula(self):
        from math import pi
        # j_s = 3 N_eff I / (pi S); k_w cancels (the solver applies it)
        m = make_motor(geometry=_slotted_geo(), N=50, coils_series=2)
        S = 0.6 * (0.75**2 - 0.70**2) / 2
        assert j_s_from_phase_current(m, 10.0) == pytest.approx(
            3 * 100 * 10.0 / (pi * S), rel=1e-12)

    def test_round_trip_exact(self):
        m = make_motor(geometry=_slotted_geo(), coils_series=2)
        j_s = j_s_from_phase_current(m, 7.3)
        assert phase_current_from_j_s(m, j_s) == pytest.approx(7.3, rel=1e-14)

    def test_missing_winding_data_raises(self):
        m = make_motor(geometry=_slotted_geo(), N=None)
        with pytest.raises(ValueError, match="N is required"):
            j_s_from_phase_current(m, 1.0)

    def test_missing_geometry_raises(self):
        m = make_motor(geometry=None)
        with pytest.raises(ValueError, match="geometry"):
            j_s_from_phase_current(m, 1.0)
