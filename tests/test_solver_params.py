"""Tests for solver parameter types and factory functions."""

import dataclasses

import pytest

from phasesweep.machines.geometry import default_inrunner, inrunner, outrunner
from phasesweep.machines.motor import DriveParams, Motor
from phasesweep.simulation.params import DriveSimParams, prepare_drive_sim
from phasesweep.solver_params import (
    j_s_from_phase_current,
    phase_current_from_j_s,
    resolve_psi_f,
    winding_transfer,
)
from phasesweep.solvers.analytical import AnalyticalParams, prepare_analytical
from phasesweep.solvers.fem_runner import FemParams, prepare_fem
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

    def test_psi_f_negative_raises(self):
        with pytest.raises(ValueError, match="psi_f"):
            DriveSimParams(
                geometry=_geo(), n_p=2,
                R_s=0.2, L_d=4e-3, L_q=4e-3, psi_f=-0.1, J=0.002,
                drive=DriveParams(U_DC=540.0, MAX_I_S=20.0, W_REF=314.16),
            )

    def test_psi_f_zero_allowed(self):
        # SynRM (reluctance-only): psi_f = 0 is a valid machine
        p = DriveSimParams(
            geometry=_geo(), n_p=2,
            R_s=0.2, L_d=4e-3, L_q=8e-3, psi_f=0.0, J=0.002,
            drive=DriveParams(U_DC=540.0, MAX_I_S=20.0, W_REF=314.16),
        )
        assert p.psi_f == 0.0

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


def _inset_geo(**kw):
    import dataclasses

    from phasesweep.machines.geometry import default_inrunner
    return dataclasses.replace(
        default_inrunner(), magnet_mount="inset", **kw)


class TestInsetGating:

    def test_analytical_gates_inset(self):
        motor = make_motor(geometry=_inset_geo(), alpha_p=0.7, B_rem=1.2)
        with pytest.raises(ValueError, match="inset rotors.*not.*supported"):
            prepare_analytical(motor)

    def test_fem_accepts_inset(self):
        p = prepare_fem(
            make_motor(geometry=_inset_geo(), alpha_p=0.7, B_rem=1.2))
        assert p.geometry.magnet_mount == "inset"

    def test_fem_params_inset_requires_partial_alpha_p(self):
        with pytest.raises(ValueError, match="alpha_p < 1"):
            FemParams(geometry=_inset_geo(), n_p=2, B_rem=1.2,
                      mu_r_pm=1.05, mu_r_fe=1000.0, alpha_p=1.0)

    def test_inset_requires_explicit_b_rem(self):
        # the psi_f -> B_rem derivation assumes air between arcs
        motor = make_motor(geometry=_inset_geo(), alpha_p=0.7, B_rem=None)
        with pytest.raises(ValueError, match="explicit B_rem"):
            prepare_fem(motor)

    def test_inset_requires_explicit_psi_f(self):
        motor = make_motor(geometry=_inset_geo(), alpha_p=0.7,
                           B_rem=1.2, psi_f=None)
        with pytest.raises(ValueError, match="explicit psi_f"):
            prepare_drive_sim(motor)

    def test_inset_skips_consistency_warning(self):
        import warnings
        motor = make_motor(geometry=_inset_geo(), alpha_p=0.7, B_rem=5.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            prepare_fem(motor)


def _bridge_geo(**kw):
    import dataclasses

    from phasesweep.machines.geometry import default_inrunner
    return dataclasses.replace(
        default_inrunner(), bridge_thickness=0.01, **kw)


class TestBridgeGating:

    def test_analytical_gates_bridge(self):
        motor = make_motor(geometry=_bridge_geo(), B_rem=1.2)
        with pytest.raises(ValueError, match="bridge ring"):
            prepare_analytical(motor)

    def test_fem_accepts_surface_bridge(self):
        p = prepare_fem(make_motor(geometry=_bridge_geo(), B_rem=1.2))
        assert p.geometry.bridge_thickness == 0.01
        assert p.geometry.magnet_mount == "surface"

    def test_bridge_requires_explicit_b_rem(self):
        motor = make_motor(geometry=_bridge_geo(), B_rem=None)
        with pytest.raises(ValueError, match="bridged motors.*explicit B_rem"):
            prepare_fem(motor)

    def test_bridge_requires_explicit_psi_f(self):
        motor = make_motor(geometry=_bridge_geo(), B_rem=1.2, psi_f=None)
        with pytest.raises(ValueError, match="bridged motors.*explicit psi_f"):
            prepare_drive_sim(motor)

    def test_bridge_skips_consistency_warning(self):
        import warnings
        motor = make_motor(geometry=_bridge_geo(), B_rem=5.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            prepare_fem(motor)


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

    def test_derivation_error_names_missing_geometry(self):
        # Datasheet-only motor: B_rem and the winding are set, but with no
        # geometry the B_rem -> psi_f transfer cannot run. The error must
        # name the geometry, not claim psi_f's own absence as the cause.
        motor = Motor(name="ds", geometry=None, n_p=4, R_s=0.1, L_d=1e-3,
                      L_q=1e-3, B_rem=1.2, N=20, k_w=0.95, L_stk=0.05)
        with pytest.raises(ValueError, match=r"missing for derivation: \[geometry\]"):
            resolve_psi_f(motor)

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
        from phasesweep.solvers.fem_mesh import slot_source_moment
        geo = _slotted_geo()  # slot_width_ratio default 0.6, no opening taper
        expected = 0.6 * (0.75**2 - 0.70**2) / 2
        assert slot_source_moment(geo) == pytest.approx(expected, rel=1e-12)

    def test_moment_tapered_sums_opening_and_body(self):
        from phasesweep.defaults import SLOT_OPENING_FRACTION
        from phasesweep.solvers.fem_mesh import slot_source_moment
        geo = _slotted_geo(slot_opening_width=0.055)  # opening ratio ~0.15
        w_open = geo.slot_opening_ratio
        r_step = 0.70 + SLOT_OPENING_FRACTION * 0.05
        expected = (w_open * (r_step**2 - 0.70**2) / 2
                    + 0.6 * (0.75**2 - r_step**2) / 2)
        assert slot_source_moment(geo) == pytest.approx(expected, rel=1e-12)

    def test_moment_outrunner_slots_go_inward(self):
        from phasesweep.solvers.fem_mesh import slot_source_moment
        geo = outrunner(r_outer=0.10, r_rotor=0.08, r_magnet=0.06,
                        r_stator=0.04, r_inner=0.02, n_slots=6,
                        slot_depth=0.005)
        expected = 0.6 * (0.04**2 - 0.035**2) / 2
        assert slot_source_moment(geo) == pytest.approx(expected, rel=1e-12)

    def test_moment_no_slot_faces_raises(self):
        from phasesweep.solvers.fem_mesh import slot_source_moment
        with pytest.raises(ValueError, match="slot faces"):
            slot_source_moment(default_inrunner())  # n_slots = 0
        with pytest.raises(ValueError, match="slot faces"):
            slot_source_moment(_slotted_geo(slot_depth=0.0))

    def test_moment_degenerate_slot_pole_combo_raises(self):
        # When 2*n_p ≡ 0 mod n_slots the comb cross terms
        # do not cancel (+63.7% at duty 0.5) — combos are 3-phase-
        # degenerate, so the guard is garbage-in insurance
        from phasesweep.solvers.fem_mesh import slot_source_moment
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


# ---------------------------------------------------------------------------
# Halbach schema (Phase 1)
# ---------------------------------------------------------------------------

class TestFemParamsHalbach:

    def test_halbach_construction(self):
        import math
        p = FemParams(
            geometry=_geo(), n_p=10, B_rem=1.35, mu_r_pm=1.09, mu_r_fe=1000,
            magnetization="halbach", halbach_segments=3,
            halbach_angles_rad=(0.0, math.radians(-45), math.radians(-135)),
        )
        assert p.magnetization == "halbach"
        assert p.halbach_segments == 3
        assert len(p.halbach_angles_rad) == 3

    def test_halbach_ideal_no_angles(self):
        p = FemParams(
            geometry=_geo(), n_p=4, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=1000,
            magnetization="halbach", halbach_segments=4,
        )
        assert p.halbach_angles_rad is None

    def test_halbach_without_segments_raises(self):
        with pytest.raises(ValueError, match="halbach_segments"):
            FemParams(
                geometry=_geo(), n_p=4, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=1000,
                magnetization="halbach",
            )

    def test_halbach_alpha_p_not_one_raises(self):
        with pytest.raises(ValueError, match="alpha_p"):
            FemParams(
                geometry=_geo(), n_p=4, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=1000,
                magnetization="halbach", halbach_segments=3, alpha_p=0.8,
            )

    def test_halbach_angles_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length"):
            FemParams(
                geometry=_geo(), n_p=4, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=1000,
                magnetization="halbach", halbach_segments=3,
                halbach_angles_rad=(0.0, -0.7),
            )

    def test_radial_default_unchanged(self):
        p = FemParams(geometry=_geo(), n_p=4, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=1000)
        assert p.magnetization == "radial"
        assert p.halbach_segments is None
        assert p.halbach_angles_rad is None


class TestPrepareHalbach:

    def test_prepare_fem_halbach_passthrough(self):
        import math

        from phasesweep.machines.motor import Motor
        m = Motor(
            name="h", geometry=_geo(), n_p=10, B_rem=1.35,
            magnetization="halbach", halbach_segments=3,
            halbach_angles_deg=(0.0, -45.0, -135.0),
        )
        p = prepare_fem(m)
        assert p.magnetization == "halbach"
        assert p.halbach_segments == 3
        assert p.halbach_angles_rad == pytest.approx(
            (0.0, math.radians(-45), math.radians(-135)), rel=1e-12,
        )

    def test_prepare_fem_halbach_ideal(self):
        from phasesweep.machines.motor import Motor
        m = Motor(
            name="h", geometry=_geo(), n_p=4, B_rem=1.2,
            magnetization="halbach", halbach_segments=4,
        )
        p = prepare_fem(m)
        assert p.magnetization == "halbach"
        assert p.halbach_segments == 4
        assert p.halbach_angles_rad is None

    def test_prepare_fem_radial_unchanged(self):
        p = prepare_fem(make_motor(B_rem=1.2, psi_f=None))
        assert p.magnetization == "radial"
        assert p.halbach_segments is None
        assert p.halbach_angles_rad is None

    def test_prepare_analytical_halbach_raises(self):
        from phasesweep.machines.motor import Motor
        m = Motor(
            name="h", geometry=_geo(), n_p=4, B_rem=1.2,
            magnetization="halbach", halbach_segments=3,
        )
        with pytest.raises(ValueError, match="radial magnetization only"):
            prepare_analytical(m)

    def test_resolve_b_rem_halbach_psi_f_only_raises(self):
        from phasesweep.machines.motor import Motor
        m = Motor(
            name="h", geometry=_geo(), n_p=4, psi_f=0.1,
            N=50, k_w=0.966, L_stk=0.10, coils_series=1,
            magnetization="halbach", halbach_segments=3,
        )
        with pytest.raises(ValueError, match="explicit B_rem"):
            prepare_fem(m)

    def test_resolve_b_rem_halbach_explicit_ok(self):
        from phasesweep.machines.motor import Motor
        m = Motor(
            name="h", geometry=_geo(), n_p=4, B_rem=1.2,
            magnetization="halbach", halbach_segments=3,
        )
        p = prepare_fem(m)
        assert p.B_rem == 1.2

    def test_resolve_psi_f_halbach_b_rem_only_raises(self):
        from phasesweep.machines.motor import Motor
        from phasesweep.models.rated_torque import prepare_rated_torque
        m = Motor(
            name="h", geometry=_geo(), n_p=4,
            B_rem=1.2, I_rated=10.0,
            N=50, k_w=0.966, L_stk=0.10, coils_series=1,
            magnetization="halbach", halbach_segments=3,
        )
        with pytest.raises(ValueError, match="explicit psi_f"):
            prepare_rated_torque(m)

    def test_resolve_psi_f_halbach_explicit_ok(self):
        from phasesweep.machines.motor import Motor
        from phasesweep.models.rated_torque import prepare_rated_torque
        m = Motor(
            name="h", geometry=_geo(), n_p=4,
            B_rem=1.2, psi_f=0.1, I_rated=10.0,
            magnetization="halbach", halbach_segments=3,
        )
        p = prepare_rated_torque(m)
        assert p.psi_f == 0.1

    def test_fem_params_radial_rejects_halbach_segments(self):
        with pytest.raises(ValueError, match="halbach_segments must be None"):
            FemParams(
                geometry=_geo(), n_p=4, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=1000,
                magnetization="radial", halbach_segments=3,
            )

    def test_fem_params_radial_rejects_halbach_angles(self):
        with pytest.raises(ValueError, match="halbach_angles_rad must be None"):
            FemParams(
                geometry=_geo(), n_p=4, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=1000,
                magnetization="radial", halbach_angles_rad=(0.0,),
            )

    def test_prepare_demag_screen_halbach_passes(self):
        from phasesweep.machines.motor import Motor
        from phasesweep.models.demag_screen import prepare_demag_screen
        geo = inrunner(
            r_outer=0.06, r_stator=0.048, r_magnet=0.04, r_rotor=0.03,
            n_slots=12, slot_depth=0.005,
        )
        m = Motor(
            name="h", geometry=geo, n_p=4,
            B_rem=1.2, B_knee=-0.5,
            N=50, k_w=0.966, L_stk=0.10, coils_series=1,
            magnetization="halbach", halbach_segments=3,
            halbach_angles_deg=(0.0, -60.0, -120.0),
        )
        p = prepare_demag_screen(m)
        assert p.fem.magnetization == "halbach"
        assert p.fem.halbach_segments == 3


# ---------------------------------------------------------------------------
# psi_f = 0 (SynRM) across the circuit-model params types
# ---------------------------------------------------------------------------


class TestSynRMParams:

    def test_rated_torque_params_accept_psi_f_zero(self):
        from phasesweep.models.rated_torque import RatedTorqueParams
        p = RatedTorqueParams(n_p=2, psi_f=0.0, I_rated=10.0,
                              L_d=8e-3, L_q=4e-3)
        assert p.psi_f == 0.0

    def test_stall_torque_params_accept_psi_f_zero(self):
        from phasesweep.models.rated_torque import StallTorqueParams
        p = StallTorqueParams(n_p=2, psi_f=0.0, I_stall=20.0,
                              L_d=8e-3, L_q=4e-3)
        assert p.psi_f == 0.0

    def test_torque_speed_params_accept_psi_f_zero_salient(self):
        from phasesweep.models.torque_speed import TorqueSpeedParams
        p = TorqueSpeedParams(n_p=2, psi_f=0.0, R_s=0.2, L_d=8e-3,
                              L_q=4e-3, I_peak=20.0, U_max=311.0)
        assert p.psi_f == 0.0

    def test_thermal_duty_params_accept_psi_f_zero(self):
        from phasesweep.models.thermal_duty import ThermalDutyParams
        p = ThermalDutyParams(n_p=2, psi_f=0.0, R_s=0.5, p_s1_budget=100.0,
                              budget_source="rated_current",
                              duty_profile=((5.0, 10.0),),
                              L_d=8e-3, L_q=4e-3)
        assert p.psi_f == 0.0

    def test_params_reject_negative_psi_f(self):
        from phasesweep.models.rated_torque import RatedTorqueParams
        with pytest.raises(ValueError, match="psi_f"):
            RatedTorqueParams(n_p=2, psi_f=-0.1, I_rated=10.0)

    def test_field_solvers_reject_psi_f_zero(self):
        # SynRM has no PM field — the psi_f -> B_rem derivation must not
        # produce B_rem = 0 and fail later with a confusing message
        motor = make_motor(psi_f=0.0)
        with pytest.raises(ValueError, match="reluctance machine"):
            prepare_analytical(motor)
        with pytest.raises(ValueError, match="reluctance machine"):
            prepare_fem(motor)


class TestDerate:
    """One temperature rule for every circuit model: the
    same motor thermal fields give the same psi_f / R_s to rated_torque,
    stall_torque, torque_speed, thermal_duty, drive_sim and
    operating_point, and the same range guard fires in all of them."""

    HOT = dict(alpha_Br=-0.0012, magnet_temp=100.0, winding_temp_limit=120.0,
               I_rated=10.0)
    PSI_FACTOR = 1 - 0.0012 * 80.0
    R_FACTOR = 1 + 0.00393 * 100.0

    def test_every_factory_sees_the_same_temperatures(self):
        from phasesweep.models.operating_point import operating_point, prepare_operating_point
        from phasesweep.models.rated_torque import prepare_rated_torque, prepare_stall_torque
        from phasesweep.models.thermal_duty import prepare_thermal_duty
        from phasesweep.models.torque_speed import prepare_torque_speed

        m = make_motor(**self.HOT)
        psi_f, R_s = 0.1 * self.PSI_FACTOR, 0.2 * self.R_FACTOR
        assert prepare_rated_torque(m).psi_f == pytest.approx(psi_f)
        st = prepare_stall_torque(m)
        assert (st.psi_f, st.I_stall_em) == (
            pytest.approx(psi_f), pytest.approx(540.0 / (3**0.5 * R_s)))
        ts = prepare_torque_speed(m)
        assert (ts.psi_f, ts.R_s) == (pytest.approx(psi_f), pytest.approx(R_s))
        td = prepare_thermal_duty(m, ((1.0, 1.0),))
        assert (td.psi_f, td.R_s) == (pytest.approx(psi_f), pytest.approx(R_s))
        ds = prepare_drive_sim(m)
        assert (ds.psi_f, ds.R_s) == (pytest.approx(psi_f), pytest.approx(R_s))
        op = prepare_operating_point(m)
        assert (op.psi_f, op.R_s) == (0.1, 0.2)  # cold inputs, derated at solve
        r = operating_point(op.n_p, op.psi_f, op.R_s, op.L_d, op.L_q, op.I_max,
                            op.U_max, op.U_DC, 10.0, 0.5, flux_map=op.flux_map,
                            winding_temp=op.winding_temp,
                            magnet_temp=op.magnet_temp, alpha_Br=op.alpha_Br)
        assert (r.psi_f_op, r.R_s_op) == (pytest.approx(psi_f), pytest.approx(R_s))

    def test_cold_motor_is_untouched(self):
        from phasesweep.models.rated_torque import prepare_stall_torque
        from phasesweep.models.torque_speed import prepare_torque_speed

        m = make_motor(alpha_Br=-0.0012)  # material data alone derates nothing
        assert prepare_torque_speed(m).R_s == 0.2
        assert prepare_torque_speed(m).psi_f == 0.1
        assert prepare_stall_torque(m).I_stall_em == pytest.approx(540.0 / (3**0.5 * 0.2))
        assert prepare_drive_sim(m) == prepare_drive_sim(make_motor())

    @pytest.mark.parametrize("model", [
        "rated_torque", "stall_torque", "torque_speed", "thermal_duty",
        "drive_sim", "demag_screen",
    ])
    def test_range_guard_fires_in_every_model(self, model):
        """A Motor-accepted (alpha_Br, magnet_temp) pair that derates psi_f
        below zero is refused by every factory, not only operating_point."""
        from phasesweep.machines.configs import load_motor
        from phasesweep.models.demag_screen import prepare_demag_screen
        from phasesweep.models.rated_torque import prepare_rated_torque, prepare_stall_torque
        from phasesweep.models.thermal_duty import prepare_thermal_duty
        from phasesweep.models.torque_speed import prepare_torque_speed
        from tests.conftest import REPO_ROOT

        extreme = dict(alpha_Br=-0.009, magnet_temp=140.0)  # factor -0.08
        factories = {
            "rated_torque": lambda m: prepare_rated_torque(m),
            "stall_torque": lambda m: prepare_stall_torque(m),
            "torque_speed": lambda m: prepare_torque_speed(m),
            "thermal_duty": lambda m: prepare_thermal_duty(m, ((1.0, 1.0),)),
            "drive_sim": lambda m: prepare_drive_sim(m),
            "demag_screen": lambda m: prepare_demag_screen(m),
        }
        if model == "demag_screen":
            base = load_motor(REPO_ROOT / "motors/creator_case_pmsm.toml")
            motor = dataclasses.replace(base, alpha_B_knee=-0.001, **extreme)
        else:
            motor = make_motor(I_rated=10.0, **extreme)
        with pytest.raises(ValueError, match="outside the linear alpha_Br range"):
            factories[model](motor)

    def test_derate_guards_inputs(self):
        from phasesweep.solver_params import derate

        assert derate(None, 0.1, None, None, None) == (None, 0.1)
        assert derate(0.2, 0.1, 120.0, None, -0.0012) == (
            pytest.approx(0.2 * self.R_FACTOR), 0.1)
        with pytest.raises(ValueError, match="alpha_Br is not"):
            derate(0.2, 0.1, None, 100.0, None)
        with pytest.raises(ValueError, match="must be finite"):
            derate(0.2, 0.1, float("nan"), None, None)
        with pytest.raises(ValueError, match="below zero"):
            derate(0.2, 0.1, -300.0, None, None)


def test_salient_pair_rejects_one_sided_inductance():
    from phasesweep.solver_params import salient_pair
    assert salient_pair(make_motor(L_d=None, L_q=None)) == (None, None)
    with pytest.raises(ValueError, match="set together"):
        salient_pair(make_motor(L_d=None))
    with pytest.raises(ValueError, match="set together"):
        salient_pair(make_motor(L_q=None))
