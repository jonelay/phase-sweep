"""im_fem_harmonic / im_fem_locked_rotor: full-machine
time-harmonic induction FEM on the cage mesh with a per-slot winding.

Winding algebra and rejections run without ngsolve; the field gates
need the fem extra and use the 8-bar / 12-slot test cage (q = 1, full
pitch). The CREATOR comparison lives in
tests/crossval/test_creator_im_crossval.py.
"""

from dataclasses import replace as dc_replace
from math import nan, pi, sin

import numpy as np
import pytest

from phasesweep.machines.geometry import inrunner
from phasesweep.machines.motor import InductionParams, Motor
from phasesweep.models.im_circuit import (
    identify_from_tests,
    identify_locked_rotor,
)
from phasesweep.models.im_fem_harmonic import (
    balanced_currents,
    layout_winding_factor,
    prepare_im_fem_harmonic,
    winding_matrix,
)
from phasesweep.registry import MODEL_REGISTRY
from phasesweep.sweep_types import RunConfig, compute_run_id
from tests.conftest import make_motor, requires_fem

_CAGE_KW = dict(r_outer=0.05, r_stator=0.032, r_magnet=0.031, r_rotor=0.031,
                r_inner=0.011, n_bars=8, bar_width=0.0034, bar_height=0.01,
                n_slots=12, slot_depth=0.008,
                end_ring_axial_length=0.008, end_ring_radial_height=0.008)
_SIG = 3e7


def make_cage_motor(geo_overrides=None, **overrides) -> Motor:
    geo = inrunner(**{**_CAGE_KW, **(geo_overrides or {})})
    ind = dict(sigma_bar=_SIG, f_rated=50.0, V_LL_rated=100.0)
    ind.update(overrides.pop("induction", {}))
    fields = dict(name="test cage", geometry=geo, n_p=2, N=10, coils_series=4,
                  k_w=1.0, R_s=0.5, L_stk=0.1, induction=InductionParams(**ind))
    fields.update(overrides)
    return Motor(**fields)


def _k_dp(q: int, pitch: int, pole_pitch: int) -> float:
    """Textbook distribution × pitch factor for a 60° belt."""
    gamma = pi / (3 * q)
    k_d = sin(q * gamma / 2) / (q * sin(gamma / 2))
    return k_d * sin(pitch / pole_pitch * pi / 2)


# ---------------------------------------------------------------------------
# Winding algebra (no FEM)
# ---------------------------------------------------------------------------

class TestWinding:

    @pytest.mark.parametrize("n_slots, n_p, pitch", [
        (36, 2, 9), (36, 2, 7), (36, 2, 8), (12, 2, 3), (24, 1, 10), (48, 4, 5),
    ])
    def test_layout_factor_is_kd_times_kp(self, n_slots, n_p, pitch):
        C = winding_matrix(n_slots, n_p, pitch)
        q = n_slots // (6 * n_p)
        assert layout_winding_factor(C, n_p) == pytest.approx(
            _k_dp(q, pitch, n_slots // (2 * n_p)), rel=1e-9)

    def test_creator_layout(self):
        """36 slots / 4 poles / 7-of-9 pitch: k_w = 0.9598 × 0.9397 = 0.9019."""
        C = winding_matrix(36, 2, 7)
        assert layout_winding_factor(C, 2) == pytest.approx(0.90191, abs=1e-4)
        assert C.shape == (3, 36)
        assert np.abs(C).sum(axis=1).tolist() == [24, 24, 24]   # 2 sides × 12 coils
        assert C.sum(axis=1).tolist() == [0, 0, 0]              # every coil closes
        assert np.abs(C).sum(axis=0).max() == 2                 # two layers per slot

    def test_three_phase_symmetry(self):
        """Phases B and C are phase A rotated by 120° / 240° electrical:
        n_slots / (3 n_p) slots and twice that."""
        C = winding_matrix(36, 2, 7)
        shift = 36 // (3 * 2)
        assert np.array_equal(np.roll(C[0], shift), C[1])
        assert np.array_equal(np.roll(C[0], 2 * shift), C[2])

    def test_positive_sequence(self):
        """The fundamental MMF of the balanced set rotates toward +θ."""
        C = winding_matrix(12, 2, 3)
        I_ph = balanced_currents(1.0)
        theta = 2 * pi * np.arange(12) / 12
        mmf = C.T @ I_ph                                    # per-slot ampere-turns
        # a wave cos(ωt − n_p θ) has slot phasors ∝ e^{−j n_p θ_k}
        forward = np.sum(mmf * np.exp(+2j * theta))
        backward = np.sum(mmf * np.exp(-2j * theta))
        assert abs(forward) > 1e3 * abs(backward)

    def test_rejections(self):
        with pytest.raises(ValueError, match="integral-slot"):
            winding_matrix(30, 2, 7)                     # q = 2.5
        with pytest.raises(ValueError, match="pole pitch"):
            winding_matrix(36, 2, 10)
        with pytest.raises(ValueError, match="pole pitch"):
            winding_matrix(36, 2, 0)

    def test_balanced_currents(self):
        I_ph = balanced_currents(2.0)
        assert abs(I_ph).tolist() == pytest.approx([2 * 2**0.5] * 3)
        assert abs(I_ph.sum()) < 1e-12


# ---------------------------------------------------------------------------
# Parameter validation (no FEM)
# ---------------------------------------------------------------------------

class TestPrepare:

    def test_full_pitch_default_and_fields(self):
        p = prepare_im_fem_harmonic(make_cage_motor())
        assert p.k_w_layout == pytest.approx(1.0)
        assert p.turns_per_side == pytest.approx(40 * 3 / 12)
        assert p.V_ph_rms == pytest.approx(100 / 3**0.5)
        assert p.ring is not None and p.leakage_split == 0.5

    def test_rejects_missing_ring_section(self):
        with pytest.raises(ValueError, match="end_ring_axial_length, end_ring_radial_height"):
            prepare_im_fem_harmonic(make_cage_motor(
                {"end_ring_axial_length": None, "end_ring_radial_height": None}))

    def test_design_letter_becomes_split(self):
        p = prepare_im_fem_harmonic(make_cage_motor(induction={"design_letter": "B"}))
        assert p.leakage_split == "B"

    @pytest.mark.parametrize("field", ["N", "coils_series"])
    def test_missing_turns_raise_named_error(self, field):
        """n_eff names the missing field; a bare assert used to fire first."""
        with pytest.raises(ValueError, match=field):
            prepare_im_fem_harmonic(make_cage_motor(**{field: None}))

    def test_rejects_wrong_k_w(self):
        with pytest.raises(ValueError, match="disagrees with the motor's k_w"):
            prepare_im_fem_harmonic(make_cage_motor(k_w=0.9))

    def test_short_pitch_changes_layout_factor(self):
        m = make_cage_motor(k_w=0.866, coil_pitch_slots=2)
        assert prepare_im_fem_harmonic(m).k_w_layout == pytest.approx(sin(pi / 3), rel=1e-6)

    @pytest.mark.parametrize("overrides, msg", [
        ({"R_s": None}, "R_s"),
        ({"k_w": None}, "k_w"),
        ({"L_stk": None}, "L_stk"),
        ({"induction": {"V_LL_rated": None}}, "V_LL_rated"),
        ({"induction": {"sigma_bar": None}}, "sigma_bar"),
    ])
    def test_missing_fields(self, overrides, msg):
        with pytest.raises(ValueError, match=msg):
            prepare_im_fem_harmonic(make_cage_motor(**overrides))

    def test_rejects_unslotted_stator(self):
        with pytest.raises(ValueError, match="stator slots"):
            prepare_im_fem_harmonic(make_cage_motor({"n_slots": 0, "slot_depth": 0.0}))

    def test_rejects_synchronous_motor(self):
        with pytest.raises(ValueError, match="requires an \\[induction\\]"):
            prepare_im_fem_harmonic(make_motor())
        for name in ("im_fem_harmonic", "im_fem_locked_rotor"):
            with pytest.raises(ValueError):
                MODEL_REGISTRY[name].validate(make_motor())

    def test_r_fe_threads_through(self):
        p = prepare_im_fem_harmonic(make_cage_motor(induction={"R_fe": 500.0}))
        assert p.R_fe == 500.0

    def test_r_fe_none_by_default(self):
        p = prepare_im_fem_harmonic(make_cage_motor())
        assert p.R_fe is None

    def test_r_fe_rejects_bad_values(self):
        import dataclasses as dc
        p = prepare_im_fem_harmonic(make_cage_motor())
        with pytest.raises(ValueError, match="R_fe"):
            dc.replace(p, R_fe=-1.0)
        with pytest.raises(ValueError, match="R_fe"):
            dc.replace(p, R_fe=0.0)
        with pytest.raises(ValueError, match="R_fe"):
            dc.replace(p, R_fe=nan)

    def test_params_reject_bad_matrix(self):
        p = prepare_im_fem_harmonic(make_cage_motor())
        import dataclasses as dc
        with pytest.raises(ValueError, match="winding matrix shape"):
            dc.replace(p, C=p.C[:, :6])

    def test_registry_metadata(self):
        for name in ("im_fem_harmonic", "im_fem_locked_rotor"):
            info = MODEL_REGISTRY[name]
            assert info.cost == "slow"
            assert "maxh_fraction" in info.hash_fields
            assert {"induction", "geometry", "R_s", "k_w"} <= info.needs
        assert "slip" in MODEL_REGISTRY["im_fem_harmonic"].needs_runtime

    def test_run_id_tracks_slip_and_mesh(self):
        m = make_cage_motor()
        a = compute_run_id(RunConfig(motor=m, model="im_fem_harmonic", slip=0.05))
        b = compute_run_id(RunConfig(motor=m, model="im_fem_harmonic", slip=0.1))
        c = compute_run_id(RunConfig(motor=m, model="im_fem_harmonic", slip=0.05,
                                     maxh_fraction=0.03))
        d = compute_run_id(RunConfig(motor=m, model="im_fem_harmonic", slip=0.05,
                                     n_theta=720))
        assert len({a, b, c}) == 3 and a == d

    def test_harmonic_needs_slip(self):
        from phasesweep.models.im_fem_harmonic import _require_slip
        with pytest.raises(ValueError, match="needs RunConfig.slip"):
            _require_slip(RunConfig(motor=make_cage_motor(), model="im_fem_harmonic"))


# ---------------------------------------------------------------------------
# Identification helper (no FEM)
# ---------------------------------------------------------------------------

class TestLockedRotorHelper:

    def test_matches_identify_from_tests(self):
        """The factored step reproduces the full identification's
        locked-rotor outputs exactly."""
        full = identify_from_tests(R_1=2.33, no_load=(230.0, 3.0, 300.0),
                                   locked_rotor=(40.0, 6.0, 500.0),
                                   f_no_load=50.0, f_locked_rotor=20.0,
                                   leakage_split="B")
        step = identify_locked_rotor(R_1=2.33, locked_rotor=(40.0, 6.0, 500.0),
                                     f_locked_rotor=20.0, leakage_split="B")
        assert step.R_2 == full.R_2
        assert step.L_sigma_s == full.L_sigma_s
        assert step.L_sigma_r == full.L_sigma_r
        assert step.pf_locked_rotor == full.pf_locked_rotor
        assert step.stator_leakage_fraction == 0.4
        assert step.R_lr == pytest.approx(500 / (3 * 36))
        assert step.X_lr == pytest.approx(((40 / 6) ** 2 - step.R_lr**2) ** 0.5)

    def test_rejections(self):
        with pytest.raises(ValueError, match="identify_locked_rotor: R_1"):
            identify_locked_rotor(R_1=0, locked_rotor=(40.0, 6.0, 500.0), f_locked_rotor=20.0)
        with pytest.raises(ValueError, match="does not exceed R_1"):
            identify_locked_rotor(R_1=5.0, locked_rotor=(40.0, 6.0, 500.0), f_locked_rotor=20.0)
        with pytest.raises(ValueError, match="unknown NEMA"):
            identify_locked_rotor(R_1=1.0, locked_rotor=(40.0, 6.0, 500.0),
                                  f_locked_rotor=20.0, leakage_split="E")
        with pytest.raises(ValueError, match="power factor >= 1"):
            identify_locked_rotor(R_1=1.0, locked_rotor=(40.0, 6.0, 800.0), f_locked_rotor=20.0)


# ---------------------------------------------------------------------------
# Field gates — 8-bar / 12-slot test cage
# ---------------------------------------------------------------------------

@requires_fem
class TestFieldGates:

    @pytest.fixture(scope="class")
    def setup(self):
        from phasesweep.solvers import fem_mesh
        from phasesweep.solvers.fem_mesh import get_or_build_cage_mesh

        fem_mesh.clear_mesh_cache()
        motor = make_cage_motor()
        p = prepare_im_fem_harmonic(motor)
        return motor, p, get_or_build_cage_mesh(p.geometry, 0.08)

    @pytest.fixture(scope="class")
    def locked(self, setup):
        from phasesweep.models.im_fem_harmonic import solve_operating_point
        _m, p, mesh = setup
        return solve_operating_point(mesh, p, 50.0, 1.0)

    def test_power_balance_at_standstill(self, locked):
        """No shaft power at s = 1: the field power equals bar + ring loss,
        and the Poynting flux through the gap returns it."""
        op = locked
        assert op.p_field_per_A2 == pytest.approx(op.p_bar_per_A2 + op.p_ring_per_A2, rel=1e-6)
        assert op.p_gap_per_A2 == pytest.approx(op.p_field_per_A2, rel=2e-2)
        assert op.p_ring_per_A2 > 0

    def test_impedance_is_the_power_balance(self, locked, setup):
        """3 Re(Z) = 3 R_1 + field power per A²rms, and the IEEE-112 step
        on a 1 A rms virtual point identifies R_2 = cage loss / 3."""
        _m, p, _mesh = setup
        Z = p.R_1 + locked.Z_int
        assert 3 * Z.real == pytest.approx(3 * p.R_1 + locked.p_field_per_A2, rel=1e-6)
        lr = identify_locked_rotor(R_1=p.R_1, locked_rotor=(abs(Z), 1.0, 3 * Z.real),
                                   f_locked_rotor=50.0)
        assert lr.R_2 == pytest.approx((locked.p_bar_per_A2 + locked.p_ring_per_A2) / 3, rel=1e-6)
        assert lr.L_sigma_s + lr.L_sigma_r == pytest.approx(Z.imag / (2 * pi * 50), rel=1e-9)

    def test_phase_asymmetry_reported(self, locked, setup):
        """12 slots are 3-fold symmetric; 8 bars are not, so at standstill
        the three phases see different bar positions and the per-phase
        impedances spread (~11% here, 6% on the 36/28 CREATOR machine at
        s = 1, 1.5% at rated slip). Real blocked-rotor tests show the
        same rotor-position dependence; the mean Z is what is reported,
        the spread beside it. It does not vanish at s = 0: the rotor
        slotting itself is n_bars-fold, not 3-fold, symmetric."""
        assert 0.02 < locked.phase_asym < 0.2
        from phasesweep.models.im_fem_harmonic import solve_operating_point
        _m, p, mesh = setup
        assert 0.0 < solve_operating_point(mesh, p, 50.0, 0.0).phase_asym < 0.2

    def test_torque_positive_and_near_power_torque(self, locked, setup):
        """Positive-sequence currents drive the rotor toward +θ. The
        Maxwell-stress torque and n_p·P_ag/ω differ by the single-slip
        harmonic share — bounded, not equated."""
        _m, p, _mesh = setup
        tau_power = p.n_p * locked.p_field_per_A2 / locked.omega
        assert locked.tau_per_A2 > 0
        assert 0.8 < locked.tau_per_A2 / tau_power < 1.2

    def test_synchronous_point_has_no_rotor_current(self, setup):
        from phasesweep.models.im_fem_harmonic import solve_operating_point
        _m, p, mesh = setup
        nl = solve_operating_point(mesh, p, 50.0, 0.0)
        assert nl.p_bar_per_A2 == pytest.approx(0.0, abs=1e-9)
        assert nl.I_bar_rms_max_per_A < 1e-9
        assert nl.p_field_per_A2 == pytest.approx(0.0, abs=1e-6)
        assert nl.Z_int.real == pytest.approx(0.0, abs=1e-9 * abs(nl.Z_int))

    def test_locked_rotor_leakage_below_no_load_inductance(self, setup, locked):
        """The rotor short-circuits the magnetizing path at s = 1."""
        from phasesweep.models.im_fem_harmonic import solve_operating_point
        _m, p, mesh = setup
        nl = solve_operating_point(mesh, p, 50.0, 0.0)
        assert locked.Z_int.imag < 0.5 * nl.Z_int.imag

    def test_deep_bar_rises_with_frequency(self, setup, locked):
        from phasesweep.models.im_fem_harmonic import solve_operating_point
        _m, p, mesh = setup
        lo = solve_operating_point(mesh, p, 5.0, 1.0)
        R_lo = (lo.p_bar_per_A2 + lo.p_ring_per_A2) / 3
        R_hi = (locked.p_bar_per_A2 + locked.p_ring_per_A2) / 3
        assert R_hi > R_lo * 1.02

    def test_ideal_ring_carries_no_ring_loss(self, setup):
        from phasesweep.models.im_fem_harmonic import solve_operating_point
        _m, p0, mesh = setup
        p = dc_replace(p0, ring=None)
        op = solve_operating_point(mesh, p, 50.0, 1.0)
        assert op.p_ring_per_A2 == 0.0
        assert op.p_field_per_A2 == pytest.approx(op.p_bar_per_A2, rel=1e-6)

    def test_runner_outputs(self, setup):
        motor, p, _mesh = setup
        rc = RunConfig(motor=motor, model="im_fem_harmonic", maxh_fraction=0.08, slip=1.0)
        out = MODEL_REGISTRY["im_fem_harmonic"].fn(rc)
        assert set(out) == MODEL_REGISTRY["im_fem_harmonic"].produces
        assert out["I_1_rms"] == pytest.approx(p.V_ph_rms / abs(complex(out["Z_phase_re"],
                                                                        out["Z_phase_im"])))
        assert out["p_in"] == pytest.approx(out["p_cu_s"] + out["p_ag"], rel=1e-6)
        assert out["p_mech"] == pytest.approx(0.0, abs=1e-6 * out["p_in"])
        assert out["ring_model"] == "segment" and out["f_rotor_Hz"] == 50.0
        grid = np.array(out["B_mag_grid"], dtype=float)
        assert grid.shape == (120, 120) and len(out["grid_coords_list"]) == 120
        inside = np.isfinite(grid)
        assert inside.sum() > 0.6 * grid.size and np.nanmax(grid) > 0

    def test_locked_rotor_runner_outputs(self, setup):
        motor, p, _mesh = setup
        rc = RunConfig(motor=motor, model="im_fem_locked_rotor", maxh_fraction=0.08)
        out = MODEL_REGISTRY["im_fem_locked_rotor"].fn(rc)
        assert set(out) == MODEL_REGISTRY["im_fem_locked_rotor"].produces
        assert out["f_locked_rotor_list"] == [50.0]
        assert out["leakage_split_used"] == 0.5 and out["R_1"] == p.R_1
        L_ew = out["L_ew_H"]
        assert L_ew > 0
        assert out["L_sigma_s_list"][0] == pytest.approx(out["L_sigma_r_list"][0] + L_ew)
        assert out["I_no_load_rms_linear_iron"] < out["V_phase_rms"] / abs(
            complex(out["Z_re_list"][0], out["Z_im_list"][0]))
        assert out["L_no_load_linear_iron"] > out["L_sigma_s_list"][0] + out["L_sigma_r_list"][0]

    def test_r_fe_none_gives_zero_p_fe(self, setup):
        """R_fe=None (default) produces p_fe=0 and unchanged eta."""
        from phasesweep.models.im_fem_harmonic import (
            operating_point_at_voltage,
            solve_operating_point,
        )
        _m, p, mesh = setup
        assert p.R_fe is None
        op = solve_operating_point(mesh, p, 50.0, 0.05)
        pt = operating_point_at_voltage(op, p, p.V_ph_rms)
        assert pt["p_fe"] == 0.0
        assert pt["R_fe_used"] is None
        assert pt["r_fe_over_x_m"] is None
        assert pt["eta_circuit"] == pytest.approx(
            pt["p_mech"] / pt["p_in"] if pt["p_in"] > 0 else 0.0)

    def test_r_fe_produces_core_loss(self, setup):
        """R_fe > 0 produces p_fe > 0 and a self-consistent power balance."""
        import dataclasses as dc

        from phasesweep.models.im_fem_harmonic import (
            operating_point_at_voltage,
            solve_operating_point,
        )
        _m, p, mesh = setup
        p_rfe = dc.replace(p, R_fe=200.0)
        op = solve_operating_point(mesh, p_rfe, 50.0, 0.05)
        pt_rfe = operating_point_at_voltage(op, p_rfe, p_rfe.V_ph_rms)
        pt_none = operating_point_at_voltage(op, p, p.V_ph_rms)
        assert pt_rfe["p_fe"] > 0
        assert pt_rfe["R_fe_used"] == 200.0
        assert pt_rfe["r_fe_over_x_m"] is not None and pt_rfe["r_fe_over_x_m"] > 0
        assert pt_rfe["eta_circuit"] < pt_none["eta_circuit"]
        assert pt_rfe["I_1_rms"] != pytest.approx(pt_none["I_1_rms"])
        # Power balance: p_in = p_cu_s + p_fe + p_ag
        assert pt_rfe["p_in"] == pytest.approx(
            pt_rfe["p_cu_s"] + pt_rfe["p_fe"] + pt_rfe["p_ag"], rel=1e-9)

    def test_terminal_impedance_adds_the_stator_branch(self, setup):
        """Operating points are 2D-only; the circuit owner adds R_1 + jωL_ew."""
        from phasesweep.models.im_fem_harmonic import (
            operating_point_at_voltage,
            solve_operating_point,
        )
        _m, p, mesh = setup
        assert p.L_ew > 0
        op = solve_operating_point(mesh, p, 50.0, 0.05)
        pt = operating_point_at_voltage(op, p, p.V_ph_rms)
        Z_t = complex(pt["Z_phase_re"], pt["Z_phase_im"])
        assert Z_t - op.Z_int == pytest.approx(complex(p.R_1, op.omega * p.L_ew), rel=1e-12)
        assert pt["I_1_rms"] == pytest.approx(p.V_ph_rms / abs(Z_t), rel=1e-12)

    def test_no_load_sweep_l_ew_is_terminal_not_magnetizing(self, setup):
        """Reported no-load L carries L_ew; identified L_m does not (the
        identification point drops the L_ew voltage, so linear L_m is the
        same with and without L_ew)."""
        from phasesweep.models.im_fem_harmonic import (
            no_load_sweep,
            solve_operating_point,
        )
        _m, p, mesh = setup
        lr = solve_operating_point(mesh, p, p.f_rated, 1.0)
        Z_lr = p.R_1 + lr.Z_int
        lr_point = (abs(Z_lr), 1.0, 3 * Z_lr.real)
        nl = solve_operating_point(mesh, p, p.f_rated, 0.0)
        with_ew = no_load_sweep([mesh], p, lr_point, nonlinear=False)
        no_ew = no_load_sweep([mesh], dc_replace(p, L_ew=0.0), lr_point, nonlinear=False)
        L_2d = nl.Z_int.imag / nl.omega
        assert with_ew["L_no_load_list"][0] == pytest.approx(L_2d + p.L_ew, rel=1e-9)
        assert no_ew["L_no_load_list"][0] == pytest.approx(L_2d, rel=1e-9)
        assert with_ew["L_m_list"] == pytest.approx(no_ew["L_m_list"], rel=1e-9)
        assert with_ew["I_no_load_rms_list"][0] < no_ew["I_no_load_rms_list"][0]


@requires_fem
class TestSaturation:
    """Energy-equivalent effective-reluctivity fixed point (F5) on the
    8-bar test cage. Rated voltage of the test cage overdrives the iron
    (5 T linear), so the checks run at a fraction of it."""

    @pytest.fixture(scope="class")
    def setup(self):
        from phasesweep.solvers import fem_mesh
        from phasesweep.solvers.fem_mesh import get_or_build_cage_mesh

        fem_mesh.clear_mesh_cache()
        motor = make_cage_motor(mu_r_fe=2000.0)
        p = prepare_im_fem_harmonic(motor)
        return motor, p, get_or_build_cage_mesh(p.geometry, 0.08)

    def test_linear_curve_reproduces_linear_solve(self, setup):
        """A straight-line B-H table at mu_r_fe must converge in one
        step to the linear solve, whatever the voltage."""
        import dataclasses as dc

        from phasesweep.models.im_fem_harmonic import (
            solve_operating_point,
            solve_saturated_operating_point,
        )
        from phasesweep.solvers.fem_materials import MU0, BHCurve

        _m, p, mesh = setup
        nu = 1 / (MU0 * p.mu_r_fe)
        p_lin = dc.replace(p, bh=BHCurve(B=np.array([0.0, 1.0, 100.0]),
                                         H=np.array([0.0, 1.0, 100.0]) * nu))
        lin = solve_operating_point(mesh, p, 50.0, 0.3)
        op, st = solve_saturated_operating_point(mesh, p_lin, 50.0, 0.3, p.V_ph_rms)
        assert st.iterations == 1 and st.residual < 1e-9
        assert op.Z_int == pytest.approx(lin.Z_int, rel=1e-9)
        assert op.tau_per_A2 == pytest.approx(lin.tau_per_A2, rel=1e-9)
        assert st.I_rms == pytest.approx(
            p.V_ph_rms / abs(p.R_1 + 2j * pi * 50.0 * p.L_ew + lin.Z_int), rel=1e-9)
        assert st.B_peak_iron_max > 1.0

    def test_saturation_raises_current_and_warm_start(self, setup):
        """Generic curve on the gap-dominated test cage: at low voltage
        the current follows the curve's initial slope (above mu_r_fe =
        2000 → less current than linear); at 2.4 T peak the current runs
        ahead of the voltage, the reactance drops, the peak iron field
        stays below the linear extrapolation, and a warm start from the
        converged state returns it in a few iterations."""
        from phasesweep.models.im_fem_harmonic import (
            circuit_currents,
            iron_peak_B,
            solve_operating_point,
            solve_saturated_operating_point,
        )

        _m, p, mesh = setup
        lin = solve_operating_point(mesh, p, 50.0, 0.0)
        V_lo, V_hi = 0.1 * p.V_ph_rms, 0.5 * p.V_ph_rms
        I_lin_lo, I_lin_hi = (abs(circuit_currents(lin, p, V)["I_int"]) for V in (V_lo, V_hi))
        op_lo, st_lo = solve_saturated_operating_point(mesh, p, 50.0, 0.0, V_lo)
        op_hi, st_hi = solve_saturated_operating_point(mesh, p, 50.0, 0.0, V_hi)
        assert st_lo.I_rms < I_lin_lo
        assert st_lo.residual < 1e-2 and st_hi.residual < 1e-2
        assert st_hi.I_rms / st_lo.I_rms > 1.005 * V_hi / V_lo            # 1.009
        assert st_hi.B_peak_iron_max < iron_peak_B(lin, I_lin_hi)
        assert st_hi.B_peak_iron_max > 2.0                                # 2.40
        assert op_hi.Z_int.imag < 0.995 * op_lo.Z_int.imag           # 0.984
        _op_w, st_w = solve_saturated_operating_point(mesh, p, 50.0, 0.0, V_hi, state=st_hi)
        assert st_w.iterations <= 5
        assert st_w.I_rms == pytest.approx(st_hi.I_rms, rel=2e-3)

    @pytest.mark.parametrize("R_fe", [None, 200.0])
    def test_saturates_on_the_circuit_owners_branch_current(self, setup, R_fe):
        """The converged field is solved at exactly the FEM-branch current
        the circuit owner sends at that voltage (R_1, L_ew and R_fe all
        in), and that is I_int, not the terminal current."""
        import dataclasses as dc

        from phasesweep.models.im_fem_harmonic import (
            operating_point_at_voltage,
            solve_saturated_operating_point,
        )

        _m, p0, mesh = setup
        p = dc.replace(p0, R_fe=R_fe)
        V = 0.3 * p.V_ph_rms
        op, st = solve_saturated_operating_point(mesh, p, 50.0, 0.0, V)
        pt = operating_point_at_voltage(op, p, V)
        assert st.I_rms == pytest.approx(pt["I_int_rms"], rel=1e-9)
        if R_fe is not None:
            assert pt["I_int_rms"] < pt["I_1_rms"]

    def test_non_convergence_raises(self, setup):
        from phasesweep.models.im_fem_harmonic import solve_saturated_operating_point

        _m, p, mesh = setup
        with pytest.raises(RuntimeError, match="did not converge"):
            solve_saturated_operating_point(mesh, p, 50.0, 0.0, 0.35 * p.V_ph_rms, max_iter=2)

    def test_runner_nonlinear_outputs(self, setup):
        motor, p, _mesh = setup
        motor = make_cage_motor(mu_r_fe=2000.0, induction=dict(V_LL_rated=20.0))
        rc = RunConfig(motor=motor, model="im_fem_harmonic", maxh_fraction=0.08,
                       slip=1.0, nonlinear=True)
        out = MODEL_REGISTRY["im_fem_harmonic"].fn(rc)
        assert set(out) == MODEL_REGISTRY["im_fem_harmonic"].produces
        assert out["iron_model"] == "bh" and out["bh_iterations"] > 1
        assert 0 < out["B_peak_iron_max"] < 2.5
        assert out["I_1_rms"] == pytest.approx(out["V_phase_rms"] / abs(complex(
            out["Z_phase_re"], out["Z_phase_im"])))
        assert out["p_in"] == pytest.approx(out["p_cu_s"] + out["p_ag"], rel=1e-6)
        lin = MODEL_REGISTRY["im_fem_harmonic"].fn(dc_replace(rc, nonlinear=False))
        assert lin["iron_model"] == "linear" and lin["bh_iterations"] == 0
        assert lin["B_peak_iron_max"] > 0
        assert compute_run_id(rc) != compute_run_id(dc_replace(rc, nonlinear=False))
        assert "nonlinear" in MODEL_REGISTRY["im_fem_harmonic"].hash_fields
        assert "nonlinear" in MODEL_REGISTRY["im_fem_locked_rotor"].hash_fields

    @pytest.mark.slow
    def test_locked_rotor_runner_no_load_sweep(self, setup):
        """The no-load sweep identifies L_m through identify_from_tests:
        flat under linear iron; with the curve it rises toward μ_max and
        then falls once the teeth saturate (a few % on this gap-dominated
        cage, whose top point sits at ~2.6 T)."""
        motor = make_cage_motor(mu_r_fe=2000.0, induction=dict(V_LL_rated=50.0))
        rc = RunConfig(motor=motor, model="im_fem_locked_rotor", maxh_fraction=0.08)
        lin = MODEL_REGISTRY["im_fem_locked_rotor"].fn(rc)
        assert set(lin) == MODEL_REGISTRY["im_fem_locked_rotor"].produces
        assert lin["iron_model"] == "linear"
        assert lin["V_no_load_list"] == pytest.approx(
            [f * lin["V_phase_rms"] for f in (0.3, 0.5, 0.7, 0.85, 1.0, 1.1)])
        assert lin["L_m_list"] == pytest.approx([lin["L_m_list"][0]] * 6, rel=1e-9)
        assert lin["L_no_load_list"][0] == pytest.approx(lin["L_no_load_linear_iron"])
        assert lin["L_m_list"][0] == pytest.approx(
            lin["L_no_load_linear_iron"] - lin["L_sigma_s_list"][0], rel=1e-9)  # L_ew in both
        assert lin["bh_iterations_list"] == [0.0] * 6
        nl = MODEL_REGISTRY["im_fem_locked_rotor"].fn(dc_replace(rc, nonlinear=True))
        assert set(nl) == set(lin) and nl["iron_model"] == "bh"
        assert nl["R_2_list"] == lin["R_2_list"]                 # locked rotor stays linear
        assert all(n >= 1 for n in nl["bh_iterations_list"])
        L = nl["L_m_list"]
        assert L[-1] < 0.985 * max(L) and L[-1] < L[-2] < L[-3]
        assert nl["I_no_load_rms_list"][-1] / nl["I_no_load_rms_list"][0] > 1.005 * 1.1 / 0.3
        assert nl["B_peak_iron_no_load_list"][-1] > 2.0 > nl["B_peak_iron_no_load_list"][0]


class TestGuards:
    """Input validation."""

    @pytest.mark.parametrize("m", [1, 2, 5, 6])
    def test_non_three_phase_winding_rejected(self, m):
        """The belt sequence is the three-phase 60 degree one, so any other m
        used to return a matrix whose extra rows were all zero."""
        with pytest.raises(ValueError, match="three-phase"):
            winding_matrix(36, 2, 9, m=m)

    def test_three_phase_still_accepted(self):
        assert winding_matrix(36, 2, 9, m=3).shape == (3, 36)

    @pytest.mark.parametrize("field", ["sigma_bar", "mu_r_fe", "L_stk", "k_w",
                                       "R_1", "f_rated", "V_ph_rms"])
    def test_nan_rejected(self, field):
        """NaN fails every comparison, so a bare `value <= 0` lets it past.
        ImCircuitParams already used isfinite; these did not."""
        p = prepare_im_fem_harmonic(make_cage_motor())
        with pytest.raises(ValueError, match="finite"):
            dc_replace(p, **{field: nan})

    def test_missing_ring_height_raises(self):
        """The ideal ring is the superconducting limit: torque stays right but
        the ring's own I^2R would leave the loss split, so the factory refuses."""
        m = make_cage_motor(geo_overrides=dict(end_ring_radial_height=None))
        with pytest.raises(ValueError, match="needs \\[geometry\\] end_ring_radial_height"):
            prepare_im_fem_harmonic(m)


# ---------------------------------------------------------------------------
# Multi-slice skew
# ---------------------------------------------------------------------------

class TestSkewOffsets:
    """Slice placement and guards (no FEM)."""

    def test_unskewed_is_one_plain_slice(self):
        from phasesweep.models.im_fem_harmonic import skew_offsets
        assert skew_offsets(None) == (0.0,)

    def test_midpoint_rule_centered(self):
        from phasesweep.models.im_fem_harmonic import skew_offsets
        offs = skew_offsets(0.5, 5)
        assert len(offs) == 5
        assert sum(offs) == pytest.approx(0.0, abs=1e-15)
        assert offs[2] == pytest.approx(0.0, abs=1e-15)   # odd N: center slice
        assert offs == tuple(sorted(offs))
        assert offs[0] == pytest.approx(-0.5 * (0.5 - 0.1))
        assert offs[-1] == pytest.approx(+0.5 * (0.5 - 0.1))

    def test_offset_placement_has_sinc_property(self):
        """The midpoint-rule offsets satisfy mean(e^{j n_p δ}) =
        sin(x)/(N sin(x/N)) → sin(x)/x — a property of the placement
        formula, NOT a claim that the decoupled solver produces the
        physical sinc skew factor (it does not; see module docstring)."""
        from phasesweep.models.im_fem_harmonic import skew_offsets
        n_p, skew = 2, 2 * pi / 12                    # one slot pitch
        x = n_p * skew / 2
        analytic = sin(x) / x
        k5 = abs(np.mean(np.exp(1j * n_p * np.array(skew_offsets(skew, 5)))))
        k21 = abs(np.mean(np.exp(1j * n_p * np.array(skew_offsets(skew, 21)))))
        assert analytic < k21 < k5
        assert k5 == pytest.approx(analytic, rel=5e-3)

    def test_too_few_slices_raise(self):
        from phasesweep.models.im_fem_harmonic import skew_offsets
        with pytest.raises(ValueError, match=">= 2 slices"):
            skew_offsets(0.5, 1)

    def test_nonlinear_with_skew_rejected_before_any_solve(self):
        from phasesweep.models.im_fem_harmonic import (
            run_im_fem_harmonic,
            run_im_fem_locked_rotor,
        )
        m = make_cage_motor(geo_overrides={"skew_angle": 2 * pi / 12})
        with pytest.raises(ValueError, match="not implemented"):
            run_im_fem_harmonic(RunConfig(motor=m, model="im_fem_harmonic",
                                          slip=1.0, nonlinear=True))
        with pytest.raises(ValueError, match="not implemented"):
            run_im_fem_locked_rotor(RunConfig(motor=m, model="im_fem_locked_rotor",
                                              nonlinear=True))


@requires_fem
class TestSkewFieldGates:
    """Decoupled multi-slice solve on the 8-bar / 12-slot test cage,
    skewed by one stator slot pitch (2π/12 mech). Tests rotor-position
    averaging of the bar/slot interaction, not the sinc coupling."""

    _SKEW = 2 * pi / 12

    @pytest.fixture(scope="class")
    def setup(self):
        from phasesweep.models.im_fem_harmonic import _slice_meshes
        from phasesweep.solvers import fem_mesh

        fem_mesh.clear_mesh_cache()
        plain = prepare_im_fem_harmonic(make_cage_motor())
        skewed = prepare_im_fem_harmonic(
            make_cage_motor(geo_overrides={"skew_angle": self._SKEW}))
        return plain, skewed, _slice_meshes(plain, 0.08), _slice_meshes(skewed, 0.08)

    def test_slice_meshes(self, setup):
        """Unskewed = one mesh; skewed = _N_SKEW_SLICES distinct meshes
        whose bar0 sits at the slice offset."""
        from math import atan2

        from ngsolve import Integrate, x, y

        from phasesweep.models.im_fem_harmonic import _N_SKEW_SLICES, skew_offsets
        _plain, skewed, m_plain, m_skew = setup
        assert len(m_plain) == 1 and len(m_skew) == _N_SKEW_SLICES
        offsets = skew_offsets(skewed.geometry.skew_angle)
        for mesh, off in zip(m_skew, offsets):
            area = Integrate(1, mesh, definedon=mesh.Materials("bar0"))
            cx = Integrate(x, mesh, definedon=mesh.Materials("bar0")) / area
            cy = Integrate(y, mesh, definedon=mesh.Materials("bar0")) / area
            assert atan2(cy, cx) == pytest.approx(off, abs=1e-3)

    def test_single_mesh_multislice_is_the_plain_solve(self, setup):
        from phasesweep.models.im_fem_harmonic import (
            solve_operating_point,
            solve_operating_point_multislice,
        )
        plain, _skewed, m_plain, _m_skew = setup
        a = solve_operating_point(m_plain[0], plain, 50.0, 1.0)
        b = solve_operating_point_multislice(m_plain, plain, 50.0, 1.0)
        assert b.Z_int == a.Z_int and b.tau_per_A2 == a.tau_per_A2
        assert b.p_bar_per_A2 == a.p_bar_per_A2

    def test_skew_reduces_torque(self, setup):
        """One slot pitch of skew reduces the locked-rotor torque per A²
        through rotor-position averaging of the discrete bar/slot
        interaction (NOT the classical sinc coupling — the decoupled
        solve does not enforce axial bar-current continuity)."""
        from phasesweep.models.im_fem_harmonic import (
            solve_operating_point,
            solve_operating_point_multislice,
        )
        plain, skewed, m_plain, m_skew = setup
        a = solve_operating_point(m_plain[0], plain, 50.0, 1.0)
        b = solve_operating_point_multislice(m_skew, skewed, 50.0, 1.0)
        assert 0.75 < b.tau_per_A2 / a.tau_per_A2 < 0.99
        assert abs(b.Z_int) == pytest.approx(abs(a.Z_int), rel=0.2)
        assert b.p_field_per_A2 == pytest.approx(
            b.p_bar_per_A2 + b.p_ring_per_A2, rel=1e-6)   # balance per slice survives

    def test_runner_outputs_skewed(self, setup):
        motor = make_cage_motor(geo_overrides={"skew_angle": self._SKEW})
        rc = RunConfig(motor=motor, model="im_fem_harmonic", maxh_fraction=0.08, slip=1.0)
        out = MODEL_REGISTRY["im_fem_harmonic"].fn(rc)
        assert set(out) == MODEL_REGISTRY["im_fem_harmonic"].produces
        assert out["skew_model"] == "multi-slice" and out["n_skew_slices"] == 5
        plain = MODEL_REGISTRY["im_fem_harmonic"].fn(
            RunConfig(motor=make_cage_motor(), model="im_fem_harmonic",
                      maxh_fraction=0.08, slip=1.0))
        assert plain["skew_model"] == "none" and plain["n_skew_slices"] == 0
        assert out["tau"] / out["I_1_rms"]**2 < plain["tau"] / plain["I_1_rms"]**2
        assert compute_run_id(rc) != compute_run_id(
            RunConfig(motor=make_cage_motor(), model="im_fem_harmonic",
                      maxh_fraction=0.08, slip=1.0))
