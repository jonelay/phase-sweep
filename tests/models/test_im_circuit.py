"""Induction-machine equivalent-circuit tier.

Unit tests for InductionParams / the [induction] TOML section / the
topology gates, and physics tests for solve_im_circuit /
thevenin_breakdown / the im_equivalent_circuit + im_torque_slip registry
models. The CREATOR IM anchor gates live in
tests/crossval/test_creator_im_crossval.py.
"""

import dataclasses as dc

import pytest

from phasesweep.machines.configs import motor_from_dict
from phasesweep.machines.motor import InductionParams, Motor
from phasesweep.machines.perturbation import perturb_motor
from phasesweep.models.im_circuit import (
    ImCircuitParams,
    prepare_im_circuit,
    run_im_torque_slip,
    solve_im_circuit,
    thevenin_breakdown,
)
from phasesweep.registry import MODEL_REGISTRY
from phasesweep.sweep_types import RunConfig, compute_run_id
from tests.conftest import make_motor


def make_im_params(**overrides) -> ImCircuitParams:
    """CREATOR-IM-shaped constant-parameter circuit for unit tests."""
    defaults = dict(
        n_p=2, R_1=2.33, R_2=1.012, L_sigma_s=0.0114, L_sigma_r=0.0106,
        L_m=0.226, V_LL=400.0, f_supply=50.0,
    )
    defaults.update(overrides)
    return ImCircuitParams(**defaults)


def make_im_motor(**induction_overrides) -> Motor:
    defaults = dict(
        R_2=1.012, L_m=0.226, L_sigma_s=0.0114, L_sigma_r=0.0106,
        f_rated=50.0, s_rated=0.047, V_LL_rated=400.0,
    )
    defaults.update(induction_overrides)
    return Motor(name="test im", geometry=None, n_p=2, R_s=2.33,
                 induction=InductionParams(**defaults))


# ---------------------------------------------------------------------------
# InductionParams / Motor validation
# ---------------------------------------------------------------------------

class TestInductionParams:

    @pytest.mark.parametrize("field", ["R_2", "L_m", "L_sigma_s",
                                       "L_sigma_r", "f_rated", "V_LL_rated"])
    def test_rejects_nonpositive(self, field):
        with pytest.raises(ValueError, match="must be > 0"):
            InductionParams(**{field: -1.0})

    def test_rejects_bad_s_rated(self):
        with pytest.raises(ValueError, match="s_rated"):
            InductionParams(s_rated=1.5)

    def test_rejects_bad_design_letter(self):
        with pytest.raises(ValueError, match="design_letter"):
            InductionParams(design_letter="E")

    def test_rejects_non_monotone_table(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            InductionParams(R_2_vs_f=((10, 1.0), (10, 1.1)))

    def test_rejects_zero_x_table_row(self):
        # x = 0 would silently shadow the scalar f = 0 anchor
        with pytest.raises(ValueError, match="must be > 0"):
            InductionParams(R_2_vs_f=((0.0, 1.0), (10, 1.1)))

    def test_n_bars_int_normalized(self):
        # 28.0 vs 28 must not split one machine into two config_ids
        a = make_im_motor(n_bars=28)
        b = make_im_motor(n_bars=28.0)
        assert b.induction.n_bars == 28
        assert a.config_id == b.config_id
        with pytest.raises(ValueError, match="integer"):
            InductionParams(n_bars=28.5)

    def test_rejects_single_row_table(self):
        with pytest.raises(ValueError, match=">= 2 rows"):
            InductionParams(R_2_vs_f=((10, 1.0),))

    def test_mutual_exclusion_with_pm_fields(self):
        with pytest.raises(ValueError, match="induction"):
            Motor(name="x", geometry=None, n_p=2, psi_f=0.1,
                  induction=InductionParams(R_2=1.0))

    def test_pm_motor_dict_has_no_induction_key(self):
        assert "induction" not in make_motor().to_dict()

    def test_config_id_appends_only(self):
        base = Motor(name="x", geometry=None, n_p=2, R_s=2.33)
        with_im = dc.replace(base, induction=InductionParams(R_2=1.0))
        assert base.config_id != with_im.config_id

    def test_config_id_hashes_tables(self):
        a = make_im_motor(R_2_vs_f=((10.0, 1.07), (50.0, 1.78)))
        b = make_im_motor(R_2_vs_f=((10.0, 1.07), (50.0, 1.79)))
        assert a.config_id != b.config_id

    def test_round_trip_preserves_config_id(self):
        # int-vs-float normalization: TOML delivers 10, JSON round-trips
        # 10.0 — the hash must not care
        m = make_im_motor(
            f_rated=50, V_LL_rated=400,
            R_2_vs_f=((10, 1.07), (50, 1.78)),
        )
        m2 = Motor.from_dict(m.to_dict())
        assert m2 == m
        assert m2.config_id == m.config_id

    def test_toml_induction_section_loads(self):
        raw = {
            "circuit": {"n_p": 2, "R_s": 2.33},
            "induction": {
                "R_2": 1.012, "L_m": 0.226, "L_sigma_s": 0.0114,
                "L_sigma_r": 0.0106, "f_rated": 50.0, "s_rated": 0.047,
                "V_LL_rated": 400.0,
                "R_2_vs_f": [[10, 1.07], [50, 1.78]],
            },
        }
        m = motor_from_dict(raw, label="test", default_name="im")
        assert m.induction is not None
        assert m.induction.R_2_vs_f == ((10.0, 1.07), (50.0, 1.78))

    def test_toml_bad_induction_names_section(self):
        raw = {"circuit": {"n_p": 2}, "induction": {"R_2": -1.0}}
        with pytest.raises(ValueError, match=r"\[induction\]"):
            motor_from_dict(raw, label="test", default_name="im")


# ---------------------------------------------------------------------------
# Symmetric topology gates
# ---------------------------------------------------------------------------

_PM_FACTORIES = {
    "prepare_analytical": "phasesweep.solvers.analytical",
    "prepare_analytical_axial": "phasesweep.solvers.analytical_axial",
    "prepare_fem_axial": "phasesweep.solvers.fem_axial",
    "prepare_fem": "phasesweep.solvers.fem_runner",
    "prepare_drive_sim": "phasesweep.simulation.params",
    "prepare_rated_torque": "phasesweep.models.rated_torque",
    "prepare_stall_torque": "phasesweep.models.rated_torque",
    "prepare_torque_speed": "phasesweep.models.torque_speed",
    "prepare_demag_screen": "phasesweep.models.demag_screen",
    "_check_thermal_duty_motor": "phasesweep.models.thermal_duty",
}
# prepare_iron_loss is deliberately absent: iron_loss runs on induction
# machines as a stator lump at the rated supply (test_iron_loss.py
# TestInductionPath) — the "eleven symmetric gates" are now ten.


class TestTopologyGates:

    @pytest.mark.parametrize("factory", _PM_FACTORIES)
    def test_pm_factories_reject_induction_motor(self, factory):
        import importlib
        mod = importlib.import_module(_PM_FACTORIES[factory])
        with pytest.raises(ValueError, match="induction"):
            getattr(mod, factory)(make_im_motor())

    def test_im_factory_rejects_pm_motor(self):
        with pytest.raises(ValueError, match="synchronous"):
            prepare_im_circuit(make_motor())

    def test_im_factory_names_missing_fields(self):
        m = Motor(name="x", geometry=None, n_p=2, R_s=2.33,
                  induction=InductionParams(R_2=1.0))
        with pytest.raises(ValueError, match="L_m"):
            prepare_im_circuit(m)

    def test_im_factory_needs_r_s(self):
        m = Motor(name="x", geometry=None, n_p=2,
                  induction=InductionParams(R_2=1.0))
        with pytest.raises(ValueError, match="R_s"):
            prepare_im_circuit(m)


# ---------------------------------------------------------------------------
# Circuit physics
# ---------------------------------------------------------------------------

class TestCircuitPhysics:

    def test_power_balance_exact(self):
        op = solve_im_circuit(make_im_params(), 0.05)
        assert op.p_in == pytest.approx(op.p_cu_s + op.p_ag, rel=1e-12)
        assert op.p_ag == pytest.approx(op.p_cu_r + op.p_mech, rel=1e-12)
        assert op.p_cu_r == pytest.approx(0.05 * op.p_ag, rel=1e-12)

    @pytest.mark.parametrize("slip", [0.0, -0.1, 1.5])
    def test_rejects_slip_outside_motoring_range(self, slip):
        with pytest.raises(ValueError, match="slip"):
            solve_im_circuit(make_im_params(), slip)

    def test_small_slip_torque_linear(self):
        # In the R_2/s-dominated region, tau is proportional to slip
        p = make_im_params()
        t1 = solve_im_circuit(p, 0.002).tau
        t2 = solve_im_circuit(p, 0.004).tau
        assert t2 / t1 == pytest.approx(2.0, rel=0.02)

    def test_tau_max_independent_of_r2(self):
        """The classic invariant: breakdown TORQUE does not move with
        rotor resistance; breakdown SLIP scales with it."""
        p = make_im_params()
        s0, t0 = thevenin_breakdown(p)
        for scale in (0.5, 2.0, 5.0):
            s, t = thevenin_breakdown(dc.replace(p, R_2=p.R_2 * scale))
            assert t == pytest.approx(t0, rel=1e-12)
            assert s == pytest.approx(s0 * scale, rel=1e-12)

    def test_swept_breakdown_matches_thevenin_closed_form(self):
        # Constant parameters: the numeric sweep must land on the exact
        # closed form (grid resolution limits the slip match)
        m = make_im_motor()
        out = run_im_torque_slip(RunConfig(motor=m, model="im_torque_slip"))
        assert out["tau_breakdown"] == pytest.approx(
            out["tau_breakdown_thevenin"], rel=1e-3)
        assert out["slip_breakdown"] == pytest.approx(
            out["slip_breakdown_thevenin"], rel=0.02)

    def test_two_pole_machine_synchronous_speed(self):
        # n_p = 1 is legal for the IM tier: synchronous mechanical speed
        # doubles and, all else equal, torque at a given slip halves
        m4 = make_im_motor()
        m2 = dc.replace(m4, n_p=1)
        out4 = run_im_torque_slip(RunConfig(motor=m4, model="im_torque_slip"))
        out2 = run_im_torque_slip(RunConfig(motor=m2, model="im_torque_slip"))
        assert out2["tau_breakdown_thevenin"] == pytest.approx(
            out4["tau_breakdown_thevenin"] / 2, rel=1e-12)
        assert out2["slip_breakdown_thevenin"] == pytest.approx(
            out4["slip_breakdown_thevenin"], rel=1e-12)

    def test_deep_bar_table_anchored_and_clamped(self):
        p = make_im_params(R_2_vs_f=((10.0, 2.0), (50.0, 4.0)))
        # f_r above the table clamps to the last row
        op_hi = solve_im_circuit(p, 1.0, f_supply=80.0)
        assert op_hi.R_2_used == pytest.approx(4.0)
        # f_r -> 0 anchors at the scalar (DC) value
        op_lo = solve_im_circuit(p, 1e-6)
        assert op_lo.R_2_used == pytest.approx(p.R_2, rel=1e-3)
        # in between: linear
        op_mid = solve_im_circuit(p, 0.6, f_supply=50.0)  # f_r = 30
        assert op_mid.R_2_used == pytest.approx(3.0)

    def test_lm_saturation_fixed_point(self):
        table = ((0.5, 0.26), (3.0, 0.20), (7.0, 0.13))
        p = make_im_params(L_m_vs_i=table)
        op = solve_im_circuit(p, 0.05)
        assert 0.13 <= op.L_m_used <= 0.26
        # magnetizing current and table row must be self-consistent
        from phasesweep.models.im_circuit import _interp_anchored
        assert op.L_m_used == pytest.approx(
            _interp_anchored(table, op.I_m_rms, table[0][1]), rel=1e-6)
        # higher voltage -> more saturation -> lower L_m
        op_hi = solve_im_circuit(p, 0.05, V_ph_rms=1.3 * op.V_phase_rms)
        assert op_hi.L_m_used < op.L_m_used


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------

class TestRegistry:

    def test_models_registered(self):
        for key in ("im_equivalent_circuit", "im_torque_slip"):
            info = MODEL_REGISTRY[key]
            assert info.cost == "fast"
            assert "induction" in info.needs
            info.validate(make_im_motor())

    def test_equivalent_circuit_requires_slip(self):
        rc = RunConfig(motor=make_im_motor(), model="im_equivalent_circuit")
        with pytest.raises(ValueError, match="slip"):
            MODEL_REGISTRY["im_equivalent_circuit"].fn(rc)

    def test_slip_hashes_into_run_id(self):
        m = make_im_motor()
        a = compute_run_id(RunConfig(motor=m, model="im_equivalent_circuit",
                                     slip=0.05))
        b = compute_run_id(RunConfig(motor=m, model="im_equivalent_circuit",
                                     slip=0.10))
        assert a != b

    def test_run_config_round_trips_slip(self):
        rc = RunConfig(motor=make_im_motor(), model="im_equivalent_circuit",
                       slip=0.05)
        rc2 = RunConfig.from_dict(rc.to_dict())
        assert rc2.slip == 0.05
        assert compute_run_id(rc2) == compute_run_id(rc)

    def test_torque_slip_outputs_match_produces(self):
        rc = RunConfig(motor=make_im_motor(), model="im_torque_slip")
        out = MODEL_REGISTRY["im_torque_slip"].fn(rc)
        assert set(out) == MODEL_REGISTRY["im_torque_slip"].produces

    def test_equivalent_circuit_outputs_match_produces(self):
        rc = RunConfig(motor=make_im_motor(), model="im_equivalent_circuit",
                       slip=0.05)
        out = MODEL_REGISTRY["im_equivalent_circuit"].fn(rc)
        assert set(out) == MODEL_REGISTRY["im_equivalent_circuit"].produces


# ---------------------------------------------------------------------------
# Perturbation
# ---------------------------------------------------------------------------

class TestPerturbation:

    def test_perturb_scales_scalar_and_table(self):
        m = make_im_motor(R_2_vs_f=((10.0, 2.0), (50.0, 4.0)))
        p = perturb_motor(m, "R_2", 0.05)
        assert p is not None
        assert p.induction.R_2 == pytest.approx(1.012 * 1.05)
        assert p.induction.R_2_vs_f[1][1] == pytest.approx(4.0 * 1.05)

    def test_perturb_im_param_on_pm_motor_returns_none(self):
        assert perturb_motor(make_motor(), "R_2", 0.05) is None

    def test_perturb_pm_param_on_im_motor_returns_none(self):
        assert perturb_motor(make_im_motor(), "psi_f", 0.05) is None


# ---------------------------------------------------------------------------
# Power-balance invariant
# ---------------------------------------------------------------------------

class TestPowerBalance:

    def test_stator_balance_across_slip(self):
        """Real input power splits exactly into stator copper loss plus
        airgap power — the magnetizing branch is lossless, so
        p_in = p_cu_s + p_ag is a circuit-theory identity that fails if
        any of the three independently-computed expressions drifts.
        (The rotor-side split p_ag = p_cu_r + p_mech is definitional —
        slip*p_ag — so only its consistency is asserted, not treated as
        an independent invariant.) Measured residual ~3e-16."""
        p = make_im_params()
        for slip in (0.001, 0.01, 0.047, 0.15, 0.5, 1.0):
            op = solve_im_circuit(p, slip=slip)
            assert op.p_in == pytest.approx(op.p_cu_s + op.p_ag, rel=1e-9)
            assert op.p_cu_r + op.p_mech == pytest.approx(op.p_ag, rel=1e-12)
            assert 0.0 < op.pf <= 1.0
            assert 0.0 <= op.eta_circuit < 1.0

    def test_stator_balance_with_saturation_table(self):
        """The L_m(I_m) damped fixed point must exit with currents
        recomputed from the final L_m — a stale-impedance exit would
        break the same stator balance."""
        p = make_im_params(
            L_m_vs_i=((1.0, 0.24), (3.0, 0.226), (6.0, 0.19)))
        for slip in (0.01, 0.047, 0.3, 1.0):
            op = solve_im_circuit(p, slip=slip)
            assert op.p_in == pytest.approx(op.p_cu_s + op.p_ag, rel=1e-9)


# ---------------------------------------------------------------------------
# identify_from_tests
# ---------------------------------------------------------------------------

def _series_test_points(R_1, R_2, L_ss, L_sr, L_m, R_extra,
                        f_lr, f_nl, V_lr, V_nl):
    """Exact synthetic test points from the series limits of the
    circuit: locked rotor with the magnetizing branch open (X_m -> inf)
    and no-load with the rotor branch open (R_2/s -> inf), plus a
    series stand-in resistance R_extra for the rotational losses."""
    from math import pi

    Z_lr = complex(R_1 + R_2, 2 * pi * f_lr * (L_ss + L_sr))
    I_lr = V_lr / abs(Z_lr)
    P_lr = 3 * I_lr**2 * (R_1 + R_2)
    Z_nl = complex(R_1 + R_extra, 2 * pi * f_nl * (L_ss + L_m))
    I_nl = V_nl / abs(Z_nl)
    P_nl = 3 * I_nl**2 * (R_1 + R_extra)
    return (V_nl, I_nl, P_nl), (V_lr, I_lr, P_lr)


class TestIdentifyFromTests:

    def test_series_round_trip_exact(self):
        """Parameters -> synthetic series test points -> identification
        recovers them exactly (non-0.5 split, reduced-frequency locked
        rotor, no-load at a different frequency)."""
        from phasesweep.models.im_circuit import identify_from_tests

        R_1, R_2 = 1.5, 2.5
        L_ss, L_sr, L_m = 0.012, 0.008, 0.25
        R_extra = 5.0
        no_load, locked_rotor = _series_test_points(
            R_1, R_2, L_ss, L_sr, L_m, R_extra,
            f_lr=12.5, f_nl=50.0, V_lr=20.0, V_nl=230.0)
        ident = identify_from_tests(
            R_1=R_1, no_load=no_load, locked_rotor=locked_rotor,
            f_no_load=50.0, f_locked_rotor=12.5,
            leakage_split=L_ss / (L_ss + L_sr))
        assert ident.R_2 == pytest.approx(R_2, rel=1e-12)
        assert ident.L_sigma_s == pytest.approx(L_ss, rel=1e-12)
        assert ident.L_sigma_r == pytest.approx(L_sr, rel=1e-12)
        assert ident.L_m == pytest.approx(L_m, rel=1e-12)
        _V, I_nl, _P = no_load
        assert ident.P_rot_W == pytest.approx(3 * I_nl**2 * R_extra,
                                              rel=1e-12)
        assert ident.P_core_W is None and ident.P_fw_W is None

    def test_full_circuit_magnetizing_shunt_bias(self):
        """Against a locked-rotor point solved with the FULL circuit,
        the shunt-blind identification under-estimates R_2 by the
        documented factor ~(X_m/(X_m + X_2))² — the bias is real,
        small, and attributable."""
        from math import pi

        from phasesweep.models.im_circuit import identify_from_tests

        p = make_im_params(R_1=2.0, R_2=3.0, L_sigma_s=0.01,
                           L_sigma_r=0.01, L_m=0.3)
        op = solve_im_circuit(p, slip=1.0, V_ph_rms=50.0)
        # The no-load P below is fabricated (only the current is
        # series-consistent) — this test asserts R_2 only, which the
        # no-load point does not enter.
        ident = identify_from_tests(
            R_1=p.R_1,
            no_load=(230.0, 230.0 / abs(complex(
                p.R_1, 2 * pi * 50 * (p.L_sigma_s + p.L_m))), 500.0),
            locked_rotor=(50.0, op.I_1_rms, op.p_in),
            f_no_load=50.0, f_locked_rotor=50.0)
        X_m = 2 * pi * 50 * p.L_m
        X_2 = 2 * pi * 50 * p.L_sigma_r
        assert ident.R_2 < p.R_2
        assert ident.R_2 == pytest.approx(
            p.R_2 * (X_m / (X_m + X_2)) ** 2, rel=0.02)

    def test_design_letter_split(self):
        """NEMA design letter B assigns the 0.4/0.6 stator/rotor
        leakage split; the total is preserved."""
        from phasesweep.models.im_circuit import identify_from_tests

        no_load, locked_rotor = _series_test_points(
            2.0, 3.0, 0.01, 0.01, 0.3, 4.0,
            f_lr=50.0, f_nl=50.0, V_lr=50.0, V_nl=230.0)
        a = identify_from_tests(
            R_1=2.0, no_load=no_load, locked_rotor=locked_rotor,
            f_no_load=50.0, f_locked_rotor=50.0, leakage_split="b")
        assert a.stator_leakage_fraction == 0.4
        assert a.L_sigma_s / (a.L_sigma_s + a.L_sigma_r) == pytest.approx(0.4)
        assert a.L_sigma_s + a.L_sigma_r == pytest.approx(0.02, rel=1e-12)

    def test_p_fw_splits_rotational_loss(self):
        from phasesweep.models.im_circuit import identify_from_tests

        no_load, locked_rotor = _series_test_points(
            2.0, 3.0, 0.01, 0.01, 0.3, 4.0,
            f_lr=50.0, f_nl=50.0, V_lr=50.0, V_nl=230.0)
        ident = identify_from_tests(
            R_1=2.0, no_load=no_load, locked_rotor=locked_rotor,
            f_no_load=50.0, f_locked_rotor=50.0, P_fw=10.0)
        assert ident.P_fw_W == 10.0
        assert ident.P_core_W == pytest.approx(ident.P_rot_W - 10.0)

    @pytest.mark.parametrize("kwargs,match", [
        (dict(leakage_split="E"), "design letter"),
        (dict(leakage_split=1.0), "must be in"),
        (dict(leakage_split=0.0), "must be in"),
        (dict(R_1=-1.0), "R_1 must be"),
        (dict(f_locked_rotor=0.0), "f_locked_rotor"),
        (dict(P_fw=1e6), "P_fw must be"),
    ])
    def test_bad_inputs_raise(self, kwargs, match):
        from phasesweep.models.im_circuit import identify_from_tests

        no_load, locked_rotor = _series_test_points(
            2.0, 3.0, 0.01, 0.01, 0.3, 4.0,
            f_lr=50.0, f_nl=50.0, V_lr=50.0, V_nl=230.0)
        good = dict(R_1=2.0, no_load=no_load, locked_rotor=locked_rotor,
                    f_no_load=50.0, f_locked_rotor=50.0)
        with pytest.raises(ValueError, match=match):
            identify_from_tests(**{**good, **kwargs})

    def test_inconsistent_data_raises(self):
        from phasesweep.models.im_circuit import identify_from_tests

        good = dict(R_1=2.0, f_no_load=50.0, f_locked_rotor=50.0)
        # power factor >= 1: wrong voltage convention signature
        with pytest.raises(ValueError, match="power factor"):
            identify_from_tests(
                **good, no_load=(230.0, 3.0, 500.0),
                locked_rotor=(50.0, 6.0, 1e6))
        # locked-rotor resistance below R_1
        with pytest.raises(ValueError, match="does not exceed R_1"):
            identify_from_tests(
                **good, no_load=(230.0, 3.0, 500.0),
                locked_rotor=(50.0, 6.0, 100.0))
        # no-load reactance below the identified stator leakage
        no_load, locked_rotor = _series_test_points(
            2.0, 3.0, 0.05, 0.05, 0.3, 4.0,
            f_lr=50.0, f_nl=50.0, V_lr=200.0, V_nl=230.0)
        with pytest.raises(ValueError, match="stator leakage"):
            identify_from_tests(
                R_1=2.0, f_no_load=50.0, f_locked_rotor=50.0,
                no_load=(230.0, 8.0, 800.0), locked_rotor=locked_rotor,
                leakage_split=0.99)


class TestSeparateNoLoadLosses:

    def test_exact_recovery(self):
        """P = 3I²R_1 + P_fw + k·V² is the separation model — the
        V²-intercept recovers P_fw exactly."""
        from phasesweep.models.im_circuit import separate_no_load_losses

        R_1, P_fw, k = 2.0, 35.0, 0.004
        points = [(V, V / 80.0, 3 * (V / 80.0) ** 2 * R_1 + P_fw + k * V**2)
                  for V in (100.0, 150.0, 200.0, 230.0)]
        assert separate_no_load_losses(points, R_1) == pytest.approx(
            P_fw, rel=1e-12)

    def test_too_few_or_degenerate_points_raise(self):
        from phasesweep.models.im_circuit import separate_no_load_losses

        with pytest.raises(ValueError, match="at least 2"):
            separate_no_load_losses([(230.0, 3.0, 500.0)], 2.0)
        with pytest.raises(ValueError, match="same voltage"):
            separate_no_load_losses(
                [(230.0, 3.0, 500.0), (230.0, 3.1, 520.0)], 2.0)


class TestStrayLoadResidual:
    """IEEE-112 method-B stray load loss residual (Boldea Layer 1b)."""

    def test_exact_zero_from_circuit_model(self):
        """The circuit model's own operating point has P_sll = 0 by
        construction — the segregation accounts for everything."""
        from phasesweep.models.im_circuit import compute_stray_load_residual

        p = make_im_params(R_fe=1420.0)
        op = solve_im_circuit(p, 0.05)
        res = compute_stray_load_residual(
            P_in=op.p_in, I_1_rms=op.I_1_rms, slip=op.slip,
            P_out=op.p_mech, R_1=p.R_1, P_fe=op.p_fe, P_fw=0.0)
        assert res.P_sll_W == pytest.approx(0.0, abs=1e-6)

    def test_positive_residual_with_unmodeled_loss(self):
        """When measured P_out is less than the circuit predicts (real
        machine has stray losses), the residual is positive."""
        from phasesweep.models.im_circuit import compute_stray_load_residual

        p = make_im_params(R_fe=1420.0)
        op = solve_im_circuit(p, 0.05)
        P_sll_injected = 20.0
        res = compute_stray_load_residual(
            P_in=op.p_in, I_1_rms=op.I_1_rms, slip=op.slip,
            P_out=op.p_mech - P_sll_injected, R_1=p.R_1,
            P_fe=op.p_fe, P_fw=0.0)
        assert res.P_sll_W == pytest.approx(P_sll_injected, rel=1e-9)
        assert res.P_sll_pct > 0

    def test_negative_residual_raises(self):
        """A significantly negative P_sll = inconsistent segregation."""
        from phasesweep.models.im_circuit import compute_stray_load_residual

        p = make_im_params(R_fe=1420.0)
        op = solve_im_circuit(p, 0.05)
        with pytest.raises(ValueError, match="negative residual"):
            compute_stray_load_residual(
                P_in=op.p_in, I_1_rms=op.I_1_rms, slip=op.slip,
                P_out=op.p_mech * 1.5, R_1=p.R_1,
                P_fe=op.p_fe, P_fw=0.0)

    def test_slightly_negative_within_tolerance(self):
        """A residual within 0.5% of P_in is tolerated (measurement
        noise in real data)."""
        from phasesweep.models.im_circuit import compute_stray_load_residual

        p = make_im_params(R_fe=1420.0)
        op = solve_im_circuit(p, 0.05)
        nudge = 0.001 * op.p_in
        res = compute_stray_load_residual(
            P_in=op.p_in, I_1_rms=op.I_1_rms, slip=op.slip,
            P_out=op.p_mech + nudge, R_1=p.R_1,
            P_fe=op.p_fe, P_fw=0.0)
        assert res.P_sll_W < 0
        assert res.P_sll_W == pytest.approx(-nudge, rel=1e-9)

    @pytest.mark.parametrize("P_rated_kW,expected_pct", [
        (5.0, 1.8), (90.0, 1.8), (375.0, 1.5),
        (500.0, 1.2), (1850.0, 1.2), (2000.0, 0.9),
    ])
    def test_ieee_112_assumed_value_table(self, P_rated_kW, expected_pct):
        """IEEE-112 assumed stray load loss by power rating."""
        from phasesweep.models.im_circuit import _ieee_112_assumed_sll_pct

        assert _ieee_112_assumed_sll_pct(P_rated_kW) == expected_pct

    def test_ieee_112_assumed_returned_in_result(self):
        from phasesweep.models.im_circuit import compute_stray_load_residual

        p = make_im_params(R_fe=1420.0)
        op = solve_im_circuit(p, 0.05)
        res = compute_stray_load_residual(
            P_in=op.p_in, I_1_rms=op.I_1_rms, slip=op.slip,
            P_out=op.p_mech, R_1=p.R_1, P_fe=op.p_fe, P_fw=0.0,
            P_rated_kW=5.0)
        assert res.ieee_112_assumed_pct == 1.8

    def test_no_rated_power_gives_none(self):
        from phasesweep.models.im_circuit import compute_stray_load_residual

        p = make_im_params(R_fe=1420.0)
        op = solve_im_circuit(p, 0.05)
        res = compute_stray_load_residual(
            P_in=op.p_in, I_1_rms=op.I_1_rms, slip=op.slip,
            P_out=op.p_mech, R_1=p.R_1, P_fe=op.p_fe, P_fw=0.0)
        assert res.ieee_112_assumed_pct is None

    def test_without_r_fe_core_loss_zero(self):
        """Without R_fe, caller passes P_fe=0 — residual is still
        meaningful (core loss is then lumped into the residual)."""
        from phasesweep.models.im_circuit import compute_stray_load_residual

        p = make_im_params()
        op = solve_im_circuit(p, 0.05)
        res = compute_stray_load_residual(
            P_in=op.p_in, I_1_rms=op.I_1_rms, slip=op.slip,
            P_out=op.p_mech, R_1=p.R_1, P_fe=0.0, P_fw=0.0)
        assert res.P_sll_W == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize("bad_kwargs,match", [
        (dict(P_in=-1), "P_in"),
        (dict(I_1_rms=-1), "I_1_rms"),
        (dict(R_1=-1), "R_1"),
        (dict(slip=0), "slip"),
        (dict(slip=1.5), "slip"),
        (dict(P_out=-1), "P_out"),
        (dict(P_fe=-1), "P_fe"),
        (dict(P_fw=-1), "P_fw"),
        (dict(P_out=float("nan")), "finite"),
        (dict(P_fe=float("inf")), "finite"),
        (dict(P_fw=float("nan")), "finite"),
        (dict(slip=float("nan")), "finite"),
    ])
    def test_bad_inputs_raise(self, bad_kwargs, match):
        from phasesweep.models.im_circuit import compute_stray_load_residual

        p = make_im_params(R_fe=1420.0)
        op = solve_im_circuit(p, 0.05)
        good = dict(P_in=op.p_in, I_1_rms=op.I_1_rms, slip=op.slip,
                    P_out=op.p_mech, R_1=p.R_1, P_fe=op.p_fe, P_fw=0.0)
        with pytest.raises(ValueError, match=match):
            compute_stray_load_residual(**{**good, **bad_kwargs})

    def test_p_out_zero_negative_residual_no_crash(self):
        """P_out=0 (locked rotor) with a negative residual must raise
        ValueError, not ZeroDivisionError."""
        from phasesweep.models.im_circuit import compute_stray_load_residual

        with pytest.raises(ValueError, match="negative residual"):
            compute_stray_load_residual(
                P_in=100.0, I_1_rms=1.0, slip=1.0,
                P_out=0.0, R_1=2.0, P_fe=50.0, P_fw=200.0)

    def test_fw_reduces_residual(self):
        """Adding P_fw to the segregation takes it out of the residual."""
        from phasesweep.models.im_circuit import compute_stray_load_residual

        p = make_im_params(R_fe=1420.0)
        op = solve_im_circuit(p, 0.05)
        P_fw = 30.0
        P_sll_extra = 10.0
        P_out_measured = op.p_mech - P_fw - P_sll_extra
        res_no_fw = compute_stray_load_residual(
            P_in=op.p_in, I_1_rms=op.I_1_rms, slip=op.slip,
            P_out=P_out_measured, R_1=p.R_1, P_fe=op.p_fe, P_fw=0.0)
        res_with_fw = compute_stray_load_residual(
            P_in=op.p_in, I_1_rms=op.I_1_rms, slip=op.slip,
            P_out=P_out_measured, R_1=p.R_1, P_fe=op.p_fe, P_fw=P_fw)
        assert res_no_fw.P_sll_W == pytest.approx(P_fw + P_sll_extra, rel=1e-9)
        assert res_with_fw.P_sll_W == pytest.approx(P_sll_extra, rel=1e-9)


class TestJurkovicFixture:
    """Jurkovic, "Induction Motor Parameters Extraction" (MIT
    Kirtley-hosted teaching note): 2 hp 4-pole 208 V machine, no-load
    (208 V L-L, 1.5 A, 49 W) and locked-rotor (90 V/phase, 6.1 A,
    724 W) at 60 Hz, R_s = 2.3 Ω, 50/50 split — an independent numeric
    fixture for the identification algebra.

    The note's own derived values reproduce to ~1% where the formulas
    coincide (R_r 4.2 Ω, X_eq 13.3 Ω, L_ls = L_lr 17.5 mH). Its L_m
    0.23 H does NOT use the series route — it divides V by I·sin(φ)
    and skips the stator-leakage drop — so the series-route value
    (0.194 H) is pinned instead, bracketed by the note's two tests
    (0.23 first, 0.323 driven-rotor). The note's rated-torque
    validation is deliberately not reproduced: its 11.2 Nm nameplate
    is inconsistent with 2 hp at 1725 rpm (8.3 Nm) and its
    Pg = 678 W -> "10.6 Nm" step does not follow from its own circuit
    (Pg/omega_s = 3.6 Nm)."""

    @pytest.fixture(scope="class")
    def ident(self):
        from math import sqrt

        from phasesweep.models.im_circuit import identify_from_tests

        return identify_from_tests(
            R_1=2.3,
            no_load=(208.0 / sqrt(3), 1.5, 49.0),
            locked_rotor=(90.0, 6.1, 724.0),
            f_no_load=60.0, f_locked_rotor=60.0)

    def test_rotor_resistance(self, ident):
        assert ident.R_2 == pytest.approx(4.2, rel=0.01)

    def test_leakage(self, ident):
        assert ident.L_sigma_s == pytest.approx(17.5e-3, rel=0.01)
        assert ident.L_sigma_r == ident.L_sigma_s

    def test_locked_rotor_power_factor(self, ident):
        assert ident.pf_locked_rotor == pytest.approx(0.44, abs=0.005)

    def test_magnetizing_inductance_series_route(self, ident):
        assert ident.L_m == pytest.approx(0.194, abs=0.001)
        assert 0.23 * 0.7 < ident.L_m < 0.23


# ---------------------------------------------------------------------------
# R_fe core-loss shunt (Boldea Layer 1a)
# ---------------------------------------------------------------------------

class TestRFeCoreLossShunt:
    """Tests for the parallel R_fe core-loss resistance across L_m."""

    def test_r_fe_none_backward_compatible(self):
        """R_fe=None produces p_fe=0, R_fe_used=None, and identical
        outputs to the original circuit."""
        p_old = make_im_params()
        p_new = make_im_params(R_fe=None)
        assert p_old == p_new
        op = solve_im_circuit(p_new, 0.05)
        assert op.p_fe == 0.0
        assert op.R_fe_used is None

    def test_r_fe_produces_nonzero_p_fe(self):
        """A finite R_fe produces measurable core loss."""
        # R_fe = 20 * X_m ~ 20 * 2*pi*50*0.226 ~ 1420 Ohm — realistic
        R_fe = 1420.0
        p = make_im_params(R_fe=R_fe)
        op = solve_im_circuit(p, 0.05)
        assert op.p_fe > 0
        assert op.R_fe_used == R_fe

    def test_power_balance_with_r_fe(self):
        """p_in = p_cu_s + p_fe + p_ag holds exactly when R_fe is present.
        P_fe is dissipated in the shunt upstream of the rotor, so
        p_mech = (1-s)*p_ag (no P_fe subtraction)."""
        R_fe = 1420.0
        p = make_im_params(R_fe=R_fe)
        for slip in (0.001, 0.01, 0.047, 0.15, 0.5, 1.0):
            op = solve_im_circuit(p, slip=slip)
            # Stator balance: input = stator copper + core + airgap
            assert op.p_in == pytest.approx(
                op.p_cu_s + op.p_fe + op.p_ag, rel=1e-9)
            # Rotor balance: p_mech = (1-s)*p_ag, not subtracted by p_fe
            assert op.p_mech == pytest.approx(
                (1 - slip) * op.p_ag, rel=1e-9)
            # Consistency: tau * omega_mech == p_mech
            from math import pi as mpi
            w_m = 2 * mpi * op.f_supply * (1 - slip) / p.n_p
            assert op.p_mech == pytest.approx(op.tau * w_m, rel=1e-12)
            # Core loss identity: 3 * |V_m|^2 / R_fe
            # (recompute V_m from the operating point)
            from math import pi as mpi
            w = 2 * mpi * op.f_supply
            Z_1 = complex(p.R_1, w * op.L_sigma_s_used)
            I_1 = op.V_phase_rms / (
                Z_1 + _z_mag_test(w, op.L_m_used, R_fe)
                * complex(op.R_2_used / slip, w * op.L_sigma_r_used)
                / (_z_mag_test(w, op.L_m_used, R_fe)
                   + complex(op.R_2_used / slip, w * op.L_sigma_r_used)))
            V_m = op.V_phase_rms - I_1 * Z_1
            assert op.p_fe == pytest.approx(
                3 * abs(V_m) ** 2 / R_fe, rel=1e-9)

    def test_r_fe_increases_input_current(self):
        """R_fe is a load in parallel with L_m: total stator current
        must increase compared to the R_fe=None case."""
        p_no = make_im_params()
        p_fe = make_im_params(R_fe=1420.0)
        op_no = solve_im_circuit(p_no, 0.05)
        op_fe = solve_im_circuit(p_fe, 0.05)
        assert op_fe.I_1_rms > op_no.I_1_rms

    def test_r_fe_reduces_efficiency(self):
        """Core loss lowers the circuit efficiency."""
        p_no = make_im_params()
        p_fe = make_im_params(R_fe=1420.0)
        op_no = solve_im_circuit(p_no, 0.05)
        op_fe = solve_im_circuit(p_fe, 0.05)
        assert op_fe.eta_circuit < op_no.eta_circuit

    @pytest.mark.parametrize("slip", [0.001, 0.01, 0.05, 0.15, 0.5, 1.0])
    def test_p_fe_scales_with_voltage_squared_linear_iron(self, slip):
        """With linear iron (no L_m saturation table), p_fe must scale
        as V^2 to machine precision — the circuit is linear in V so
        all phasors scale proportionally (Boldea Layer 2c invariant)."""
        p = make_im_params(R_fe=1420.0)
        op1 = solve_im_circuit(p, slip, V_ph_rms=200.0)
        op2 = solve_im_circuit(p, slip, V_ph_rms=100.0)
        assert op2.p_fe / op1.p_fe == pytest.approx(0.25, rel=1e-12)

    def test_p_fe_v_squared_with_deep_bar_tables(self):
        """V^2 scaling holds with deep-bar R_2(f) and leakage tables —
        those depend on frequency, not voltage (Boldea 2c corollary)."""
        p = make_im_params(
            R_fe=1420.0,
            R_2_vs_f=((10.0, 2.0), (50.0, 4.0)),
            L_sigma_r_vs_f=((10.0, 0.012), (50.0, 0.009)),
        )
        for slip in (0.01, 0.05, 0.5, 1.0):
            op1 = solve_im_circuit(p, slip, V_ph_rms=200.0)
            op2 = solve_im_circuit(p, slip, V_ph_rms=100.0)
            assert op2.p_fe / op1.p_fe == pytest.approx(0.25, rel=1e-12)

    def test_r_fe_validation_rejects_nonpositive(self):
        """R_fe must be > 0."""
        with pytest.raises(ValueError, match="R_fe"):
            ImCircuitParams(n_p=2, R_1=2.33, R_2=1.012, L_sigma_s=0.0114,
                            L_sigma_r=0.0106, L_m=0.226, V_LL=400.0,
                            f_supply=50.0, R_fe=-100.0)

    def test_r_fe_on_induction_params_rejects_nonpositive(self):
        """InductionParams.R_fe must be > 0."""
        with pytest.raises(ValueError, match="R_fe"):
            InductionParams(R_fe=-100.0)

    def test_config_id_distinguishes_r_fe(self):
        """Two otherwise identical motors with different R_fe values
        must receive different config_ids (cache identity)."""
        a = make_im_motor()  # R_fe=None
        b = dc.replace(a, induction=dc.replace(
            a.induction, R_fe=1420.0))
        c = dc.replace(a, induction=dc.replace(
            a.induction, R_fe=2000.0))
        assert a.config_id != b.config_id
        assert b.config_id != c.config_id
        assert a.config_id != c.config_id

    def test_run_id_differs_with_r_fe(self):
        """Different R_fe must produce different run IDs for the same
        model and slip."""
        a = make_im_motor()
        b = dc.replace(a, induction=dc.replace(
            a.induction, R_fe=1420.0))
        rc_a = RunConfig(motor=a, model="im_equivalent_circuit", slip=0.05)
        rc_b = RunConfig(motor=b, model="im_equivalent_circuit", slip=0.05)
        assert compute_run_id(rc_a) != compute_run_id(rc_b)

    def test_r_fe_threads_through_prepare_im_circuit(self):
        """prepare_im_circuit must carry R_fe from InductionParams."""
        m = dc.replace(make_im_motor(), induction=dc.replace(
            make_im_motor().induction, R_fe=1420.0))
        p = prepare_im_circuit(m)
        assert p.R_fe == 1420.0

    def test_r_fe_none_threads_through(self):
        """R_fe=None must be carried through as None."""
        p = prepare_im_circuit(make_im_motor())
        assert p.R_fe is None

    def test_registry_equivalent_circuit_r_fe_outputs(self):
        """The registry runner emits r_fe_over_x_m when R_fe is set."""
        m = dc.replace(make_im_motor(), induction=dc.replace(
            make_im_motor().induction, R_fe=1420.0))
        rc = RunConfig(motor=m, model="im_equivalent_circuit", slip=0.05)
        out = MODEL_REGISTRY["im_equivalent_circuit"].fn(rc)
        assert out["R_fe_used"] == 1420.0
        assert out["p_fe"] > 0
        assert out["r_fe_over_x_m"] is not None
        # Boldea ~20 for this parameter set
        from math import pi as mpi
        X_m = 2 * mpi * 50.0 * out["L_m_used"]
        assert out["r_fe_over_x_m"] == pytest.approx(1420.0 / X_m, rel=1e-9)

    def test_registry_equivalent_circuit_no_r_fe_outputs(self):
        """Without R_fe, the runner emits None/0 for the new keys."""
        rc = RunConfig(motor=make_im_motor(),
                       model="im_equivalent_circuit", slip=0.05)
        out = MODEL_REGISTRY["im_equivalent_circuit"].fn(rc)
        assert out["R_fe_used"] is None
        assert out["p_fe"] == 0.0
        assert out["r_fe_over_x_m"] is None

    def test_registry_torque_slip_r_fe_outputs(self):
        """The torque-slip runner emits p_fe_list and R_fe_used."""
        m = dc.replace(make_im_motor(), induction=dc.replace(
            make_im_motor().induction, R_fe=1420.0))
        rc = RunConfig(motor=m, model="im_torque_slip")
        out = MODEL_REGISTRY["im_torque_slip"].fn(rc)
        assert out["R_fe_used"] == 1420.0
        assert len(out["p_fe_list"]) == len(out["slip_list"])
        assert all(pf >= 0 for pf in out["p_fe_list"])

    def test_r_fe_boldea_ratio_diagnostic(self):
        """R_fe / X_m ~ 20 for a realistic motor. Warn outside 5-100
        (soft diagnostic, not a hard assert)."""
        from math import pi as mpi
        L_m = 0.226
        X_m = 2 * mpi * 50.0 * L_m
        # Ratio ~20 -> R_fe ~20 * X_m
        R_fe_20 = 20.0 * X_m
        m = dc.replace(make_im_motor(), induction=dc.replace(
            make_im_motor().induction, R_fe=R_fe_20))
        rc = RunConfig(motor=m, model="im_equivalent_circuit", slip=0.05)
        out = MODEL_REGISTRY["im_equivalent_circuit"].fn(rc)
        assert 15 < out["r_fe_over_x_m"] < 25

    def test_r_fe_toml_round_trip(self):
        """R_fe survives TOML dict round-trip."""
        m = dc.replace(make_im_motor(), induction=dc.replace(
            make_im_motor().induction, R_fe=1420.0))
        d = m.to_dict()
        assert d["induction"]["R_fe"] == 1420.0
        m2 = Motor.from_dict(d)
        assert m2.induction.R_fe == 1420.0
        assert m2.config_id == m.config_id

    def test_saturation_table_with_r_fe(self):
        """L_m(I_m) saturation iteration must converge when R_fe is
        present — I_m is the inductive magnetizing current, not the
        total magnetizing branch current."""
        table = ((0.5, 0.26), (3.0, 0.20), (7.0, 0.13))
        p = make_im_params(R_fe=1420.0, L_m_vs_i=table)
        op = solve_im_circuit(p, 0.05)
        assert 0.13 <= op.L_m_used <= 0.26
        # The saturation key is I_m through jX_m, not the total branch
        from phasesweep.models.im_circuit import _interp_anchored
        assert op.L_m_used == pytest.approx(
            _interp_anchored(table, op.I_m_rms, table[0][1]), rel=1e-6)


def _z_mag_test(w: float, L_m: float, R_fe: float | None) -> complex:
    """Test helper: magnetizing branch impedance."""
    Z_jxm = complex(0.0, w * L_m)
    if R_fe is None:
        return Z_jxm
    return Z_jxm * R_fe / (Z_jxm + R_fe)
