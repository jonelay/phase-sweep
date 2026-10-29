"""motor_thermal: three-node lumped network coupled to the operating point."""

from __future__ import annotations

import dataclasses
import pickle
import tomllib
from math import exp, hypot, pi

import numpy as np
import pytest

from phasesweep.machines.configs import motor_from_dict
from phasesweep.machines.motor import DriveParams, FluxMap, Motor, TempLimit, ThermalNetwork
from phasesweep.models.flux_map import flux_map_psi_f
from phasesweep.models.iron_loss import iron_loss_at, run_iron_loss
from phasesweep.models.motor_thermal import (
    CoupledSolveError,
    InfeasibleDemand,
    NoConvergence,
    ThermalRunaway,
    _modes,
    _runaway,
    assemble_network,
    coupled_steady,
    periodic_peak,
    periodic_start,
    prepare_motor_thermal,
    run_motor_thermal,
    segment_map,
    steady_temps,
)
from phasesweep.models.operating_point import run_operating_point
from phasesweep.models.thermal_duty import transient_peak
from phasesweep.registry import MODEL_REGISTRY
from phasesweep.sweep_types import RunConfig, compute_run_id
from tests.conftest import REPO_ROOT, make_motor
from tests.models.test_torque_speed import _make_linear_flux_map

_RPM = 2 * pi / 60
_TN = ThermalNetwork(G_WFe=42.5, G_FeS=30.0, G_FeM=8.0,
                     C_W=2700.0, C_Fe=6300.0, C_M=6500.0)
_IRON = dict(k_h=0.02, k_e=6e-5, alpha_fe=1.8, m_core=5.0, B_core=1.0)


def _motor(**kw):
    base = dict(alpha_Br=-0.0012, thermal_network=_TN, **_IRON)
    base.update(kw)
    return make_motor(**base)


@pytest.fixture(scope="module")
def i3():
    from phasesweep import load_motor
    return load_motor(REPO_ROOT / "motors/bmw_i3.toml")


# ---------------------------------------------------------------------------
# ThermalNetwork on the Motor
# ---------------------------------------------------------------------------

class TestThermalNetwork:

    @pytest.mark.parametrize("field", ["G_WFe", "G_FeS", "G_FeM", "C_W", "C_Fe", "C_M"])
    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_rejects_non_positive_and_non_finite(self, field, bad):
        with pytest.raises(ValueError, match=field):
            ThermalNetwork(**{**_TN.to_dict(), field: bad})

    def test_round_trip(self):
        assert ThermalNetwork.from_dict(_TN.to_dict()) == _TN
        tagged = dataclasses.replace(_TN, winding_basis="ETD")
        assert tagged.to_dict()["winding_basis"] == "ETD"
        assert ThermalNetwork.from_dict(tagged.to_dict()) == tagged

    def test_bases_unhashed(self, i3):
        # Provenance tags, like the [thermal] _basis/_source tags: setting
        # one keeps the config_id and every run ID. Each basis is checked
        # against a network without it (the i3 block ships winding_basis).
        bare = dataclasses.replace(i3.thermal_network, winding_basis=None,
                                   magnet_basis=None)
        i3 = dataclasses.replace(i3, thermal_network=bare)
        for tn in (dataclasses.replace(bare, winding_basis="ETD"),
                   dataclasses.replace(bare, magnet_basis="resistance")):
            assert dataclasses.replace(i3, thermal_network=tn).config_id == i3.config_id
        tn = dataclasses.replace(bare, winding_basis="ETD", magnet_basis="resistance")
        tagged = dataclasses.replace(i3, thermal_network=tn)
        assert tagged.config_id == i3.config_id
        for model, kw in (("torque_speed", {}),
                          ("motor_thermal", dict(w_mech=733.0, tau_ref=100.0,
                                                 T_coolant=60.0))):
            rc = RunConfig(motor=i3, model=model, **kw)
            assert (compute_run_id(dataclasses.replace(rc, motor=tagged))
                    == compute_run_id(rc))
        assert Motor.from_dict(tagged.to_dict()) == tagged

    def test_config_id_append_only(self, i3):
        without = dataclasses.replace(i3, thermal_network=None)
        assert "thermal_network" not in without.to_dict()
        assert without.config_id != i3.config_id
        assert Motor.from_dict(i3.to_dict()) == i3

    def test_motors_without_block_serialize_unchanged(self):
        # Identity of block-less motors is pinned elsewhere: the committed
        # run ids in tests/run_id_vectors.json (test_run_id_compat) and the
        # block-less i3 rows in record_vectors.json (test_record_vectors).
        from phasesweep import load_motor
        blocks = 0
        for path in sorted((REPO_ROOT / "motors").glob("*.toml")):
            m = load_motor(path)
            if m.thermal_network is None:
                assert "thermal_network" not in m.to_dict()
            else:
                blocks += 1
        assert blocks == 1  # only the i3 carries a block

    def test_loader_strict_keys(self):
        raw = tomllib.loads((REPO_ROOT / "motors/bmw_i3.toml").read_text())
        raw["thermal_network"] = {**_TN.to_dict(), "G_WS": 1.0}
        with pytest.raises(ValueError, match=r"\[thermal_network\] unknown field"):
            motor_from_dict(raw, label="i3", default_name="i3")
        raw["thermal_network"] = {**_TN.to_dict(), "G_FeM": 0.0}
        with pytest.raises(ValueError, match=r"\[thermal_network\] G_FeM"):
            motor_from_dict(raw, label="i3", default_name="i3")

    def test_runconfig_t_coolant(self):
        m = _motor()
        with pytest.raises(ValueError, match="T_coolant"):
            RunConfig(motor=m, model="motor_thermal", T_coolant=float("nan"))
        rc = RunConfig(motor=m, model="motor_thermal", w_mech=50.0, tau_ref=1.0,
                       T_coolant=40.0)
        assert RunConfig.from_dict(rc.to_dict()) == rc
        assert "T_coolant" not in RunConfig(motor=m, model="motor_thermal").to_dict()
        other = dataclasses.replace(rc, T_coolant=41.0)
        assert compute_run_id(rc) != compute_run_id(other)


# ---------------------------------------------------------------------------
# pure network functions
# ---------------------------------------------------------------------------

class TestNetwork:

    def test_steady_closed_form(self):
        G, g_s, _ = assemble_network(_TN)
        p_cu, p_fe, T_s = 300.0, 600.0, 60.0
        T = steady_temps(G, g_s, np.array([p_cu, p_fe, 0.0]), T_s)
        T_Fe = T_s + (p_cu + p_fe) / _TN.G_FeS
        assert T[1] == pytest.approx(T_Fe, rel=1e-12)
        assert T[0] == pytest.approx(T_Fe + p_cu / _TN.G_WFe, rel=1e-12)
        assert T[2] == pytest.approx(T_Fe, rel=1e-12)

    def test_segment_map_matches_fine_euler(self):
        G, g_s, C = assemble_network(_TN)
        P, T_s, dt = np.array([300.0, 600.0, 0.0]), 60.0, 200.0
        T0 = np.array([90.0, 70.0, 65.0])
        M, v = segment_map(G, g_s, C, P, T_s, dt)
        n = 200_000
        T = T0.copy()
        h = dt / n
        for _ in range(n):
            T = T + h * (-(G @ T) + P + g_s * T_s) / C
        assert M @ T0 + v == pytest.approx(T, rel=1e-6)

    def test_periodic_start_matches_brute_force_march(self):
        G, g_s, C = assemble_network(_TN)
        segs = [(np.array([900.0, 600.0, 0.0]), 120.0), (np.array([50.0, 300.0, 0.0]), 400.0)]
        maps = [segment_map(G, g_s, C, P, 60.0, dt) for P, dt in segs]
        T = np.full(3, 60.0)
        for _ in range(20_000):
            for M, v in maps:
                T = M @ T + v
        assert periodic_start(G, g_s, C, 60.0, segs) == pytest.approx(T, rel=1e-9)

    def test_peak_inside_a_segment_is_resolved(self):
        # Two-node probe: a fast node fed by a hot pulse then coupled to a
        # slow node overshoots inside the long segment; the endpoints miss it.
        G = np.array([[1.0 + 0.5, -0.5], [-0.5, 0.5 + 0.02]])
        g_s = np.array([1.0, 0.02])
        C = np.array([1.0, 200.0])
        segs = [(np.array([40.0, 0.0]), 5.0), (np.array([0.0, 0.0]), 100.0)]
        peak, T0 = periodic_peak(G, g_s, C, 0.0, segs)
        maps = [segment_map(G, g_s, C, P, 0.0, dt) for P, dt in segs]
        ends = [T0]
        for M, v in maps:
            ends.append(M @ ends[-1] + v)
        endpoint_peak = np.max(np.array(ends), axis=0)
        # brute force on a 1 ms grid
        T, ref = T0.copy(), T0.copy()
        for P, dt in segs:
            n = int(dt / 1e-3)
            M, v = segment_map(G, g_s, C, P, 0.0, dt / n)
            for _ in range(n):
                T = M @ T + v
                ref = np.maximum(ref, T)
        assert peak == pytest.approx(ref, rel=1e-4)
        assert peak[1] > endpoint_peak[1] * 1.01

    def test_near_singular_scalar_is_cancellation_safe(self):
        # exp(-dt/tau) rounds to 1 - 1e-15 here; T_ss - M T_ss loses digits
        # (0.889 for 0.9 before the expm1 form).
        G, g_s, C = np.array([[1.0]]), np.array([1.0]), np.array([1e15])
        T0 = periodic_start(G, g_s, C, 0.0, [(np.array([0.9]), 1.0)])
        assert T0[0] == pytest.approx(0.9, rel=1e-12)

    def test_mixed_timescales_keep_the_fast_mode(self):
        # Node 2 is far slower than the cycle (flat at its average); node 1
        # still cycles. Reference: node 1 marched with node 2 clamped.
        G = np.array([[2.0, -1.0], [-1.0, 1.0]])
        g_s, C = np.array([1.0, 0.0]), np.array([1.0, 1e30])
        segs = [(np.array([1.0, 0.0]), 1.0), (np.array([0.0, 0.0]), 1.0)]
        T0 = periodic_start(G, g_s, C, 0.0, segs)
        T1, T2 = 0.0, 0.5
        for _ in range(200):
            for P, dt in segs:
                T_ss = (T2 + P[0]) / 2.0
                T1 = T_ss + (T1 - T_ss) * exp(-2.0 * dt)
        assert T0 == pytest.approx([T1, 0.5], rel=1e-9)

    def test_asymmetric_conductance_rejected(self):
        G = np.array([[2.0, -1.0], [-0.5, 1.0]])
        with pytest.raises(ValueError, match="symmetric"):
            periodic_start(G, np.array([1.0, 0.0]), np.array([1.0, 1.0]), 0.0,
                           [(np.array([1.0, 0.0]), 1.0)])

    def test_one_node_reproduces_thermal_duty_s1_budget(self):
        # thermal_resistance path: P_S1 = (limit - ambient) / r_th puts the
        # single node exactly at the limit.
        r_th, limit, ambient = 0.5, 130.0, 20.0
        G, g_s = np.array([[1.0 / r_th]]), np.array([1.0 / r_th])
        T = steady_temps(G, g_s, np.array([(limit - ambient) / r_th]), ambient)
        assert T[0] == pytest.approx(limit, rel=1e-12)

    @pytest.mark.parametrize("tau", [30.0, 1e30])
    def test_one_node_reproduces_thermal_duty_transient(self, tau):
        # 1x1 network: G = g_s = 1, C = tau, T_s = 0, so T == u = P/P_S1.
        u, dts = [0.9, 0.2, 1.4], [20.0, 40.0, 5.0]
        G, g_s, C = np.array([[1.0]]), np.array([1.0]), np.array([tau])
        segs = [(np.array([u_k]), dt) for u_k, dt in zip(u, dts)]
        peak, T0 = periodic_peak(G, g_s, C, 0.0, segs)
        expected = transient_peak(u, dts, tau)
        assert peak[0] == pytest.approx(expected, rel=1e-9)
        if tau > 1e20:
            assert exp(-sum(dts) / tau) == 1.0  # the singular branch
            assert T0[0] == pytest.approx(expected, rel=1e-12)

    def test_disconnected_network_rejected(self):
        with pytest.raises(ValueError, match="G_FeM"):
            ThermalNetwork(**{**_TN.to_dict(), "G_FeM": 0.0})


# ---------------------------------------------------------------------------
# iron_loss_at and psi_d / psi_q
# ---------------------------------------------------------------------------

class TestLossInputs:

    def test_iron_loss_at_equals_run_iron_loss_at_w_ref(self):
        m = make_motor(**_IRON)
        ref = run_iron_loss(RunConfig(motor=m, model="iron_loss"))["p_fe"]
        got = iron_loss_at(m.n_p, m.k_h, m.k_e, m.alpha_fe, m.m_core, m.B_core,
                           m.drive.W_REF, 1.0)
        assert got == ref

    def test_iron_loss_at_scaling(self):
        k_h, k_e, a = 0.02, 6e-5, 1.8
        hyst = lambda w, r: iron_loss_at(2, k_h, 0.0 + 1e-300, a, 5.0, 1.0, w, r)  # noqa: E731
        eddy = lambda w, r: iron_loss_at(2, 1e-300, k_e, a, 5.0, 1.0, w, r)  # noqa: E731
        assert hyst(200.0, 1.0) / hyst(100.0, 1.0) == pytest.approx(2.0)
        assert eddy(200.0, 1.0) / eddy(100.0, 1.0) == pytest.approx(4.0)
        assert hyst(100.0, 1.2) / hyst(100.0, 1.0) == pytest.approx(1.2 ** a)
        assert eddy(100.0, 1.2) / eddy(100.0, 1.0) == pytest.approx(1.44)

    @pytest.mark.parametrize("with_map", [False, True])
    def test_psi_dq_satisfy_voltage_equations(self, with_map):
        m = make_motor(I_rated=10.0)
        if with_map:
            m = dataclasses.replace(m, flux_map=_make_linear_flux_map(m.psi_f, m.L_d, m.L_q, 30.0))
        r = run_operating_point(RunConfig(motor=m, model="operating_point",
                                          w_mech=100.0, tau_ref=1.0))
        w_e = m.n_p * 100.0
        assert r["u_d"] == pytest.approx(r["R_s_op"] * r["i_d"] - w_e * r["psi_q"], rel=1e-9)
        assert r["u_q"] == pytest.approx(r["R_s_op"] * r["i_q"] + w_e * r["psi_d"], rel=1e-9)
        assert set(r) == MODEL_REGISTRY["operating_point"].produces


# ---------------------------------------------------------------------------
# params factory and coupled solve
# ---------------------------------------------------------------------------

class TestPrepare:

    def test_psi_ref_constant_l(self):
        p = prepare_motor_thermal(_motor())
        assert p.psi_ref == p.op.psi_f

    def test_psi_ref_map_with_origin(self):
        # The map's psi_d(0, 0) deliberately differs from psi_f (3 %, under
        # the 5 % warning) so the two references are distinguishable.
        m = _motor()
        fm = _make_linear_flux_map(1.03 * m.psi_f, m.L_d, m.L_q, 30.0)
        assert fm.has_origin
        p = prepare_motor_thermal(dataclasses.replace(m, flux_map=fm))
        assert p.psi_ref == flux_map_psi_f(fm)
        assert p.psi_ref != p.op.psi_f

    def test_psi_ref_hot_map_referred_to_20C(self):
        """A map extracted hot pairs with B_core at 20 °C, the reference the
        off-map psi_f uses — not at the map's own temperature."""
        m = _motor()
        fm = dataclasses.replace(
            _make_linear_flux_map(m.psi_f, m.L_d, m.L_q, 30.0), magnet_temp_ref_C=80.0)
        p = prepare_motor_thermal(dataclasses.replace(m, flux_map=fm))
        assert p.psi_ref == pytest.approx(
            flux_map_psi_f(fm) / (1.0 + m.alpha_Br * 60.0), rel=1e-12)

    def test_map_without_origin_rejected(self):
        m = _motor()
        fm = _make_linear_flux_map(m.psi_f, m.L_d, m.L_q, 30.0)
        d = fm.to_dict()
        d["i_d_A"] = [x - 1.0 for x in d["i_d_A"]]
        no_origin = FluxMap.from_dict(d)
        assert not no_origin.has_origin
        with pytest.raises(ValueError, match="origin node"):
            prepare_motor_thermal(dataclasses.replace(m, flux_map=no_origin))

    def test_map_not_covering_falls_back_to_psi_f(self):
        m = _motor()
        fm = _make_linear_flux_map(m.psi_f, m.L_d, m.L_q, 5.0)
        p = prepare_motor_thermal(dataclasses.replace(m, flux_map=fm))
        assert p.psi_ref == p.op.psi_f

    @pytest.mark.parametrize("kw, match", [
        ({"thermal_network": None}, r"\[thermal_network\]"),
        ({"alpha_Br": None}, "alpha_Br"),
        ({"k_e": None}, "k_e"),
    ])
    def test_missing_inputs_fail_validate(self, kw, match):
        with pytest.raises(ValueError, match=match):
            MODEL_REGISTRY["motor_thermal"].validate(_motor(**kw))

    def test_params_reject_non_positive(self):
        p = prepare_motor_thermal(_motor())
        with pytest.raises(ValueError, match="psi_ref"):
            dataclasses.replace(p, psi_ref=0.0)

    def test_params_reject_unknown_limit_node(self):
        p = prepare_motor_thermal(_motor())
        with pytest.raises(ValueError, match="unknown node 'iron'"):
            dataclasses.replace(p, limits={"iron": TempLimit(200.0)})


class TestCoupled:

    def test_fixed_point_is_self_consistent(self):
        p = prepare_motor_thermal(_motor())
        r = coupled_steady(p, 100.0, 2.0, 40.0)
        G, g_s, _ = assemble_network(_TN)
        T = steady_temps(G, g_s, np.array([r["p_cu"], r["p_fe"], 0.0]), 40.0)
        assert [r["T_winding"], r["T_iron"], r["T_magnet"]] == pytest.approx(list(T))
        assert r["T_magnet"] == pytest.approx(r["T_iron"], rel=1e-12)
        assert r["residual_K"] <= 0.01
        assert r["p_elec"] == pytest.approx(r["p_shaft"] + r["p_cu"] + r["p_fe"])
        assert r["psi_ratio"] == pytest.approx(hypot(r["psi_f_op"] + p.op.L_d * r["i_d"],
                                                     p.op.L_q * r["i_q"]) / p.psi_ref)

    def test_hotter_coolant_hotter_winding_and_more_copper_loss(self):
        p = prepare_motor_thermal(_motor())
        cold, hot = (coupled_steady(p, 100.0, 2.0, T) for T in (20.0, 80.0))
        assert hot["T_winding"] > cold["T_winding"] + 59.0
        assert hot["p_cu"] > cold["p_cu"]
        assert hot["R_s_op"] > cold["R_s_op"]

    def test_derating_range_exit_is_runaway(self):
        # Reachable cold, no fixed point: a huge R_FeS gives a first-step
        # rise past the linear alpha_Br range (psi_f <= 0 near 850 degC).
        m = _motor(thermal_network=ThermalNetwork(
            G_WFe=42.5, G_FeS=0.01, G_FeM=8.0, C_W=1.0, C_Fe=1.0, C_M=1.0))
        p = prepare_motor_thermal(m)
        with pytest.raises(ThermalRunaway, match="derating range") as e:
            coupled_steady(p, 100.0, 2.0, 20.0)
        assert e.value.iteration == 2 and e.value.T[2] > 800.0
        assert isinstance(e.value, ValueError)  # ValueError callers keep working

    def test_growing_residual_is_runaway_not_non_convergence(self):
        # G_FeS = 0.1: loop gain above 1, residual grows every iterate.
        p = prepare_motor_thermal(_motor(thermal_network=dataclasses.replace(_TN, G_FeS=0.1)))
        with pytest.raises(ThermalRunaway, match="residual grew") as e:
            coupled_steady(p, 100.0, 2.0, 20.0, max_iter=4)
        assert e.value.iteration == 4 and e.value.residual_K > 0.0

    def test_non_convergence_raises_with_state(self):
        p = prepare_motor_thermal(_motor())
        with pytest.raises(NoConvergence, match="no convergence") as e:
            coupled_steady(p, 100.0, 2.0, 40.0, max_iter=1)
        assert "residual" in str(e.value)
        assert not isinstance(e.value, ThermalRunaway)
        assert e.value.iteration == 1 and len(e.value.T) == 3

    def test_cold_infeasible_is_infeasible_at_iteration_1(self):
        p = prepare_motor_thermal(_motor())
        with pytest.raises(InfeasibleDemand) as e:
            coupled_steady(p, 100.0, 1e4, 40.0)
        assert e.value.iteration == 1 and e.value.T == (40.0, 40.0, 40.0)
        assert e.value.residual_K is None
        assert e.value.at_coolant is True

    def test_infeasible_on_the_climb_is_not_at_coolant(self, i3):
        # i3 1000 rpm / 240 N.m / 60 degC: feasible cold, the magnet heats
        # the demand out of the envelope at iteration 3.
        with pytest.raises(InfeasibleDemand) as e:
            coupled_steady(prepare_motor_thermal(i3), 1000 * _RPM, 240.0, 60.0)
        assert e.value.iteration == 3 and e.value.at_coolant is False

    def test_errors_pickle_with_attributes(self):
        for cls in (InfeasibleDemand, ThermalRunaway, NoConvergence):
            err = cls("msg", iteration=4, T=(1.0, 2.0, 3.0), residual_K=0.5)
            back = pickle.loads(pickle.dumps(err))
            assert type(back) is cls and str(back) == "msg"
            assert (back.iteration, back.T, back.residual_K, back.at_coolant) == (
                4, (1.0, 2.0, 3.0), 0.5, False)
        err.add_note("extra")
        err.tag = 5
        back = pickle.loads(pickle.dumps(err))
        assert back.__notes__ == ["extra"] and back.tag == 5

    def test_budget_exhausted_while_contracting_is_no_convergence(self):
        # Base motor converges in ~6 iterations with shrinking residuals:
        # a 3-iteration budget ends without the runaway signature.
        p = prepare_motor_thermal(_motor())
        with pytest.raises(NoConvergence) as e:
            coupled_steady(p, 100.0, 2.0, 40.0, max_iter=3)
        assert not isinstance(e.value, ThermalRunaway) and e.value.iteration == 3

    @pytest.mark.parametrize("T_c", [float("nan"), float("inf"), -float("inf")])
    def test_non_finite_coolant_is_input_error(self, T_c):
        p = prepare_motor_thermal(_motor())
        with pytest.raises(ValueError, match="T_coolant must be finite") as e:
            coupled_steady(p, 100.0, 2.0, T_c)
        assert not isinstance(e.value, CoupledSolveError)

    def test_coolant_outside_derating_range_is_input_error(self):
        # psi_f <= 0 near 850 degC with alpha_Br -0.0012.
        p = prepare_motor_thermal(_motor())
        with pytest.raises(ValueError, match="derating range") as e:
            coupled_steady(p, 100.0, 2.0, 900.0)
        assert not isinstance(e.value, CoupledSolveError)

    @pytest.mark.parametrize("n", [0, -1])
    def test_max_iter_below_one_rejected(self, n):
        with pytest.raises(ValueError, match="max_iter must be >= 1"):
            coupled_steady(prepare_motor_thermal(_motor()), 100.0, 2.0, 40.0, max_iter=n)

    @pytest.mark.parametrize("kw, match", [
        ({"tol_K": float("inf")}, "tol_K"), ({"tol_K": 0.0}, "tol_K"),
        ({"tol_K": float("nan")}, "tol_K"), ({"relax": 0.0}, "relax"),
        ({"relax": float("nan")}, "relax"), ({"relax": 1.5}, "relax"),
    ])
    def test_solver_controls_rejected(self, kw, match):
        with pytest.raises(ValueError, match=match) as e:
            coupled_steady(prepare_motor_thermal(_motor()), 100.0, 2.0, 40.0, **kw)
        assert not isinstance(e.value, CoupledSolveError)

    def test_converged_state_outside_derating_range_is_runaway(self):
        # psi_f reaches 0 at 220 degC (alpha_Br -0.005): from 219.999 degC
        # coolant the first update is under tol_K but lands the magnet past
        # 220 degC. The reported state is checked, not only the iterate.
        m = _motor(psi_f=1.0, R_s=0.001, L_d=0.01, L_q=0.01, alpha_Br=-0.005,
                   thermal_network=ThermalNetwork(G_WFe=1, G_FeS=1, G_FeM=1,
                                                  C_W=1, C_Fe=1, C_M=1),
                   drive=DriveParams(U_DC=100.0, MAX_I_S=10.0, W_REF=314.159265358979))
        with pytest.raises(ThermalRunaway, match="derating range") as e:
            coupled_steady(prepare_motor_thermal(m), 0.0, 1.5e-5, 219.999)
        assert e.value.iteration == 2 and e.value.T[2] > 220.0

    def test_converged_state_outside_envelope_is_infeasible(self):
        # Stiff network (first update ~1e-3 K, under tol_K) at the current
        # limit: tau_max = 1.5 n_p psi_f(40 degC) I_max = 5.856 N.m. Just
        # under it the demand is feasible at coolant but not at the
        # slightly hotter converged magnet, which is the reported state.
        tn = ThermalNetwork(G_WFe=1e5, G_FeS=1e5, G_FeM=1, C_W=1, C_Fe=1, C_M=1)
        p = prepare_motor_thermal(_motor(thermal_network=tn))
        tau_max = 1.5 * 2 * 0.1 * (1 - 0.0012 * 20) * 20.0
        with pytest.raises(InfeasibleDemand, match="converged") as e:
            coupled_steady(p, 100.0, tau_max * (1 - 5e-7), 40.0)
        assert e.value.iteration == 2 and not e.value.at_coolant
        assert coupled_steady(p, 100.0, tau_max * (1 - 1e-5), 40.0)["iterations"] == 1

    def test_budget_error_reports_the_last_evaluated_iterate(self):
        with pytest.raises(NoConvergence) as e:
            coupled_steady(prepare_motor_thermal(_motor()), 100.0, 2.0, 40.0, max_iter=1)
        assert e.value.at_coolant and e.value.T == (40.0, 40.0, 40.0)

    def test_budget_final_proposal_out_of_range_is_runaway_at_any_budget(self):
        # Standstill, G_FeS 1e-4: the first relaxed step puts the magnet
        # near 7520 degC. Budget 1 and 2 classify it the same way.
        m = _motor(psi_f=1.0, R_s=1.0, L_d=0.01, L_q=0.01,
                   thermal_network=ThermalNetwork(G_WFe=1, G_FeS=1e-4, G_FeM=1,
                                                  C_W=1, C_Fe=1, C_M=1),
                   drive=DriveParams(U_DC=1000.0, MAX_I_S=100.0, W_REF=314.159265358979))
        p = prepare_motor_thermal(m)
        errs = []
        for n in (1, 2):
            with pytest.raises(ThermalRunaway, match="derating range") as e:
                coupled_steady(p, 0.0, 3.0, 20.0, max_iter=n)
            errs.append(e.value)
        assert [x.iteration for x in errs] == [2, 2] and errs[0].T == errs[1].T
        assert errs[0].T[2] > 7000.0

    def test_motor_operating_temperatures_are_not_range_checked(self):
        # motor_thermal solves both nodes: a scenario magnet_temp past the
        # linear alpha_Br range, or an implausible winding limit, neither
        # blocks the solve nor moves it; the limit shows as headroom.
        base = coupled_steady(prepare_motor_thermal(_motor()), 100.0, 2.0, 40.0)
        for kw in ({"magnet_temp": 900.0}, {"winding_temp_limit": -300.0}):
            r = coupled_steady(prepare_motor_thermal(_motor(**kw)), 100.0, 2.0, 40.0)
            assert r["T_winding"] == base["T_winding"] and r["p_cu"] == base["p_cu"]
        r = coupled_steady(prepare_motor_thermal(_motor(magnet_temp=200.0, magnet_temp_max=150.0)),
                           100.0, 2.0, 40.0)
        assert r["headroom_magnet_K"] == pytest.approx(150.0 - base["T_magnet"])

    @pytest.mark.parametrize("cooling, expected", [(False, ThermalRunaway),
                                                   (True, NoConvergence)])
    def test_heating_is_every_node(self, monkeypatch, cooling, expected):
        # Scripted network response: residual doubles every iterate while the
        # magnet node cools (or heats). One cooling node is not runaway.
        import phasesweep.models.motor_thermal as mt
        T = np.full(3, 40.0)
        step = {"d": np.array([1.0, 1.0, -0.1 if cooling else 0.1])}

        def scripted(G, g_s, P, T_s):
            nonlocal T
            T_new = T + step["d"]
            T = T + 0.5 * step["d"]
            step["d"] = step["d"] * np.array([2.0, 2.0, 1.0])
            return T_new

        monkeypatch.setattr(mt, "steady_temps", scripted)
        with pytest.raises(expected) as e:
            coupled_steady(prepare_motor_thermal(_motor()), 100.0, 2.0, 40.0, max_iter=4)
        assert e.value.iteration == 4


class TestRunawayRule:

    def test_growing_and_heating(self):
        assert _runaway([1.0, 2.0, 3.0], [True, True, True])

    def test_shrinking(self):
        assert not _runaway([3.0, 2.0, 1.0], [True, True, True])

    def test_growing_with_one_node_not_heating(self):
        assert not _runaway([1.0, 2.0, 3.0], [True, False, True])

    def test_fewer_than_three(self):
        assert not _runaway([1.0, 2.0], [True, True])

    def test_only_the_last_three_count(self):
        assert _runaway([9.0, 1.0, 2.0, 3.0], [False, True, True, True])

    def test_stalled_residual_is_not_growth(self):
        assert not _runaway([2.0, 2.0, 2.0], [True, True, True])

    @pytest.mark.parametrize("resid", [[1.0, 3.0, 2.0], [1.0, 1.0, 2.0]])
    def test_growth_must_be_consecutive(self, resid):
        assert not _runaway(resid, [True, True, True])

    def test_runner_produces_and_runtime_guards(self):
        m = _motor()
        rc = RunConfig(motor=m, model="motor_thermal", w_mech=100.0, tau_ref=2.0,
                       T_coolant=40.0)
        r = MODEL_REGISTRY["motor_thermal"].fn(rc)
        assert set(r) == MODEL_REGISTRY["motor_thermal"].produces
        assert r == run_motor_thermal(rc)
        for missing in ("w_mech", "tau_ref", "T_coolant"):
            with pytest.raises(ValueError, match=f"RunConfig.{missing}"):
                run_motor_thermal(dataclasses.replace(rc, **{missing: None}))

    def test_run_id_moves_with_every_runtime_field_only(self):
        m = _motor()
        base = RunConfig(motor=m, model="motor_thermal", w_mech=100.0, tau_ref=2.0,
                         T_coolant=40.0)
        ids = {compute_run_id(dataclasses.replace(base, **kw))
               for kw in ({}, {"w_mech": 101.0}, {"tau_ref": 2.1}, {"T_coolant": 41.0})}
        assert len(ids) == 4
        assert compute_run_id(dataclasses.replace(base, n_theta=90)) == compute_run_id(base)
        no_wref = dataclasses.replace(m, drive=DriveParams(U_DC=540.0, MAX_I_S=20.0))
        assert compute_run_id(dataclasses.replace(base, motor=no_wref)) == compute_run_id(base)


class TestLimits:

    def test_no_limits_gives_none_verdicts(self):
        r = coupled_steady(prepare_motor_thermal(_motor()), 100.0, 2.0, 40.0)
        assert (r["headroom_winding_K"], r["headroom_magnet_K"],
                r["binding_limit"], r["within_s1"]) == (None, None, None, None)

    def test_headroom_from_the_motor_and_binding_node(self):
        m = _motor(winding_temp_limit=155.0, magnet_temp_max=60.0)
        r = coupled_steady(prepare_motor_thermal(m), 100.0, 2.0, 40.0)
        assert r["headroom_winding_K"] == pytest.approx(155.0 - r["T_winding"], abs=0)
        assert r["headroom_magnet_K"] == pytest.approx(60.0 - r["T_magnet"], abs=0)
        smaller = min(("winding", "magnet"), key=lambda n: r[f"headroom_{n}_K"])
        assert r["binding_limit"] == smaller
        assert r["within_s1"] is (min(r["headroom_winding_K"], r["headroom_magnet_K"]) >= 0)

    def test_over_limit_is_a_result_not_an_error(self):
        m = _motor(winding_temp_limit=40.5)  # the point rises ~1 K
        r = coupled_steady(prepare_motor_thermal(m), 100.0, 2.0, 40.0)
        assert r["headroom_winding_K"] < 0 and r["within_s1"] is False
        assert r["binding_limit"] == "winding"

    def test_limits_do_not_move_the_solve(self):
        # The limit's value is not a derating temperature for motor_thermal:
        # the solved state is identical with and without it.
        p = prepare_motor_thermal(_motor())
        base = coupled_steady(p, 100.0, 2.0, 40.0)
        lim = coupled_steady(p, 100.0, 2.0, 40.0,
                             limits={"winding": TempLimit(130.0), "magnet": TempLimit(90.0)})
        keys = set(base) - {"headroom_winding_K", "headroom_magnet_K",
                            "binding_limit", "within_s1"}
        assert {k: base[k] for k in keys} == {k: lim[k] for k in keys}
        assert lim["within_s1"] is True

    def test_override_replaces_and_none_drops(self):
        p = prepare_motor_thermal(_motor(winding_temp_limit=155.0, magnet_temp_max=150.0))
        r = coupled_steady(p, 100.0, 2.0, 40.0,
                           limits={"winding": TempLimit(170.0, "ETD", "assumed"),
                                   "magnet": None})
        assert r["headroom_winding_K"] == pytest.approx(170.0 - r["T_winding"])
        assert r["headroom_magnet_K"] is None and r["binding_limit"] == "winding"

    def test_basis_mismatch_rejected_unknown_unchecked(self):
        tn = dataclasses.replace(_TN, winding_basis="ETD")
        p = prepare_motor_thermal(_motor(thermal_network=tn))
        with pytest.raises(ValueError, match="basis 'hot_spot'"):
            coupled_steady(p, 100.0, 2.0, 40.0,
                           limits={"winding": TempLimit(180.0, "hot_spot")})
        for lim in (TempLimit(180.0, "ETD"), TempLimit(180.0)):
            coupled_steady(p, 100.0, 2.0, 40.0, limits={"winding": lim})
        with pytest.raises(ValueError, match="basis 'resistance'"):
            prepare_motor_thermal(_motor(thermal_network=tn, winding_temp_limit=155.0,
                                         winding_temp_limit_basis="resistance"))

    @pytest.mark.parametrize("lim", [TempLimit(200.0), None])
    def test_unknown_node_rejected(self, lim):
        with pytest.raises(ValueError, match="unknown node 'iron'"):
            coupled_steady(prepare_motor_thermal(_motor()), 100.0, 2.0, 40.0,
                           limits={"iron": lim})

    def test_mixed_sign_headroom_is_not_within(self):
        p = prepare_motor_thermal(_motor())
        r = coupled_steady(p, 100.0, 2.0, 40.0)
        r = coupled_steady(p, 100.0, 2.0, 40.0,
                           limits={"winding": TempLimit(r["T_winding"] + 5.0),
                                   "magnet": TempLimit(r["T_magnet"] - 5.0)})
        assert r["headroom_winding_K"] > 0 > r["headroom_magnet_K"]
        assert r["within_s1"] is False and r["binding_limit"] == "magnet"

    def test_zero_headroom_is_within(self):
        p = prepare_motor_thermal(_motor())
        r = coupled_steady(p, 100.0, 2.0, 40.0)
        r = coupled_steady(p, 100.0, 2.0, 40.0,
                           limits={"winding": TempLimit(r["T_winding"])})
        assert r["headroom_winding_K"] == 0.0 and r["within_s1"] is True

    def test_tie_reports_winding(self):
        p = prepare_motor_thermal(_motor())
        r = coupled_steady(p, 100.0, 2.0, 40.0)
        r = coupled_steady(p, 100.0, 2.0, 40.0,
                           limits={"winding": TempLimit(r["T_winding"] + 5.0),
                                   "magnet": TempLimit(r["T_magnet"] + 5.0)})
        assert r["headroom_winding_K"] == r["headroom_magnet_K"]  # exact tie
        assert r["binding_limit"] == "winding"

    def test_registered_runner_reports_motor_limits(self):
        m = _motor(winding_temp_limit=155.0)
        r = run_motor_thermal(RunConfig(motor=m, model="motor_thermal", w_mech=100.0,
                                        tau_ref=2.0, T_coolant=40.0))
        assert r["headroom_winding_K"] == pytest.approx(155.0 - r["T_winding"])
        assert r["headroom_magnet_K"] is None


class TestLimitFields:

    def test_templimit_vocabulary(self):
        with pytest.raises(ValueError, match="basis"):
            TempLimit(150.0, basis="sensor")
        with pytest.raises(ValueError, match="source"):
            TempLimit(150.0, source="guessed")
        for bad in (float("nan"), float("inf"), -float("inf")):
            with pytest.raises(ValueError, match="finite"):
                TempLimit(bad)

    @pytest.mark.parametrize("kw, match", [
        ({"winding_temp_limit_basis": "ETD"}, "winding_temp_limit is not"),
        ({"magnet_temp_max_source": "published"}, "magnet_temp_max is not"),
        ({"winding_temp_limit": 155.0, "winding_temp_limit_basis": "rtd"}, "basis"),
        ({"magnet_temp_max": 150.0, "magnet_temp_max_source": "datasheet"}, "source"),
    ])
    def test_motor_rejects_bad_tags(self, kw, match):
        with pytest.raises(ValueError, match=match):
            _motor(**kw)

    def test_temp_limits_property(self):
        m = _motor(winding_temp_limit=155.0, winding_temp_limit_basis="ETD",
                   winding_temp_limit_source="published", magnet_temp_max=150.0)
        assert m.temp_limits == {"winding": TempLimit(155.0, "ETD", "published"),
                                 "magnet": TempLimit(150.0)}
        assert _motor().temp_limits == {}

    def test_hashing_limit_append_only_tags_unhashed(self):
        m = _motor()
        assert _motor(magnet_temp_max=150.0).config_id != m.config_id
        tagged = _motor(winding_temp_limit=155.0, winding_temp_limit_basis="ETD",
                        winding_temp_limit_source="assumed")
        assert tagged.config_id == _motor(winding_temp_limit=155.0).config_id
        assert Motor.from_dict(tagged.to_dict()) == tagged
        tn = dataclasses.replace(_TN, winding_basis="ETD")
        assert "winding_basis" not in _TN.to_dict()
        assert ThermalNetwork.from_dict(tn.to_dict()) == tn
        with pytest.raises(ValueError, match="winding_basis"):
            dataclasses.replace(_TN, winding_basis="guess")

    def test_loader_reads_thermal_tags(self):
        d = tomllib.loads((REPO_ROOT / "motors/bmw_i3.toml").read_text())
        d["thermal"] = {**d.get("thermal", {}), "winding_temp_limit": 170.0,
                        "winding_temp_limit_basis": "ETD",
                        "winding_temp_limit_source": "assumed",
                        "magnet_temp_max": 150.0}
        d["thermal_network"]["winding_basis"] = "ETD"
        m = motor_from_dict(d, label="i3", default_name="i3")
        assert m.temp_limits["winding"] == TempLimit(170.0, "ETD", "assumed")
        assert m.thermal_network is not None and m.thermal_network.winding_basis == "ETD"


# ---------------------------------------------------------------------------
# BMW i3 anchor: Burress 2017 continuous test at 7000 rpm
# ---------------------------------------------------------------------------

class TestBmwI3Anchor:
    """Pins the fit of `[thermal_network] G_FeS` to the three digitized
    winding plateaus (scripts/fit_i3_thermal_network.py, coolant assumed
    60 degC; the opening no-load winding reading of 63.2 degC bounds the
    coolant from above if the motor is at equilibrium there; assumed
    sensitivity bracket 55-63). The fit absorbs every loss the
    model does not carry (AC copper, PWM, mechanical, the uncalibrated
    [iron] B_core), so this is a pin of the fit, not a validation of the
    network: the residual pattern (-3.3, -3.8, +4.1 K, predicted minus
    measured) is the loss-model gap — the model rises 30.5 K from 25 to
    75 kW where the measurement rises 23.1 K — and at the unfitted
    no-load segment the model sits +10.7 K above the trace on its own
    no-load iron loss."""

    RPM = 7000.0
    W = RPM * _RPM
    # (shaft power kW, digitized plateau degC, prediction at the fitted block)
    PLATEAUS = ((25.0, 83.4, 80.05), (50.0, 95.3, 91.52), (75.0, 106.5, 110.54))
    COOLANT = 60.0

    @pytest.fixture(scope="class")
    def params(self, i3):
        return prepare_motor_thermal(i3)

    @pytest.mark.parametrize("P_kW, measured, predicted", PLATEAUS)
    def test_plateaus_within_the_fit_residual(self, params, P_kW, measured, predicted):
        r = coupled_steady(params, self.W, P_kW * 1e3 / self.W, self.COOLANT)
        assert r["op_source"] == "flux_map"
        assert r["iterations"] < 50
        assert r["T_winding"] == pytest.approx(predicted, abs=0.05)
        assert abs(r["T_winding"] - measured) < 4.5
        assert r["T_magnet"] == pytest.approx(r["T_iron"], rel=1e-12)

    def test_iron_loss_follows_the_flux_ratio(self, params):
        low, high = (coupled_steady(params, self.W, P * 1e3 / self.W, self.COOLANT)
                     for P in (25e3 / 1e3, 75e3 / 1e3))
        assert low["regime"] == "mtpa" and high["regime"] == "field_weakening"
        assert 1.0 < low["psi_ratio"] < high["psi_ratio"] < 1.1
        assert high["p_fe"] > low["p_fe"]

    def test_monotone_in_coolant_over_the_bracket(self, params):
        T = [coupled_steady(params, self.W, 75e3 / self.W, T_c)["T_winding"]
             for T_c in (55.0, 60.0, 65.0)]
        assert T[0] < T[1] < T[2]
        assert T[2] - T[0] == pytest.approx(11.3, abs=0.3)

    def test_bus_override_default_is_exact_and_binds_only_in_fw(self, params):
        # Plan finding 1: an 0.8 % bus drop (2.84 V) moves the field-weakening
        # point and leaves MTPA untouched.
        U = params.op.U_DC
        fw, mtpa = 75e3 / self.W, 25e3 / self.W
        assert (coupled_steady(params, self.W, fw, self.COOLANT)
                == coupled_steady(params, self.W, fw, self.COOLANT, U_DC=U))
        base, low = (coupled_steady(params, self.W, fw, self.COOLANT, U_DC=u)
                     for u in (U, U - 2.84))
        assert low["regime"] == "field_weakening"
        assert low["T_winding"] > base["T_winding"]
        assert low["p_cu"] > base["p_cu"]
        base, low = (coupled_steady(params, self.W, mtpa, self.COOLANT, U_DC=u)
                     for u in (U, U - 2.84))
        assert base["regime"] == "mtpa"
        assert low["T_winding"] == pytest.approx(base["T_winding"], abs=1e-9)
        for bad in (0.0, float("nan")):
            with pytest.raises(ValueError, match="U_DC override"):
                coupled_steady(params, self.W, mtpa, self.COOLANT, U_DC=bad)

    def test_block_provenance_and_modes(self, i3):
        # The fitted block's basis (provenance, unhashed) and the modes of
        # C^-1 G (scripts/check_i3_thermal_transient.py marches them).
        tn = i3.thermal_network
        assert tn.winding_basis == "ETD" and tn.magnet_basis is None
        G, _, C = assemble_network(tn)
        lam, _, _ = _modes(G, C)
        assert sorted(1.0 / lam) == pytest.approx([39.7, 205.4, 1014.9], abs=0.1)

    def test_registry_runner_and_run_id(self, i3):
        rc = RunConfig(motor=i3, model="motor_thermal", w_mech=self.W,
                       tau_ref=75e3 / self.W, T_coolant=self.COOLANT)
        r = MODEL_REGISTRY["motor_thermal"].fn(rc)
        assert set(r) == MODEL_REGISTRY["motor_thermal"].produces
        assert r["T_winding"] == pytest.approx(110.54, abs=0.05)
        assert compute_run_id(rc) != compute_run_id(dataclasses.replace(rc, T_coolant=55.0))
