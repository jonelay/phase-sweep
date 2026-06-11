"""Unit tests for build_sim() and extract_metrics()."""

import dataclasses
from pathlib import Path

import numpy as np
import pytest
from motulator.drive.utils import Step

from phasesweep.configs import load_motor
from phasesweep.geometry import default_inrunner
from phasesweep.motor import DriveParams, Motor
from phasesweep.sim import SimPlan, build_sim, extract_metrics, plan_sim, plan_torque_sim
from phasesweep.solver_params import DriveSimParams, prepare_drive_sim

MOTOR = Motor(
    name="test-SPMSM", geometry=dataclasses.replace(default_inrunner(), n_slots=12),
    n_p=2, R_s=0.2, L_d=4e-3, L_q=4e-3, psi_f=0.1, J=0.002,
    N=50, k_w=0.966, L_stk=0.10, I_rated=10.0, coils_series=1,
)
PARAMS = prepare_drive_sim(MOTOR)
PLAN = plan_sim(PARAMS)
W_REF = PARAMS.drive.W_REF


def test_build_sim_returns_simulation():
    sim = build_sim(PARAMS, PLAN)
    from motulator.drive.model import Simulation
    assert isinstance(sim, Simulation)


@pytest.fixture(scope="module")
def short_result():
    short_plan = dataclasses.replace(PLAN, t_stop=0.5)
    sim = build_sim(PARAMS, short_plan)
    return short_plan, sim.simulate(t_stop=0.5)


@pytest.fixture(scope="module")
def full_result():
    sim = build_sim(PARAMS, PLAN)
    return sim.simulate(t_stop=PLAN.t_stop)


@pytest.fixture(scope="module")
def full_result_heavy_load():
    heavy_plan = plan_sim(PARAMS, load_fraction=1.0)
    sim = build_sim(PARAMS, heavy_plan)
    return heavy_plan, sim.simulate(t_stop=heavy_plan.t_stop)


def test_extract_metrics_keys(short_result):
    short_plan, res = short_result
    m = extract_metrics(res, plan=short_plan, w_ref=W_REF)
    assert set(m.keys()) == {"t_settle", "i_ss", "speed_droop", "tau_peak"}


def test_extract_metrics_i_ss_positive(full_result):
    m = extract_metrics(full_result, plan=PLAN, w_ref=W_REF)
    assert m["i_ss"] > 0


def test_extract_metrics_tau_peak_positive(short_result):
    short_plan, res = short_result
    m = extract_metrics(res, plan=short_plan, w_ref=W_REF)
    assert m["tau_peak"] > 0


def test_extract_metrics_short_sim_nan_contract(short_result, recwarn):
    """Short sim (t_stop < load_time) must return NaN without numpy warnings.

    The short fixture stops before the load step, so pre_load / post_load
    windows are empty. Empty-window metrics must be explicit NaN, not silent
    mean-of-empty-slice artifacts.
    """
    short_plan, res = short_result
    m = extract_metrics(res, plan=short_plan, w_ref=W_REF)
    assert np.isnan(m["speed_droop"])
    assert np.isfinite(m["i_ss"])
    assert not any(
        "empty slice" in str(w.message) or "invalid value" in str(w.message)
        for w in recwarn.list
    )


def test_extract_metrics_settle_time_finite(full_result):
    m = extract_metrics(full_result, plan=PLAN, w_ref=W_REF)
    assert np.isfinite(m["t_settle"])
    assert m["t_settle"] > 0


def test_extract_metrics_all_positive(full_result):
    m = extract_metrics(full_result, plan=PLAN, w_ref=W_REF)
    for key in ("t_settle", "i_ss", "speed_droop", "tau_peak"):
        assert np.isfinite(m[key]), f"{key} is not finite"
        assert m[key] > 0, f"{key} is not positive"


def test_speed_droop_increases_with_load(full_result, full_result_heavy_load):
    m_default = extract_metrics(full_result, plan=PLAN, w_ref=W_REF)
    heavy_plan, heavy_res = full_result_heavy_load
    m_heavy = extract_metrics(heavy_res, plan=heavy_plan, w_ref=W_REF)
    assert m_heavy["speed_droop"] > m_default["speed_droop"]


def test_i_ss_increases_with_load(full_result, full_result_heavy_load):
    m_default = extract_metrics(full_result, plan=PLAN, w_ref=W_REF)
    heavy_plan, heavy_res = full_result_heavy_load
    m_heavy = extract_metrics(heavy_res, plan=heavy_plan, w_ref=W_REF)
    assert m_heavy["i_ss"] > m_default["i_ss"]


def test_build_sim_derived_psi_f():
    """Bug regression: psi_f=None + B_rem -> derive -> simulate -> valid metrics."""
    from phasesweep.geometry import default_inrunner
    from phasesweep.motor import Motor

    motor = Motor(
        name="test_derived", geometry=default_inrunner(),
        n_p=2, R_s=0.2, L_d=4e-3, L_q=4e-3, psi_f=None, B_rem=0.5,
        J=0.002, N=10, k_w=0.966, L_stk=0.10, coils_series=1,
    )
    params = prepare_drive_sim(motor)
    assert params.psi_f > 0
    plan = plan_sim(params)
    sim = build_sim(params, plan)
    res = sim.simulate(t_stop=plan.t_stop)
    m = extract_metrics(res, plan=plan, w_ref=params.drive.W_REF)
    assert m["tau_peak"] > 0


# ---------------------------------------------------------------------------
# SimPlan + plan_sim
# ---------------------------------------------------------------------------

class TestPlanSim:

    def test_plan_sim_returns_simplan(self):
        plan = plan_sim(PARAMS)
        assert isinstance(plan, SimPlan)
        assert plan.alpha_s > 0
        assert plan.alpha_c > 0
        assert plan.T_s > 0

    def test_plan_sim_timing_positive(self):
        plan = plan_sim(PARAMS)
        assert plan.load_torque > 0
        assert plan.load_time > plan.speed_step_time
        assert plan.t_stop > plan.load_time
        assert plan.tau_m > 0

    def test_plan_sim_zero_r_s_does_not_crash(self):
        params = dataclasses.replace(PARAMS, R_s=0.0)
        plan = plan_sim(params)
        assert plan.T_s > 0
        assert np.isfinite(plan.alpha_c)

    def test_plan_sim_load_fraction(self):
        plan_half = plan_sim(PARAMS, load_fraction=0.5)
        plan_full = plan_sim(PARAMS, load_fraction=1.0)
        assert plan_full.load_torque == pytest.approx(2 * plan_half.load_torque)

    def test_plan_sim_accel_window_after_speed_step(self):
        plan = plan_sim(PARAMS)
        assert plan.accel_window[0] == plan.speed_step_time
        assert plan.accel_window[1] > plan.accel_window[0]

    def test_plan_sim_scales_with_motor_size(self):
        """Small motor gets shorter timing than large motor."""
        small = Motor(
            name="small", geometry=dataclasses.replace(default_inrunner(), n_slots=12),
            n_p=2, R_s=5.0, L_d=0.5e-3, L_q=0.5e-3, psi_f=0.01, J=1e-5,
            N=50, k_w=0.966, L_stk=0.02,
        )
        large = Motor(
            name="large", geometry=dataclasses.replace(default_inrunner(), n_slots=12),
            n_p=2, R_s=0.05, L_d=10e-3, L_q=10e-3, psi_f=0.5, J=0.1,
            N=50, k_w=0.966, L_stk=0.20,
        )
        plan_s = plan_sim(prepare_drive_sim(small))
        plan_l = plan_sim(prepare_drive_sim(large))
        assert plan_s.t_stop < plan_l.t_stop
        assert plan_s.load_torque < plan_l.load_torque

    def test_plan_sim_to_dict_roundtrip(self):
        plan = plan_sim(PARAMS)
        d = plan.to_dict()
        plan2 = SimPlan.from_dict(d)
        assert plan2.load_torque == plan.load_torque
        assert plan2.t_stop == plan.t_stop
        assert plan2.accel_window == plan.accel_window
        assert plan2.alpha_s == plan.alpha_s
        assert plan2.alpha_c == plan.alpha_c
        assert plan2.T_s == plan.T_s

    def test_plan_sim_alpha_s_scales_with_inertia(self):
        small_j = Motor(
            name="small-J", geometry=dataclasses.replace(default_inrunner(), n_slots=12),
            n_p=2, R_s=0.2, L_d=4e-3, L_q=4e-3, psi_f=0.1, J=1e-5,
            N=50, k_w=0.966, L_stk=0.10, coils_series=1,
        )
        large_j = Motor(
            name="large-J", geometry=dataclasses.replace(default_inrunner(), n_slots=12),
            n_p=2, R_s=0.2, L_d=4e-3, L_q=4e-3, psi_f=0.1, J=0.01,
            N=50, k_w=0.966, L_stk=0.10, coils_series=1,
        )
        plan_s = plan_sim(prepare_drive_sim(small_j))
        plan_l = plan_sim(prepare_drive_sim(large_j))
        assert plan_s.alpha_s > plan_l.alpha_s

    def test_plan_sim_override_with_replace(self):
        plan = plan_sim(PARAMS)
        overridden = dataclasses.replace(plan, load_torque=99.0)
        assert overridden.load_torque == 99.0
        assert overridden.t_stop == plan.t_stop

    def test_plan_sim_controller_tuning_bounds(self):
        plan = plan_sim(PARAMS)
        assert 2 * np.pi * 200 <= plan.alpha_c <= 2 * np.pi * 5000
        assert 10e-6 <= plan.T_s <= 125e-6
        assert plan.alpha_c * plan.T_s < 0.3  # stability invariant

    def test_plan_sim_T_s_scales_with_tau_e(self):
        """Motor with small L_d/R_s gets shorter T_s."""
        fast_e = Motor(
            name="fast-electric", geometry=dataclasses.replace(default_inrunner(), n_slots=12),
            n_p=2, R_s=1.0, L_d=50e-6, L_q=50e-6, psi_f=0.1, J=0.002,
            N=50, k_w=0.966, L_stk=0.10, coils_series=1,
        )  # tau_e = 50us
        slow_e = Motor(
            name="slow-electric", geometry=dataclasses.replace(default_inrunner(), n_slots=12),
            n_p=2, R_s=0.2, L_d=4e-3, L_q=4e-3, psi_f=0.1, J=0.002,
            N=50, k_w=0.966, L_stk=0.10, coils_series=1,
        )  # tau_e = 20ms
        plan_fast = plan_sim(prepare_drive_sim(fast_e))
        plan_slow = plan_sim(prepare_drive_sim(slow_e))
        assert plan_fast.T_s < plan_slow.T_s
        assert plan_fast.alpha_c > plan_slow.alpha_c

    def test_plan_sim_from_dict_legacy_compat(self):
        """Legacy serialized SimPlan missing controller fields."""
        old = {
            "load_torque": 1.0, "load_time": 0.5, "t_stop": 1.0,
            "speed_step_time": 0.05, "settle_threshold": 0.05,
            "ss_window": 0.1, "droop_window": 0.1,
            "accel_window": [0.05, 0.3], "tau_m": 0.1,
        }
        plan = SimPlan.from_dict(old)
        assert plan.alpha_s == pytest.approx(2 * np.pi * 4)
        assert plan.alpha_c == pytest.approx(2 * np.pi * 200)
        assert plan.T_s == 125e-6


# ---------------------------------------------------------------------------
# Full pipeline: plan_sim -> build_sim -> extract_metrics
# ---------------------------------------------------------------------------

class TestPlanSimIntegration:

    @pytest.fixture(scope="class")
    def plan_result(self):
        plan = plan_sim(PARAMS)
        sim = build_sim(PARAMS, plan)
        res = sim.simulate(t_stop=plan.t_stop)
        return plan, res

    def test_metrics_finite_and_positive(self, plan_result):
        plan, res = plan_result
        m = extract_metrics(res, plan=plan, w_ref=W_REF)
        for key in ("t_settle", "i_ss", "speed_droop", "tau_peak"):
            assert np.isfinite(m[key]), f"{key} is not finite"
            assert m[key] > 0, f"{key} is not positive"

    def test_heavy_load_increases_droop(self, plan_result):
        plan_default, res_default = plan_result
        plan_heavy = plan_sim(PARAMS, load_fraction=1.0)
        sim_heavy = build_sim(PARAMS, plan_heavy)
        res_heavy = sim_heavy.simulate(t_stop=plan_heavy.t_stop)
        m_default = extract_metrics(res_default, plan=plan_default, w_ref=W_REF)
        m_heavy = extract_metrics(res_heavy, plan=plan_heavy, w_ref=W_REF)
        assert m_heavy["speed_droop"] > m_default["speed_droop"]


# ---------------------------------------------------------------------------
# Actuator regression: adaptive alpha_s prevents motulator crash
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestActuatorRegression:

    @pytest.fixture(scope="class", params=["actuator_aluminum_rotor", "actuator_steel_rotor"])
    def actuator_metrics(self, request):
        motor_path = Path(__file__).parent.parent / "motors" / f"{request.param}.toml"
        if not motor_path.exists():
            pytest.skip(f"{motor_path} not found")
        motor = load_motor(motor_path)
        params = prepare_drive_sim(motor)
        plan = plan_sim(params)
        sim = build_sim(params, plan)
        res = sim.simulate(t_stop=plan.t_stop)
        return extract_metrics(res, plan=plan, w_ref=params.drive.W_REF)

    def test_actuator_sim_completes(self, actuator_metrics):
        for key in ("t_settle", "i_ss", "speed_droop", "tau_peak"):
            assert np.isfinite(actuator_metrics[key]), f"{key} is not finite"
            assert actuator_metrics[key] > 0, f"{key} is not positive"


# ---------------------------------------------------------------------------
# Torque-control mode
# ---------------------------------------------------------------------------

class TestPlanTorqueSim:

    def test_returns_simplan(self):
        plan = plan_torque_sim(PARAMS, tau_ref=0.5)
        assert isinstance(plan, SimPlan)

    def test_defaults_load_to_tau_ref(self):
        plan = plan_torque_sim(PARAMS, tau_ref=0.5)
        assert plan.load_torque == 0.5

    def test_load_override(self):
        plan = plan_torque_sim(PARAMS, tau_ref=0.5, load_torque=0.0)
        assert plan.load_torque == 0.0

    def test_tau_ref_nonpositive_raises(self):
        with pytest.raises(ValueError, match="tau_ref"):
            plan_torque_sim(PARAMS, tau_ref=0.0)

    def test_step_time_and_window_propagated(self):
        plan = plan_torque_sim(PARAMS, tau_ref=0.5, step_time=0.1, ss_window=0.08)
        assert plan.load_time == 0.1
        assert plan.ss_window == 0.08
        assert plan.t_stop > plan.load_time + plan.ss_window

    def test_controller_tuning_bounds(self):
        plan = plan_torque_sim(PARAMS, tau_ref=0.5)
        assert 2 * np.pi * 200 <= plan.alpha_c <= 2 * np.pi * 5000
        assert 10e-6 <= plan.T_s <= 125e-6


class TestBuildSimTorqueMode:

    def test_torque_ref_returns_simulation(self):
        plan = plan_torque_sim(PARAMS, tau_ref=0.5)
        torque_ref = Step(step_time=plan.load_time, step_value=0.5, initial_value=0)
        sim = build_sim(PARAMS, plan, torque_ref=torque_ref)
        from motulator.drive.model import Simulation
        assert isinstance(sim, Simulation)

    def test_torque_mode_runs(self):
        """End-to-end: torque step simulates without crash, produces nonzero current."""
        plan = plan_torque_sim(PARAMS, tau_ref=0.3)
        torque_ref = Step(step_time=plan.load_time, step_value=0.3, initial_value=0)
        sim = build_sim(PARAMS, plan, torque_ref=torque_ref)
        res = sim.simulate(t_stop=plan.t_stop)
        i_s = np.abs(res.mdl.machine.i_s_ab)
        t = res.mdl.t
        ss_mask = t >= (plan.t_stop - plan.ss_window)
        assert ss_mask.any()
        assert float(np.mean(i_s[ss_mask])) > 0

    def test_speed_mode_unchanged_by_torque_addition(self):
        """Speed-mode call without torque_ref produces same result as before refactor."""
        sim = build_sim(PARAMS, PLAN)
        res = sim.simulate(t_stop=PLAN.t_stop)
        m = extract_metrics(res, plan=PLAN, w_ref=W_REF)
        assert m["i_ss"] > 0
        assert m["tau_peak"] > 0


class TestTorqueAnchor:
    """Analytical anchor for torque-mode cross-checks against an external
    plant model.

    Scenario: torque step = 3 Nm at id*=0, steady state.
    Target: |i_s| = tau / (1.5 * n_p * psi_f) = 0.549 A within 3%
    (loose enough to accommodate a switched-PWM external reference).
    """

    def test_steady_state_current_matches_analytical(self):
        params = DriveSimParams(
            n_p=2,
            R_s=5.2, L_d=0.063, L_q=0.133,
            psi_f=1.819, J=0.0011,
            drive=DriveParams(U_DC=400.0, MAX_I_S=5.0),
        )
        tau_ref = 3.0
        plan = plan_torque_sim(params, tau_ref=tau_ref)
        torque_ref = Step(step_time=plan.load_time, step_value=tau_ref, initial_value=0)
        sim = build_sim(params, plan, torque_ref=torque_ref)
        res = sim.simulate(t_stop=plan.t_stop)

        t = res.mdl.t
        i_s = np.abs(res.mdl.machine.i_s_ab)
        ss_mask = t >= (plan.t_stop - plan.ss_window)
        i_ss = float(np.mean(i_s[ss_mask]))

        analytical = tau_ref / (1.5 * params.n_p * params.psi_f)
        assert abs(i_ss - analytical) / analytical < 0.03
