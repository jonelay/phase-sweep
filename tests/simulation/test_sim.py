"""Unit tests for build_sim() and extract_metrics()."""

from importlib.util import find_spec

import pytest

if not find_spec("motulator"):
    pytest.skip("requires phasesweep[sim] (motulator not installed)", allow_module_level=True)

import dataclasses
from pathlib import Path

import numpy as np
from motulator.drive.utils import Step

from phasesweep.machines.configs import load_motor
from phasesweep.machines.geometry import default_inrunner
from phasesweep.machines.motor import DriveParams, Motor
from phasesweep.simulation.sim import (
    SimPlan,
    build_sim,
    extract_metrics,
    extract_waveforms,
    plan_sim,
    plan_torque_sim,
)
from phasesweep.solver_params import DriveSimParams, prepare_drive_sim

MOTOR = Motor(
    name="test-SPMSM", geometry=dataclasses.replace(default_inrunner(), n_slots=12),
    n_p=2, R_s=0.2, L_d=4e-3, L_q=4e-3, psi_f=0.1, J=0.002,
    N=50, k_w=0.966, L_stk=0.10, I_rated=10.0, coils_series=1,
    drive=DriveParams(U_DC=540.0, MAX_I_S=20.0, W_REF=314.159265358979),
)
PARAMS = prepare_drive_sim(MOTOR)
PLAN = plan_sim(PARAMS)
W_REF = PARAMS.drive.W_REF


def test_build_sim_returns_simulation():
    sim = build_sim(PARAMS, PLAN)
    from motulator.drive.model import Simulation
    assert isinstance(sim, Simulation)


def test_build_sim_rejects_reverse_salient():
    """motulator's MTPA locus is silently wrong for L_d > L_q (zero- or
    negative-torque root) — build_sim must refuse until fixed upstream."""
    params = dataclasses.replace(PARAMS, L_d=8e-3, L_q=4e-3)
    with pytest.raises(ValueError, match="reverse-salient"):
        build_sim(params)


def test_build_sim_rejects_missing_drive_fields():
    """DriveSimParams validates motor constants but not drive fields — a
    directly constructed instance must not carry U_DC=None into
    motulator's VoltageSourceConverter."""
    params = dataclasses.replace(
        PARAMS, drive=DriveParams(MAX_I_S=20.0, W_REF=100.0))
    with pytest.raises(ValueError, match="U_DC"):
        build_sim(params)


def test_plan_sim_rejects_missing_drive_fields():
    """plan_sim's bare assert vanished under python -O — must be a
    ValueError like prepare_drive_sim's."""
    params = dataclasses.replace(PARAMS, drive=DriveParams())
    with pytest.raises(ValueError, match="U_DC, MAX_I_S, W_REF"):
        plan_sim(params)


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


def test_extract_waveforms_downsampled_and_aligned(short_result):
    """Dashboard sim-waveforms traces: capped length, equal
    lengths across quantities, monotonic time, |i_s| non-negative."""
    _short_plan, res = short_result
    wf = extract_waveforms(res, max_points=500)
    assert set(wf) == {"t_list", "w_M_list", "tau_M_list", "i_s_abs_list"}
    n = len(wf["t_list"])
    assert 0 < n <= 500
    assert all(len(v) == n for v in wf.values())
    assert wf["t_list"] == sorted(wf["t_list"])
    assert all(i >= 0 for i in wf["i_s_abs_list"])


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


def _mock_result(t, w_M):
    from types import SimpleNamespace
    n = len(t)
    return SimpleNamespace(mdl=SimpleNamespace(
        t=np.array(t),
        mechanics=SimpleNamespace(w_M=np.array(w_M)),
        machine=SimpleNamespace(
            i_s_ab=np.ones(n, dtype=complex),
            tau_M=np.ones(n),
        ),
    ))


_SETTLE_PLAN = SimPlan(
    alpha_s=100, alpha_c=100, T_s=1e-4, speed_step_time=0.2,
    load_time=1.5, load_torque=0.1, t_stop=2.0, tau_m=0.01,
    settle_threshold=0.05, ss_window=0.2, droop_window=0.3,
    accel_window=(0.2, 0.8),
)


def test_settle_time_underdamped():
    """First entry is at t=0.3, but signal leaves band and re-enters at t=0.7."""
    t = np.linspace(0.0, 2.0, 2001)
    w_ref = 100.0
    w = np.where(t < 0.2, 0.0, w_ref)
    overshoot = (t >= 0.3) & (t <= 0.7)
    w = np.where(overshoot, w_ref * 1.10, w)
    res = _mock_result(t, w)
    m = extract_metrics(res, plan=_SETTLE_PLAN, w_ref=w_ref)
    assert m["t_settle"] >= 0.5


def test_settle_time_never_settles():
    """Signal oscillates and ends outside the band."""
    t = np.linspace(0.0, 2.0, 2001)
    w_ref = 100.0
    w = np.where(t < 0.2, 0.0, w_ref * (1.0 + 0.10 * np.sin(20 * t)))
    res = _mock_result(t, w)
    m = extract_metrics(res, plan=_SETTLE_PLAN, w_ref=w_ref)
    assert np.isnan(m["t_settle"])


def test_settle_time_immediately_settled():
    """Speed is already within band right at the step time."""
    t = np.linspace(0.0, 2.0, 2001)
    w_ref = 100.0
    w = np.where(t < 0.2, 0.0, w_ref)
    res = _mock_result(t, w)
    m = extract_metrics(res, plan=_SETTLE_PLAN, w_ref=w_ref)
    assert m["t_settle"] == pytest.approx(0.0, abs=0.002)


def test_build_sim_derived_psi_f():
    """Bug regression: psi_f=None + B_rem -> derive -> simulate -> valid metrics."""
    from phasesweep.machines.geometry import default_inrunner
    from phasesweep.machines.motor import Motor

    motor = Motor(
        name="test_derived", geometry=default_inrunner(),
        n_p=2, R_s=0.2, L_d=4e-3, L_q=4e-3, psi_f=None, B_rem=0.5,
        J=0.002, N=10, k_w=0.966, L_stk=0.10, coils_series=1,
        drive=DriveParams(U_DC=540.0, MAX_I_S=20.0, W_REF=314.159265358979),
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
            drive=DriveParams(U_DC=540.0, MAX_I_S=20.0, W_REF=314.16),
        )
        large = Motor(
            name="large", geometry=dataclasses.replace(default_inrunner(), n_slots=12),
            n_p=2, R_s=0.05, L_d=10e-3, L_q=10e-3, psi_f=0.5, J=0.1,
            N=50, k_w=0.966, L_stk=0.20,
            drive=DriveParams(U_DC=540.0, MAX_I_S=20.0, W_REF=314.16),
        )
        plan_s = plan_sim(prepare_drive_sim(small))
        plan_l = plan_sim(prepare_drive_sim(large))
        assert plan_s.t_stop < plan_l.t_stop
        assert plan_s.load_torque < plan_l.load_torque

    def test_plan_sim_warns_on_infeasible_speed_setpoint(self):
        motor = Motor(
            name="fw", geometry=dataclasses.replace(default_inrunner(), n_slots=12),
            n_p=2, R_s=0.5, L_d=1e-3, L_q=1e-3, psi_f=0.5, J=1e-3,
            N=50, k_w=0.966, L_stk=0.05,
            drive=DriveParams(U_DC=100.0, MAX_I_S=10.0, W_REF=400.0),
        )
        with pytest.warns(UserWarning, match="field weakening"):
            plan_sim(prepare_drive_sim(motor))

    def test_plan_sim_T_s_capped_by_electrical_period(self):
        """High-pole/high-speed: T_s respects 20 samples per electrical period."""
        motor = Motor(
            name="hispeed", geometry=dataclasses.replace(default_inrunner(), n_slots=12),
            n_p=20, R_s=5.0, L_d=0.5e-3, L_q=0.5e-3, psi_f=0.005, J=1e-5,
            N=50, k_w=0.966, L_stk=0.02,
            drive=DriveParams(U_DC=540.0, MAX_I_S=10.0, W_REF=1000.0),
        )
        plan = plan_sim(prepare_drive_sim(motor))
        T_elec = 2 * np.pi / (20 * 1000.0)
        assert plan.T_s == pytest.approx(T_elec / 20)
        assert plan.T_s < 20e-6  # tighter than the tau_e/5-derived 20 us

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
            drive=DriveParams(U_DC=540.0, MAX_I_S=20.0, W_REF=314.16),
        )
        large_j = Motor(
            name="large-J", geometry=dataclasses.replace(default_inrunner(), n_slots=12),
            n_p=2, R_s=0.2, L_d=4e-3, L_q=4e-3, psi_f=0.1, J=0.01,
            N=50, k_w=0.966, L_stk=0.10, coils_series=1,
            drive=DriveParams(U_DC=540.0, MAX_I_S=20.0, W_REF=314.16),
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
            drive=DriveParams(U_DC=540.0, MAX_I_S=20.0, W_REF=314.16),
        )  # tau_e = 50us
        slow_e = Motor(
            name="slow-electric", geometry=dataclasses.replace(default_inrunner(), n_slots=12),
            n_p=2, R_s=0.2, L_d=4e-3, L_q=4e-3, psi_f=0.1, J=0.002,
            N=50, k_w=0.966, L_stk=0.10, coils_series=1,
            drive=DriveParams(U_DC=540.0, MAX_I_S=20.0, W_REF=314.16),
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

    def test_metrics_finite_and_positive(self, full_result):
        m = extract_metrics(full_result, plan=PLAN, w_ref=W_REF)
        for key in ("t_settle", "i_ss", "speed_droop", "tau_peak"):
            assert np.isfinite(m[key]), f"{key} is not finite"
            assert m[key] > 0, f"{key} is not positive"

    def test_heavy_load_increases_droop(self, full_result, full_result_heavy_load):
        heavy_plan, heavy_res = full_result_heavy_load
        m_default = extract_metrics(full_result, plan=PLAN, w_ref=W_REF)
        m_heavy = extract_metrics(heavy_res, plan=heavy_plan, w_ref=W_REF)
        assert m_heavy["speed_droop"] > m_default["speed_droop"]


# ---------------------------------------------------------------------------
# Outrunner regression: adaptive alpha_s prevents motulator crash
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestOutrunnerRegression:

    @pytest.fixture(scope="class", params=["outrunner_14mm_steel"])
    def outrunner_metrics(self, request):
        motor_path = Path(__file__).parent.parent.parent / "motors" / f"{request.param}.toml"
        if not motor_path.exists():
            pytest.skip(f"{motor_path} not found")
        motor = load_motor(motor_path)
        params = prepare_drive_sim(motor)
        plan = plan_sim(params)
        sim = build_sim(params, plan)
        res = sim.simulate(t_stop=plan.t_stop)
        return extract_metrics(res, plan=plan, w_ref=params.drive.W_REF)

    def test_outrunner_sim_completes(self, outrunner_metrics):
        for key in ("t_settle", "i_ss", "speed_droop", "tau_peak"):
            assert np.isfinite(outrunner_metrics[key]), f"{key} is not finite"
            assert outrunner_metrics[key] > 0, f"{key} is not positive"


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

    Scenario: torque step = 3 Nm, steady state. The controller runs MTPA (not
    id*=0); for THIS param set psi_f is large relative to the saliency, so the
    reluctance saving is negligible (<<1%) and the MTPA current collapses onto
    the magnet-only value.
    Target: |i_s| ~= tau / (1.5 * n_p * psi_f) = 0.549 A within 3%
    (loose enough to accommodate a switched-PWM external reference). The
    salient case where this approximation breaks is covered by
    test_steady_state_current_tracks_salient_mtpa below.
    """

    def test_steady_state_current_matches_analytical(self):
        params = DriveSimParams(
            n_p=2,
            R_s=5.2, L_d=0.063, L_q=0.133,
            psi_f=1.819, J=0.0011,
            drive=DriveParams(U_DC=400.0, MAX_I_S=5.0, W_REF=314.16),
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

    def test_steady_state_current_tracks_salient_mtpa(self):
        """On a strongly salient machine the torque-mode steady state sits on
        the MTPA locus — BELOW the magnet-only tau/(1.5 n_p psi_f) by the
        reluctance saving — and matches the closed-form current_for_torque.

        The sibling test above agrees with the magnet-only anchor only because
        its psi_f is huge (reluctance negligible). Here, on the most salient
        in-repo anchor (Kollmorgen B-104-B, Lq/Ld=2.06) at nameplate torque,
        the saving is ~3.5%, so the magnet-only anchor would be wrong and the
        MTPA closed form is required. Tight tolerance also locks the
        motulator<->closed-form MTPA contract: a motulator upgrade that shifted
        its MTPA reference generator would trip this.
        """
        from phasesweep.models.thermal_duty import current_for_torque

        m = load_motor("data/kollmorgen_b104b/kollmorgen_b104b_ipm.toml")
        params = DriveSimParams(
            n_p=m.n_p, R_s=m.R_s, L_d=m.L_d, L_q=m.L_q, psi_f=m.psi_f,
            # datasheet rotor inertia not published; timing-only, does not
            # affect the held-rotor steady state being asserted here
            J=3e-4,
            drive=DriveParams(
                U_DC=m.drive.U_DC, MAX_I_S=m.drive.MAX_I_S,
                W_REF=m.drive.W_REF, I_LIMIT=m.drive.I_LIMIT),
        )
        tau_ref = 1.57  # nameplate rated torque

        plan = plan_torque_sim(params, tau_ref=tau_ref)
        torque_ref = Step(step_time=plan.load_time, step_value=tau_ref, initial_value=0)
        res = build_sim(params, plan, torque_ref=torque_ref).simulate(t_stop=plan.t_stop)

        t = res.mdl.t
        i_s = np.abs(res.mdl.machine.i_s_ab)
        tau_M = res.mdl.machine.tau_M
        ss_mask = t >= (plan.t_stop - plan.ss_window)
        i_ss = float(np.mean(i_s[ss_mask]))
        tau_ss = float(np.mean(tau_M[ss_mask]))

        i_mtpa = current_for_torque(
            params.n_p, params.psi_f, params.L_d, params.L_q, tau_ref)
        i_magnet = tau_ref / (1.5 * params.n_p * params.psi_f)

        # tracks the closed-form MTPA current (not magnet-only) and the torque ref
        assert abs(i_ss - i_mtpa) / i_mtpa < 2e-3
        assert abs(tau_ss - tau_ref) / tau_ref < 2e-3
        # reluctance is genuinely exploited: MTPA current sits below magnet-only
        assert i_mtpa < i_magnet
        assert (i_magnet - i_mtpa) / i_magnet > 0.02
