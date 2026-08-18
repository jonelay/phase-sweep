"""Tests for the drive_sim_two_mass model — two-mass mechanical system."""

from importlib.util import find_spec

import pytest

if not find_spec("motulator"):
    pytest.skip("requires phasesweep[sim] (motulator not installed)", allow_module_level=True)

import dataclasses
import json

import numpy as np

from phasesweep.machines.geometry import default_inrunner
from phasesweep.machines.motor import DriveParams, Motor
from phasesweep.registry import MODEL_REGISTRY
from phasesweep.simulation.sim import (
    SimPlan,
    build_sim,
    extract_metrics,
    extract_waveforms,
    plan_sim,
    plan_two_mass_sim,
)
from phasesweep.solver_params import TwoMassLoad, prepare_drive_sim
from phasesweep.sweep_types import RunConfig, compute_run_id

MOTOR = Motor(
    name="test-SPMSM",
    geometry=dataclasses.replace(default_inrunner(), n_slots=12),
    n_p=2, R_s=0.2, L_d=4e-3, L_q=4e-3, psi_f=0.1, J=0.002,
    N=50, k_w=0.966, L_stk=0.10, I_rated=10.0, coils_series=1,
    drive=DriveParams(U_DC=540.0, MAX_I_S=20.0, W_REF=314.159265358979),
)

SAARAKKALA_LOAD = TwoMassLoad(
    J_L=0.005, K_S=700.0, C_S=0.01, B_L=0.0,
)

PARAMS = prepare_drive_sim(MOTOR)


# ---------------------------------------------------------------------------
# Registry structure (pattern: test_demag_screen.py)
# ---------------------------------------------------------------------------

class TestRegistryEntry:

    def test_entry_exists(self):
        assert "drive_sim_two_mass" in MODEL_REGISTRY

    def test_source_computed(self):
        assert MODEL_REGISTRY["drive_sim_two_mass"].source == "computed"

    def test_cost_medium(self):
        assert MODEL_REGISTRY["drive_sim_two_mass"].cost == "medium"

    def test_produces_superset_of_drive_sim(self):
        base = MODEL_REGISTRY["drive_sim"].produces
        two_mass = MODEL_REGISTRY["drive_sim_two_mass"].produces
        assert base < two_mass
        assert {"w_L_list", "tau_S_list", "tau_S_peak"} <= two_mass

    def test_hash_fields_include_load_mech(self):
        assert "load_mech" in MODEL_REGISTRY["drive_sim_two_mass"].hash_fields

    def test_fn_callable(self):
        assert MODEL_REGISTRY["drive_sim_two_mass"].fn is not None


# ---------------------------------------------------------------------------
# TwoMassLoad validation
# ---------------------------------------------------------------------------

class TestTwoMassLoadValidation:

    def test_valid_construction(self):
        load = TwoMassLoad(J_L=0.005, K_S=700.0, C_S=0.01)
        assert load.J_L == 0.005
        assert load.B_L == 0.0

    def test_rejects_negative_J_L(self):
        with pytest.raises(ValueError, match="J_L"):
            TwoMassLoad(J_L=-0.001, K_S=700.0, C_S=0.01)

    def test_rejects_zero_J_L(self):
        with pytest.raises(ValueError, match="J_L"):
            TwoMassLoad(J_L=0.0, K_S=700.0, C_S=0.01)

    def test_rejects_negative_K_S(self):
        with pytest.raises(ValueError, match="K_S"):
            TwoMassLoad(J_L=0.005, K_S=-1.0, C_S=0.01)

    def test_rejects_negative_C_S(self):
        with pytest.raises(ValueError, match="C_S"):
            TwoMassLoad(J_L=0.005, K_S=700.0, C_S=-0.01)

    def test_rejects_negative_B_L(self):
        with pytest.raises(ValueError, match="B_L"):
            TwoMassLoad(J_L=0.005, K_S=700.0, C_S=0.01, B_L=-1.0)

    def test_round_trip(self):
        load = TwoMassLoad(J_L=0.005, K_S=700.0, C_S=0.01, B_L=0.1)
        restored = TwoMassLoad.from_dict(load.to_dict())
        assert restored == load


# ---------------------------------------------------------------------------
# plan_two_mass_sim
# ---------------------------------------------------------------------------

class TestPlanTwoMassSim:

    def test_returns_sim_plan(self):
        plan = plan_two_mass_sim(PARAMS, SAARAKKALA_LOAD)
        assert isinstance(plan, SimPlan)

    def test_alpha_s_below_antiresonance(self):
        plan = plan_two_mass_sim(PARAMS, SAARAKKALA_LOAD)
        w_ar = np.sqrt(SAARAKKALA_LOAD.K_S / SAARAKKALA_LOAD.J_L)
        assert plan.alpha_s <= w_ar / 5

    def test_tau_m_uses_total_inertia(self):
        plan = plan_two_mass_sim(PARAMS, SAARAKKALA_LOAD)
        J_total = PARAMS.J + SAARAKKALA_LOAD.J_L
        single_plan = plan_sim(PARAMS)
        # tau_m should be larger (more inertia)
        assert plan.tau_m > single_plan.tau_m
        # check the formula directly
        from phasesweep.models.rated_torque import magnet_torque_constant
        k_t = magnet_torque_constant(PARAMS.n_p, PARAMS.psi_f)
        I_limit = PARAMS.drive.I_LIMIT if PARAMS.drive.I_LIMIT is not None else PARAMS.drive.MAX_I_S
        tau_peak = k_t * I_limit
        expected = J_total * PARAMS.drive.W_REF / tau_peak
        assert abs(plan.tau_m - expected) < 1e-12


# ---------------------------------------------------------------------------
# Frequency anchor (Saarakkala & Hinkkanen 2015)
# ---------------------------------------------------------------------------

class TestFrequencyAnchor:
    """Closed-form resonance/antiresonance from the Saarakkala 2015 parameter set,
    verified against the FFT of the simulated motor speed."""

    J_M = 0.005
    J_L = 0.005
    K_S = 700.0
    C_S = 0.01

    @pytest.fixture(scope="class")
    def two_mass_result(self):
        motor = dataclasses.replace(MOTOR, J=self.J_M)
        params = prepare_drive_sim(motor)
        load = TwoMassLoad(J_L=self.J_L, K_S=self.K_S, C_S=self.C_S)
        plan = plan_two_mass_sim(params, load)
        sim = build_sim(params, plan, load_mech=load)
        return sim.simulate(t_stop=plan.t_stop), plan

    def test_resonance_in_shaft_torque_fft(self, two_mass_result):
        """FFT of shaft torque after the load step should peak near the
        closed-form resonance frequency."""
        res, plan = two_mass_result
        t = np.asarray(res.mdl.t)
        tau_S = np.asarray(res.mdl.mechanics.tau_S)
        mask = t >= plan.load_time
        sig = tau_S[mask] - np.mean(tau_S[mask])
        dt = np.mean(np.diff(t[mask]))
        freqs = np.fft.rfftfreq(len(sig), d=dt)
        spectrum = np.abs(np.fft.rfft(sig))
        above_20 = freqs > 20
        peak_idx = np.argmax(spectrum[above_20])
        f_peak = freqs[above_20][peak_idx]
        f_res_theory = (1 / (2 * np.pi)) * np.sqrt(
            self.K_S * (self.J_M + self.J_L) / (self.J_M * self.J_L)
        )
        assert abs(f_peak - f_res_theory) < 10.0

    def test_closed_form_frequencies_match_literature(self):
        f_res = (1 / (2 * np.pi)) * np.sqrt(
            self.K_S * (self.J_M + self.J_L) / (self.J_M * self.J_L)
        )
        f_ar = (1 / (2 * np.pi)) * np.sqrt(self.K_S / self.J_L)
        assert abs(f_res - 84.2) < 0.1
        assert abs(f_ar - 59.6) < 0.1

    def test_constructor_wiring_pre_simulate(self):
        """Verify TwoMassLoad fields reach motulator before simulate()."""
        motor = dataclasses.replace(MOTOR, J=self.J_M)
        params = prepare_drive_sim(motor)
        load = TwoMassLoad(J_L=self.J_L, K_S=self.K_S, C_S=self.C_S, B_L=0.5)
        plan = plan_two_mass_sim(params, load)
        sim = build_sim(params, plan, load_mech=load)
        mech = sim.mdl.mechanics
        assert mech.J_M == self.J_M
        assert mech.J_L == self.J_L
        assert mech.K_S == self.K_S
        assert mech.C_S == self.C_S
        assert mech.B_L == 0.5

    def test_simulation_runs_without_error(self, two_mass_result):
        res, plan = two_mass_result
        w_M = np.asarray(res.mdl.mechanics.w_M)
        assert len(w_M) > 100
        assert np.all(np.isfinite(w_M))

    def test_two_mass_waveforms_present(self, two_mass_result):
        res, plan = two_mass_result
        w_L = np.asarray(res.mdl.mechanics.w_L)
        tau_S = np.asarray(res.mdl.mechanics.tau_S)
        assert len(w_L) > 100
        assert len(tau_S) > 100

    def test_extract_metrics_two_mass(self, two_mass_result):
        res, plan = two_mass_result
        metrics = extract_metrics(
            res, plan=plan, w_ref=PARAMS.drive.W_REF, two_mass=True,
        )
        assert "tau_S_peak" in metrics
        assert metrics["tau_S_peak"] > 0

    def test_extract_waveforms_two_mass(self, two_mass_result):
        res, plan = two_mass_result
        wf = extract_waveforms(res, two_mass=True)
        assert "w_L_list" in wf
        assert "tau_S_list" in wf
        assert len(wf["w_L_list"]) > 0


# ---------------------------------------------------------------------------
# Rigid-limit regression: very stiff shaft → converges to single-mass
# ---------------------------------------------------------------------------

class TestRigidLimit:

    @pytest.fixture(scope="class")
    def single_mass_metrics(self):
        plan = plan_sim(PARAMS)
        sim = build_sim(PARAMS, plan)
        res = sim.simulate(t_stop=plan.t_stop)
        return extract_metrics(res, plan=plan, w_ref=PARAMS.drive.W_REF)

    @pytest.fixture(scope="class")
    def rigid_two_mass_metrics(self):
        load = TwoMassLoad(J_L=0.002, K_S=1e6, C_S=100.0)
        plan = plan_two_mass_sim(PARAMS, load)
        sim = build_sim(PARAMS, plan, load_mech=load)
        res = sim.simulate(t_stop=plan.t_stop)
        return extract_metrics(
            res, plan=plan, w_ref=PARAMS.drive.W_REF, two_mass=True,
        )

    def test_i_ss_converges(self, single_mass_metrics, rigid_two_mass_metrics):
        if np.isnan(single_mass_metrics["i_ss"]) or np.isnan(rigid_two_mass_metrics["i_ss"]):
            pytest.skip("NaN in i_ss")
        assert abs(rigid_two_mass_metrics["i_ss"] - single_mass_metrics["i_ss"]) < 1.0


# ---------------------------------------------------------------------------
# Run-ID separation
# ---------------------------------------------------------------------------

class TestRunIdSeparation:

    def test_different_load_mech_different_run_id(self):
        plan = plan_sim(PARAMS)
        load_a = TwoMassLoad(J_L=0.005, K_S=700.0, C_S=0.01)
        load_b = TwoMassLoad(J_L=0.010, K_S=700.0, C_S=0.01)
        rc_a = RunConfig(motor=MOTOR, model="drive_sim_two_mass",
                         sim_plan=plan, load_mech=load_a)
        rc_b = RunConfig(motor=MOTOR, model="drive_sim_two_mass",
                         sim_plan=plan, load_mech=load_b)
        assert compute_run_id(rc_a) != compute_run_id(rc_b)

    def test_no_load_mech_matches_pinned_hash(self):
        plan = plan_sim(PARAMS)
        rc = RunConfig(motor=MOTOR, model="drive_sim", sim_plan=plan)
        assert compute_run_id(rc) == "7d6c4420911d"


# ---------------------------------------------------------------------------
# RunConfig serialization round-trip
# ---------------------------------------------------------------------------

class TestSimRunnerGuards:

    def test_rejects_load_mech_on_drive_sim(self):
        from phasesweep.simulation.sim_runner import _run_sim_impl
        plan = plan_sim(PARAMS)
        load = TwoMassLoad(J_L=0.005, K_S=700.0, C_S=0.01)
        rc = RunConfig(motor=MOTOR, model="drive_sim",
                       sim_plan=plan, load_mech=load)
        with pytest.raises(ValueError, match="drive_sim_two_mass"):
            _run_sim_impl(rc)

    def test_rejects_two_mass_without_load_mech(self):
        from phasesweep.simulation.sim_runner import _run_sim_impl
        plan = plan_sim(PARAMS)
        rc = RunConfig(motor=MOTOR, model="drive_sim_two_mass", sim_plan=plan)
        with pytest.raises(ValueError, match="load_mech"):
            _run_sim_impl(rc)


class TestRunConfigRoundTrip:

    def test_load_mech_survives_round_trip(self):
        load = TwoMassLoad(J_L=0.005, K_S=700.0, C_S=0.01, B_L=0.1)
        plan = plan_sim(PARAMS)
        rc = RunConfig(motor=MOTOR, model="drive_sim_two_mass",
                       sim_plan=plan, load_mech=load)
        d = rc.to_dict()
        assert "load_mech" in d
        rc2 = RunConfig.from_dict(d)
        assert rc2.load_mech == load

    def test_json_serializable(self):
        load = TwoMassLoad(J_L=0.005, K_S=700.0, C_S=0.01)
        plan = plan_sim(PARAMS)
        rc = RunConfig(motor=MOTOR, model="drive_sim_two_mass",
                       sim_plan=plan, load_mech=load)
        s = json.dumps(rc.to_dict())
        rc2 = RunConfig.from_dict(json.loads(s))
        assert rc2.load_mech == load
