"""Cross-validation: Jilin University 5.5 kW SynRM (pure reluctance, no magnets).

Source: I-f control paper, PMC11679518 (2024), Jilin University.
Tables 1 & 2: R_s, J, L_d, L_q at rated current.

This motor has psi_f = 0 (no permanent magnets): the reluctance-only
anchor for the circuit-model chain. MTPA math is validated directly and
through the runners (rated_torque, torque_speed, thermal_duty); drive_sim
accepts the anchor's L_d > L_q SynRM convention (the one motulator's
psi_f = 0 angle search assumes).
"""

from math import pi, sqrt

import pytest

from phasesweep.machines.configs import load_motor
from phasesweep.models.rated_torque import mtpa_gamma, mtpa_torque
from tests.conftest import requires_sim

TOML = "data/synrm_5p5kw/synrm_5p5kw.toml"

N_P = 2
R_S = 0.25
L_D = 0.02883
L_Q = 0.00755
I_RATED_RMS = 12.5
I_RATED_PEAK = I_RATED_RMS * sqrt(2)
J = 0.108
POWER_KW = 5.5


class TestSynRMCrossval:

    def test_toml_loads(self):
        m = load_motor(TOML)
        assert m.name == "Jilin SynRM 5.5 kW"
        assert m.n_p == N_P
        assert m.L_d == L_D
        assert m.L_q == L_Q
        assert m.psi_f == 0.0
        assert m.geometry is None

    def test_saliency_ratio_at_rated_current(self):
        """L_d/L_q = 3.82 at rated current (strong saliency, L_d > L_q)."""
        assert L_D / L_Q == pytest.approx(3.82, rel=0.01)

    def test_mtpa_angle_45_degrees(self):
        """Pure SynRM MTPA is at 45 electrical degrees from q-axis.
        For L_d > L_q (reverse salient), gamma < 0 (positive i_d)."""
        gamma = mtpa_gamma(0.0, L_D, L_Q, I_RATED_PEAK)
        assert abs(gamma) == pytest.approx(pi / 4, rel=0.01), (
            f"SynRM MTPA angle should be +/-45 deg, got {gamma * 180 / pi:.1f} deg"
        )

    def test_mtpa_torque_pure_reluctance(self):
        """With psi_f = 0, torque = (3/2) * n_p * (L_d - L_q) * i_d * i_q.
        At gamma = -45 deg: i_d = I_s * sin(45) = I_s/sqrt(2),
        i_q = I_s * cos(45) = I_s/sqrt(2), so
        T = (3/2) * n_p * (L_d - L_q) * I_s^2 / 2."""
        gamma = mtpa_gamma(0.0, L_D, L_Q, I_RATED_PEAK)
        tau = mtpa_torque(N_P, 0.0, L_D, L_Q, I_RATED_PEAK, gamma)
        tau_expected = 1.5 * N_P * (L_D - L_Q) * I_RATED_PEAK**2 / 2
        assert tau == pytest.approx(tau_expected, rel=0.01)
        assert tau > 0, "Reluctance torque must be positive"

    def test_torque_physically_reasonable(self):
        """MTPA torque at rated current with saturated inductances.
        T = (3/2)*n_p*(L_d-L_q)*I_s^2/2 = 1.5*2*(0.02883-0.00755)*17.68^2/2
        = ~10 Nm. Low vs the 5.5 kW nameplate (~35 Nm at 1500 rpm) because
        these are heavily-saturated inductance values; the unsaturated L_d
        is 44 mH (50% higher) and the test paper only reached ~189 rpm."""
        gamma = mtpa_gamma(0.0, L_D, L_Q, I_RATED_PEAK)
        tau = mtpa_torque(N_P, 0.0, L_D, L_Q, I_RATED_PEAK, gamma)
        assert 5.0 < tau < 30.0, f"tau={tau:.2f} Nm outside expected range"

    def test_rated_torque_runner(self):
        """rated_torque through the registry: reluctance-only MTPA at
        I_rated. tau = 0.75 * n_p * (L_d - L_q) * I^2 = 9.975 Nm at the
        saturated (rated-current) inductances; gamma = -45 deg (L_d > L_q,
        magnetizing i_d)."""
        from phasesweep.registry import MODEL_REGISTRY
        from phasesweep.sweep_types import RunConfig
        m = load_motor(TOML)
        r = MODEL_REGISTRY["rated_torque"].fn(
            RunConfig(motor=m, model="rated_torque"))
        expected = 0.75 * N_P * (L_D - L_Q) * m.I_rated**2
        assert r["tau_mtpa"] == pytest.approx(expected, rel=1e-9)
        assert r["gamma_opt_deg"] == pytest.approx(-45.0, rel=1e-9)
        assert r["k_T"] == 0.0

    def test_torque_speed_runner(self):
        """torque_speed through the registry. Base speed ~413 rad/s (well
        above the 157 rad/s rated point — the paper only drove it to
        ~189 rpm); characteristic current 0 <= I so the MTPV branch keeps
        the envelope torque nonzero at any speed (max_speed None). Peak
        envelope power ~4.77 kW vs the 5.5 kW nameplate — low because the
        Table 2 inductances are at worst-case saturation."""
        from phasesweep.registry import MODEL_REGISTRY
        from phasesweep.sweep_types import RunConfig
        m = load_motor(TOML)
        r = MODEL_REGISTRY["torque_speed"].fn(
            RunConfig(motor=m, model="torque_speed"))
        assert r["max_speed_peak"] is None
        assert r["max_speed_cont"] is None
        assert r["base_speed_peak"] == pytest.approx(413.4, rel=1e-3)
        assert r["base_speed_peak"] > 157.08  # rated speed inside MTPA region
        assert 4.5e3 < r["p_max_peak"] < 5.0e3
        assert all(t > 0.0 for t in r["tau_curve_peak"][1:])

    def test_thermal_duty_runner(self):
        """thermal_duty at the rated MTPA torque: the reluctance-only
        inverse maps the segment back to exactly I_rated, so the
        rated-current budget path closes at over_budget_ratio = 1."""
        from phasesweep.models.thermal_duty import run_thermal_duty
        from phasesweep.sweep_types import RunConfig
        m = load_motor(TOML)
        tau_rated = 0.75 * N_P * (L_D - L_Q) * m.I_rated**2
        r = run_thermal_duty(RunConfig(
            motor=m, model="thermal_duty",
            duty_profile=((tau_rated, 60.0),)))
        assert r["budget_source"] == "rated_current"
        # exactly 1 up to float roundtrip (sqrt of the squared current)
        assert r["over_budget_ratio"] == pytest.approx(1.0, rel=1e-12)

    @requires_sim
    def test_drive_sim_plan_and_build(self):
        """plan_sim sizes tau_peak from the reluctance MTPA torque, and
        build_sim accepts the anchor's L_d > L_q SynRM convention (the one
        motulator's psi_f = 0 angle search assumes). Swapping the axes must
        be rejected."""
        import dataclasses

        from phasesweep.simulation.params import prepare_drive_sim
        from phasesweep.simulation.sim import build_sim, plan_sim
        params = prepare_drive_sim(load_motor(TOML))
        plan = plan_sim(params)
        # I_limit = MAX_I_S = 17.678 A (TOML rounding vs I_rated_peak)
        tau_peak = 0.75 * N_P * (L_D - L_Q) * 17.678**2
        assert plan.load_torque == pytest.approx(0.5 * tau_peak, rel=1e-9)
        build_sim(params, plan)  # must not raise
        swapped = dataclasses.replace(params, L_d=L_Q, L_q=L_D)
        with pytest.raises(ValueError, match="L_d > L_q SynRM convention"):
            build_sim(swapped)
