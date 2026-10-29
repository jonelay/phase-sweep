"""im_design_audit: Boldea-derived diagnostic bands for IM design.

Unit tests exercise the audit checks on synthetic motors. The CREATOR
regression test lives in tests/crossval/test_creator_im_crossval.py.
"""

from math import sqrt

import pytest

from phasesweep.audit.im_design_audit import AuditResult, im_design_audit
from phasesweep.machines.motor import InductionParams, Motor


def _make_im(**overrides) -> Motor:
    ind_kw = dict(
        R_2=1.0, L_m=0.2, L_sigma_s=0.01, L_sigma_r=0.01,
        sigma_bar=3e7, f_rated=50.0, s_rated=0.04, V_LL_rated=400.0,
    )
    ind_kw.update(overrides.pop("induction", {}))
    fields = dict(
        name="test IM", geometry=None, n_p=2, R_s=2.0,
        I_rated=10.0 * sqrt(2),
        induction=InductionParams(**ind_kw),
    )
    fields.update(overrides)
    return Motor(**fields)


def _make_sweep(tau_rated=10.0, tau_breakdown=25.0,
                i_lr=50.0, i_rated=8.0, s_rated=0.04):
    return {
        "tau_rated": tau_rated,
        "tau_breakdown": tau_breakdown,
        "I_locked_rotor_rms": i_lr,
        "I_1_rated_rms": i_rated,
        "slip_rated": s_rated,
    }


class TestAuditBasics:

    def test_synchronous_motor_rejected(self):
        m = Motor(name="sync", geometry=None, n_p=2)
        with pytest.raises(ValueError, match="induction"):
            im_design_audit(m)

    def test_minimal_motor_returns_empty(self):
        m = Motor(name="bare", geometry=None, n_p=2,
                  induction=InductionParams())
        results = im_design_audit(m)
        assert results == []

    def test_all_checks_have_required_fields(self):
        m = _make_im()
        results = im_design_audit(m, _make_sweep())
        names = [r.name for r in results]
        assert "S_n_pct" in names
        assert "T_ek_over_T_en" in names
        assert "I_start_over_I_n" in names
        assert "I_0_over_I_n" in names

    def test_results_are_audit_result(self):
        m = _make_im()
        for r in im_design_audit(m, _make_sweep()):
            assert isinstance(r, AuditResult)
            assert r.verdict in ("pass", "info", "warn")


class TestRatedSlip:

    def test_typical_small_motor_passes(self):
        m = _make_im(induction=dict(s_rated=0.05))
        sweep = _make_sweep(tau_rated=3.0, s_rated=0.05)
        results = {r.name: r for r in im_design_audit(m, sweep)}
        r = results["S_n_pct"]
        assert r.value == pytest.approx(5.0)
        assert r.verdict == "pass"

    def test_large_motor_low_slip_passes(self):
        m = _make_im(induction=dict(s_rated=0.015))
        sweep = _make_sweep(tau_rated=500.0, s_rated=0.015)
        results = {r.name: r for r in im_design_audit(m, sweep)}
        r = results["S_n_pct"]
        assert r.value == pytest.approx(1.5)
        assert r.verdict == "pass"

    def test_unusually_high_slip_warns(self):
        m = _make_im(induction=dict(s_rated=0.12))
        sweep = _make_sweep(tau_rated=500.0, s_rated=0.12)
        results = {r.name: r for r in im_design_audit(m, sweep)}
        assert results["S_n_pct"].verdict == "warn"

    def test_no_torque_slip_still_runs_without_band(self):
        m = _make_im(induction=dict(s_rated=0.04))
        results = {r.name: r for r in im_design_audit(m)}
        r = results["S_n_pct"]
        assert r.band is None
        assert r.verdict == "info"


class TestBreakdownRatio:

    def test_typical_ratio_passes(self):
        m = _make_im(induction=dict(design_letter="B"))
        sweep = _make_sweep(tau_rated=10.0, tau_breakdown=22.0)
        results = {r.name: r for r in im_design_audit(m, sweep)}
        r = results["T_ek_over_T_en"]
        assert r.value == pytest.approx(2.2)
        assert r.verdict == "pass"

    def test_low_ratio_warns(self):
        m = _make_im(induction=dict(design_letter="B"))
        sweep = _make_sweep(tau_rated=10.0, tau_breakdown=12.0)
        results = {r.name: r for r in im_design_audit(m, sweep)}
        assert results["T_ek_over_T_en"].verdict == "warn"

    def test_no_design_letter_uses_wide_band(self):
        m = _make_im()
        sweep = _make_sweep(tau_rated=10.0, tau_breakdown=20.0)
        results = {r.name: r for r in im_design_audit(m, sweep)}
        r = results["T_ek_over_T_en"]
        assert r.band == (1.5, 5.0)
        assert r.verdict == "pass"


class TestStartingCurrent:

    def test_typical_nema_b_passes(self):
        m = _make_im(induction=dict(design_letter="B"))
        sweep = _make_sweep(i_lr=48.0, i_rated=8.0)
        results = {r.name: r for r in im_design_audit(m, sweep)}
        r = results["I_start_over_I_n"]
        assert r.value == pytest.approx(6.0)
        assert r.verdict == "pass"

    def test_high_inrush_warns(self):
        m = _make_im(induction=dict(design_letter="B"))
        sweep = _make_sweep(i_lr=80.0, i_rated=8.0)
        results = {r.name: r for r in im_design_audit(m, sweep)}
        assert results["I_start_over_I_n"].verdict == "warn"


class TestNoLoadCurrent:

    def test_reasonable_ratio(self):
        m = _make_im()
        results = {r.name: r for r in im_design_audit(m)}
        assert "I_0_over_I_n" in results

    def test_missing_I_rated_skips(self):
        m = _make_im(I_rated=None)
        results = {r.name: r for r in im_design_audit(m)}
        assert "I_0_over_I_n" not in results


class TestAirgapFlux:

    def test_skipped_without_geometry(self):
        m = _make_im()
        results = {r.name: r for r in im_design_audit(m)}
        assert "B_g_T" not in results

    def test_runs_with_geometry(self):
        from phasesweep.machines.geometry import inrunner
        geo = inrunner(r_outer=0.1, r_stator=0.063, r_magnet=0.0626,
                       r_rotor=0.0626, r_inner=0.02,
                       n_slots=36, slot_depth=0.015, n_bars=28,
                       bar_width=0.003, bar_height=0.01,
                       end_ring_axial_length=0.008,
                       end_ring_radial_height=0.008)
        m = _make_im(geometry=geo, N=18, coils_series=12, k_w=0.902,
                     L_stk=0.1)
        results = {r.name: r for r in im_design_audit(m)}
        r = results["B_g_T"]
        assert 0.3 < r.value < 1.2
        assert r.verdict in ("pass", "warn")
        # Peak, not pole average: E = 4.44 f N k_w (2/π) B_pk τ_p L
        from math import pi, sqrt

        from phasesweep.solver_params import n_eff
        ind = m.induction
        b_avg = (ind.V_LL_rated / sqrt(3)) / (
            4.44 * ind.f_rated * n_eff(m) * m.k_w * (pi * 0.063 / m.n_p) * m.L_stk)
        assert r.value == pytest.approx(b_avg * pi / 2)


class TestSaturationFactor:

    def test_informational_with_L_m_vs_i(self):
        lm_table = ((0.5, 0.25), (1.0, 0.23), (3.0, 0.18), (6.0, 0.12))
        m = _make_im(induction=dict(L_m_vs_i=lm_table))
        results = {r.name: r for r in im_design_audit(m)}
        r = results["K_s"]
        assert r.verdict == "info"
        assert r.value == pytest.approx(0.25 / 0.2)

    def test_skipped_without_table(self):
        m = _make_im()
        results = {r.name: r for r in im_design_audit(m)}
        assert "K_s" not in results
