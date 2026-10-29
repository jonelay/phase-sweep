"""operating_point model: minimum current for a torque demand at a speed."""

from __future__ import annotations

import dataclasses
from math import cos, pi, radians, sin, sqrt

import numpy as np
import pytest

from phasesweep.machines.motor import DriveParams, FluxMap
from phasesweep.models.flux_map import flux_map_at_magnet_temp, flux_map_psi_f
from phasesweep.models.operating_point import (
    OperatingPointParams,
    operating_point,
    prepare_operating_point,
    run_operating_point,
)
from phasesweep.models.rated_torque import run_rated_torque
from phasesweep.models.thermal_duty import (
    psi_f_at_magnet_temp,
    r_s_at_operating_temp,
)
from phasesweep.models.torque_speed import base_speed, envelope_at_speed
from phasesweep.registry import MODEL_REGISTRY
from phasesweep.sweep_types import RunConfig, compute_run_id
from tests.conftest import REPO_ROOT, make_motor
from tests.models.test_torque_speed import _make_linear_flux_map

# salient reference machine (test_torque_speed's energy-balance case)
_NP, _PSI, _RS, _LD, _LQ = 4, 0.1, 0.05, 2e-3, 4e-3
_IMAX, _UMAX = 50.0, 100.0
_UDC = _UMAX * sqrt(3)
_RPM = 2 * pi / 60


def _op(w_mech, tau_ref, **kw):
    return operating_point(_NP, _PSI, _RS, _LD, _LQ, _IMAX, _UMAX, _UDC,
                           w_mech, tau_ref, **kw)


def _energy_residual(r):
    p_em = 1.5 * (r.u_d * r.i_d + r.u_q * r.i_q) - r.p_cu
    return abs(p_em - r.tau * r.w_mech) / max(abs(r.tau * r.w_mech), 1.0)


def _assert_feasible(r):
    assert r.I_s_peak <= _IMAX * (1 + 1e-9)
    assert r.u_mag <= r.u_max * (1 + 1e-9)
    assert r.m_index <= 2 / sqrt(3) * (1 + 1e-9)
    assert _energy_residual(r) < 1e-9
    assert r.I_s_rms == pytest.approx(r.I_s_peak / sqrt(2))
    assert r.p_elec == pytest.approx(r.p_shaft + r.p_cu)


@pytest.fixture(scope="module")
def i3():
    from phasesweep import load_motor
    return load_motor(REPO_ROOT / "motors/bmw_i3.toml")


@pytest.fixture(scope="module")
def i3_params(i3):
    return prepare_operating_point(i3)


def _i3_point(p, rpm, tau, **kw):
    return operating_point(p.n_p, p.psi_f, p.R_s, p.L_d, p.L_q, p.I_max,
                           p.U_max, p.U_DC, rpm * _RPM, tau,
                           flux_map=kw.pop("flux_map", p.flux_map), **kw)


# ---------------------------------------------------------------------------
# constant-L path
# ---------------------------------------------------------------------------

class TestAnalytical:

    def test_mtpa_branch_matches_rated_torque_angle(self):
        m = make_motor(I_rated=10.0, L_d=2e-3, L_q=4e-3)
        rated = run_rated_torque(RunConfig(motor=m, model="rated_torque"))
        g = radians(rated["gamma_opt_deg"])
        w = 0.1 * base_speed(m.n_p, m.psi_f, m.R_s, m.L_d, m.L_q, 10.0,
                             m.drive.U_DC / sqrt(3))
        r = run_operating_point(RunConfig(motor=m, model="operating_point",
                                          w_mech=w, tau_ref=rated["tau_mtpa"]))
        assert r["regime"] == "mtpa" and r["op_source"] == "analytical"
        assert r["i_d"] == pytest.approx(-10.0 * sin(g), rel=1e-9)
        assert r["i_q"] == pytest.approx(10.0 * cos(g), rel=1e-9)
        assert r["I_s_peak"] == pytest.approx(10.0, rel=1e-9)
        assert r["gamma_deg"] == pytest.approx(rated["gamma_opt_deg"], rel=1e-9)

    @pytest.mark.parametrize("tau_ref", [1.0, 4.0, 8.0])
    def test_field_weakening_sits_on_voltage_boundary(self, tau_ref):
        w_b = base_speed(_NP, _PSI, _RS, _LD, _LQ, _IMAX, _UMAX)
        r = _op(2.0 * w_b, tau_ref)
        assert r.regime == "field_weakening"
        assert r.u_mag == pytest.approx(_UMAX, rel=1e-9)
        assert r.i_d < 0.0
        _assert_feasible(r)

    def test_audit_counterexample_non_monotone_contour(self):
        """Both bracket ends sit at 100.6 V; a plain bisection fails, the
        three-stage search lands on the feasible root."""
        r = operating_point(2, 0.1, 0.01, 0.001, 0.001, 200.0, 50.0,
                            50.0 * sqrt(3), 500.0, 3.0)
        assert r.regime == "field_weakening"
        assert r.i_q == pytest.approx(10.0, rel=1e-9)
        assert r.i_d == pytest.approx(-51.2175, abs=1e-3)
        assert r.u_mag == pytest.approx(50.0, rel=1e-9)

    def test_torque_above_envelope_raises_with_envelope_value(self):
        w_b = base_speed(_NP, _PSI, _RS, _LD, _LQ, _IMAX, _UMAX)
        w = 2.5 * w_b
        tau_env = envelope_at_speed(_NP, _PSI, _RS, _LD, _LQ, _IMAX, _UMAX, w)[0]
        with pytest.raises(ValueError, match=f"{tau_env:.4g}"):
            _op(w, 1.5 * tau_env)
        r = _op(w, 0.999 * tau_env)
        _assert_feasible(r)

    def test_torque_beyond_mtpa_current_raises(self):
        with pytest.raises(ValueError, match="above I_max"):
            _op(1.0, 1e3)

    def test_zero_torque_below_and_above_no_load_base_speed(self):
        w_nl = _UMAX / (_NP * _PSI)
        low = _op(0.5 * w_nl, 0.0)
        assert (low.i_d, low.i_q, low.I_s_peak) == (0.0, 0.0, 0.0)
        assert low.regime == "mtpa" and low.pf is None
        assert low.p_cu == 0.0 and low.p_shaft == 0.0
        high = _op(2.0 * w_nl, 0.0)
        assert high.regime == "field_weakening" and high.i_q == 0.0
        assert high.i_d < 0.0
        assert high.u_mag == pytest.approx(_UMAX, rel=1e-9)
        assert high.pf is not None

    def test_zero_torque_beyond_current_limit_raises(self):
        # characteristic current psi_f/L_d = 100 A > I_max = 50 A
        with pytest.raises(ValueError, match="beyond I_max"):
            operating_point(_NP, _PSI, _RS, 1e-3, 2e-3, _IMAX, _UMAX, _UDC,
                            1e4, 0.0)

    def test_temperature_derates_both_paths(self):
        cold = _op(100.0, 5.0)
        hot = _op(100.0, 5.0, winding_temp=120.0, magnet_temp=100.0,
                  alpha_Br=-0.0012)
        assert hot.R_s_op == pytest.approx(r_s_at_operating_temp(_RS, 120.0))
        assert hot.psi_f_op == pytest.approx(
            psi_f_at_magnet_temp(_PSI, 100.0, -0.0012))
        assert hot.I_s_peak > cold.I_s_peak
        assert hot.p_cu > cold.p_cu
        _assert_feasible(hot)

    def test_magnet_temp_without_alpha_br_raises(self):
        with pytest.raises(ValueError, match="alpha_Br"):
            _op(100.0, 5.0, magnet_temp=100.0)

    def test_reverse_salient_field_weakening_matches_contour_scan(self):
        """L_d > L_q puts the contour pole at negative i_d inside the
        bracket; the result must match a dense scan of the contour."""
        psi_f, L_d, L_q = 0.08, 4e-3, 2e-3
        pole = -psi_f / (L_d - L_q)
        assert -_IMAX < pole < 0.0
        w = 3.0 * base_speed(_NP, psi_f, _RS, L_d, L_q, _IMAX, _UMAX)
        tau = 1.0
        r = operating_point(_NP, psi_f, _RS, L_d, L_q, _IMAX, _UMAX, _UDC, w, tau)
        assert r.regime == "field_weakening" and r.i_d > pole
        assert r.u_mag == pytest.approx(_UMAX, rel=1e-9)
        _assert_feasible(r)
        i_d = np.linspace(pole + 1e-6, _IMAX, 200001)
        i_q = tau / (1.5 * _NP * (psi_f + (L_d - L_q) * i_d))
        from phasesweep.models.torque_speed import dq_voltage
        ok = (np.hypot(i_d, i_q) <= _IMAX) & (
            dq_voltage(_NP, psi_f, _RS, L_d, L_q, i_d, i_q, w) <= _UMAX)
        I_scan = np.min(np.hypot(i_d, i_q)[ok])
        assert r.I_s_peak <= I_scan * (1 + 1e-9)
        assert r.I_s_peak == pytest.approx(I_scan, rel=1e-4)

    def test_outputs_are_finite_and_signed(self):
        r = _op(150.0, 6.0)
        d = r.to_dict()
        for k, v in d.items():
            if isinstance(v, float):
                assert np.isfinite(v), k
        assert d["f_e_Hz"] == pytest.approx(_NP * 150.0 / (2 * pi))
        assert 0.0 < d["pf"] <= 1.0
        assert d["gamma_deg"] > 0.0  # demagnetizing i_d on L_q > L_d


# ---------------------------------------------------------------------------
# flux-map path
# ---------------------------------------------------------------------------

class TestFluxMap:

    @pytest.mark.parametrize("L_d, L_q, tol", [(_LD, _LD, 1e-6), (_LD, _LQ, 1e-4)])
    def test_linear_map_reproduces_constant_l(self, L_d, L_q, tol):
        fm = _make_linear_flux_map(_PSI, L_d, L_q, _IMAX)
        w_b = base_speed(_NP, _PSI, _RS, L_d, L_q, _IMAX, _UMAX)
        for w, tau in ((0.3 * w_b, 5.0), (1.8 * w_b, 3.0)):
            a = operating_point(_NP, _PSI, _RS, L_d, L_q, _IMAX, _UMAX, _UDC, w, tau)
            m = operating_point(_NP, _PSI, _RS, L_d, L_q, _IMAX, _UMAX, _UDC,
                                w, tau, flux_map=fm)
            assert m.op_source == "flux_map" and a.op_source == "analytical"
            assert m.regime == a.regime
            assert m.I_s_peak == pytest.approx(a.I_s_peak, rel=tol)
            assert m.i_d == pytest.approx(a.i_d, rel=tol, abs=1e-6)
            _assert_feasible(m)

    def test_map_not_covering_current_limit_falls_back(self):
        fm = _make_linear_flux_map(_PSI, _LD, _LQ, 0.5 * _IMAX)
        r = _op(100.0, 5.0, flux_map=fm)
        assert r.op_source == "analytical"

    def test_linear_map_temperature_shift_is_exact(self):
        fm = _make_linear_flux_map(_PSI, _LD, _LQ, _IMAX)
        kw = dict(magnet_temp=100.0, alpha_Br=-0.0012)
        a = _op(150.0, 5.0, **kw)
        m = _op(150.0, 5.0, flux_map=fm, **kw)
        assert m.psi_f_op == pytest.approx(a.psi_f_op, rel=1e-12)
        assert m.I_s_peak == pytest.approx(a.I_s_peak, rel=1e-4)

    def test_map_zero_torque_and_winding_temp(self):
        fm = _make_linear_flux_map(_PSI, _LD, _LQ, _IMAX)
        w_nl = _UMAX / (_NP * _PSI)
        low = _op(0.5 * w_nl, 0.0, flux_map=fm)
        assert low.op_source == "flux_map" and low.regime == "mtpa"
        assert low.I_s_peak == 0.0 and low.pf is None
        high = _op(2.0 * w_nl, 0.0, flux_map=fm)
        a = _op(2.0 * w_nl, 0.0)
        assert high.regime == "field_weakening"
        assert high.i_d == pytest.approx(a.i_d, rel=1e-6)
        hot = _op(150.0, 5.0, flux_map=fm, winding_temp=155.0)
        cold = _op(150.0, 5.0, flux_map=fm)
        assert hot.R_s_op == pytest.approx(r_s_at_operating_temp(_RS, 155.0))
        assert hot.p_cu > cold.p_cu

    def test_non_monotone_map_finds_both_roots_and_keeps_the_smaller(self):
        """psi_d = psi_f + L_d i_d - c i_q^2 makes torque rise then fall in
        i_q, so a demand below the peak has two roots per i_d."""
        from phasesweep.models.operating_point import _MapSolver
        c = _PSI / 2700.0
        i_d = np.linspace(-_IMAX, 0, 9)
        i_q = np.linspace(0, _IMAX, 9)
        fm = FluxMap(i_d_A=tuple(i_d), i_q_A=tuple(i_q),
                     psi_d_Wb=tuple(tuple(_PSI + _LD * d - c * q * q for d in i_d) for q in i_q),
                     psi_q_Wb=tuple(tuple(_LQ * q for _ in i_d) for q in i_q))
        w = 0.2 * base_speed(_NP, _PSI, _RS, _LD, _LQ, _IMAX, _UMAX)
        rows, iq = _MapSolver(_NP, _RS, _IMAX, _UMAX, fm, w, 8.0).roots(np.array([0.0]))
        assert len(rows) == 2 and iq.min() < 20.0 < iq.max()
        r = _op(w, 8.0, flux_map=fm)
        assert r.regime == "mtpa" and r.i_q < 20.0
        assert r.I_s_peak == pytest.approx(13.716, abs=2e-3)
        _assert_feasible(r)

    def test_map_zero_torque_is_zero_torque(self, i3_params):
        """A d-axis-only point carries torque where psi_q(i_d, 0) != 0; the
        contour search must land on the true zero-torque contour."""
        from phasesweep.models.flux_map import make_flux_map_interpolators
        i_d, i_q = (-10.0, -7.0, -3.0, -1.0), (0.0, 3.0, 7.0, 10.0)
        fm = FluxMap(i_d_A=i_d, i_q_A=i_q,
                     psi_d_Wb=tuple(tuple(0.1 for _ in i_d) for _ in i_q),
                     psi_q_Wb=tuple(tuple(-0.01 for _ in i_d) for _ in i_q))
        r = operating_point(2, 0.1, 0.01, 1e-3, 1e-3, 10.0, 100.0, 100.0 * sqrt(3),
                            100.0, 0.0, flux_map=fm)
        assert r.op_source == "flux_map" and r.i_q > 0.0
        assert _energy_residual(r) < 1e-9
        r = _i3_point(i3_params, 11000, 0.0)
        pd, pq = make_flux_map_interpolators(i3_params.flux_map)
        pt = np.array([[r.i_d, r.i_q]])
        tau_map = 1.5 * i3_params.n_p * (pd(pt)[0] * r.i_q - pq(pt)[0] * r.i_d)
        assert r.regime == "field_weakening" and 0.0 < r.i_q < 1.0
        assert abs(tau_map) < 1e-9
        assert abs(1.5 * (r.u_d * r.i_d + r.u_q * r.i_q) - r.p_cu) < 1e-6

    def test_no_origin_map_with_magnet_temp_rejected_before_the_solve(self):
        i_d = np.linspace(-_IMAX, -5.0, 7)
        i_q = np.linspace(0, _IMAX, 7)
        fm = FluxMap(i_d_A=tuple(i_d), i_q_A=tuple(i_q),
                     psi_d_Wb=tuple(tuple(_PSI + _LD * d for d in i_d) for _ in i_q),
                     psi_q_Wb=tuple(tuple(_LQ * q for _ in i_d) for q in i_q))
        assert _op(100.0, 5.0, flux_map=fm).op_source == "flux_map"
        with pytest.raises(ValueError, match="origin node"):
            _op(100.0, 5.0, flux_map=fm, magnet_temp=100.0, alpha_Br=-0.0012)
        m = make_motor(n_p=_NP, psi_f=_PSI, R_s=_RS, L_d=_LD, L_q=_LQ, flux_map=fm,
                       magnet_temp=100.0, alpha_Br=-0.0012,
                       drive=DriveParams(U_DC=_UDC, MAX_I_S=_IMAX))
        with pytest.raises(ValueError, match="origin node"):
            prepare_operating_point(m)

    def test_map_entirely_below_current_limit_falls_back(self):
        i_d, i_q = (-40.0, -30.0, -25.0, -20.0), (0.0, 10.0, 20.0, 40.0)
        fm = FluxMap(i_d_A=i_d, i_q_A=i_q,
                     psi_d_Wb=tuple(tuple(_PSI for _ in i_d) for _ in i_q),
                     psi_q_Wb=tuple(tuple(0.0 for _ in i_d) for _ in i_q))
        r = operating_point(_NP, _PSI, _RS, _LD, _LQ, 10.0, _UMAX, _UDC, 10.0, 0.5, flux_map=fm)
        assert r.op_source == "analytical"

    def test_flux_map_at_magnet_temp_helper(self):
        fm = _make_linear_flux_map(_PSI, _LD, _LQ, _IMAX)
        assert flux_map_at_magnet_temp(fm, None, None) is fm
        shifted = flux_map_at_magnet_temp(fm, 100.0, -0.0012)
        assert flux_map_psi_f(shifted) == pytest.approx(
            psi_f_at_magnet_temp(_PSI, 100.0, -0.0012))
        assert shifted.psi_q_Wb == fm.psi_q_Wb
        ref = dataclasses.replace(fm, magnet_temp_ref_C=100.0)
        assert flux_map_at_magnet_temp(ref, 100.0, -0.0012).psi_d_Wb == fm.psi_d_Wb
        with pytest.raises(ValueError, match="alpha_Br"):
            flux_map_at_magnet_temp(fm, 100.0, None)

    def test_flux_map_ref_temp_round_trip(self):
        fm = _make_linear_flux_map(_PSI, _LD, _LQ, _IMAX)
        assert "magnet_temp_ref_C" not in fm.to_dict()
        assert FluxMap.from_dict(fm.to_dict()) == fm
        ref = dataclasses.replace(fm, magnet_temp_ref_C=80.0)
        assert ref.to_dict()["magnet_temp_ref_C"] == 80.0
        assert FluxMap.from_dict(ref.to_dict()) == ref
        with pytest.raises(ValueError, match="finite"):
            dataclasses.replace(fm, magnet_temp_ref_C=float("nan"))


# ---------------------------------------------------------------------------
# BMW i3 (Gierczynski 7x7 map)
# ---------------------------------------------------------------------------

_I3_DEMO_POINTS = [(1000, 250.0), (3000, 250.0), (4800, 250.0),
                   (4800, 125.0), (8000, 80.0), (11000, 40.0)]


class TestBmwI3:

    def test_i3_config_id_unchanged_by_ref_temp_field(self, i3):
        assert i3.flux_map.magnet_temp_ref_C is None
        assert "magnet_temp_ref_C" not in i3.flux_map.to_dict()

    @pytest.mark.parametrize("rpm, tau", _I3_DEMO_POINTS)
    def test_demo_points_feasible_on_map(self, i3_params, rpm, tau):
        r = _i3_point(i3_params, rpm, tau)
        assert r.op_source == "flux_map"
        assert r.I_s_peak <= i3_params.I_max * (1 + 1e-9)
        assert r.u_mag <= i3_params.U_max * (1 + 1e-9)
        assert _energy_residual(r) < 1e-9

    def test_map_and_constant_l_disagree_at_4800_125(self, i3_params):
        """The regime-clean disagreement point: both paths are MTPA and
        the secant L_d/L_q over-predicts the current by ~9 %."""
        m = _i3_point(i3_params, 4800, 125.0)
        a = _i3_point(i3_params, 4800, 125.0, flux_map=None)
        assert m.regime == a.regime == "mtpa"
        assert abs(m.I_s_peak - a.I_s_peak) / m.I_s_peak > 0.06

    def test_magnet_temperature_raises_current_and_lowers_base_speed(self, i3_params):
        cold = _i3_point(i3_params, 4800, 125.0, magnet_temp=20.0, alpha_Br=-0.0012)
        ref = _i3_point(i3_params, 4800, 125.0)
        hot = _i3_point(i3_params, 4800, 125.0, magnet_temp=100.0, alpha_Br=-0.0012)
        assert cold.I_s_peak == pytest.approx(ref.I_s_peak, rel=1e-12)
        assert hot.I_s_peak > cold.I_s_peak
        assert hot.psi_f_op < cold.psi_f_op
        # no-load base speed: the zero-torque point leaves MTPA earlier hot
        w_nl_cold = i3_params.U_max / (i3_params.n_p * cold.psi_f_op)
        w_nl_hot = i3_params.U_max / (i3_params.n_p * hot.psi_f_op)
        assert w_nl_hot > w_nl_cold
        mid = 0.5 * (w_nl_cold + w_nl_hot)
        assert _i3_point(i3_params, mid / _RPM, 0.0,
                         magnet_temp=20.0, alpha_Br=-0.0012).regime == "field_weakening"
        assert _i3_point(i3_params, mid / _RPM, 0.0,
                         magnet_temp=100.0, alpha_Br=-0.0012).regime == "mtpa"

    def test_runner_through_registry(self, i3):
        rc = RunConfig(motor=i3, model="operating_point",
                       w_mech=4800 * _RPM, tau_ref=125.0)
        r = MODEL_REGISTRY["operating_point"].fn(rc)
        assert r["op_source"] == "flux_map"
        assert r["I_s_peak"] == pytest.approx(267.957, abs=0.01)


# ---------------------------------------------------------------------------
# RunConfig / registry contract
# ---------------------------------------------------------------------------

class TestContract:

    def test_outputs_match_produces(self):
        m = make_motor(I_rated=10.0)
        r = run_operating_point(RunConfig(motor=m, model="operating_point",
                                          w_mech=50.0, tau_ref=1.0))
        assert set(r) == MODEL_REGISTRY["operating_point"].produces

    def test_run_ids_differ_across_demand(self):
        m = make_motor()
        ids = {compute_run_id(RunConfig(motor=m, model="operating_point",
                                        w_mech=w, tau_ref=t))
               for w, t in ((50.0, 1.0), (60.0, 1.0), (50.0, 2.0))}
        assert len(ids) == 3

    @pytest.mark.parametrize("kw", [{"w_mech": 50.0}, {"tau_ref": 1.0}, {}])
    def test_runner_rejects_missing_demand(self, kw):
        with pytest.raises(ValueError, match="RunConfig"):
            run_operating_point(RunConfig(motor=make_motor(), model="operating_point", **kw))

    @pytest.mark.parametrize("field", ["w_mech", "tau_ref"])
    @pytest.mark.parametrize("bad", [float("nan"), -1.0, float("inf")])
    def test_runconfig_rejects_bad_demand(self, field, bad):
        with pytest.raises(ValueError, match=field):
            RunConfig(motor=make_motor(), model="operating_point", **{field: bad})

    def test_params_factory_and_guards(self):
        p = prepare_operating_point(make_motor())
        assert p.U_max == pytest.approx(540.0 / sqrt(3))
        assert p.I_max == 20.0 and p.U_DC == 540.0
        with pytest.raises(ValueError, match="alpha_Br"):
            prepare_operating_point(make_motor(magnet_temp=100.0))
        with pytest.raises(ValueError, match="U_DC"):
            dataclasses.replace(p, U_DC=0.0)
        with pytest.raises(ValueError, match="L_d"):
            OperatingPointParams(n_p=2, psi_f=0.1, R_s=0.1, L_d=0.0, L_q=1e-3,
                                 I_max=10.0, U_max=100.0, U_DC=173.0)
        with pytest.raises(ValueError, match="needs"):
            prepare_operating_point(make_motor(L_d=None, L_q=None))
        with pytest.raises(ValueError, match="U_DC"):
            prepare_operating_point(make_motor(drive=DriveParams(MAX_I_S=20.0)))

    @pytest.mark.parametrize("kw, match", [
        ({"winding_temp": float("nan")}, "winding_temp"),
        ({"magnet_temp": float("nan"), "alpha_Br": -0.0012}, "magnet_temp"),
        ({"magnet_temp": 100.0, "alpha_Br": float("nan")}, "alpha_Br"),
        ({"magnet_temp": 100.0, "alpha_Br": 5.0}, "alpha_Br"),
        ({"magnet_temp": 220.0, "alpha_Br": -0.009}, "psi_f"),
        ({"winding_temp": -300.0}, "R_s"),
        ({"U_DC": 10.0}, "U_DC/sqrt"),
    ])
    def test_params_and_solver_reject_unphysical_derating(self, kw, match):
        base = dict(n_p=_NP, psi_f=_PSI, R_s=_RS, L_d=_LD, L_q=_LQ, I_max=_IMAX,
                    U_max=_UMAX, U_DC=_UDC)
        with pytest.raises(ValueError, match=match):
            OperatingPointParams(**{**base, **kw})
        with pytest.raises(ValueError, match=match):
            operating_point(_NP, _PSI, _RS, _LD, _LQ, _IMAX, _UMAX, kw.pop("U_DC", _UDC),
                            100.0, 0.0, **kw)

    @pytest.mark.parametrize("field", ["winding_temp_limit", "magnet_temp", "magnet_temp_max"])
    def test_motor_rejects_non_finite_temperatures(self, field):
        with pytest.raises(ValueError, match=field):
            make_motor(**{field: float("nan")})


# ---------------------------------------------------------------------------
# i3 peak-torque envelope vs the map's unknown extraction temperature
# ---------------------------------------------------------------------------

class TestI3EnvelopeBracket:
    """The Gierczynski map carries no extraction temperature. With magnets
    at 100 degC the reachable peak torque at MAX_I_S depends on what
    reference the shift is taken from: the 20 degC default derates the
    map and the 250 N.m nameplate is lost; a 100 degC reference leaves
    the map alone and the nameplate holds. Pins the bracket, not the
    machine."""

    RPM = 1000

    def test_ref_20_loses_the_nameplate(self, i3, i3_params):
        assert i3_params.flux_map.magnet_temp_ref_C is None
        with pytest.raises(ValueError, match=r"not reachable.*envelope torque.*24[0-9]\."):
            _i3_point(i3_params, self.RPM, 250.0, magnet_temp=100.0, alpha_Br=i3.alpha_Br)
        r = _i3_point(i3_params, self.RPM, 240.0, magnet_temp=100.0, alpha_Br=i3.alpha_Br)
        assert r.op_source == "flux_map" and r.regime == "mtpa"

    def test_ref_100_keeps_the_nameplate(self, i3, i3_params):
        fm = dataclasses.replace(i3_params.flux_map, magnet_temp_ref_C=100.0)
        r = _i3_point(i3_params, self.RPM, 250.0, magnet_temp=100.0, alpha_Br=i3.alpha_Br,
                      flux_map=fm)
        assert r.op_source == "flux_map" and r.I_s_peak <= i3_params.I_max


class TestI3OneTemperatureRule:
    """Every circuit model sees the same magnet temperature: the i3 at
    magnet_temp = 100 gives one peak torque from rated_torque (at
    I_rated = MAX_I_S), stall_torque, the torque_speed envelope below
    base speed, and the operating_point envelope message — 243 N.m
    against the 20 degC map reference, 258 N.m cold. Before the shared
    derating, rated_torque reported the cold 258 beside an
    operating_point that rejected 250."""

    @pytest.fixture(scope="class")
    def hot(self, i3):
        return dataclasses.replace(i3, I_rated=i3.drive.MAX_I_S, magnet_temp=100.0)

    @staticmethod
    def _peaks(motor):
        def run(model):
            return MODEL_REGISTRY[model].fn(RunConfig(motor=motor, model=model))

        ts = run("torque_speed")
        return {
            "rated_torque": run("rated_torque")["tau_mtpa"],
            "stall_torque": run("stall_torque")["tau_stall"],
            "torque_speed": ts["tau_curve_peak"][0],
        }, ts["envelope_source"]

    def test_hot_peak_agrees_across_models(self, hot):
        peaks, source = self._peaks(hot)
        assert source == "flux_map"
        assert all(p == pytest.approx(243.2, abs=0.1) for p in peaks.values()), peaks
        p = prepare_operating_point(hot)
        with pytest.raises(ValueError, match=r"envelope torque at this speed is 243\.2"):
            operating_point(p.n_p, p.psi_f, p.R_s, p.L_d, p.L_q, p.I_max, p.U_max,
                            p.U_DC, 1000 * _RPM, 250.0, flux_map=p.flux_map,
                            magnet_temp=p.magnet_temp, alpha_Br=p.alpha_Br)

    def test_cold_peak_is_the_map_value(self, hot):
        peaks, _ = self._peaks(dataclasses.replace(hot, magnet_temp=None))
        assert all(p == pytest.approx(258.1, abs=0.1) for p in peaks.values()), peaks

    def test_hot_and_cold_run_ids_differ(self, hot):
        cold = dataclasses.replace(hot, magnet_temp=None)
        for model in ("rated_torque", "stall_torque", "torque_speed"):
            assert (compute_run_id(RunConfig(motor=hot, model=model))
                    != compute_run_id(RunConfig(motor=cold, model=model)))
