"""make_dq_maps: the complex d-q flux/current callables built from a FluxMap
for the saturated drive-sim machine model (no motulator needed)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from phasesweep.machines.configs import load_motor
from phasesweep.machines.motor import FluxMap, Motor
from phasesweep.models.flux_map import (
    _invert_on_grid,
    cold_psi_f,
    make_dq_maps,
    make_flux_map_interpolators,
    map_covers,
)
from phasesweep.models.rated_torque import _mtpa_result, run_rated_torque, run_stall_torque
from phasesweep.sweep_types import RunConfig

ROOT = Path(__file__).resolve().parent.parent.parent
BMW = ROOT / "motors" / "bmw_i3.toml"

PSI_F, L_D, L_Q, I_MAX = 0.1, 4e-3, 6e-3, 20.0


def _linear_map(psi_f=PSI_F, L_d=L_D, L_q=L_Q, I_max=I_MAX, n=5) -> FluxMap:
    i_d = tuple(float(x) for x in np.linspace(-I_max, 0, n))
    i_q = tuple(float(x) for x in np.linspace(0, I_max, n))
    return FluxMap(
        i_d_A=i_d, i_q_A=i_q,
        psi_d_Wb=tuple(tuple(psi_f + L_d * d for d in i_d) for _ in i_q),
        psi_q_Wb=tuple(tuple(L_q * q for _ in i_d) for q in i_q),
    )


@pytest.fixture(scope="module")
def bmw_maps():
    fm = load_motor(BMW).flux_map
    assert fm is not None
    return fm, *make_dq_maps(fm)


class TestLinearMap:
    """A constant-L surface is reproduced exactly, inside and outside the
    grid, in both directions."""

    def test_flux_matches_constant_l_everywhere(self):
        psi, _ = make_dq_maps(_linear_map())
        i = np.array([-5 + 3j, 0j, -25 + 30j, 4 - 8j, -20 + 0j])
        expected = PSI_F + L_D * i.real + 1j * L_Q * i.imag
        np.testing.assert_allclose(psi(i), expected, atol=1e-12)

    def test_current_inverts_flux(self):
        psi, cur = make_dq_maps(_linear_map())
        i = np.array([-5 + 3j, -19 + 19j, 2 - 4j, 0j])
        np.testing.assert_allclose(cur(psi(i)), i, atol=1e-6)

    def test_scalar_in_scalar_out(self):
        psi, cur = make_dq_maps(_linear_map())
        p = psi(-3 + 4j)
        assert isinstance(p, complex)
        assert isinstance(cur(p), complex)
        assert cur(p) == pytest.approx(-3 + 4j, abs=1e-6)

    def test_psi_f_is_map_origin(self):
        psi, cur = make_dq_maps(_linear_map())
        assert psi(0j) == pytest.approx(PSI_F)
        assert cur(PSI_F + 0j) == pytest.approx(0j, abs=1e-6)

    def test_cached_per_map(self):
        fm = _linear_map()
        assert make_dq_maps(fm) is make_dq_maps(_linear_map())
        assert make_dq_maps(fm) is not make_dq_maps(_linear_map(L_q=7e-3))


class TestBmwMap:

    def test_inside_grid_matches_cubic_interpolators(self, bmw_maps):
        """The dq-maps surface (mirrored to full i_q span) must agree with
        the one-quadrant cubic interpolators inside the grid. Not exact:
        the mirrored grid changes cubic spline boundary conditions, but
        the difference is negligible (<0.01%)."""
        fm, psi, _ = bmw_maps
        pd, pq = make_flux_map_interpolators(fm)
        i = np.array([-401 + 399j, -100 + 50j, -550 + 580j, 0 + 0j])
        pts = np.column_stack((i.real, i.imag))
        np.testing.assert_allclose(psi(i), pd(pts) + 1j * pq(pts), rtol=5e-3)

    def test_round_trip_within_grid(self, bmw_maps):
        _, psi, cur = bmw_maps
        i_d, i_q = np.meshgrid(np.linspace(-600, 0, 13),
                               np.linspace(-600, 600, 13), indexing="ij")
        i = (i_d + 1j * i_q).ravel()
        err = np.abs(cur(psi(i)) - i)
        assert err.max() < 2.0
        assert err.mean() < 0.3

    def test_negative_iq_mirrors(self, bmw_maps):
        _, psi, _ = bmw_maps
        p_pos = psi(-200 + 300j)
        p_neg = psi(-200 - 300j)
        p_zero = psi(-200 + 0j)
        assert p_neg.real == pytest.approx(p_pos.real, rel=1e-10)
        assert p_neg.imag == pytest.approx(2 * p_zero.imag - p_pos.imag, rel=1e-10)

    def test_dpsi_d_diq_zero_at_iq_zero(self, bmw_maps):
        """∂ψ_d/∂i_q at i_q = 0 must be ~0 (even symmetry). The old
        runtime mirror gave ±8.2e-6 Wb/A; the data-mirrored surface
        should give machine-epsilon-level residual."""
        _, psi, _ = bmw_maps
        h = 1e-3
        for i_d in [-400.0, -200.0, 0.0]:
            dp = (psi(i_d + h * 1j) - psi(i_d - h * 1j)) / (2 * h)
            assert abs(dp.real) < 1e-10

    def test_beyond_grid_is_linear_and_continuous(self, bmw_maps):
        _, psi, _ = bmw_maps
        p0, p1, p2 = psi(0j), psi(50 + 0j), psi(100 + 0j)
        p_in = psi(-1e-6 + 0j)
        assert abs(p0 - p_in) < 1e-6
        assert p2 - p1 == pytest.approx(p1 - p0, rel=1e-6)
        assert p1.real > p0.real

    def test_current_map_extrapolates_beyond_flux_grid(self, bmw_maps):
        _, psi, cur = bmw_maps
        i = cur(0j)
        assert np.isfinite(i)
        assert i.real < -600

    def test_beyond_grid_q_edge_linear_and_continuous(self, bmw_maps):
        """Linear extension on the +q edge (beyond i_q_max) and a corner
        (beyond both axes) — complements the +d-axis-only test above."""
        _, psi, _ = bmw_maps
        p0 = psi(-300 + 600j)
        p1 = psi(-300 + 650j)
        p2 = psi(-300 + 700j)
        assert abs(p0 - psi(-300 + (600 - 1e-6) * 1j)) < 1e-4
        assert p2 - p1 == pytest.approx(p1 - p0, rel=1e-4)
        assert p1.imag > p0.imag

        c0 = psi(50 + 650j)
        c1 = psi(100 + 700j)
        c2 = psi(150 + 750j)
        assert c2 - c1 == pytest.approx(c1 - c0, rel=1e-3)


class TestInvertConvergenceGuard:

    def test_unconverged_newton_raises(self):
        """_invert_on_grid raises ValueError when max_iter is too low."""
        psi, _ = make_dq_maps(_linear_map())
        target = np.array([PSI_F - L_D * 5 + 1j * L_Q * 3])
        i0 = np.array([0j])
        with pytest.raises(ValueError, match="inverse Newton did not converge"):
            _invert_on_grid(psi, target, i0, step_max=0.1, max_iter=1)


ALPHA = -0.0012


class TestMapReferenceTemperature:
    """The map origin is psi_f at magnet_temp_ref_C; the scalar derating
    rule is referenced to 20 °C, so a hot-extracted map is referred back."""

    def test_cold_psi_f_refers_to_20C(self):
        fm = replace(_linear_map(), magnet_temp_ref_C=80.0)
        assert cold_psi_f(fm, ALPHA) == pytest.approx(PSI_F / (1 + ALPHA * 60.0))

    @pytest.mark.parametrize("ref, alpha", [(None, ALPHA), (20.0, ALPHA), (80.0, None)])
    def test_cold_psi_f_passthrough(self, ref, alpha):
        fm = replace(_linear_map(), magnet_temp_ref_C=ref)
        assert cold_psi_f(fm, alpha) == PSI_F

    def test_map_only_motor_at_ref_temp_is_consistent(self):
        """Non-salient linear map at its own reference temperature: the
        scalar k_T (from psi_f) and the map MTPA torque must agree. Taking
        the map origin as 20 °C put them 7 % apart."""
        fm = replace(_linear_map(L_q=L_D), magnet_temp_ref_C=80.0)
        m = Motor(name="map-only", n_p=4, geometry=None, flux_map=fm, I_rated=10.0,
                  magnet_temp=80.0, alpha_Br=ALPHA)
        r = run_rated_torque(RunConfig(motor=m, model="rated_torque"))
        assert r["mtpa_source"] == "flux_map"
        assert r["k_T_effective"] == pytest.approx(r["k_T"], rel=1e-9)
        assert r["k_T"] == pytest.approx(1.5 * 4 * PSI_F, rel=1e-12)


    def test_torque_speed_psi_f_at_ref_temp_is_map_origin(self):
        """i3 map declared at 80 °C, run at 80 °C with psi_f unset: the
        envelope psi_f is the map origin (was −7.2 %, audit V-08)."""
        from phasesweep.models.flux_map import flux_map_psi_f
        from phasesweep.models.torque_speed import prepare_torque_speed

        i3 = load_motor(BMW)
        assert i3.flux_map is not None
        fm = replace(i3.flux_map, magnet_temp_ref_C=80.0)
        m = replace(i3, psi_f=None, flux_map=fm, magnet_temp=80.0, alpha_Br=ALPHA)
        assert prepare_torque_speed(m).psi_f == pytest.approx(flux_map_psi_f(fm), rel=1e-12)


class TestMtpaCoverage:

    def test_map_covers_rule(self):
        fm = _linear_map()
        assert map_covers(fm, I_MAX)
        assert not map_covers(fm, I_MAX * 1.01)

    def test_past_grid_falls_back_to_analytical(self):
        """Past the grid the feasible gamma window narrows and the numeric
        optimum pins to the edge; the analytical path runs instead."""
        r = _mtpa_result(4, PSI_F, 1.25 * I_MAX, L_D, L_Q, _linear_map())
        assert r["mtpa_source"] == "analytical"
        inside = _mtpa_result(4, PSI_F, I_MAX, L_D, L_Q, _linear_map())
        assert inside["mtpa_source"] == "flux_map"

    def test_stall_reports_electromagnetic_source(self):
        """i3: I_stall (drive limit) is on the map; the electromagnetic
        stall current (tens of kA) is far past it."""
        m = load_motor(BMW)
        r = run_stall_torque(RunConfig(motor=m, model="stall_torque"))
        assert r["mtpa_source"] == "flux_map"
        assert r["mtpa_source_electromagnetic"] == "analytical"
