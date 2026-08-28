"""Tests for fem_materials.py — the shared B-H curve and reluctivity spline."""

from importlib.util import find_spec

import pytest

if not find_spec("ngsolve"):
    pytest.skip("requires phasesweep[fem] (ngsolve not installed)", allow_module_level=True)

import numpy as np

from phasesweep.solvers.fem_materials import _BH_B, _BH_H, _MU0, _bh_nu
from tests.conftest import requires_fem

# Module-level guard covers missing ngsolve; the marker makes the module
# selectable by lane.
pytestmark = requires_fem


# ---------------------------------------------------------------------------
# B-H curve and _bh_nu (Issue 1)
# ---------------------------------------------------------------------------

def test_bh_table_monotonic():
    assert np.all(np.diff(_BH_H) > 0)


def test_bh_spline_monotone_dense():
    """H(B) must be strictly increasing on a dense grid — the Newton
    prerequisite (monotone, C1 BH interpolant), guaranteed by PCHIP on
    the monotone table but guarded here at spline level, not just
    table level."""
    from phasesweep.solvers.fem_materials import _bh_H_of_B

    B = np.linspace(0.0, 3.0, 30001)
    assert np.all(np.diff(_bh_H_of_B(B)) > 0)


def test_bh_nu_prime_matches_finite_difference():
    """Analytic d nu/dB agrees with a central difference of the clipped
    nu everywhere except the two grid points straddling a clip
    transition (where the FD smears the kink)."""
    from phasesweep.solvers.fem_materials import _bh_nu_prime

    B = np.linspace(0.02, 3.0, 6000)
    h = 1e-7
    fd = (_bh_nu(B + h) - _bh_nu(B - h)) / (2 * h)
    an = _bh_nu_prime(B)
    active = an != 0.0
    interior = active & np.roll(active, 1) & np.roll(active, -1)
    np.testing.assert_allclose(an[interior], fd[interior], rtol=1e-4)
    # and the clipped plateau reports exactly zero slope
    assert np.all(fd[~active & np.roll(~active, 1) & np.roll(~active, -1)]
                  == 0.0)


def test_bh_nu_at_zero():
    nu = _bh_nu(np.array([0.0]))
    assert np.isfinite(nu[0])
    assert nu[0] > 0


def test_bh_nu_at_saturation():
    nu = _bh_nu(np.array([2.5]))
    nu_linear = _BH_H[1] / _BH_B[1]
    assert nu[0] > 10 * nu_linear


def test_bh_nu_linear_regime():
    nu = _bh_nu(np.array([0.1]))
    mu_r_initial = _BH_B[1] / (_MU0 * _BH_H[1])
    expected_nu = 1 / (_MU0 * mu_r_initial)
    np.testing.assert_allclose(nu[0], expected_nu, rtol=0.10)


# ---------------------------------------------------------------------------
# Inverse curve and Newton tangent (C4 mutation-run kills: _bh_B_of_H and
# _bh_dB_dH previously had no fast test — only the slow wedge Newton
# solves exercised them)
# ---------------------------------------------------------------------------

def test_bh_B_of_H_inverts_H_of_B():
    """Round trip B -> H -> B over the table interior and the vacuum-slope
    extension: the bisection (60 halvings, ~2e-18 bracket) and the
    mirrored beyond-table linear form must invert exactly."""
    from phasesweep.solvers.fem_materials import _bh_B_of_H, _bh_H_of_B

    B = np.concatenate([
        np.linspace(1e-6, _BH_B[-1] - 1e-9, 500),
        np.linspace(_BH_B[-1] + 0.01, 5.0, 50),  # vacuum-slope extension
    ])
    np.testing.assert_allclose(_bh_B_of_H(_bh_H_of_B(B)), B,
                               rtol=1e-10, atol=1e-12)
    # H = 0 maps to B = 0 exactly
    assert _bh_B_of_H(np.array([0.0]))[0] == 0.0


def test_bh_dB_dH_matches_finite_difference():
    """The wedge Newton's analytic tangent dB/dH agrees with a central
    difference of the effective curve B(H) = mu(H)*H away from the clip
    kinks, sits on the mu_max plateau secant at low H, and floors at mu0
    deep in saturation."""
    from phasesweep.solvers.fem_materials import (
        _bh_B_of_H,
        _bh_dB_dH,
        _bh_H_of_B,
        _bh_mu_h,
    )

    H = np.logspace(0.5, 5.5, 2000)
    h = H * 1e-6
    fd = (_bh_B_of_H(H + h) - _bh_B_of_H(H - h)) / (2 * h)
    an = _bh_dB_dH(H)
    # compare where the clip is comfortably inactive (analytic == spline
    # tangent); wide margins keep the FD step clear of the plateau kinks
    nu_0 = _BH_H[1] / _BH_B[1]
    B = _bh_B_of_H(H)
    nu_raw = _bh_H_of_B(B) / np.maximum(B, 1e-12)
    interior = ((nu_raw > nu_0 * 1.2) & (nu_raw < 0.3 / _MU0)
                & (H < _BH_H[-1] * 0.98) & (an > _MU0 * 1.01))
    assert interior.sum() > 500
    np.testing.assert_allclose(an[interior], fd[interior], rtol=1e-3)
    # Near the table end the spline is briefly steeper than vacuum
    # (1/H' < mu0): the SPD floor must bind there — analytic exactly mu0
    # while the raw spline tangent sits below it.
    floored = (an == _MU0) & (H < _BH_H[-1] * 0.98)
    assert floored.sum() > 10
    assert np.all(fd[floored] < _MU0)
    # H -> 0 corner: mu lands on 1/H'(0) = 1/300 (the PCHIP end slope,
    # just under the mu_max = 1/250 secant clip) — the essential contract
    # is that it never falls to the wrong (mu0) end of the clip.
    H_zero = np.array([1e-9, 1e-6])
    np.testing.assert_allclose(_bh_mu_h(H_zero), 1.0 / 300.0, rtol=1e-6)
    np.testing.assert_allclose(_bh_dB_dH(H_zero), 1.0 / 300.0, rtol=1e-6)
    # floor: never below mu0
    assert np.all(_bh_dB_dH(np.logspace(0, 6, 200)) >= _MU0 * (1 - 1e-15))
