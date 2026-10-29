"""Tests for fem_materials.py — the shared B-H curve and reluctivity spline."""

from importlib.util import find_spec

import pytest

if not find_spec("ngsolve"):
    pytest.skip("requires phasesweep[fem] (ngsolve not installed)", allow_module_level=True)

import numpy as np

from phasesweep.solvers.fem_materials import _BH_B, _BH_H, MU0, bh_nu
from tests.conftest import requires_fem

# Module-level guard covers missing ngsolve; the marker makes the module
# selectable by lane.
pytestmark = requires_fem


# ---------------------------------------------------------------------------
# B-H curve and bh_nu (Issue 1)
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
    from phasesweep.solvers.fem_materials import bh_nu_prime

    B = np.linspace(0.02, 3.0, 6000)
    h = 1e-7
    fd = (bh_nu(B + h) - bh_nu(B - h)) / (2 * h)
    an = bh_nu_prime(B)
    active = an != 0.0
    interior = active & np.roll(active, 1) & np.roll(active, -1)
    np.testing.assert_allclose(an[interior], fd[interior], rtol=1e-4)
    # and the clipped plateau reports exactly zero slope
    assert np.all(fd[~active & np.roll(~active, 1) & np.roll(~active, -1)]
                  == 0.0)


def test_bh_nu_at_zero():
    nu = bh_nu(np.array([0.0]))
    assert np.isfinite(nu[0])
    assert nu[0] > 0


def test_bh_nu_at_saturation():
    nu = bh_nu(np.array([2.5]))
    nu_linear = _BH_H[1] / _BH_B[1]
    assert nu[0] > 10 * nu_linear


def test_bh_nu_linear_regime():
    nu = bh_nu(np.array([0.1]))
    mu_r_initial = _BH_B[1] / (MU0 * _BH_H[1])
    expected_nu = 1 / (MU0 * mu_r_initial)
    np.testing.assert_allclose(nu[0], expected_nu, rtol=0.10)


# ---------------------------------------------------------------------------
# Inverse curve and Newton tangent (C4 mutation-run kills: _bh_B_of_H and
# bh_dB_dH previously had no fast test — only the slow wedge Newton
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
        _bh_H_of_B,
        bh_dB_dH,
        bh_mu_h,
    )

    H = np.logspace(0.5, 5.5, 2000)
    h = H * 1e-6
    fd = (_bh_B_of_H(H + h) - _bh_B_of_H(H - h)) / (2 * h)
    an = bh_dB_dH(H)
    # compare where the clip is comfortably inactive (analytic == spline
    # tangent); wide margins keep the FD step clear of the plateau kinks
    nu_0 = _BH_H[1] / _BH_B[1]
    B = _bh_B_of_H(H)
    nu_raw = _bh_H_of_B(B) / np.maximum(B, 1e-12)
    interior = ((nu_raw > nu_0 * 1.2) & (nu_raw < 0.3 / MU0)
                & (H < _BH_H[-1] * 0.98) & (an > MU0 * 1.01))
    assert interior.sum() > 500
    np.testing.assert_allclose(an[interior], fd[interior], rtol=1e-3)
    # Near the table end the spline is briefly steeper than vacuum
    # (1/H' < mu0): the SPD floor must bind there — analytic exactly mu0
    # while the raw spline tangent sits below it.
    floored = (an == MU0) & (H < _BH_H[-1] * 0.98)
    assert floored.sum() > 10
    assert np.all(fd[floored] < MU0)
    # H -> 0 corner: mu lands on 1/H'(0) = 1/300 (the PCHIP end slope,
    # just under the mu_max = 1/250 secant clip) — the essential contract
    # is that it never falls to the wrong (mu0) end of the clip.
    H_zero = np.array([1e-9, 1e-6])
    np.testing.assert_allclose(bh_mu_h(H_zero), 1.0 / 300.0, rtol=1e-6)
    np.testing.assert_allclose(bh_dB_dH(H_zero), 1.0 / 300.0, rtol=1e-6)
    # floor: never below mu0
    assert np.all(bh_dB_dH(np.logspace(0, 6, 200)) >= MU0 * (1 - 1e-15))


# ---------------------------------------------------------------------------
# BHCurve: per-motor tables and the cycle-averaged reluctivity (F5)
# ---------------------------------------------------------------------------

from phasesweep.solvers.fem_materials import (  # noqa: E402
    GENERIC_BH,
    BHCurve,
    bh_curve_for,
    bh_dB_dH,
    bh_mu_h,
    bh_nu_prime,
)

_M800_ROWS = [[68, 0.10], [157, 1.00], [283, 1.40], [521, 1.50], [1489, 1.60],
              [4057, 1.70]]


def test_generic_wrappers_are_the_generic_curve():
    B = np.linspace(0.0, 2.5, 400)
    H = np.logspace(0, 6, 300)
    np.testing.assert_array_equal(bh_nu(B), GENERIC_BH.nu(B))
    np.testing.assert_array_equal(bh_nu_prime(B), GENERIC_BH.nu_prime(B))
    np.testing.assert_array_equal(bh_mu_h(H), GENERIC_BH.mu_h(H))
    np.testing.assert_array_equal(bh_dB_dH(H), GENERIC_BH.dB_dH(H))
    assert GENERIC_BH.mu_r_initial == pytest.approx(1 / (MU0 * 250.0))


def test_from_rows_prepends_origin_and_validates():
    c = BHCurve.from_rows(_M800_ROWS)
    assert c.B[0] == 0.0 and c.H[0] == 0.0 and len(c.B) == len(_M800_ROWS) + 1
    assert c.mu_r_initial == pytest.approx(1170.3, rel=1e-3)
    with pytest.raises(ValueError, match="strictly"):
        BHCurve.from_rows([[68, 0.1], [60, 0.2], [100, 0.3]])
    with pytest.raises(ValueError, match="strictly"):
        BHCurve.from_rows([[68, 0.1], [80, 0.1], [100, 0.3]])
    with pytest.raises(ValueError, match="pairs"):
        BHCurve.from_rows([[68, 0.1, 3]])
    with pytest.raises(ValueError, match="origin"):
        BHCurve(B=np.array([0.1, 0.5, 1.0]), H=np.array([68.0, 120.0, 157.0]))


def test_nu_secant_keeps_rising_permeability():
    """The floored ``nu`` reads M800-50A as linear at μ_r 1170 up to
    ~1.5 T; the unfloored secant carries the μ_max ≈ 5000 hump the
    magnetizing inductance depends on. Both cap at 1/μ0."""
    c = BHCurve.from_rows(_M800_ROWS)
    B = np.array([0.0, 0.1, 1.0, 1.4, 1.6, 3.0])
    mu_r = 1 / (MU0 * c.nu_secant(B))
    assert mu_r[1] == pytest.approx(1170.3, rel=1e-3)
    assert mu_r[2] == pytest.approx(5069, rel=1e-3)
    assert mu_r[2] > mu_r[3] > mu_r[4] > 1.0
    assert mu_r[0] == pytest.approx(1 / (MU0 * c.Hprime_of_B(np.zeros(1))[0]))
    assert c.nu_secant(np.array([500.0]))[0] == pytest.approx(1 / MU0, rel=1e-2)
    assert np.all(c.nu(B[1:4]) == pytest.approx(c.nu_0))


def test_nu_harmonic_limits():
    """Circular locus = secant at the constant |B|; a linear medium is
    returned unchanged; a pulsating field returns the fundamental of H
    over B_peak; ν is monotone in the drive once past the knee."""
    c = BHCurve.from_rows(_M800_ROWS)
    B = np.array([0.3, 1.0, 1.45, 1.8])
    # circular: S = 2|B|², D = 0
    np.testing.assert_allclose(c.nu_harmonic(2 * B**2, 0 * B), c.nu_secant(B), rtol=1e-12)
    # linear medium (H = ν B straight line) for any locus
    nu_lin = 1 / (MU0 * 800.0)
    lin = BHCurve(B=np.array([0.0, 1.0, 2.0]), H=np.array([0.0, 1.0, 2.0]) * nu_lin)
    S = np.array([0.5, 2.0, 3.0])
    D = np.array([0.0, 1.0, 3.0])
    np.testing.assert_allclose(lin.nu_harmonic(S, D), nu_lin, rtol=1e-12)
    # pulsating: S = D = B_peak²; fundamental of H(B_peak cos θ) / B_peak
    theta = np.linspace(0, 2 * np.pi, 20001)
    for Bp in B:
        H_t = np.sign(np.cos(theta)) * c.H_of_B(Bp * np.abs(np.cos(theta)))
        H_1 = np.trapezoid(H_t * np.cos(theta), theta) / np.pi
        assert c.nu_harmonic(np.array([Bp**2]), np.array([Bp**2]))[0] == pytest.approx(
            min(H_1 / Bp, 1 / MU0), rel=2e-3)
    # zero field: the origin tangent, finite
    assert 0 < c.nu_harmonic(np.zeros(2), np.zeros(2))[0] < 1 / MU0
    # past the knee, more drive = more reluctivity
    S_ramp = np.linspace(1.5, 3.0, 30) ** 2
    nu_ramp = c.nu_harmonic(S_ramp, S_ramp)
    assert np.all(np.diff(nu_ramp) > 0)


def test_bh_curve_for_motor():
    from phasesweep.machines.motor import Motor

    plain = Motor(name="m", geometry=None, n_p=2)
    assert bh_curve_for(plain) is GENERIC_BH
    own = Motor(name="m", geometry=None, n_p=2, bh_curve=tuple(tuple(r) for r in _M800_ROWS))
    assert bh_curve_for(own).mu_r_initial == pytest.approx(1170.3, rel=1e-3)
