"""Generic soft-iron B-H curve and magnetic constants.

Shared by the radial 2D solver, the axial multi-plane solver, and the
3D wedge solver — one representative soft-steel curve, not
grade-specific data.

The table is interpolated with a monotone PCHIP spline (fem model v8;
previously piecewise-linear np.interp). Monotone, C1 H(B) is the
standard prerequisite for Newton BH solves — it keeps the differential
reluctivity finite and the assembled Jacobian well-posed — and the
spline coefficients give the Newton solvers their d nu/dB and dB/dH
analytically instead of by finite difference.
"""

from __future__ import annotations

from math import pi

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import PchipInterpolator

_MU0: float = 4e-7 * pi

# Generic soft magnetic steel B-H curve (representative, not grade-specific).
# Source: composite of textbook soft iron data; suitable for design exploration.
_BH_B = np.array([
    0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.1, 1.2,
    1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2,
])
_BH_H = np.array([
    0., 25., 40., 60., 80., 110., 160., 210., 310.,
    480., 900., 1800., 3800., 7500., 15000., 30000., 60000., 120000., 200000.,
])

# PCHIP preserves the monotonicity of the (monotone) table, so H(B) is
# strictly increasing and dH/dB > 0 on the table interior.
_BH_SPLINE = PchipInterpolator(_BH_B, _BH_H)
_BH_SPLINE_D = _BH_SPLINE.derivative()


def _bh_H_of_B(B_mag: NDArray) -> NDArray:
    """H(|B|): PCHIP inside the table, vacuum differential slope beyond.

    Beyond the table end the curve continues with
    H = H_end + (B − B_end)/μ0, so the secant ν = H/B stays monotone and
    approaches 1/μ0 from below (clamping H instead would make deeply
    saturated iron look magnetically softer with rising B).
    """
    B = np.asarray(B_mag, dtype=float)
    H = _BH_SPLINE(np.clip(B, 0.0, _BH_B[-1]))
    beyond = B > _BH_B[-1]
    if np.any(beyond):
        H = np.where(beyond, _BH_H[-1] + (B - _BH_B[-1]) / _MU0, H)
    return H


def _bh_Hprime_of_B(B_mag: NDArray) -> NDArray:
    """dH/dB(|B|): spline derivative inside the table, 1/μ0 beyond."""
    B = np.asarray(B_mag, dtype=float)
    Hp = _BH_SPLINE_D(np.clip(B, 0.0, _BH_B[-1]))
    return np.where(B > _BH_B[-1], 1.0 / _MU0, Hp)


def _bh_nu(B_mag: NDArray) -> NDArray:
    """Reluctivity ν(|B|) = H(|B|)/|B| from the generic soft iron B-H curve.

    Secant reluctivity on the PCHIP curve (see ``_bh_H_of_B`` for the
    beyond-table extension), clipped to [ν₀, 1/μ0] where ν₀ is the
    first-segment secant slope H₁/B₁.
    """
    H = _bh_H_of_B(B_mag)
    B_safe = np.maximum(B_mag, 1e-12)
    nu = H / B_safe
    nu_0 = _BH_H[1] / _BH_B[1]
    return np.clip(nu, nu_0, 1 / _MU0)


def _bh_nu_prime(B_mag: NDArray) -> NDArray:
    """d ν/dB of the clipped secant reluctivity ``_bh_nu``, analytic.

    ν' = (H'(B)·B − H(B))/B² from the spline coefficients inside the
    table and the vacuum slope beyond; zero wherever the [ν₀, 1/μ0]
    clip is active (matching what the previous finite difference of the
    clipped ν measured).
    """
    B = np.asarray(B_mag, dtype=float)
    H = _bh_H_of_B(B)
    B_safe = np.maximum(B, 1e-12)
    nu_raw = H / B_safe
    nu_0 = _BH_H[1] / _BH_B[1]
    unclipped = (nu_raw > nu_0) & (nu_raw < 1 / _MU0)
    nu_p = (_bh_Hprime_of_B(B) * B_safe - H) / B_safe**2
    return np.where(unclipped, nu_p, 0.0)


def _bh_B_of_H(H_mag: NDArray) -> NDArray:
    """B(|H|): exact numeric inverse of ``_bh_H_of_B``.

    Bisection on the monotone spline inside the table (60 halvings of
    [0, B_end] ≈ 2e-18 T bracket, far below solver tolerance), the
    mirrored vacuum-slope linear form beyond the table end.
    """
    H = np.asarray(H_mag, dtype=float)
    B = np.zeros_like(H)
    beyond = H > _BH_H[-1]
    if np.any(beyond):
        B = np.where(beyond, _BH_B[-1] + (H - _BH_H[-1]) * _MU0, B)
    inside = (~beyond) & (H > 0.0)
    if np.any(inside):
        Hi = H[inside]
        lo = np.zeros_like(Hi)
        hi = np.full_like(Hi, _BH_B[-1])
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            under = _BH_SPLINE(mid) < Hi
            lo = np.where(under, mid, lo)
            hi = np.where(under, hi, mid)
        B[inside] = 0.5 * (lo + hi)
    return B


def _bh_mu_h(H_mag: NDArray) -> NDArray:
    """Permeability μ(|H|) — the exact H-space inverse of ``_bh_nu``.

    The scalar-potential wedge solver knows |H| = |grad phi| directly, so
    its BH fixed point μ = 1/ν(μ·|H|) resolves to μ = 1/ν(B(|H|)) with
    B(|H|) the numeric inverse of the same spline (``_bh_B_of_H``) —
    inverse by construction, so the wedge Picard and Newton iterations
    share one solution with the 2D solver's curve. ν's [ν₀, 1/μ0] clip
    inverts to the [μ0, μ_max] bounds automatically. At H = 0 exactly,
    B = 0 gives ν = ν₀ and hence μ = μ_max; for H → 0⁺ the limit is
    1/H'(0) (the spline tangent sits above the first-segment secant), a
    benign measure-zero discontinuity at the origin.
    """
    return 1.0 / _bh_nu(_bh_B_of_H(H_mag))


def _bh_dB_dH(H_mag: NDArray) -> NDArray:
    """Tangent slope dB/dH(|H|) of the effective curve B(H) = μ(H)·H.

    Analytic counterpart of the wedge Newton's previous central
    difference: 1/H'(B(H)) from the spline where ν is unclipped, the
    plateau secant (μ_max at the low-H clip, μ0 at the vacuum cap)
    where the clip is active — there the effective B(H) is linear in H,
    so the tangent equals the secant and the anisotropic Jacobian
    correction vanishes. Floored at μ0 so the Jacobian stays SPD.
    """
    H = np.asarray(H_mag, dtype=float)
    B = _bh_B_of_H(H)
    B_safe = np.maximum(B, 1e-12)
    nu_raw = _bh_H_of_B(B) / B_safe
    nu_0 = _BH_H[1] / _BH_B[1]
    unclipped = (nu_raw > nu_0) & (nu_raw < 1 / _MU0)
    dB_dH = np.where(unclipped, 1.0 / _bh_Hprime_of_B(B), 1.0 / nu_0)
    dB_dH = np.where(nu_raw >= 1 / _MU0, _MU0, dB_dH)
    return np.maximum(dB_dH, _MU0)
