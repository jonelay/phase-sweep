"""Analytical air-gap field model and slotting/end-effect corrections.

Primary reference: Zhu, Howe & Chan (2002) improved analytical model
(see docs/references.md). Carter factor per Zhu & Howe (1993) Part III.
"""

from __future__ import annotations

import warnings
from math import atan, exp, log, pi, sin, sqrt, tanh

import numpy as np
from numpy.typing import NDArray

from phasesweep.machines.geometry import Geometry


def zhu_howe_Br(
    theta: NDArray[np.floating],
    n_p: int,
    B_rem: float,
    r_eval: float | None = None,
    *,
    r_stator: float,
    r_magnet: float,
    r_rotor: float,
    mu_r_pm: float = 1.05,
    alpha_p: float = 1.0,
) -> NDArray[np.floating]:
    """Zhu, Howe & Chan (2002) improved-model air-gap radial flux density.

    Convention: B_rem is physical remanence; the source is the fundamental
    of square-wave radial magnetization, M_1 = (4/π)(B_rem/μ0)·sin(πα_p/2)
    (paper Eq 7a M_rn at n=1). Matches the FEM solver's uniform-magnitude
    per-pole-alternating arc source exactly at the fundamental.

    The radii are required keywords: silent physical defaults (a 0.70 m
    machine) were a footgun for partial-kwargs callers.
    """
    if n_p < 2:
        raise ValueError(f"n_p={n_p} not supported (singularity at n²-1=0)")

    if r_eval is None:
        r_eval = (r_stator + r_magnet) / 2

    n = n_p

    # Square-wave magnetization fundamental: (4/π)·sin(πα_p/2) is the
    # Fourier coefficient of a ±1 square wave with pole-arc ratio α_p
    # (paper Eq 7a). k_mp = 1.0 when α_p = 1 (full-pitch magnets).
    k_mp = sin(pi * alpha_p / 2)
    bp = n * (4 / pi) * B_rem * k_mp / (n**2 - 1)

    K = _zhu_howe_order_amplitude(
        n, bp, r_eval,
        r_stator=r_stator, r_magnet=r_magnet, r_rotor=r_rotor,
        mu_r_pm=mu_r_pm,
    )
    return K * np.cos(n * theta)


def _zhu_howe_order_amplitude(
    n: int | float, bp: float, r: float, *,
    r_stator: float, r_magnet: float, r_rotor: float, mu_r_pm: float,
) -> float:
    """Radial-field amplitude at r for one spatial order n (paper Eq 17).

    bp is the order's magnetization source term n·μ0·M_rn/(n²−1); the
    two-boundary solve is identical for every order.
    """
    p, q, s = r_rotor, r_magnet, r_stator
    mu = mu_r_pm

    a11 = q**n + p**(2*n) * q**(-n)
    a12 = -(q**n + s**(2*n) * q**(-n))
    b1 = -bp * (p**(n+1) * q**(-n) / n + q)

    a21 = (1/mu) * (q**(n-1) - p**(2*n) * q**(-(n+1)))
    a22 = -(q**(n-1) - s**(2*n) * q**(-(n+1)))
    b2 = -(bp/n) * (1/mu) * (1 - p**(n+1) * q**(-(n+1)))

    det = a11 * a22 - a12 * a21
    C2 = (a11 * b2 - a21 * b1) / det

    return C2 * n * (r**(n-1) + s**(2*n) * r**(-(n+1)))


def zhu_howe_Br_series(
    theta: NDArray[np.floating],
    n_p: int,
    B_rem: float,
    r_eval: float | None = None,
    *,
    r_stator: float,
    r_magnet: float,
    r_rotor: float,
    mu_r_pm: float = 1.05,
    alpha_p: float = 1.0,
    max_harmonic: int = 29,
    max_order: int | None = None,
) -> NDArray[np.floating]:
    """Odd-harmonic series of the square-wave magnetization.

    Sums each odd harmonic k of the ±1 square wave — spatial order
    m = k·n_p, Fourier coefficient (4/(kπ))·sin(kπα_p/2) (paper Eq 7a
    M_rn) — through the per-order Zhu/Howe/Chan transfer function. The
    k = 1 term is exactly zhu_howe_Br; the fundamental bin is unchanged.

    max_harmonic bounds k (1/k² coefficient decay plus geometric radial
    attenuation make the tail negligible); max_order additionally drops
    orders m ≥ max_order so a sampled waveform cannot alias (pass
    n_theta // 2 when the result feeds an FFT).
    """
    if n_p < 2:
        raise ValueError(f"n_p={n_p} not supported (singularity at n²-1=0)")
    if r_eval is None:
        r_eval = (r_stator + r_magnet) / 2

    # The per-order amplitude is scale-invariant in the radii; normalizing
    # by r_magnet keeps high-order powers O(ratio^m) instead of
    # O(meters^m), which under/overflows by m ~ a few hundred. The k = 1
    # fundamental is unaffected beyond float round-off (zhu_howe_Br keeps
    # the raw-radii path so psi_f/B_rem derivations stay bit-identical).
    q0 = r_magnet
    B = np.zeros_like(np.asarray(theta, dtype=float))
    K_fund: float | None = None
    for k in range(1, max_harmonic + 1, 2):
        m = k * n_p
        if max_order is not None and m >= max_order:
            break
        coeff = sin(k * pi * alpha_p / 2)
        if abs(coeff) < 1e-12:
            continue  # eliminated harmonic (e.g. k=3 at alpha_p=2/3)
        bp = m * (4 / (k * pi)) * B_rem * coeff / (m**2 - 1)
        try:
            K = _zhu_howe_order_amplitude(
                m, bp, r_eval / q0,
                r_stator=r_stator / q0, r_magnet=1.0, r_rotor=r_rotor / q0,
                mu_r_pm=mu_r_pm,
            )
        except (OverflowError, ZeroDivisionError):
            break  # past the numerically representable order range
        if not np.isfinite(K):
            break
        if K_fund is None:
            K_fund = K
        elif abs(K) < 1e-12 * abs(K_fund):
            continue  # negligible term; k cap bounds the loop, not this
        B += K * np.cos(m * theta)
    return B


def carter_factor(
    r_stator: float,
    n_slots: int,
    slot_opening_ratio: float,
    g_prime: float,
) -> float:
    """Carter coefficient for PM machine per Zhu & Howe 1993 III Eq 16.

    slot_opening_ratio is the slot OPENING over the slot pitch at the bore —
    not the slot body width ratio, which over-corrects.
    g_prime is the effective airgap: g + h_m / mu_r_pm.
    Returns 1.0 when n_slots == 0.
    """
    if n_slots == 0:
        return 1.0
    tau_s = 2 * pi * r_stator / n_slots
    b_o = slot_opening_ratio * tau_s
    u = b_o / (2 * g_prime)
    gamma = (4 / pi) * (u * atan(u) - log(sqrt(1 + u**2)))
    return tau_s / (tau_s - gamma * g_prime)


def carter_adjusted_radii(geo: Geometry, mu_r_pm: float) -> tuple[float, float, float]:
    """Carter modified-geometry correction: (r_stator, r_magnet, k_c).

    Widens the air gap by delta_g = (k_c - 1)·g' by moving the stator bore
    away from the magnets; r_magnet is deliberately unchanged. Returns the
    original radii with k_c = 1.0 for smooth bores. Uses the slot opening
    ratio at the bore, falling back to slot_width_ratio when unset.
    """
    r_stator = geo.r_stator
    r_magnet = geo.r_magnet
    sor = geo.slot_opening_ratio if geo.slot_opening_ratio > 0 else geo.slot_width_ratio
    if not (geo.n_slots > 0 and sor > 0 and geo.slot_depth > 0):
        return r_stator, r_magnet, 1.0
    if geo.slot_opening_ratio <= 0:
        warnings.warn(
            "slot_opening_ratio unset; Carter factor falling back to "
            f"slot_width_ratio={geo.slot_width_ratio:.3g} (slot body width), "
            "which over-corrects — set slot_opening_ratio to the bore "
            "opening fraction",
            stacklevel=2,
        )
    g = abs(r_stator - r_magnet)
    h_m = abs(r_magnet - geo.r_rotor)
    g_prime = g + h_m / mu_r_pm
    k_c = carter_factor(r_stator, geo.n_slots, sor, g_prime)
    delta_g = (k_c - 1) * g_prime
    if geo.topology == "inrunner":
        return r_stator + delta_g, r_magnet, k_c
    return r_stator - delta_g, r_magnet, k_c


def end_effect_factor(L_stk: float, g_eff: float) -> float:
    """Heuristic end-effect correction for short-stack machines.

    Returns k_end in (0, 1] — multiply B_g1 by k_end to account for axial
    flux leakage at stack ends.  k_end → 1 for L_stk >> g_eff.

    In-house exponential form on the effective-airgap scale — NOT a
    published formula. Informational-only metric; limits are sane
    (→1 for long stacks, →0 as L_stk → 0).

    Uncalibrated: use together with end_effect_factor_pole_pitch as a
    two-form bracket, not as a point estimate.

    g_eff is the effective airgap: g + h_m / mu_r_pm.
    """
    if L_stk <= 0 or g_eff <= 0:
        return 1.0
    ratio = pi * L_stk / (2 * g_eff)
    return 1.0 - (2 * g_eff) / (pi * L_stk) * (1.0 - exp(-ratio))


def end_effect_factor_pole_pitch(L_stk: float, tau_p: float) -> float:
    """Pole-pitch-scale end-effect form: k_end = 1 − tanh(x)/x, x = πL/(2τ_p).

    MISATTRIBUTED for PM flux linkage. The primary
    source Russell & Norsworthy (1958) is "Eddy currents and wall losses in
    screened-rotor induction motors" (Proc. IEE Part A, 10.1049/pi-a.1958.0036)
    — an eddy-current effective-length factor for unlaminated/screened
    INDUCTION rotors, not a magnet flux-linkage end-effect. The PM end-effect
    is instead a one-sided reduction on the AIR-GAP scale (~g/L per end),
    recoverable by magnet overhang; it does not act on the pole-pitch scale and
    cannot over-correct to the −35% this form predicts at L/τ_p ≈ 1.8. Retained
    only as a conservative lower bound / historical bracket edge — do NOT treat
    as physical. Use end_effect_factor (gap-scale) for the magnitude. Same
    limits (→1 for L_stk >> τ_p, →0 as L_stk → 0).
    """
    if L_stk <= 0 or tau_p <= 0:
        return 1.0
    x = pi * L_stk / (2 * tau_p)
    return 1.0 - tanh(x) / x


def _derive_B_rem(
    psi_f: float, n_p: int, N: int, k_w: float, L_stk: float,
    *,
    r_stator: float,
    r_magnet: float,
    r_rotor: float,
    mu_r_pm: float = 1.05,
    alpha_p: float = 1.0,
    r_stator_c: float | None = None,
    r_magnet_c: float | None = None,
) -> float:
    """Invert psi_f to magnet remanence B_rem via Zhu & Howe transfer ratio.

    Mirrors psi_f_carter: Carter-adjusted radii (r_stator_c, r_magnet_c)
    for the field solution when given, original bore for the evaluation
    point and the winding formula.
    """
    tr = zhu_howe_Br(
        np.array([0.0]), n_p, B_rem=1.0, r_eval=r_stator,
        r_stator=r_stator_c if r_stator_c is not None else r_stator,
        r_magnet=r_magnet_c if r_magnet_c is not None else r_magnet,
        r_rotor=r_rotor, mu_r_pm=mu_r_pm,
        alpha_p=alpha_p,
    )[0]
    B_ag_peak = psi_f * n_p / (2 * N * k_w * r_stator * L_stk)
    return B_ag_peak / tr
