"""Soft-iron B-H curves and magnetic constants.

``BHCurve`` wraps one monotone (B, H) table: a PCHIP spline H(B) (fem
model v8; previously piecewise-linear np.interp), its secant and
differential reluctivities, the numeric inverse B(H) the scalar-
potential wedge solver needs, and the cycle-averaged reluctivity the
time-harmonic induction solver uses for saturation. Monotone, C1 H(B)
is the standard prerequisite for Newton BH solves — it keeps the
differential reluctivity finite and the assembled Jacobian well-posed —
and the spline coefficients give the Newton solvers their d nu/dB and
dB/dH analytically instead of by finite difference.

``GENERIC_BH`` is the representative soft-steel curve (not grade-
specific) shared by the radial 2D solver, the axial multi-plane solver,
and the 3D wedge solver; the module-level ``_bh_*`` functions are its
methods and keep those solvers' results unchanged. A motor may carry
its own table (``[materials] bh_curve``, ``bh_curve_for``), read today
by the induction FEM models only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import pi
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import PchipInterpolator

if TYPE_CHECKING:
    from phasesweep.machines.motor import Motor

__all__ = [
    "GENERIC_BH",
    "MU0",
    "BHCurve",
    "bh_curve_for",
    "bh_dB_dH",
    "bh_mu_h",
    "bh_nu",
    "bh_nu_prime",
]

MU0: float = 4e-7 * pi

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

_N_CYCLE = 48    # midpoint samples over half a cycle for nu_harmonic


@dataclass(frozen=True)
class BHCurve:
    """One monotone B-H table with its spline machinery.

    ``B`` (T) and ``H`` (A/m) start at the origin and increase strictly.
    Beyond the table end the curve continues with the vacuum slope,
    H = H_end + (B − B_end)/μ0, so the secant ν = H/B stays monotone and
    approaches 1/μ0 from below (clamping H instead would make deeply
    saturated iron look magnetically softer with rising B).
    """

    B: NDArray[np.floating]
    H: NDArray[np.floating]
    _spline: Any = field(init=False, repr=False, compare=False)
    _spline_d: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        B = np.asarray(self.B, dtype=float)
        H = np.asarray(self.H, dtype=float)
        if B.ndim != 1 or B.shape != H.shape or len(B) < 3:
            raise ValueError("BHCurve needs matching 1-D B and H tables of >= 3 points")
        if B[0] != 0.0 or H[0] != 0.0:
            raise ValueError("BHCurve tables must start at the origin (0, 0)")
        if np.any(np.diff(B) <= 0) or np.any(np.diff(H) <= 0):
            raise ValueError("BHCurve tables must increase strictly in both B and H")
        if not (np.all(np.isfinite(B)) and np.all(np.isfinite(H))):
            raise ValueError("BHCurve tables must be finite")
        object.__setattr__(self, "B", B)
        object.__setattr__(self, "H", H)
        spline = PchipInterpolator(B, H)
        object.__setattr__(self, "_spline", spline)
        object.__setattr__(self, "_spline_d", spline.derivative())

    @classmethod
    def from_rows(cls, rows: Any) -> BHCurve:
        """Build from (H_A_per_m, B_T) rows as a motor TOML lists them;
        the origin is prepended when the table does not start there."""
        arr = np.asarray(rows, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("bh_curve rows must be (H, B) pairs")
        H, B = arr[:, 0], arr[:, 1]
        # Checked here, not on BHCurve: a hand-built linear table may run
        # past 3 T on purpose, but a lamination table never does.
        if B[-1] > 3.0:
            raise ValueError(
                f"bh_curve B ends at {B[-1]:g} T (> 3 T, beyond any steel) — "
                f"are the columns swapped? TOML bh_curve rows are (H, B)")
        if not (H[0] == 0.0 and B[0] == 0.0):
            H = np.concatenate(([0.0], H))
            B = np.concatenate(([0.0], B))
        return cls(B=B, H=H)

    @property
    def nu_0(self) -> float:
        """Reluctivity floor: the first-segment secant slope H₁/B₁."""
        return float(self.H[1] / self.B[1])

    @property
    def mu_r_initial(self) -> float:
        return 1.0 / (MU0 * self.nu_0)

    def H_of_B(self, B_mag: NDArray) -> NDArray:
        """H(|B|): PCHIP inside the table, vacuum differential slope beyond."""
        B = np.asarray(B_mag, dtype=float)
        H = self._spline(np.clip(B, 0.0, self.B[-1]))
        beyond = B > self.B[-1]
        if np.any(beyond):
            H = np.where(beyond, self.H[-1] + (B - self.B[-1]) / MU0, H)
        return H

    def Hprime_of_B(self, B_mag: NDArray) -> NDArray:
        """dH/dB(|B|): spline derivative inside the table, 1/μ0 beyond."""
        B = np.asarray(B_mag, dtype=float)
        Hp = self._spline_d(np.clip(B, 0.0, self.B[-1]))
        return np.where(B > self.B[-1], 1.0 / MU0, Hp)

    def nu(self, B_mag: NDArray) -> NDArray:
        """Secant reluctivity ν(|B|) = H(|B|)/|B|, clipped to [ν₀, 1/μ0]."""
        H = self.H_of_B(B_mag)
        B_safe = np.maximum(B_mag, 1e-12)
        return np.clip(H / B_safe, self.nu_0, 1 / MU0)

    def nu_secant(self, B_mag: NDArray) -> NDArray:
        """Secant reluctivity H(|B|)/|B| capped at 1/μ0 but NOT floored at
        ν₀: the rising permeability between the initial slope and μ_max
        (M800-50A: μ_r 1170 at 0.1 T, ~5000 at 1 T) is kept. As B → 0 the
        secant tends to the spline tangent at the origin. The floor in
        ``nu`` is the magnetostatic PM solvers' convention (kept for their
        cached results); a magnetizing inductance is set by exactly this
        rising part, so the harmonic path reads the curve as published."""
        B_safe = np.maximum(np.asarray(B_mag, dtype=float), 1e-9)
        return np.minimum(self.H_of_B(B_safe) / B_safe, 1 / MU0)

    def nu_prime(self, B_mag: NDArray) -> NDArray:
        """d ν/dB of the clipped secant reluctivity, analytic: (H'B − H)/B²
        from the spline coefficients inside the table and the vacuum slope
        beyond; zero wherever the [ν₀, 1/μ0] clip is active."""
        B = np.asarray(B_mag, dtype=float)
        H = self.H_of_B(B)
        B_safe = np.maximum(B, 1e-12)
        nu_raw = H / B_safe
        unclipped = (nu_raw > self.nu_0) & (nu_raw < 1 / MU0)
        nu_p = (self.Hprime_of_B(B) * B_safe - H) / B_safe**2
        return np.where(unclipped, nu_p, 0.0)

    def B_of_H(self, H_mag: NDArray) -> NDArray:
        """B(|H|): exact numeric inverse of ``H_of_B`` — bisection on the
        monotone spline inside the table (60 halvings of [0, B_end], a
        ~2e-18 T bracket), the mirrored vacuum-slope form beyond."""
        H = np.asarray(H_mag, dtype=float)
        B = np.zeros_like(H)
        beyond = H > self.H[-1]
        if np.any(beyond):
            B = np.where(beyond, self.B[-1] + (H - self.H[-1]) * MU0, B)
        inside = (~beyond) & (H > 0.0)
        if np.any(inside):
            Hi = H[inside]
            lo = np.zeros_like(Hi)
            hi = np.full_like(Hi, self.B[-1])
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                under = self._spline(mid) < Hi
                lo = np.where(under, mid, lo)
                hi = np.where(under, hi, mid)
            B[inside] = 0.5 * (lo + hi)
        return B

    def mu_h(self, H_mag: NDArray) -> NDArray:
        """Permeability μ(|H|) = 1/ν(B(|H|)) — the exact H-space inverse of
        ``nu``, so the wedge fixed point and the 2D solver share one
        solution. ν's clip inverts to the [μ0, μ_max] bounds. At H = 0
        exactly B = 0 gives μ_max; for H → 0⁺ the limit is 1/H'(0), a
        benign measure-zero discontinuity at the origin."""
        return 1.0 / self.nu(self.B_of_H(H_mag))

    def dB_dH(self, H_mag: NDArray) -> NDArray:
        """Tangent slope dB/dH(|H|) of the effective curve B(H) = μ(H)·H:
        1/H'(B(H)) where ν is unclipped, the plateau secant where the
        clip is active (there the effective B(H) is linear in H and the
        anisotropic Jacobian correction vanishes). Floored at μ0 so the
        Jacobian stays SPD."""
        H = np.asarray(H_mag, dtype=float)
        B = self.B_of_H(H)
        B_safe = np.maximum(B, 1e-12)
        nu_raw = self.H_of_B(B) / B_safe
        unclipped = (nu_raw > self.nu_0) & (nu_raw < 1 / MU0)
        dB_dH = np.where(unclipped, 1.0 / self.Hprime_of_B(B), 1.0 / self.nu_0)
        dB_dH = np.where(nu_raw >= 1 / MU0, MU0, dB_dH)
        return np.maximum(dB_dH, MU0)

    def nu_harmonic(self, S: NDArray, D: NDArray) -> NDArray:
        """Cycle-averaged (energy-equivalent) reluctivity of a phasor field.

        For B(t) = Re(B̂ e^{jωt}) the magnitude traces an ellipse,
        |B(t)|² = ½(S + D cos 2ωt) with S = |B̂_x|² + |B̂_y|² and
        D = |B̂_x² + B̂_y²| (peak² = ½(S + D), min² = ½(S − D)). The linear
        medium that stores the same cycle-averaged energy ⟨H·B⟩/2 at the
        same field has ν_eff = ⟨ν(|B|)|B|²⟩ / ⟨|B|²⟩ with ν the unfloored
        secant (``nu_secant``). It is the secant reluctivity for a
        circular locus (D = 0, |B| constant — the exact time-harmonic
        solution of an isotropic curve) and the fundamental of H(t) over
        B_peak for a linearly polarized one (D = S), the classic
        single-harmonic equivalent. Midpoint rule over half a cycle
        (even, periodic integrand).
        """
        S = np.asarray(S, dtype=float)
        D = np.minimum(np.asarray(D, dtype=float), S)
        theta = (np.arange(_N_CYCLE) + 0.5) * pi / _N_CYCLE
        b2 = 0.5 * (S[..., None] + D[..., None] * np.cos(theta))
        b2 = np.maximum(b2, 0.0)
        b = np.sqrt(b2)
        num = np.mean(self.nu_secant(b) * b2, axis=-1)
        den = np.maximum(0.5 * S, 1e-24)
        return np.where(S > 0, num / den, self.nu_secant(np.zeros(1))[0])


GENERIC_BH = BHCurve(B=_BH_B, H=_BH_H)

# PCHIP preserves the monotonicity of the (monotone) table, so H(B) is
# strictly increasing and dH/dB > 0 on the table interior.
_BH_SPLINE = GENERIC_BH._spline
_BH_SPLINE_D = GENERIC_BH._spline_d


def bh_curve_for(motor: Motor) -> BHCurve:
    """The motor's own ``[materials] bh_curve`` table, or the generic curve."""
    if motor.bh_curve is None:
        return GENERIC_BH
    return BHCurve.from_rows(motor.bh_curve)


def _bh_H_of_B(B_mag: NDArray) -> NDArray:
    return GENERIC_BH.H_of_B(B_mag)


def _bh_Hprime_of_B(B_mag: NDArray) -> NDArray:
    return GENERIC_BH.Hprime_of_B(B_mag)


def bh_nu(B_mag: NDArray) -> NDArray:
    """Reluctivity ν(|B|) of the generic soft iron curve (see BHCurve.nu)."""
    return GENERIC_BH.nu(B_mag)


def bh_nu_prime(B_mag: NDArray) -> NDArray:
    return GENERIC_BH.nu_prime(B_mag)


def _bh_B_of_H(H_mag: NDArray) -> NDArray:
    return GENERIC_BH.B_of_H(H_mag)


def bh_mu_h(H_mag: NDArray) -> NDArray:
    return GENERIC_BH.mu_h(H_mag)


def bh_dB_dH(H_mag: NDArray) -> NDArray:
    return GENERIC_BH.dB_dH(H_mag)
