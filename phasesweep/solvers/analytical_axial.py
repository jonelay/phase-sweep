"""Quasi-3D Cartesian layer model for axial-flux PM machines.

The machine is unrolled at the mean radius into a linear (Cartesian)
machine and solved per spatial harmonic in the axial/circumferential
plane — Parviainen (2005) D.Sc. thesis Ch. 2 (see docs/references.md).
The per-harmonic air-gap expression is thesis Eq. (2.18) (Chung et al.
2002 / Hanselman 2003 form); Carter's coefficient multiplies the
physical air gap per thesis Eq. (2.11); PM leakage follows Qu & Lipo
(2004) as adopted in thesis Eqs. (2.15)-(2.17).

All field quantities are the axial component B_z evaluated at the
stator surface of ONE air gap. Multi-gap flux linkage (TORUS windings
that link both gaps) is handled by ``gap_linkage_factor``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from math import asin, cosh, log, pi, sin, sinh, tanh
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from phasesweep.machines.geometry import AxialFluxGeometry
from phasesweep.solver_params import reject_induction, require_axial_geometry
from phasesweep.solvers.field_solution import carter_factor

if TYPE_CHECKING:
    from typing import Any

    from phasesweep.machines.motor import Motor
    from phasesweep.sweep_types import RunConfig

__all__ = [
    "ALPHA_P_SINUSOIDAL_EQUIV",
    "AnalyticalAxialParams",
    "axial_layer_Bz_series",
    "carter_airgap_axial",
    "pm_leakage_factor",
    "prepare_analytical_axial",
    "run_analytical_axial",
    "winding_transfer_axial",
]

# Equivalent square-wave pole-arc ratio of a sinusoidally shaped magnet:
# a magnet profiled so the gap flux is sinusoidal with peak equal to the
# flat-top value has fundamental amplitude = flat-top; matching
# (4/pi)·sin(alpha·pi/2) = 1 gives alpha = (2/pi)·asin(pi/4).
ALPHA_P_SINUSOIDAL_EQUIV = (2 / pi) * asin(pi / 4)


def _layer_amplitude(
    m: float, b_src: float, g_prime: float, l_pm: float, mu_r_pm: float,
) -> float:
    """Stator-surface B_z amplitude for one wavenumber m (thesis Eq. 2.18).

    b_src is the harmonic's magnetization source amplitude
    (4·B_rem/(k·pi))·sin(k·pi·alpha_p/2). The m -> 0 limit is the 1D
    magnetic circuit b_src·l_pm/(l_pm + mu_r·g').
    """
    return b_src / (
        cosh(m * g_prime) + mu_r_pm * sinh(m * g_prime) / tanh(m * l_pm)
    )


def axial_layer_Bz_series(
    theta: NDArray[np.floating],
    n_p: int,
    B_rem: float,
    *,
    tau_p: float,
    g_prime: float,
    l_pm: float,
    mu_r_pm: float = 1.05,
    alpha_p: float = 1.0,
    max_harmonic: int = 29,
    max_order: int | None = None,
) -> NDArray[np.floating]:
    """Odd-harmonic B_z(theta) at the stator surface, theta mechanical.

    Square-wave axial magnetization of relative width alpha_p, summed
    through the per-harmonic Cartesian layer transfer function. Harmonic
    k has spatial order k·n_p and wavenumber k·pi/tau_p. max_order drops
    orders >= max_order so a sampled waveform cannot alias (pass
    n_theta // 2 when the result feeds an FFT).

    No n_p >= 2 restriction: the Cartesian solution has no polar
    (n^2 - 1) singularity.
    """
    if n_p < 1:
        raise ValueError(f"n_p must be >= 1, got {n_p}")
    if tau_p <= 0 or g_prime <= 0 or l_pm <= 0:
        raise ValueError(
            f"tau_p={tau_p}, g_prime={g_prime}, l_pm={l_pm} must all be > 0"
        )

    B = np.zeros_like(np.asarray(theta, dtype=float))
    A_fund: float | None = None
    for k in range(1, max_harmonic + 1, 2):
        order = k * n_p
        if max_order is not None and order >= max_order:
            break
        coeff = sin(k * pi * alpha_p / 2)
        if abs(coeff) < 1e-12:
            continue  # eliminated harmonic (e.g. k=3 at alpha_p=2/3)
        m = k * pi / tau_p
        try:
            A = _layer_amplitude(
                m, (4 * B_rem / (k * pi)) * coeff, g_prime, l_pm, mu_r_pm,
            )
        except OverflowError:
            break  # cosh/sinh past the representable order range
        if not np.isfinite(A):
            break
        if A_fund is None:
            A_fund = A
        elif abs(A) < 1e-12 * abs(A_fund):
            break  # geometric attenuation: terms only shrink from here
        B += A * np.cos(order * theta)
    return B


def carter_airgap_axial(
    geo: AxialFluxGeometry, mu_r_pm: float,
) -> tuple[float, float]:
    """(g_prime, k_c): Carter-corrected layer-model air gap.

    k_c is computed with the slot pitch at r_mean and the PM-effective
    gap g + l_pm/mu_r (same gamma convention as the radial
    carter_factor); it then multiplies the PHYSICAL gap only — thesis
    Eq. (2.11) g' = k_c·g — because the layer model carries the magnet
    explicitly in its coth term. Returns (geo.airgap, 1.0) for slotless
    stators. Falls back from slot_opening_ratio to slot_width_ratio with
    a warning, mirroring carter_adjusted_radii.
    """
    g = geo.airgap
    sor = (
        geo.slot_opening_ratio
        if geo.slot_opening_ratio > 0
        else geo.slot_width_ratio
    )
    if not (geo.n_slots > 0 and sor > 0 and geo.slot_depth > 0):
        return g, 1.0
    if geo.slot_opening_ratio <= 0:
        warnings.warn(
            "slot_opening_ratio unset; axial Carter factor falling back to "
            f"slot_width_ratio={geo.slot_width_ratio:.3g} (slot body width), "
            "which over-corrects — set slot_opening_width to the opening "
            "at r_mean",
            stacklevel=2,
        )
    g_c = g + geo.magnet_thickness / mu_r_pm
    k_c = carter_factor(geo.r_mean, geo.n_slots, sor, g_c)
    return k_c * g, k_c


def pm_leakage_factor(
    geo: AxialFluxGeometry,
    mu_r_pm: float,
    alpha_p: float,
    g_prime: float,
    tau_p: float,
    *,
    k_add: float = 0.95,
) -> float:
    """PM leakage factor k_sigma at r_mean (thesis Eqs. 2.15-2.17).

    Product of Qu & Lipo's air-gap leakage (magnet-to-rotor +
    magnet-to-magnet), the zigzag leakage (active only when the
    interpole gap is narrower than a slot pitch), and the additional
    coefficient k_add (thesis: 0.95 reconciles the analytical factor
    with 3D FEA for the 5 kW prototype).

    g_prime is the Carter-corrected air gap from carter_airgap_axial;
    tau_p the pole pitch at r_mean. At alpha_p = 1 the magnet-to-magnet
    term is dropped (no interpole gap to leak across).
    """
    l_pm = geo.magnet_thickness
    w_pm = alpha_p * tau_p
    interpole = tau_p - w_pm

    # Air-gap leakage, Qu & Lipo (2004) / thesis Eq. (2.15):
    # magnet-to-rotor term always present; magnet-to-magnet only with a
    # finite interpole gap.
    ln_terms = (2 * l_pm / (pi * mu_r_pm * w_pm)) * log(
        1 + pi * g_prime / l_pm
    )
    if interpole > 0:
        ln_terms += (4 * l_pm / (pi * mu_r_pm * w_pm)) * log(
            1 + pi * g_prime / interpole
        )
    k_agap = 1.0 / (
        1.0
        + mu_r_pm * (g_prime / l_pm) * (w_pm / (w_pm + 2 * g_prime)) * ln_terms
    )

    # Zigzag leakage, thesis Eq. (2.16): active when the interpole gap
    # is narrower than a slot pitch.
    k_zz = 1.0
    if geo.n_slots > 0:
        tau_slot = 2 * pi * geo.r_mean / geo.n_slots
        if 0 <= interpole < tau_slot:
            k_zz = 1.0 - (tau_slot - interpole) ** 2 / (2 * w_pm * tau_slot)
            if k_zz <= 0:
                raise ValueError(
                    f"zigzag leakage factor k_zz={k_zz:.3g} <= 0: slot pitch "
                    f"{tau_slot * 1e3:.2f} mm against magnet width "
                    f"{w_pm * 1e3:.2f} mm and interpole gap "
                    f"{interpole * 1e3:.2f} mm is outside thesis Eq. (2.16) "
                    f"(n_slots={geo.n_slots}, alpha_p={alpha_p})"
                )

    return k_agap * k_zz * k_add


# Gaps linked by one phase winding circuit, per magnetic-circuit variant.
# 1R2S / DSSR: each stator winding faces its own single gap (the second
# stator is a separate, usually parallel-connected, circuit — the
# terminal psi_f convention lives in the winding data, not here).
# 2R1S TORUS: the toroidal stator winding links both gaps.
# coreless is rejected by the solver (no stator iron boundary).
GAP_LINKAGE: dict[str, int] = {
    "1R2S": 1,
    "2R1S-NN": 2,
    "2R1S-NS": 2,
    "DSSR": 1,
}


def winding_transfer_axial(
    geo: AxialFluxGeometry, n_eff: int, k_w: float, n_p: int,
) -> float:
    """psi_f per unit peak fundamental B_z, one gap:
    2·N_eff·k_w·r_mean·active_length / n_p.

    The Cartesian analog of the radial winding transfer with
    r_stator -> r_mean and L_stk -> active_length: the fundamental flux
    per pole over the annulus is B_1·(r_out^2 - r_in^2)/n_p
    = 2·B_1·r_mean·active_length/n_p. Multiply by GAP_LINKAGE[variant]
    for windings that link both gaps.
    """
    return 2 * n_eff * k_w * geo.r_mean * geo.active_length / n_p


@dataclass(frozen=True)
class AnalyticalAxialParams:
    """Validated inputs for the quasi-3D Cartesian axial-flux layer solver.

    geometry in meters; B_rem — magnet remanence (T); mu_r_pm / mu_r_fe —
    relative permeabilities (dimensionless); alpha_p — magnet-width /
    pole-pitch ratio at r_mean ((0, 1]).

    mu_r_fe is validated and hashed but does not enter the solution —
    the layer model assumes infinitely permeable iron on both sides of
    the gap (which is also why the coreless variant is rejected here).
    """

    geometry: AxialFluxGeometry
    n_p: int
    B_rem: float
    mu_r_pm: float
    mu_r_fe: float
    alpha_p: float = 1.0

    def __post_init__(self) -> None:
        if self.n_p < 1:
            raise ValueError(f"n_p must be >= 1, got {self.n_p}")
        if self.B_rem <= 0:
            raise ValueError(f"B_rem must be > 0, got {self.B_rem}")
        if self.mu_r_pm <= 0:
            raise ValueError(f"mu_r_pm must be > 0, got {self.mu_r_pm}")
        if self.mu_r_fe <= 0:
            raise ValueError(f"mu_r_fe must be > 0, got {self.mu_r_fe}")
        if not (0 < self.alpha_p <= 1):
            raise ValueError(f"alpha_p must be in (0, 1], got {self.alpha_p}")
        if self.geometry.variant == "coreless":
            raise ValueError(
                "analytical_axial does not support the coreless variant — "
                "the layer model needs stator iron behind the gap"
            )


def prepare_analytical_axial(motor: Motor) -> AnalyticalAxialParams:
    """Validate and extract axial layer-solver params from a Motor.

    B_rem must be set explicitly — the psi_f -> B_rem inversion is
    radial-only (Zhu & Howe transfer), so there is no derivation path.
    """
    reject_induction(motor, "analytical_axial")
    geo = require_axial_geometry(motor, "analytical_axial")
    if motor.B_rem is None:
        raise ValueError(
            f"Motor '{motor.name}': analytical_axial needs an explicit "
            f"[materials] B_rem (no psi_f derivation path for axial flux)"
        )
    return AnalyticalAxialParams(
        geometry=geo,
        n_p=motor.n_p,
        B_rem=motor.B_rem,
        mu_r_pm=motor.mu_r_pm,
        mu_r_fe=motor.mu_r_fe,
        alpha_p=motor.alpha_p,
    )


def run_analytical_axial(config: RunConfig) -> dict[str, Any]:
    from phasesweep.solver_params import backemf_from_flux, backemf_source, check_nyquist, n_eff
    from phasesweep.solvers.harmonics import (
        assert_waveform_symmetry,
        compute_thd,
        harmonics_1sided,
    )

    params = prepare_analytical_axial(config.motor)
    geo = params.geometry
    n_p = params.n_p
    check_nyquist(config.n_theta, n_p, series=True)

    tau_p = pi * geo.r_mean / n_p
    g_prime, k_c = carter_airgap_axial(geo, params.mu_r_pm)

    theta = np.linspace(0, 2 * np.pi, config.n_theta, endpoint=False)
    B_z = axial_layer_Bz_series(
        theta, n_p, params.B_rem,
        tau_p=tau_p, g_prime=g_prime, l_pm=geo.magnet_thickness,
        mu_r_pm=params.mu_r_pm, alpha_p=params.alpha_p,
        max_order=config.n_theta // 2,
    )

    amps = harmonics_1sided(B_z)
    assert_waveform_symmetry(amps, n_p, label="axial B_z")
    fundamental = float(amps[n_p])
    thd_pct = compute_thd(amps, n_p)

    k_sigma_pm = float(pm_leakage_factor(
        geo, params.mu_r_pm, params.alpha_p, g_prime, tau_p,
    ))

    m = config.motor
    flux_linkage_peak = None
    if m.N is not None and m.k_w is not None and m.coils_series is not None:
        flux_linkage_peak = float(
            GAP_LINKAGE[geo.variant]
            * k_sigma_pm * fundamental
            * winding_transfer_axial(geo, n_eff(m), m.k_w, n_p)
        )

    backemf_fundamental = backemf_from_flux(m, flux_linkage_peak)

    return {
        "theta_list": theta.tolist(),
        "B_r_list": B_z.tolist(),
        "fundamental": fundamental,
        "peak_Br": float(np.max(np.abs(B_z))),
        "thd_pct": thd_pct,
        "backemf_fundamental": backemf_fundamental,
        "backemf_source": backemf_source(m, backemf_fundamental),
        "flux_linkage_peak": flux_linkage_peak,
        "k_c": float(k_c),
        "k_sigma_pm": k_sigma_pm,
    }
