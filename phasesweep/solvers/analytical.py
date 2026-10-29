"""Analytical air-gap field solver and end-effect corrections.

Pure-math field functions (Zhu & Howe, Carter factor) live in
field_solution; this module owns the solver params/prepare/run and
end-effect corrections.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, pi, tanh
from typing import TYPE_CHECKING

import numpy as np

from phasesweep.machines.geometry import Geometry
from phasesweep.solver_params import (
    reject_induction,
    require_geometry,
    resolve_b_rem,
)
from phasesweep.solvers.field_solution import (
    carter_adjusted_radii,
    zhu_howe_Br,
    zhu_howe_Br_series,
)

if TYPE_CHECKING:
    from typing import Any

    from phasesweep.machines.motor import Motor
    from phasesweep.sweep_types import RunConfig

__all__ = [
    "AnalyticalParams",
    "end_effect_factor",
    "end_effect_factor_pole_pitch",
    "end_winding_leakage",
    "prepare_analytical",
    "run_analytical",
]


def end_effect_factor(L_stk: float, g_eff: float) -> float:
    """Heuristic end-effect correction for short-stack machines.

    Returns k_end in (0, 1] for L_stk > 0 and the limit 0 at L_stk = 0 —
    multiply B_g1 by k_end to account for axial flux leakage at stack
    ends.  k_end → 1 for L_stk >> g_eff.

    In-house exponential form on the effective-airgap scale — NOT a
    published formula. Informational-only metric; limits are sane
    (→1 for long stacks, →0 as L_stk → 0).

    Uncalibrated: use together with end_effect_factor_pole_pitch as a
    two-form bracket, not as a point estimate.

    g_eff is the effective airgap: g + h_m / mu_r_pm.
    """
    if L_stk < 0:
        raise ValueError(f"L_stk must be non-negative, got {L_stk}")
    if g_eff < 0:
        raise ValueError(f"g_eff must be non-negative, got {g_eff}")
    if L_stk == 0:
        return 0.0
    if g_eff == 0:
        return 1.0
    ratio = pi * L_stk / (2 * g_eff)
    k = 1.0 - (2 * g_eff) / (pi * L_stk) * (1.0 - exp(-ratio))
    if not (0 < k <= 1):
        raise RuntimeError(
            f"end_effect_factor={k:.6f} outside (0, 1]"
        )
    return k


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
    if L_stk < 0:
        raise ValueError(f"L_stk must be non-negative, got {L_stk}")
    if tau_p < 0:
        raise ValueError(f"tau_p must be non-negative, got {tau_p}")
    if L_stk == 0:
        return 0.0
    if tau_p == 0:
        return 1.0
    x = pi * L_stk / (2 * tau_p)
    k = 1.0 - tanh(x) / x
    if not (0 < k <= 1):
        raise RuntimeError(
            f"end_effect_factor_pole_pitch={k:.6f} outside (0, 1]"
        )
    return k


def end_winding_leakage(
    r_stator: float,
    n_slots: int,
    n_p: int,
    coil_pitch_slots: int,
    N: int,
    coils_series: int,
    ew_ratio: float = pi / 2,
) -> float:
    """Stator end-winding leakage inductance (H) from Boldea & Nasar.

    Boldea, The Induction Machine Handbook, Ch. 6: per-phase, both end
    regions.  L_ew = 4 μ₀ q_total N_c² λ_ew with
    λ_ew = 0.34 q (l_ew − 0.64 y), q = slots/pole/phase, y = coil span (m),
    l_ew = end-turn length per side.  ``ew_ratio`` = l_ew / y; the default
    π/2 is the ideal semicircular end-turn (conservative lower bound).
    """
    m = 3
    q = n_slots / (2 * n_p * m)
    tau_s = 2 * pi * r_stator / n_slots
    y = coil_pitch_slots * tau_s
    l_ew = ew_ratio * y
    lam_ew = 0.34 * q * (l_ew - 0.64 * y)
    if lam_ew <= 0:
        raise ValueError(
            f"end_winding_leakage: non-positive permeance lambda_ew={lam_ew:.4e} "
            f"(ew_ratio={ew_ratio:.3f} < 0.64 threshold); this is an input error"
        )
    q_total = coils_series
    MU0 = 4 * pi * 1e-7
    return 4 * MU0 * q_total * N**2 * lam_ew


@dataclass(frozen=True)
class AnalyticalParams:
    """Validated inputs for Zhu & Howe analytical air-gap field solver.

    geometry in meters; B_rem — magnet remanence (T); mu_r_pm / mu_r_fe —
    relative permeabilities (dimensionless); alpha_p — pole-arc ratio ((0, 1]).

    mu_r_fe is validated and hashed but does not enter the analytical
    solution — Zhu & Howe assumes infinitely permeable iron. It is kept for
    parity with FemParams (where it enters the linear solve) and config_id
    stability.
    """

    geometry: Geometry
    n_p: int
    B_rem: float
    mu_r_pm: float
    mu_r_fe: float
    alpha_p: float = 1.0

    def __post_init__(self) -> None:
        if self.n_p < 2:
            raise ValueError(f"n_p must be >= 2, got {self.n_p}")
        if self.B_rem <= 0:
            raise ValueError(f"B_rem must be > 0, got {self.B_rem}")
        if self.mu_r_pm <= 0:
            raise ValueError(f"mu_r_pm must be > 0, got {self.mu_r_pm}")
        if self.mu_r_fe <= 0:
            raise ValueError(f"mu_r_fe must be > 0, got {self.mu_r_fe}")
        if not (0 < self.alpha_p <= 1):
            raise ValueError(f"alpha_p must be in (0, 1], got {self.alpha_p}")


def prepare_analytical(motor: Motor) -> AnalyticalParams:
    """Derive, validate, and extract analytical solver params from a Motor."""
    reject_induction(motor, "analytical")
    if motor.magnetization == "halbach":
        raise ValueError(
            f"Motor '{motor.name}': the analytical solver (Zhu & Howe) assumes "
            f"radial magnetization only — Halbach arrays are not supported. "
            f"Use the FEM solver instead."
        )
    geo = require_geometry(motor, "analytical")
    if geo.magnet_mount == "inset":
        raise ValueError(
            f"Motor '{motor.name}': the analytical solver (Zhu & Howe) assumes "
            f"air between magnet arcs — inset rotors (interpole iron) are not "
            f"supported. Use the FEM solver instead."
        )
    if geo.bridge_thickness is not None:
        raise ValueError(
            f"Motor '{motor.name}': the analytical solver (Zhu & Howe) does "
            f"not model a bridge ring — use the FEM solver instead."
        )
    b_rem = resolve_b_rem(motor)
    return AnalyticalParams(
        geometry=geo,
        n_p=motor.n_p,
        B_rem=b_rem,
        mu_r_pm=motor.mu_r_pm,
        mu_r_fe=motor.mu_r_fe,
        alpha_p=motor.alpha_p,
    )


def run_analytical(config: RunConfig) -> dict[str, Any]:
    from phasesweep.solver_params import backemf_from_flux, backemf_source, check_nyquist, psi_f_carter
    from phasesweep.solvers.harmonics import (
        assert_waveform_symmetry,
        compute_thd,
        harmonics_1sided,
    )

    params = prepare_analytical(config.motor)
    geo = params.geometry
    n_p = params.n_p

    r_stator, r_magnet, _k_c = carter_adjusted_radii(geo, params.mu_r_pm)
    check_nyquist(config.n_theta, n_p, series=True)
    theta = np.linspace(0, 2 * np.pi, config.n_theta, endpoint=False)
    B_r = zhu_howe_Br_series(
        theta, n_p, params.B_rem, r_eval=geo.r_ag,
        r_stator=r_stator, r_magnet=r_magnet, r_rotor=geo.r_rotor,
        mu_r_pm=params.mu_r_pm, alpha_p=params.alpha_p,
        max_order=config.n_theta // 2,
    )

    amps = harmonics_1sided(B_r)
    assert_waveform_symmetry(amps, n_p, label="analytical B_r")
    fundamental = float(amps[n_p])
    thd_pct = compute_thd(amps, n_p)

    B_r1_iron = float(zhu_howe_Br(
        np.array([0.0]), n_p, params.B_rem, r_eval=r_stator,
        r_stator=r_stator, r_magnet=r_magnet, r_rotor=geo.r_rotor,
        mu_r_pm=params.mu_r_pm, alpha_p=params.alpha_p,
    )[0])
    rB_ag = abs(geo.r_ag * fundamental)
    rB_iron = abs(r_stator * B_r1_iron)
    if rB_ag < rB_iron * (1 - 1e-10):
        raise RuntimeError(
            f"|r·B_r1| at midgap ({rB_ag:.6e}) < at iron ({rB_iron:.6e}); "
            "Zhu-Howe 2c invariant violated"
        )

    m = config.motor
    flux_linkage_peak = psi_f_carter(m, r_stator, r_magnet, params.B_rem)
    if flux_linkage_peak is not None:
        flux_linkage_peak = float(flux_linkage_peak)

    if (m.k_end is not None and m.B_rem is not None
            and flux_linkage_peak is not None):
        k_end_applied = True
        flux_linkage_peak = float(flux_linkage_peak * m.k_end)
    else:
        k_end_applied = False

    backemf_fundamental = backemf_from_flux(m, flux_linkage_peak)

    if m.k_end is not None:
        k_end = m.k_end
        k_end_source = "computed"
    elif m.L_stk is not None:
        g = abs(geo.r_stator - geo.r_magnet)
        h_m = abs(geo.r_magnet - geo.r_rotor)
        k_end = float(end_effect_factor(m.L_stk, g + h_m / params.mu_r_pm))
        k_end_source = "heuristic"
    else:
        k_end = None
        k_end_source = None

    return {
        "theta_list": theta.tolist(),
        "B_r_list": B_r.tolist(),
        "fundamental": fundamental,
        "peak_Br": float(np.max(np.abs(B_r))),
        "thd_pct": thd_pct,
        "backemf_fundamental": backemf_fundamental,
        "backemf_source": backemf_source(m, backemf_fundamental),
        "flux_linkage_peak": flux_linkage_peak,
        "k_end": k_end,
        "k_end_source": k_end_source,
        "k_end_applied": k_end_applied,
    }
