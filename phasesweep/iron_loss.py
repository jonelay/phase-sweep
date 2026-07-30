"""Bertotti two-term iron-loss model (hysteresis + classical eddy).

p_fe = k_h · f · B^alpha + k_e · f² · B²   [W/kg]

Reference: Bertotti (1988), docs/references.md "Iron Loss". The registry
model is a lumped single-B estimate: the whole core mass m_core is
assumed to cycle at peak flux density B_core and electrical frequency
f_e = n_p · W_REF / 2π. Real machines distribute B between teeth and
yoke; splitting masses is a refinement, not this model. Coefficients are
per-kg steel properties (grade + lamination specific) — fit them from
multi-frequency specific-loss tables with fit_bertotti.
"""

from __future__ import annotations

from math import pi
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from phasesweep.sweep_types import RunConfig


def bertotti_loss_density(f: float, B: float, k_h: float, k_e: float,
                          alpha: float) -> float:
    """Specific iron loss (W/kg) at electrical frequency f (Hz), peak B (T)."""
    return k_h * f * B**alpha + k_e * f**2 * B**2


def fit_bertotti(
    f: NDArray, B: NDArray, p: NDArray,
    alpha_grid: tuple[float, float, float] = (1.4, 2.4, 0.01),
) -> tuple[float, float, float]:
    """Fit (k_h, k_e, alpha) to specific-loss table rows (f Hz, B T, p W/kg).

    For fixed alpha the model is linear in (k_h, k_e); alpha is scanned on
    a grid and the relative-residual least squares winner returned.
    Rows are weighted by 1/p so low-loss points count equally — an
    unweighted fit is dominated by the high-f, high-B corner.
    """
    import numpy as np

    f = np.asarray(f, dtype=float)
    B = np.asarray(B, dtype=float)
    p = np.asarray(p, dtype=float)
    if not (np.isfinite(f).all() and np.isfinite(B).all()
            and np.isfinite(p).all()):
        raise ValueError("fit_bertotti: non-finite entry in (f, B, p) table")
    if (B <= 0).any() or (p <= 0).any():
        raise ValueError("fit_bertotti: B and p must be > 0")

    lo, hi, step = alpha_grid
    best: tuple[float, float, float, float] | None = None
    for alpha in np.arange(lo, hi + step / 2, step):
        X = np.vstack([f * B**alpha, f**2 * B**2]).T
        coef, *_ = np.linalg.lstsq(X / p[:, None], np.ones_like(p), rcond=None)
        ss = float(np.sum((X @ coef / p - 1.0) ** 2))
        if not np.isfinite(ss):
            continue
        if best is None or ss < best[0]:
            best = (ss, float(alpha), float(coef[0]), float(coef[1]))
    if best is None:
        raise ValueError("fit_bertotti: no finite fit on the alpha grid")
    _, alpha, k_h, k_e = best
    return k_h, k_e, alpha


def run_iron_loss(config: RunConfig) -> dict[str, Any]:
    """Lumped Bertotti iron loss at the reference speed W_REF."""
    from phasesweep.solver_params import prepare_iron_loss

    p = prepare_iron_loss(config.motor)
    f_e = p.n_p * p.W_REF / (2 * pi)
    p_hyst = p.m_core * p.k_h * f_e * p.B_core**p.alpha_fe
    p_eddy = p.m_core * p.k_e * f_e**2 * p.B_core**2
    total = p_hyst + p_eddy
    return {
        "p_fe": total,
        "p_fe_hysteresis": p_hyst,
        "p_fe_eddy": p_eddy,
        "loss_density_W_per_kg": total / p.m_core,
        "f_e_Hz": f_e,
    }
