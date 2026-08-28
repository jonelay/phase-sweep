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

from dataclasses import dataclass
from math import pi
from typing import TYPE_CHECKING, Any

from phasesweep.solver_params import _require_drive_fields

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from phasesweep.machines.motor import Motor
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


@dataclass(frozen=True)
class IronLossParams:
    """Validated inputs for the lumped Bertotti iron-loss model.

    k_h / k_e — per-kg hysteresis / eddy coefficients (steel grade +
    lamination specific); alpha_fe — Steinmetz exponent; m_core — core
    mass (kg) assumed to cycle at B_core (T, peak); W_REF — mechanical
    speed (rad/s) setting f_e = n_p·W_REF/2π.
    """

    n_p: int
    k_h: float
    k_e: float
    alpha_fe: float
    m_core: float
    B_core: float
    W_REF: float

    def __post_init__(self) -> None:
        if self.n_p < 2:
            raise ValueError(f"n_p must be >= 2, got {self.n_p}")
        for f in ("k_h", "k_e", "alpha_fe", "m_core", "B_core", "W_REF"):
            if getattr(self, f) <= 0:
                raise ValueError(f"{f} must be > 0, got {getattr(self, f)}")


_IRON_FIELDS = ("k_h", "k_e", "alpha_fe", "m_core", "B_core")


def prepare_iron_loss(motor: Motor) -> IronLossParams:
    """Validate and extract lumped iron-loss params from a Motor.

    Rejects induction motors even though the Bertotti lump itself is
    machine-agnostic: f_e = n_p·W_REF/2π assumes SYNCHRONOUS operation.
    An induction machine's stator frequency differs by the slip
    (f_e = n_p·ω_m / (2π·(1−s))) and its rotor iron cycles at s·f_e —
    running one through silently would under-count both.
    """
    if motor.induction is not None:
        raise ValueError(
            f"Motor '{motor.name}': iron_loss assumes synchronous operation "
            f"(f_e = n_p·W_REF/2π); this motor has an [induction] section, "
            f"where the stator frequency differs by the slip and the rotor "
            f"iron cycles at s·f_e. A slip-aware iron_loss variant does not "
            f"exist yet."
        )
    k_h, k_e, alpha_fe = motor.k_h, motor.k_e, motor.alpha_fe
    m_core, B_core = motor.m_core, motor.B_core
    if k_h is None or k_e is None or alpha_fe is None or m_core is None or B_core is None:
        missing = [f for f in _IRON_FIELDS if getattr(motor, f) is None]
        raise ValueError(
            f"Motor '{motor.name}': iron_loss needs {', '.join(missing)} "
            f"([iron] TOML section)"
        )

    _require_drive_fields(motor, "iron_loss", ("W_REF",))

    return IronLossParams(
        n_p=motor.n_p,
        k_h=k_h,
        k_e=k_e,
        alpha_fe=alpha_fe,
        m_core=m_core,
        B_core=B_core,
        W_REF=motor.drive.W_REF,  # type: ignore[arg-type]  # validated above
    )
