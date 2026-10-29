"""Bertotti two-term iron-loss model (hysteresis + classical eddy).

p_fe = k_h · f · B^alpha + k_e · f² · B²   [W/kg]

Reference: Bertotti (1988), docs/references.md "Iron Loss". The registry
model is a lumped single-B estimate: the whole core mass m_core is
assumed to cycle at peak flux density B_core and the stator electrical
frequency — f_e = n_p · W_REF / 2π for a synchronous machine, the rated
supply frequency f_rated for an induction machine (slip decouples the
stator frequency from the shaft speed). The lump is the STATOR core in
both cases: an induction rotor's fundamental iron loss at s·f_e is O(s)
of the hysteresis term at rated slip and no test separates it (no-load
runs at s ≈ 0), so it is excluded and the rotor frequency is reported
for the record. Real machines distribute B between teeth and yoke;
splitting masses is a refinement, not this model. Coefficients are
per-kg steel properties (grade + lamination specific) — fit them from
multi-frequency specific-loss tables with fit_bertotti.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, pi
from typing import TYPE_CHECKING, Any

from phasesweep.solver_params import require_drive_fields

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from phasesweep.machines.motor import Motor
    from phasesweep.sweep_types import RunConfig

__all__ = [
    "IronLossParams",
    "bertotti_loss_density",
    "fit_bertotti",
    "iron_loss_at",
    "prepare_iron_loss",
    "run_iron_loss",
]


def bertotti_loss_density(f: float, B: float, k_h: float, k_e: float,
                          alpha: float) -> float:
    """Specific iron loss (W/kg) at electrical frequency f (Hz), peak B (T)."""
    return k_h * f * B**alpha + k_e * f**2 * B**2


def _lumped_split(f_e: float, B: float, k_h: float, k_e: float, alpha_fe: float,
                  m_core: float) -> tuple[float, float]:
    """(hysteresis W, eddy W) of the m_core lump at f_e and peak B."""
    return (m_core * k_h * f_e * B**alpha_fe,
            m_core * k_e * f_e**2 * B**2)


def iron_loss_at(n_p: int, k_h: float, k_e: float, alpha_fe: float,
                 m_core: float, B_core: float, w_mech: float,
                 psi_ratio: float) -> float:
    """Lumped Bertotti stator loss (W) of a synchronous machine at
    mechanical speed w_mech (rad/s) with the core flux density scaled by
    psi_ratio = |psi_s| / psi_ref from the solved operating point.

    First-order operating-point dependence: f_e = n_p·w_mech/2π and
    B = B_core·psi_ratio (the whole lump scales with the stator flux
    linkage, teeth and yoke alike). At W_REF and ratio 1 this is exactly
    run_iron_loss's p_fe (pinned).
    """
    f_e = n_p * w_mech / (2 * pi)
    return sum(_lumped_split(f_e, B_core * psi_ratio, k_h, k_e, alpha_fe, m_core))


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
    """Lumped Bertotti stator iron loss at the reference operating point:
    W_REF for a synchronous machine, (f_rated, s_rated) for an induction
    machine."""

    p = prepare_iron_loss(config.motor)
    f_e = p.f_e
    p_hyst, p_eddy = _lumped_split(f_e, p.B_core, p.k_h, p.k_e, p.alpha_fe, p.m_core)
    total = p_hyst + p_eddy
    out: dict[str, Any] = {
        "p_fe": total,
        "p_fe_hysteresis": p_hyst,
        "p_fe_eddy": p_eddy,
        "loss_density_W_per_kg": total / p.m_core,
        "f_e_Hz": f_e,
    }
    if p.slip is not None:
        # Rotor fundamental frequency, for the record only: the lump is
        # the stator core (see module docstring). Emitted only with
        # s_rated set.
        out["f_rotor_Hz"] = p.slip * f_e
    return out


@dataclass(frozen=True)
class IronLossParams:
    """Validated inputs for the lumped Bertotti iron-loss model.

    k_h / k_e — per-kg hysteresis / eddy coefficients (steel grade +
    lamination specific); alpha_fe — Steinmetz exponent; m_core — core
    mass (kg) assumed to cycle at B_core (T, peak). Exactly one frequency
    source: W_REF — mechanical speed (rad/s) setting f_e = n_p·W_REF/2π
    (synchronous machine), or f_supply — stator supply frequency (Hz) of
    an induction machine, with slip (rated, in [0, 1), optional) reported
    as the rotor frequency s·f_supply when given.
    """

    n_p: int
    k_h: float
    k_e: float
    alpha_fe: float
    m_core: float
    B_core: float
    W_REF: float | None = None
    f_supply: float | None = None
    slip: float | None = None

    def __post_init__(self) -> None:
        if (self.W_REF is None) == (self.f_supply is None):
            raise ValueError(
                "exactly one of W_REF (synchronous) or f_supply (induction) "
                "must be set")
        if self.n_p < 1 or (self.f_supply is None and self.n_p < 2):
            bound = 1 if self.f_supply is not None else 2
            raise ValueError(f"n_p must be >= {bound}, got {self.n_p}")
        for f in ("k_h", "k_e", "alpha_fe", "m_core", "B_core", "W_REF",
                  "f_supply"):
            v = getattr(self, f)
            if v is not None and (v <= 0 or not isfinite(v)):
                raise ValueError(f"{f} must be > 0 and finite, got {v}")
        if self.slip is not None:
            if self.f_supply is None:
                raise ValueError("slip is only meaningful with f_supply")
            if not 0 <= self.slip < 1:
                raise ValueError(f"slip must be in [0, 1), got {self.slip}")

    @property
    def f_e(self) -> float:
        """Stator electrical frequency (Hz)."""
        if self.f_supply is not None:
            return self.f_supply
        assert self.W_REF is not None
        return self.n_p * self.W_REF / (2 * pi)


_IRON_FIELDS = ("k_h", "k_e", "alpha_fe", "m_core", "B_core")


def prepare_iron_loss(motor: Motor) -> IronLossParams:
    """Validate and extract lumped iron-loss params from a Motor.

    Synchronous machines take the stator frequency from the reference
    speed (`[drive] W_REF`). Induction machines take it from the rated
    supply (`[induction] f_rated`) — the slip decouples stator frequency
    from shaft speed, so W_REF is neither required nor consulted for
    them. `s_rated` is optional: the stator lump does not depend on it,
    it only sets the reported `f_rotor_Hz`.
    """
    k_h, k_e, alpha_fe = motor.k_h, motor.k_e, motor.alpha_fe
    m_core, B_core = motor.m_core, motor.B_core
    if k_h is None or k_e is None or alpha_fe is None or m_core is None or B_core is None:
        missing = [f for f in _IRON_FIELDS if getattr(motor, f) is None]
        raise ValueError(
            f"Motor '{motor.name}': iron_loss needs {', '.join(missing)} "
            f"([iron] TOML section)"
        )

    ind = motor.induction
    if ind is not None:
        if ind.f_rated is None:
            raise ValueError(
                f"Motor '{motor.name}': iron_loss on an induction machine "
                f"needs f_rated ([induction] TOML section) — the stator "
                f"frequency is the supply, not n_p·W_REF/2π"
            )
        return IronLossParams(
            n_p=motor.n_p, k_h=k_h, k_e=k_e, alpha_fe=alpha_fe,
            m_core=m_core, B_core=B_core,
            f_supply=ind.f_rated, slip=ind.s_rated,
        )

    require_drive_fields(motor, "iron_loss", ("W_REF",))
    return IronLossParams(
        n_p=motor.n_p, k_h=k_h, k_e=k_e, alpha_fe=alpha_fe,
        m_core=m_core, B_core=B_core,
        W_REF=motor.drive.W_REF,
    )
