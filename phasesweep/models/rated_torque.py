"""Rated torque and MTPA characteristic curve computation.

Implements Morimoto (1994) MTPA quadratic for salient machines (both
saliency signs), with k_T * I_rated fallback for non-salient.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import asin, cos, degrees, sin, sqrt
from typing import TYPE_CHECKING, Any

from phasesweep.solver_params import _reject_induction, _resolve_psi_f, _salient_pair

if TYPE_CHECKING:
    from phasesweep.machines.motor import Motor
    from phasesweep.sweep_types import RunConfig


def magnet_torque_constant(n_p: int, psi_f: float) -> float:
    """Three-phase magnet torque constant k_T = 1.5 · n_p · ψ_f (N·m/A_peak)."""
    return 1.5 * n_p * psi_f


def mtpa_gamma(psi_f: float, L_d: float, L_q: float, I_s: float) -> float:
    """MTPA angle in radians from q-axis. 0 = all current on q-axis.

    The + root of the Morimoto quadratic is the global optimum for both
    saliency signs: L_q > L_d gives gamma > 0 (demagnetizing i_d < 0),
    L_d > L_q gives gamma < 0 (magnetizing i_d > 0, positive reluctance
    torque).
    """
    if I_s <= 0:
        return 0.0
    dL = L_q - L_d
    if dL == 0:
        return 0.0
    sin_g = (-psi_f + sqrt(psi_f**2 + 8 * dL**2 * I_s**2)) / (4 * dL * I_s)
    sin_g = max(-1.0, min(1.0, sin_g))
    return asin(sin_g)


def mtpa_torque(n_p: float, psi_f: float, L_d: float, L_q: float,
                I_s: float, gamma: float) -> float:
    """Torque in Nm at given current magnitude and MTPA angle."""
    i_d = -I_s * sin(gamma)
    i_q = I_s * cos(gamma)
    return 1.5 * n_p * (psi_f * i_q + (L_d - L_q) * i_d * i_q)


def mtpa_curve(n_p: int, psi_f: float, L_d: float, L_q: float,
               I_rated: float, n_pts: int = 50) -> dict[str, list[float]]:
    """MTPA characteristic curve from 0.01*I_rated to 2*I_rated.

    I_rated defines the sweep range, not the evaluation point.
    """
    I_min = 0.01 * I_rated
    I_max = 2.0 * I_rated
    step = (I_max - I_min) / (n_pts - 1)

    I_list: list[float] = []
    gamma_list: list[float] = []
    angle_d_list: list[float] = []
    tau_list: list[float] = []

    for i in range(n_pts):
        I_s = I_min + i * step
        g = mtpa_gamma(psi_f, L_d, L_q, I_s)
        tau = mtpa_torque(n_p, psi_f, L_d, L_q, I_s, g)
        I_list.append(I_s)
        gamma_list.append(degrees(g))
        angle_d_list.append(90.0 + degrees(g))
        tau_list.append(tau)

    return {
        "I_curve": I_list,
        "gamma_curve_deg": gamma_list,
        "angle_d_curve_deg": angle_d_list,
        "tau_curve": tau_list,
    }


_CURVE_KEYS = ("I_curve", "gamma_curve_deg", "angle_d_curve_deg", "tau_curve")


def _mtpa_result(
    n_p: int,
    psi_f: float,
    I_s: float,
    L_d: float | None,
    L_q: float | None,
) -> dict[str, Any]:
    """Shared MTPA computation at a given current magnitude.

    Returns tau, k_T, k_T_effective (= tau / I_s, folds in reluctance
    torque), gamma_opt_deg (< 0 for reverse saliency, magnetizing i_d),
    and curve data.
    """
    k_T = magnet_torque_constant(n_p, psi_f)

    if L_d is None or L_q is None or L_q == L_d:
        if psi_f == 0.0:
            raise ValueError(
                "psi_f = 0 with no saliency (L_d == L_q or unset): "
                "machine cannot produce torque"
            )
        tau = k_T * I_s
        gamma_opt = 0.0
        Ld = L_d if L_d is not None else 0.0
        Lq = L_q if L_q is not None else 0.0
    else:
        Ld = L_d
        Lq = L_q
        g = mtpa_gamma(psi_f, Ld, Lq, I_s)
        tau = mtpa_torque(n_p, psi_f, Ld, Lq, I_s, g)
        gamma_opt = degrees(g)

    result: dict[str, Any] = {
        "tau": tau,
        "k_T": k_T,
        "k_T_effective": tau / I_s,
        "gamma_opt_deg": gamma_opt,
    }

    if Ld > 0 and Lq > 0:
        result.update(mtpa_curve(n_p, psi_f, Ld, Lq, I_s))

    return result


def run_rated_torque(config: RunConfig) -> dict[str, Any]:
    """Rated torque computation. Returns scalars + curves."""

    p = prepare_rated_torque(config.motor)
    r = _mtpa_result(p.n_p, p.psi_f, p.I_rated, p.L_d, p.L_q)

    return {
        "tau_mtpa": r["tau"],
        "k_T": r["k_T"],
        "k_T_rms": r["k_T"] * sqrt(2),
        "k_T_effective": r["k_T_effective"],
        "gamma_opt_deg": r["gamma_opt_deg"],
        **{k: r[k] for k in _CURVE_KEYS if k in r},
    }


def run_stall_torque(config: RunConfig) -> dict[str, Any]:
    """Stall torque computation (MTPA at I_stall). Returns scalars + curves.

    Emits both current-limited (tau_stall, I_stall) and electromagnetic
    (tau_stall_electromagnetic, I_stall_electromagnetic) values.
    """

    p = prepare_stall_torque(config.motor)
    r = _mtpa_result(p.n_p, p.psi_f, p.I_stall, p.L_d, p.L_q)

    saturation_ratio = None
    saturation_warning = False
    I_rated = config.motor.I_rated
    if I_rated is not None and I_rated > 0:
        saturation_ratio = p.I_stall / I_rated
        if saturation_ratio > 3.0:
            saturation_warning = True

    result: dict[str, Any] = {
        "tau_stall": r["tau"],
        "I_stall": p.I_stall,
        "k_T": r["k_T"],
        "k_T_effective": r["k_T_effective"],
        "gamma_opt_deg": r["gamma_opt_deg"],
        "saturation_ratio": saturation_ratio,
        "saturation_warning": saturation_warning,
        **{k: r[k] for k in _CURVE_KEYS if k in r},
    }

    if p.I_stall_em is not None:
        r_em = _mtpa_result(p.n_p, p.psi_f, p.I_stall_em, p.L_d, p.L_q)
        result["tau_stall_electromagnetic"] = r_em["tau"]
        result["I_stall_electromagnetic"] = p.I_stall_em

    return result


@dataclass(frozen=True)
class RatedTorqueParams:
    """Validated inputs for rated torque (MTPA at I_rated) computation.

    psi_f — PM flux linkage (Wb, peak); I_rated — rated current (A, peak);
    L_d / L_q — inductances (H), both required for the salient form.
    """

    n_p: int
    psi_f: float
    I_rated: float
    L_d: float | None = None
    L_q: float | None = None

    def __post_init__(self) -> None:
        if self.n_p < 2:
            raise ValueError(f"n_p must be >= 2, got {self.n_p}")
        if self.psi_f < 0:
            raise ValueError(f"psi_f must be >= 0, got {self.psi_f}")
        if self.I_rated <= 0:
            raise ValueError(f"I_rated must be > 0, got {self.I_rated}")
        if self.L_d is not None and self.L_d <= 0:
            raise ValueError(f"L_d must be > 0, got {self.L_d}")
        if self.L_q is not None and self.L_q <= 0:
            raise ValueError(f"L_q must be > 0, got {self.L_q}")


@dataclass(frozen=True)
class StallTorqueParams:
    """Validated inputs for stall torque (MTPA at I_stall) computation.

    psi_f — PM flux linkage (Wb, peak); L_d / L_q — inductances (H).
    I_stall: drive current limit, I_LIMIT or MAX_I_S (A, peak).
    I_stall_em: electromagnetic stall current U_DC / (sqrt(3) * R_s)
        (A, peak), None when R_s = 0.
    """

    n_p: int
    psi_f: float
    I_stall: float
    I_stall_em: float | None = None
    L_d: float | None = None
    L_q: float | None = None

    def __post_init__(self) -> None:
        if self.n_p < 2:
            raise ValueError(f"n_p must be >= 2, got {self.n_p}")
        if self.psi_f < 0:
            raise ValueError(f"psi_f must be >= 0, got {self.psi_f}")
        if self.I_stall <= 0:
            raise ValueError(f"I_stall must be > 0, got {self.I_stall}")
        if self.I_stall_em is not None and self.I_stall_em <= 0:
            raise ValueError(f"I_stall_em must be > 0, got {self.I_stall_em}")
        if self.L_d is not None and self.L_d <= 0:
            raise ValueError(f"L_d must be > 0, got {self.L_d}")
        if self.L_q is not None and self.L_q <= 0:
            raise ValueError(f"L_q must be > 0, got {self.L_q}")


def prepare_rated_torque(motor: Motor) -> RatedTorqueParams:
    """Derive, validate, and extract rated torque params from a Motor."""
    _reject_induction(motor, "rated_torque")
    psi_f = _resolve_psi_f(motor)

    if motor.I_rated is None:
        raise ValueError(
            f"Motor '{motor.name}': rated_torque needs I_rated"
        )

    L_d, L_q = _salient_pair(motor)

    return RatedTorqueParams(
        n_p=motor.n_p,
        psi_f=psi_f,
        I_rated=motor.I_rated,
        L_d=L_d,
        L_q=L_q,
    )


def prepare_stall_torque(motor: Motor) -> StallTorqueParams:
    """Derive, validate, and extract stall torque params from a Motor.

    I_stall = I_LIMIT (or MAX_I_S when I_LIMIT is not set).
    I_stall_em = U_DC / (sqrt(3) * R_s) — electromagnetic stall (no drive limit).
    """
    from math import sqrt

    _reject_induction(motor, "stall_torque")
    psi_f = _resolve_psi_f(motor)

    if motor.R_s is None:
        raise ValueError(
            f"Motor '{motor.name}': stall_torque needs R_s"
        )

    drive = motor.drive
    if drive.I_LIMIT is None and drive.MAX_I_S is None:
        raise ValueError(
            f"Motor '{motor.name}': stall_torque needs [drive] "
            f"MAX_I_S (or I_LIMIT)"
        )
    I_stall = drive.I_LIMIT if drive.I_LIMIT is not None else drive.MAX_I_S
    assert I_stall is not None  # narrowed above
    if drive.U_DC is None:
        raise ValueError(
            f"Motor '{motor.name}': stall_torque needs [drive] U_DC"
        )
    I_stall_em = drive.U_DC / (sqrt(3) * motor.R_s) if motor.R_s > 0 else None

    L_d, L_q = _salient_pair(motor)

    return StallTorqueParams(
        n_p=motor.n_p,
        psi_f=psi_f,
        I_stall=I_stall,
        I_stall_em=I_stall_em,
        L_d=L_d,
        L_q=L_q,
    )
