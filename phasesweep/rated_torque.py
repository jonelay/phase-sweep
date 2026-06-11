"""Rated torque and MTPA characteristic curve computation.

Implements Morimoto (1994) MTPA quadratic for salient machines,
with k_T * I_rated fallback for non-salient / reverse-salient.
"""

from __future__ import annotations

from math import asin, cos, degrees, sin, sqrt
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from phasesweep.sweep_types import RunConfig


def mtpa_gamma(psi_f: float, L_d: float, L_q: float, I_s: float) -> float:
    """MTPA angle in radians from q-axis. 0 = all current on q-axis."""
    if I_s <= 0:
        return 0.0
    dL = L_q - L_d
    if dL <= 0:
        return 0.0
    sin_g = (-psi_f + sqrt(psi_f**2 + 8 * dL**2 * I_s**2)) / (4 * dL * I_s)
    sin_g = max(-1.0, min(1.0, sin_g))
    return asin(sin_g)


def mtpa_torque(n_p: int, psi_f: float, L_d: float, L_q: float,
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

    Returns tau, k_T, gamma_opt_deg, and curve data.
    """
    k_T = 1.5 * n_p * psi_f

    use_simple_kt = (
        L_d is None or L_q is None
        or L_q <= L_d
    )

    if use_simple_kt:
        tau = k_T * I_s
        gamma_opt = 0.0
        Ld = L_d if L_d is not None else 0.0
        Lq = L_q if L_q is not None else 0.0
    else:
        Ld = L_d  # type: ignore[assignment]
        Lq = L_q  # type: ignore[assignment]
        g = mtpa_gamma(psi_f, Ld, Lq, I_s)
        tau = mtpa_torque(n_p, psi_f, Ld, Lq, I_s, g)
        gamma_opt = degrees(g)

    result: dict[str, Any] = {
        "tau": tau,
        "k_T": k_T,
        "gamma_opt_deg": gamma_opt,
    }

    if Ld > 0 and Lq > 0:
        result.update(mtpa_curve(n_p, psi_f, Ld, Lq, I_s))

    return result


def run_rated_torque(config: RunConfig) -> dict[str, Any]:
    """Rated torque computation. Returns scalars + curves."""
    from phasesweep.solver_params import prepare_rated_torque

    p = prepare_rated_torque(config.motor)
    r = _mtpa_result(p.n_p, p.psi_f, p.I_rated, p.L_d, p.L_q)

    return {
        "tau_mtpa": r["tau"],
        "k_T": r["k_T"],
        "k_T_rms": r["k_T"] * sqrt(2),
        "gamma_opt_deg": r["gamma_opt_deg"],
        **{k: r[k] for k in _CURVE_KEYS if k in r},
    }


def run_stall_torque(config: RunConfig) -> dict[str, Any]:
    """Stall torque computation (MTPA at I_stall). Returns scalars + curves.

    Emits both current-limited (tau_stall, I_stall) and electromagnetic
    (tau_stall_electromagnetic, I_stall_electromagnetic) values.
    """
    from phasesweep.solver_params import prepare_stall_torque

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
