"""Rated torque and MTPA characteristic curve computation.

Implements Morimoto (1994) MTPA quadratic for salient machines (both
saliency signs), with k_T * I_rated fallback for non-salient.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from math import asin, cos, degrees, pi, sin, sqrt
from typing import TYPE_CHECKING, Any

import numpy as np

from phasesweep.models.flux_map import (
    cold_psi_f,
    derated_flux_map,
    flux_map_torque,
    make_flux_map_interpolators,
    map_covers,
)
from phasesweep.solver_params import (
    derate_motor,
    reject_induction,
    resolve_psi_f,
    salient_pair,
)

if TYPE_CHECKING:
    from phasesweep.machines.motor import FluxMap, Motor
    from phasesweep.sweep_types import RunConfig

__all__ = [
    "RatedTorqueParams",
    "StallTorqueParams",
    "magnet_torque_constant",
    "mtpa_curve",
    "mtpa_gamma",
    "mtpa_numeric",
    "mtpa_torque",
    "prepare_rated_torque",
    "prepare_stall_torque",
    "resolve_psi_f_or_map",
    "run_rated_torque",
    "run_stall_torque",
]


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


def mtpa_numeric(
    n_p: int, flux_map: FluxMap, I_s: float, n_pts: int = 72,
) -> tuple[float, float, float] | None:
    """Numeric MTPA on the flux-linkage surface: sweep the current angle
    gamma from 0 to π/2 (i_d = -I_s sin γ, i_q = I_s cos γ) and return
    (tau_max, i_d_opt, i_q_opt). Only gamma values whose (i_d, i_q) lies
    within the tabulated grid are evaluated; returns None when no feasible
    point exists (caller falls back to the analytical path).
    """
    if I_s <= 0:
        return 0.0, 0.0, 0.0
    psi_d_interp, psi_q_interp = make_flux_map_interpolators(flux_map)
    gamma = np.linspace(0.0, pi / 2, n_pts)
    i_d = -I_s * np.sin(gamma)
    i_q = I_s * np.cos(gamma)
    feasible = (
        (i_d >= flux_map.i_d_A[0]) & (i_d <= flux_map.i_d_A[-1])
        & (i_q >= flux_map.i_q_A[0]) & (i_q <= flux_map.i_q_A[-1])
    )
    if not feasible.any():
        return None
    i_d_f, i_q_f = i_d[feasible], i_q[feasible]
    tau = np.asarray(flux_map_torque(n_p, psi_d_interp, psi_q_interp, i_d_f, i_q_f))
    k = int(np.argmax(tau))
    return float(tau[k]), float(i_d_f[k]), float(i_q_f[k])


def _mtpa_result(
    n_p: int,
    psi_f: float,
    I_s: float,
    L_d: float | None,
    L_q: float | None,
    flux_map: FluxMap | None = None,
) -> dict[str, Any]:
    """Shared MTPA computation at a given current magnitude.

    Returns tau, k_T, k_T_effective (= tau / I_s, folds in reluctance
    torque), gamma_opt_deg (< 0 for reverse saliency, magnetizing i_d),
    mtpa_source ("flux_map" | "analytical"), and curve data. With a
    flux_map whose grid covers the current circle (flux_map.map_covers)
    the optimum comes from the tabulated surface (mtpa_numeric); past the
    grid the search would pin to its edge, so the analytical path runs.
    k_T stays the zero-current magnet constant either way.
    """
    k_T = magnet_torque_constant(n_p, psi_f)

    if flux_map is not None and map_covers(flux_map, I_s):
        result_fm = mtpa_numeric(n_p, flux_map, I_s)
        if result_fm is not None:
            tau, i_d_opt, i_q_opt = result_fm
            return {
                "tau": tau,
                "k_T": k_T,
                "k_T_effective": tau / I_s,
                "gamma_opt_deg": degrees(np.arctan2(-i_d_opt, i_q_opt)),
                "i_d_opt": i_d_opt,
                "i_q_opt": i_q_opt,
                "mtpa_source": "flux_map",
            }

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
        # MTPA optimum can never lose to gamma=0 (pure q-axis current)
        tau_q0 = k_T * I_s
        if tau < tau_q0 * (1 - 1e-9):
            raise RuntimeError(
                f"MTPA solver regression: tau={tau:.6e} < k_T*I_s={tau_q0:.6e}"
            )

    result: dict[str, Any] = {
        "tau": tau,
        "k_T": k_T,
        "k_T_effective": tau / I_s,
        "gamma_opt_deg": gamma_opt,
        "mtpa_source": "analytical",
    }

    if Ld > 0 and Lq > 0:
        result.update(mtpa_curve(n_p, psi_f, Ld, Lq, I_s))

    return result


def run_rated_torque(config: RunConfig) -> dict[str, Any]:
    """Rated torque computation. Returns scalars + curves."""

    p = prepare_rated_torque(config.motor)
    r = _mtpa_result(p.n_p, p.psi_f, p.I_rated, p.L_d, p.L_q,
                     derated_flux_map(config.motor))

    return {
        "tau_mtpa": r["tau"],
        "k_T": r["k_T"],
        "k_T_rms": r["k_T"] * sqrt(2),
        "k_T_effective": r["k_T_effective"],
        "gamma_opt_deg": r["gamma_opt_deg"],
        "mtpa_source": r["mtpa_source"],
        **{k: r[k] for k in _CURVE_KEYS if k in r},
    }


def run_stall_torque(config: RunConfig) -> dict[str, Any]:
    """Stall torque computation (MTPA at I_stall). Returns scalars + curves.

    Emits both current-limited (tau_stall, I_stall) and electromagnetic
    (tau_stall_electromagnetic, I_stall_electromagnetic) values.
    """

    p = prepare_stall_torque(config.motor)
    flux_map = derated_flux_map(config.motor)
    r = _mtpa_result(p.n_p, p.psi_f, p.I_stall, p.L_d, p.L_q, flux_map)

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
        "mtpa_source": r["mtpa_source"],
        "saturation_ratio": saturation_ratio,
        "saturation_warning": saturation_warning,
        **{k: r[k] for k in _CURVE_KEYS if k in r},
    }

    if p.I_stall_em is not None:
        r_em = _mtpa_result(p.n_p, p.psi_f, p.I_stall_em, p.L_d, p.L_q, flux_map)
        result["tau_stall_electromagnetic"] = r_em["tau"]
        result["I_stall_electromagnetic"] = p.I_stall_em
        result["mtpa_source_electromagnetic"] = r_em["mtpa_source"]

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

    @property
    def is_reluctance(self) -> bool:
        return self.psi_f == 0.0

    def __post_init__(self) -> None:
        # NaN passes every inequality below, so finiteness comes first.
        non_finite = [f for f in ('psi_f', 'I_rated', 'L_d', 'L_q')
                      if getattr(self, f) is not None
                      and not math.isfinite(getattr(self, f))]
        if non_finite:
            raise ValueError(f"rated_torque got non-finite {', '.join(non_finite)}")
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

    @property
    def is_reluctance(self) -> bool:
        return self.psi_f == 0.0

    def __post_init__(self) -> None:
        # NaN passes every inequality below, so finiteness comes first.
        non_finite = [f for f in ('psi_f', 'I_stall', 'I_stall_em', 'L_d', 'L_q')
                      if getattr(self, f) is not None
                      and not math.isfinite(getattr(self, f))]
        if non_finite:
            raise ValueError(f"stall_torque got non-finite {', '.join(non_finite)}")
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


def resolve_psi_f_or_map(motor: Motor) -> float:
    """Cold (20 °C) psi_f from the motor, else ψ_d(0, 0) of its flux map
    referred to 20 °C (flux_map.cold_psi_f)."""
    if motor.psi_f is None and motor.flux_map is not None:
        return cold_psi_f(motor.flux_map, motor.alpha_Br)
    return resolve_psi_f(motor)


def prepare_rated_torque(motor: Motor) -> RatedTorqueParams:
    """Derive, validate, and extract rated torque params from a Motor.
    psi_f is at magnet_temp when set (solver_params.derate)."""
    reject_induction(motor, "rated_torque")
    psi_f = derate_motor(motor, None, resolve_psi_f_or_map(motor))[1]

    if motor.I_rated is None:
        raise ValueError(
            f"Motor '{motor.name}': rated_torque needs I_rated"
        )

    L_d, L_q = salient_pair(motor)

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
    I_stall_em = U_DC / (sqrt(3) * R_s) — electromagnetic stall (no drive
    limit), with R_s at winding_temp_limit and psi_f at magnet_temp when set.
    """
    from math import sqrt

    reject_induction(motor, "stall_torque")
    if motor.R_s is None:
        raise ValueError(
            f"Motor '{motor.name}': stall_torque needs R_s"
        )
    R_s, psi_f = derate_motor(motor, motor.R_s, resolve_psi_f_or_map(motor))
    assert R_s is not None

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
    I_stall_em = drive.U_DC / (sqrt(3) * R_s) if R_s > 0 else None

    L_d, L_q = salient_pair(motor)

    return StallTorqueParams(
        n_p=motor.n_p,
        psi_f=psi_f,
        I_stall=I_stall,
        I_stall_em=I_stall_em,
        L_d=L_d,
        L_q=L_q,
    )
