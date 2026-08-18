"""Torque-speed envelope model: MTPA below base speed, voltage-limited above.

Steady-state dq circuit model (motulator peak conventions):

    u_d = R_s i_d - w_e L_q i_q
    u_q = R_s i_q + w_e (psi_f + L_d i_d)

subject to i_d^2 + i_q^2 <= I_max^2 and u_d^2 + u_q^2 <= U_max^2, with
U_max = U_DC / sqrt(3) (space-vector PWM linear region, phase peak — same
convention as stall_torque's electromagnetic stall current) and
w_e = n_p * w_mech.

At each speed the envelope torque is the maximum of
tau = 1.5 n_p (psi_f + (L_d - L_q) i_d) i_q over the feasible set. Below
base speed the exact MTPA point is returned (identical to rated_torque's
tau at the same current); above, the optimum is found by a deterministic
i_d grid search with golden-section refinement — this covers field
weakening and the MTPV branch for both saliency signs without case
analysis.

Uses catalogue (cold) R_s and psi_f like rated_torque/stall_torque —
temperature derating stays a thermal_duty concern.
"""

from __future__ import annotations

from math import cos, sin, sqrt
from typing import TYPE_CHECKING, Any

from phasesweep.models.rated_torque import mtpa_gamma, mtpa_torque

if TYPE_CHECKING:
    from phasesweep.sweep_types import RunConfig

N_SPEED_PTS = 61
_ID_GRID_PTS = 601
_GOLDEN = (sqrt(5.0) - 1.0) / 2.0


def dq_voltage(n_p: int, psi_f: float, R_s: float, L_d: float, L_q: float,
               i_d: float, i_q: float, w_mech: float) -> float:
    """Steady-state stator voltage magnitude (V, phase peak)."""
    w_e = n_p * w_mech
    u_d = R_s * i_d - w_e * L_q * i_q
    u_q = R_s * i_q + w_e * (psi_f + L_d * i_d)
    return sqrt(u_d * u_d + u_q * u_q)


def _iq_voltage_max(psi_f: float, R_s: float, L_d: float, L_q: float,
                    i_d: float, w_e: float, U_max: float) -> float:
    """Largest i_q satisfying the voltage limit at fixed i_d.

    |u|^2 is quadratic in i_q with positive leading coefficient
    R_s^2 + (w_e L_q)^2; the largest root is the voltage-feasible i_q
    ceiling. Returns +inf when unconstrained (w_e = 0, R_s = 0) and
    -inf when no i_q satisfies the limit at this i_d.
    """
    psi_d = psi_f + L_d * i_d
    a = R_s * R_s + (w_e * L_q) ** 2
    if a == 0.0:
        return float("inf")
    b = 2.0 * R_s * w_e * (psi_d - L_q * i_d)
    c = (R_s * i_d) ** 2 + (w_e * psi_d) ** 2 - U_max * U_max
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return float("-inf")
    return (-b + sqrt(disc)) / (2.0 * a)


def _id_feasible_band(psi_f: float, R_s: float, L_d: float, I_max: float,
                      w_e: float, U_max: float) -> tuple[float, float] | None:
    """i_d interval where the voltage limit admits i_q >= 0.

    Solves R_s^2 i_d^2 + w_e^2 (psi_f + L_d i_d)^2 <= U_max^2 (the c <= 0
    condition of the i_q quadratic), intersected with [-I_max, I_max].
    Positive-torque optima always satisfy c <= 0: the complementary
    feasible band (b < 0, c > 0) only exists where the torque coefficient
    psi_f + (L_d - L_q) i_d has flipped sign, so nothing on the positive
    envelope is lost. Returns None when the band is empty (past max speed).
    """
    a = R_s * R_s + (w_e * L_d) ** 2
    if a == 0.0:
        return -I_max, I_max
    b = 2.0 * w_e * w_e * L_d * psi_f
    c = (w_e * psi_f) ** 2 - U_max * U_max
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None
    root = sqrt(disc)
    lo = max((-b - root) / (2.0 * a), -I_max)
    hi = min((-b + root) / (2.0 * a), I_max)
    if lo > hi:
        return None
    return lo, hi


def _tau_at_id(n_p: int, psi_f: float, R_s: float, L_d: float, L_q: float,
               I_max: float, U_max: float, w_e: float,
               i_d: float) -> tuple[float, float]:
    """(tau, i_q) at the feasible i_q ceiling for a fixed i_d; tau=0 if none."""
    i_q_circle = sqrt(max(0.0, I_max * I_max - i_d * i_d))
    i_q = min(i_q_circle, _iq_voltage_max(psi_f, R_s, L_d, L_q, i_d, w_e, U_max))
    if i_q <= 0.0:
        return 0.0, 0.0
    return 1.5 * n_p * (psi_f + (L_d - L_q) * i_d) * i_q, i_q


def envelope_at_speed(n_p: int, psi_f: float, R_s: float, L_d: float,
                      L_q: float, I_max: float, U_max: float,
                      w_mech: float) -> tuple[float, float, float]:
    """Max torque (Nm) and its (i_d, i_q) at a mechanical speed (rad/s).

    Exact MTPA point when it is voltage-feasible (below base speed);
    otherwise grid + golden-section max over i_d on the constraint
    boundary.
    """
    w_e = n_p * w_mech
    g = mtpa_gamma(psi_f, L_d, L_q, I_max)
    i_d0, i_q0 = -I_max * sin(g), I_max * cos(g)
    if dq_voltage(n_p, psi_f, R_s, L_d, L_q, i_d0, i_q0, w_mech) <= U_max * (1 + 1e-12):
        return mtpa_torque(n_p, psi_f, L_d, L_q, I_max, g), i_d0, i_q0

    band = _id_feasible_band(psi_f, R_s, L_d, I_max, w_e, U_max)
    if band is None:
        return 0.0, 0.0, 0.0
    band_lo, band_hi = band
    step = (band_hi - band_lo) / (_ID_GRID_PTS - 1)
    best_tau, best_id, best_iq = 0.0, 0.0, 0.0
    for k in range(_ID_GRID_PTS):
        i_d = band_lo + k * step
        tau, i_q = _tau_at_id(n_p, psi_f, R_s, L_d, L_q, I_max, U_max, w_e, i_d)
        if tau > best_tau:
            best_tau, best_id, best_iq = tau, i_d, i_q
    if best_tau == 0.0:
        return 0.0, 0.0, 0.0

    # Golden-section refine on the bracket around the best grid point
    lo = max(band_lo, best_id - step)
    hi = min(band_hi, best_id + step)
    for _ in range(80):
        m1 = hi - _GOLDEN * (hi - lo)
        m2 = lo + _GOLDEN * (hi - lo)
        t1, _ = _tau_at_id(n_p, psi_f, R_s, L_d, L_q, I_max, U_max, w_e, m1)
        t2, _ = _tau_at_id(n_p, psi_f, R_s, L_d, L_q, I_max, U_max, w_e, m2)
        if t1 < t2:
            lo = m1
        else:
            hi = m2
    i_d = 0.5 * (lo + hi)
    tau, i_q = _tau_at_id(n_p, psi_f, R_s, L_d, L_q, I_max, U_max, w_e, i_d)
    if tau >= best_tau:
        return tau, i_d, i_q
    return best_tau, best_id, best_iq


def base_speed(n_p: int, psi_f: float, R_s: float, L_d: float, L_q: float,
               I_max: float, U_max: float) -> float:
    """Highest mechanical speed (rad/s) where MTPA at I_max is voltage-feasible.

    |u|^2 at the fixed MTPA currents is quadratic in w_e; the positive
    root is exact (R_s included). Returns 0.0 when the resistive drop
    alone exceeds U_max (guarded against in TorqueSpeedParams).
    """
    g = mtpa_gamma(psi_f, L_d, L_q, I_max)
    i_d, i_q = -I_max * sin(g), I_max * cos(g)
    psi_d = psi_f + L_d * i_d
    psi_q = L_q * i_q
    A = psi_d * psi_d + psi_q * psi_q
    B = 2.0 * R_s * (i_q * psi_d - i_d * psi_q)
    C = R_s * R_s * I_max * I_max - U_max * U_max
    if C >= 0.0:
        return 0.0
    if A == 0.0:
        return float("inf")
    w_e = (-B + sqrt(B * B - 4.0 * A * C)) / (2.0 * A)
    return w_e / n_p


def max_speed(n_p: int, psi_f: float, R_s: float, L_d: float, L_q: float,
              I_max: float, U_max: float) -> float | None:
    """Mechanical speed (rad/s) where the envelope torque reaches zero.

    None when psi_f <= L_d * I_max: the voltage-ellipse centre lies inside
    the current circle, so the MTPV branch extends the envelope to
    unbounded speed (torque decays but never reaches zero).

    L_q does not enter (zero-torque speed sits on the d-axis); the
    parameter is kept so base_speed/max_speed share one signature.
    """
    if psi_f <= L_d * I_max:
        return None
    num = U_max * U_max - (R_s * I_max) ** 2
    if num <= 0.0:
        return 0.0
    w_e = sqrt(num) / (psi_f - L_d * I_max)
    return w_e / n_p


def _envelope_scalars(n_p: int, psi_f: float, R_s: float, L_d: float,
                      L_q: float, I_max: float,
                      U_max: float) -> tuple[float, float | None]:
    return (
        base_speed(n_p, psi_f, R_s, L_d, L_q, I_max, U_max),
        max_speed(n_p, psi_f, R_s, L_d, L_q, I_max, U_max),
    )


def run_torque_speed(config: RunConfig) -> dict[str, Any]:
    """Torque-speed envelope at the drive current limit (peak) and, when
    I_rated is set, at the continuous rating (cont). Speeds are mechanical
    rad/s; one shared speed grid spans the widest envelope."""
    from phasesweep.solver_params import prepare_torque_speed

    p = prepare_torque_speed(config.motor)
    args = (p.n_p, p.psi_f, p.R_s, p.L_d, p.L_q)

    w_base_pk, w_max_pk = _envelope_scalars(*args, p.I_peak, p.U_max)
    scalars_cont: tuple[float, float | None] | None = None
    if p.I_cont is not None:
        scalars_cont = _envelope_scalars(*args, p.I_cont, p.U_max)

    ends = [w_max_pk]
    bases = [w_base_pk]
    if scalars_cont is not None:
        ends.append(scalars_cont[1])
        bases.append(scalars_cont[0])
    finite_ends = [w for w in ends if w is not None]
    if len(finite_ends) == len(ends):
        w_end = max(finite_ends)
    else:
        w_end = 3.0 * max(bases)

    speeds = [w_end * k / (N_SPEED_PTS - 1) for k in range(N_SPEED_PTS)]
    tau_pk = [
        envelope_at_speed(*args, p.I_peak, p.U_max, w)[0] for w in speeds
    ]

    result: dict[str, Any] = {
        "speed_curve": speeds,
        "tau_curve_peak": tau_pk,
        "base_speed_peak": w_base_pk,
        "max_speed_peak": w_max_pk,
        "p_max_peak": max(t * w for t, w in zip(tau_pk, speeds)),
        "u_max": p.U_max,
        "I_env_peak": p.I_peak,
    }

    if p.I_cont is not None and scalars_cont is not None:
        tau_cont = [
            envelope_at_speed(*args, p.I_cont, p.U_max, w)[0] for w in speeds
        ]
        result["tau_curve_cont"] = tau_cont
        result["base_speed_cont"] = scalars_cont[0]
        result["max_speed_cont"] = scalars_cont[1]
        result["p_max_cont"] = max(t * w for t, w in zip(tau_cont, speeds))
        result["I_env_cont"] = p.I_cont

    return result
