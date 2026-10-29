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

R_s and psi_f (and the flux map) are at the motor's operating
temperatures — winding_temp_limit and magnet_temp through the shared
solver_params.derate, like every circuit model; cold when
the thermal fields are unset.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import cos, isfinite, sin, sqrt
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from phasesweep.models.flux_map import derated_flux_map, make_flux_map_interpolators
from phasesweep.models.rated_torque import mtpa_gamma, mtpa_numeric, mtpa_torque
from phasesweep.solver_params import derate_motor, reject_induction, resolve_psi_f

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from phasesweep.machines.motor import FluxMap, Motor
    from phasesweep.sweep_types import RunConfig

__all__ = [
    "N_SPEED_PTS",
    "TorqueSpeedParams",
    "base_speed",
    "cold_torque_speed_params",
    "dq_voltage",
    "envelope_at_speed",
    "envelope_curve",
    "envelope_curve_batch",
    "envelope_curve_map",
    "max_speed",
    "prepare_torque_speed",
    "run_torque_speed",
    "run_torque_speed_batch",
]

N_SPEED_PTS = 61
# The grid only has to *bracket* the maximum — the 80 golden-section
# contractions below resolve it to machine precision from there, so grid
# density buys bracketing robustness, not accuracy.
#
# Measured against a 2001-point reference across 10195 swept synthetic
# designs (n_p 1-48, saliency 0.01-100, R_s 1e-4..30 ohm, U_max down to
# the resistive limit R_s*I_max, 40% unbounded-CPSR): every grid from 601
# down to 31 agrees to <= 2e-11 of curve scale, with no design anywhere
# above 1e-9. The feared failure — a bracket straddling the
# current-circle/voltage-limit kink and selecting a worse basin — does not
# occur in that domain at any of those densities.
#
# 151 is therefore a conservative choice, not a measured floor. Measure at
# curve scale (max|err| / max(tau)) if you revisit it: the last speed sits
# at the collapsed band where torque is ~1e-11 N·m, so a per-lane relative
# metric turns floating-point noise there into phantom percent-level
# cliffs.
_ID_GRID_PTS = 151
_GOLDEN = (sqrt(5.0) - 1.0) / 2.0
_GOLDEN_ITERS = 80
_BISECT_ITERS = 30

# The optimizer is vectorized over the *speed* axis, with an optional
# leading design axis for the batch path: the i_d grid (when present) is
# always the last axis. Both branches are evaluated for every speed and
# selected at the end, because branching per speed is what the array form
# exists to avoid.
#
# v2 kept the search unchanged in kind and moved only the association of the
# arithmetic (<= 4e-14 relative on tau). v3 thins the bracketing grid to 151
# points (see _ID_GRID_PTS). Any edit that changes the *search* — grid
# density, contraction count, bracket — is a real physics change: bump the
# version.


def dq_voltage(n_p: float | NDArray, psi_f: float | NDArray,
               R_s: float | NDArray, L_d: float | NDArray,
               L_q: float | NDArray, i_d: float | NDArray,
               i_q: float | NDArray, w_mech: float | NDArray) -> float | NDArray:
    """Steady-state stator voltage magnitude (V, phase peak).

    Elementwise: accepts scalars or broadcastable arrays for i_d/i_q/w_mech.
    """
    w_e = n_p * w_mech
    u_d = R_s * i_d - w_e * L_q * i_q
    u_q = R_s * i_q + w_e * (psi_f + L_d * i_d)
    return np.sqrt(u_d * u_d + u_q * u_q)


class _Boundary(NamedTuple):
    """Everything the i_d optimum needs that does not depend on i_d.

    The refinement loop evaluates the same kernel ~80 times per speed, so
    every speed-only quantity is computed once here. Fields carrying `_safe`
    have the a == 0 lane (no voltage limit: w_e = 0 and R_s = 0) substituted
    with 1.0 purely to keep the division defined; `unlimited` selects that
    lane back out.
    """
    n_p: float | NDArray
    psi_f: float | NDArray
    L_d: float | NDArray
    L_q: float | NDArray
    R_s: float | NDArray
    dL: float | NDArray    # L_d - L_q
    k_t: float | NDArray   # 1.5 * n_p
    I_max2: float | NDArray
    U_max2: float | NDArray
    w_e: NDArray
    k_b: NDArray       # 2 R_s w_e
    four_a_safe: NDArray
    inv_2a_safe: NDArray
    unlimited: NDArray


def _boundary(n_p: float | NDArray, psi_f: float | NDArray,
              R_s: float | NDArray, L_d: float | NDArray,
              L_q: float | NDArray, I_max: float | NDArray,
              U_max: float | NDArray, w_e: NDArray) -> _Boundary:
    a = R_s * R_s + (w_e * L_q) * (w_e * L_q)
    unlimited = a == 0.0
    a_safe = np.where(unlimited, 1.0, a)
    return _Boundary(
        n_p=n_p, psi_f=psi_f, L_d=L_d, L_q=L_q, R_s=R_s, dL=L_d - L_q,
        k_t=1.5 * n_p, I_max2=I_max * I_max, U_max2=U_max * U_max,
        w_e=w_e, k_b=2.0 * R_s * w_e,
        four_a_safe=4.0 * a_safe, inv_2a_safe=1.0 / (2.0 * a_safe),
        unlimited=unlimited,
    )


def _id_feasible_band(psi_f: float | NDArray, R_s: float | NDArray,
                      L_d: float | NDArray, I_max: float | NDArray,
                      w_e: NDArray,
                      U_max: float | NDArray
                      ) -> tuple[NDArray, NDArray, NDArray]:
    """i_d interval where the voltage limit admits i_q >= 0, per speed.

    Solves R_s^2 i_d^2 + w_e^2 (psi_f + L_d i_d)^2 <= U_max^2 (the c <= 0
    condition of the i_q quadratic), intersected with [-I_max, I_max].
    Positive-torque optima always satisfy c <= 0: the complementary
    feasible band (b < 0, c > 0) only exists where the torque coefficient
    psi_f + (L_d - L_q) i_d has flipped sign, so nothing on the positive
    envelope is lost.

    Returns (lo, hi, empty); lo/hi are collapsed to 0 where empty is True
    (band empty — past max speed) so downstream arithmetic stays finite.
    """
    a = R_s * R_s + (w_e * L_d) * (w_e * L_d)
    b = 2.0 * w_e * w_e * L_d * psi_f
    w_psi = w_e * psi_f
    c = w_psi * w_psi - U_max * U_max
    disc = b * b - 4.0 * a * c
    a_safe = np.where(a == 0.0, 1.0, a)
    root = np.sqrt(np.where(disc < 0.0, 0.0, disc))
    lo = np.maximum((-b - root) / (2.0 * a_safe), -I_max)
    hi = np.minimum((-b + root) / (2.0 * a_safe), I_max)
    # a == 0 (w_e = 0 and R_s = 0): the limit is inactive, whole circle.
    unconstrained = a == 0.0
    lo = np.where(unconstrained, -I_max, lo)
    hi = np.where(unconstrained, I_max, hi)
    empty = np.where(unconstrained, False, disc < 0.0) | (lo > hi)
    return np.where(empty, 0.0, lo), np.where(empty, 0.0, hi), empty


def _tau_at_id(bd: _Boundary, i_d: NDArray) -> tuple[NDArray, NDArray]:
    """(tau, i_q) at the feasible i_q ceiling for each i_d; tau=0 where none.

    i_q is the smaller of the current-circle ceiling and the largest root of
    the voltage-limit quadratic in i_q (positive leading coefficient a, so
    the + root is the ceiling). No feasible i_q >= 0 gives tau = 0.
    `i_d` broadcasts against the boundary's speed axis, so this serves both
    the (speeds x i_d grid) scan and the (speeds,) refinement.
    """
    psi_d = bd.psi_f + bd.L_d * i_d
    r_i = bd.R_s * i_d
    w_psi = bd.w_e * psi_d
    b = bd.k_b * (psi_d - bd.L_q * i_d)
    c = r_i * r_i + w_psi * w_psi - bd.U_max2
    disc = b * b - bd.four_a_safe * c
    i_q = (-b + np.sqrt(np.maximum(disc, 0.0))) * bd.inv_2a_safe
    i_q = np.where(bd.unlimited, np.inf, np.where(disc < 0.0, -np.inf, i_q))
    i_q = np.minimum(np.sqrt(np.maximum(0.0, bd.I_max2 - i_d * i_d)), i_q)
    # Clamp before forming tau so the -inf lanes never reach an inf * 0.
    i_q = np.maximum(i_q, 0.0)
    return bd.k_t * (bd.psi_f + bd.dL * i_d) * i_q, i_q


def _expand(x: float | NDArray) -> float | NDArray:
    return x[..., None] if isinstance(x, np.ndarray) else x


def _envelope_core(n_p: float | NDArray, psi_f: float | NDArray,
                   R_s: float | NDArray, L_d: float | NDArray,
                   L_q: float | NDArray, I_max: float | NDArray,
                   U_max: float | NDArray, i_d0: float | NDArray,
                   i_q0: float | NDArray, tau_mtpa: float | NDArray,
                   w_mech: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """Shared envelope search: grid scan + golden-section + branch select.

    Serves two shapes: scalar params with 1D speeds (serial), and (N,1)
    params with (N,S) speeds (batch). The i_d grid always lives on the
    last axis; the MTPA point (i_d0, i_q0, tau_mtpa) arrives precomputed
    because mtpa_gamma is scalar code.
    """
    w_e = n_p * w_mech
    on_mtpa = (dq_voltage(n_p, psi_f, R_s, L_d, L_q, i_d0, i_q0, w_mech)
               <= U_max * (1 + 1e-12))

    band_lo, band_hi, empty = _id_feasible_band(psi_f, R_s, L_d, I_max,
                                                w_e, U_max)
    step = (band_hi - band_lo) / (_ID_GRID_PTS - 1)

    # Grid scan: i_d samples on the last axis; array params gain a
    # matching trailing axis so they broadcast against it.
    grid_bd = _boundary(_expand(n_p), _expand(psi_f), _expand(R_s),
                        _expand(L_d), _expand(L_q), _expand(I_max),
                        _expand(U_max), w_e[..., None])
    i_d_grid = (band_lo[..., None]
                + np.arange(float(_ID_GRID_PTS)) * step[..., None])
    tau_grid, i_q_grid = _tau_at_id(grid_bd, i_d_grid)
    # A non-finite speed, or an overflow inside the quadratic, leaves NaN
    # lanes. argmax would *select* one (NaN propagates as the maximum) and
    # `best_tau <= 0` is False for NaN, so it would escape both the no_peak
    # and dead masks and surface as a NaN torque. The scalar scan this
    # replaced rejected them implicitly: `tau > best_tau` is False for NaN.
    tau_grid = np.where(np.isnan(tau_grid), -np.inf, tau_grid)
    k_best = np.argmax(tau_grid, axis=-1)
    sel = k_best[..., None]
    best_tau = np.take_along_axis(tau_grid, sel, axis=-1)[..., 0]
    best_id = np.take_along_axis(i_d_grid, sel, axis=-1)[..., 0]
    best_iq = np.take_along_axis(i_q_grid, sel, axis=-1)[..., 0]
    # argmax takes the first maximum, matching a strict `tau > best` scan; a
    # non-positive best means no feasible positive-torque point at all.
    no_peak = best_tau <= 0.0
    best_tau = np.where(no_peak, 0.0, best_tau)
    best_id = np.where(no_peak, 0.0, best_id)
    best_iq = np.where(no_peak, 0.0, best_iq)

    # Golden-section refine on the bracket around each best grid point. The
    # golden ratio is chosen precisely so that one of the two interior probes
    # survives each contraction, so only the *new* probe is evaluated per
    # iteration — one kernel call, not two.
    bd = _boundary(n_p, psi_f, R_s, L_d, L_q, I_max, U_max, w_e)
    lo = np.maximum(band_lo, best_id - step)
    hi = np.minimum(band_hi, best_id + step)
    m1 = hi - _GOLDEN * (hi - lo)
    m2 = lo + _GOLDEN * (hi - lo)
    t1, _ = _tau_at_id(bd, m1)
    t2, _ = _tau_at_id(bd, m2)
    for _ in range(_GOLDEN_ITERS):
        # t1 < t2 → the max lies right of m1, so lo climbs to m1 and the
        # surviving probe m2 becomes the new left probe. Otherwise hi drops
        # to m2 and m1 becomes the new right probe.
        shrink_lo = t1 < t2
        lo = np.where(shrink_lo, m1, lo)
        hi = np.where(shrink_lo, hi, m2)
        kept = np.where(shrink_lo, m2, m1)
        kept_t = np.where(shrink_lo, t2, t1)
        span = _GOLDEN * (hi - lo)
        probe = np.where(shrink_lo, lo + span, hi - span)
        probe_t, _ = _tau_at_id(bd, probe)
        m1 = np.where(shrink_lo, kept, probe)
        m2 = np.where(shrink_lo, probe, kept)
        t1 = np.where(shrink_lo, kept_t, probe_t)
        t2 = np.where(shrink_lo, probe_t, kept_t)
    i_d_ref = 0.5 * (lo + hi)
    tau_ref, i_q_ref = _tau_at_id(bd, i_d_ref)

    # Refinement is only accepted when it did not lose to the grid point.
    keep_ref = tau_ref >= best_tau
    tau = np.where(keep_ref, tau_ref, best_tau)
    i_d = np.where(keep_ref, i_d_ref, best_id)
    i_q = np.where(keep_ref, i_q_ref, best_iq)

    # Select branch: MTPA wins outright; an empty band or no positive grid
    # peak means zero torque.
    dead = empty | no_peak
    tau = np.where(on_mtpa, tau_mtpa, np.where(dead, 0.0, tau))
    i_d = np.where(on_mtpa, i_d0, np.where(dead, 0.0, i_d))
    i_q = np.where(on_mtpa, i_q0, np.where(dead, 0.0, i_q))
    return tau, i_d, i_q


def envelope_curve(n_p: int, psi_f: float, R_s: float, L_d: float,
                   L_q: float, I_max: float, U_max: float,
                   w_mech: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """Envelope (tau, i_d, i_q) for a whole array of mechanical speeds.

    Exact MTPA point where it is voltage-feasible (below base speed);
    elsewhere grid + golden-section max over i_d on the constraint
    boundary. Every speed is carried through both branches and selected at
    the end, which is what makes the whole sweep one set of array ops.
    """
    w_mech = np.asarray(w_mech, dtype=float)

    # MTPA branch — the optimum currents do not depend on speed, only their
    # voltage feasibility does.
    g = mtpa_gamma(psi_f, L_d, L_q, I_max)
    i_d0, i_q0 = -I_max * sin(g), I_max * cos(g)
    tau_mtpa = mtpa_torque(n_p, psi_f, L_d, L_q, I_max, g)
    return _envelope_core(n_p, psi_f, R_s, L_d, L_q, I_max, U_max,
                          i_d0, i_q0, tau_mtpa, w_mech)


def envelope_curve_batch(n_p: NDArray, psi_f: NDArray, R_s: NDArray,
                         L_d: NDArray, L_q: NDArray, I_max: NDArray,
                         U_max: NDArray,
                         w_mech: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """Envelope curves for N designs at once: params (N,), speeds (N,S).

    One broadcast pass through the shared core; only the scalar MTPA
    point is computed per design. Returns (tau, i_d, i_q), each (N,S),
    byte-identical to per-design `envelope_curve`.
    """
    n_p = np.asarray(n_p, dtype=float).reshape(-1, 1)
    psi_f = np.asarray(psi_f, dtype=float).reshape(-1, 1)
    R_s = np.asarray(R_s, dtype=float).reshape(-1, 1)
    L_d = np.asarray(L_d, dtype=float).reshape(-1, 1)
    L_q = np.asarray(L_q, dtype=float).reshape(-1, 1)
    I_max = np.asarray(I_max, dtype=float).reshape(-1, 1)
    U_max = np.asarray(U_max, dtype=float).reshape(-1, 1)
    w_mech = np.asarray(w_mech, dtype=float)

    i_d0 = np.empty_like(I_max)
    i_q0 = np.empty_like(I_max)
    tau_mtpa = np.empty_like(I_max)
    for k in range(n_p.shape[0]):
        g = mtpa_gamma(float(psi_f[k, 0]), float(L_d[k, 0]),
                       float(L_q[k, 0]), float(I_max[k, 0]))
        i_d0[k, 0] = -float(I_max[k, 0]) * sin(g)
        i_q0[k, 0] = float(I_max[k, 0]) * cos(g)
        tau_mtpa[k, 0] = mtpa_torque(float(n_p[k, 0]), float(psi_f[k, 0]),
                                     float(L_d[k, 0]), float(L_q[k, 0]),
                                     float(I_max[k, 0]), g)
    return _envelope_core(n_p, psi_f, R_s, L_d, L_q, I_max, U_max,
                          i_d0, i_q0, tau_mtpa, w_mech)


def envelope_at_speed(n_p: int, psi_f: float, R_s: float, L_d: float,
                      L_q: float, I_max: float, U_max: float,
                      w_mech: float) -> tuple[float, float, float]:
    """Max torque (Nm) and its (i_d, i_q) at one mechanical speed (rad/s).

    Single-speed wrapper over `envelope_curve` — one implementation of the
    physics, not two. Batch callers should use `envelope_curve` directly;
    per-speed array overhead dominates this path.
    """
    tau, i_d, i_q = envelope_curve(n_p, psi_f, R_s, L_d, L_q, I_max, U_max,
                                   np.array([w_mech], dtype=float))
    return float(tau[0]), float(i_d[0]), float(i_q[0])


def _interp_psi(interp, i_d, i_q):
    """Shape-aware RGI wrapper: broadcast i_d/i_q, flatten, call, reshape."""
    shape = np.broadcast_shapes(np.shape(i_d), np.shape(i_q))
    flat_d = np.broadcast_to(np.asarray(i_d, dtype=float), shape).ravel()
    flat_q = np.broadcast_to(np.asarray(i_q, dtype=float), shape).ravel()
    return interp(np.column_stack((flat_d, flat_q))).reshape(shape)


def _bisect_iq_map(psi_d_interp, psi_q_interp, R_s, U_max2,
                   i_d, w_e, i_q_top):
    """Voltage-limited i_q ceiling via bisection on the flux map.

    Returns (i_q, feasible). All inputs broadcastable; i_q_top is the
    minimum of the current-circle ceiling and the map's i_q bound.
    """
    shape = np.broadcast_shapes(np.shape(i_d), np.shape(w_e),
                                np.shape(i_q_top))
    i_d_b = np.broadcast_to(np.asarray(i_d, dtype=float), shape)
    w_e_b = np.broadcast_to(np.asarray(w_e, dtype=float), shape)

    def vsq(iq):
        pd = _interp_psi(psi_d_interp, i_d_b, iq)
        pq = _interp_psi(psi_q_interp, i_d_b, iq)
        ud = R_s * i_d_b - w_e_b * pq
        uq = R_s * iq + w_e_b * pd
        return ud * ud + uq * uq

    lo = np.zeros(shape)
    hi = np.broadcast_to(np.asarray(i_q_top, dtype=float), shape).copy()

    g_lo = vsq(lo) - U_max2
    feasible = g_lo <= 0.0
    g_hi = vsq(hi) - U_max2
    current_limited = g_hi <= 0.0

    for _ in range(_BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        g = vsq(mid) - U_max2
        hi = np.where(g > 0, mid, hi)
        lo = np.where(g > 0, lo, mid)

    iq = np.where(current_limited,
                  np.broadcast_to(np.asarray(i_q_top, dtype=float), shape),
                  0.5 * (lo + hi))
    return np.maximum(np.where(feasible, iq, 0.0), 0.0), feasible


def _tau_at_id_map(psi_d_interp, psi_q_interp, n_p, R_s, U_max2,
                   I_max, iq_max_map, i_d, w_e):
    """Torque and i_q at the voltage-limited ceiling on the flux map."""
    i_d_a = np.asarray(i_d, dtype=float)
    i_q_top = np.minimum(
        np.sqrt(np.maximum(0.0, I_max * I_max - i_d_a * i_d_a)),
        iq_max_map,
    )
    iq, feasible = _bisect_iq_map(psi_d_interp, psi_q_interp, R_s, U_max2,
                                  i_d_a, w_e, i_q_top)
    pd = _interp_psi(psi_d_interp, i_d_a, iq)
    pq = _interp_psi(psi_q_interp, i_d_a, iq)
    tau = 1.5 * n_p * (pd * iq - pq * i_d_a)
    tau = np.where(feasible & (iq > 0), tau, 0.0)
    return tau, iq


def _envelope_map_core(psi_d_interp, psi_q_interp, n_p, R_s, I_max,
                       U_max, flux_map: FluxMap, tau_mtpa, i_d0, i_q0,
                       w_mech: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """Grid scan + golden-section on the nonlinear flux-linkage surface."""
    U_max2 = U_max * U_max
    w_e = n_p * w_mech

    pt = np.array([[i_d0, i_q0]])
    psi_d0 = float(psi_d_interp(pt)[0])
    psi_q0 = float(psi_q_interp(pt)[0])
    u_d0 = R_s * i_d0 - w_e * psi_q0
    u_q0 = R_s * i_q0 + w_e * psi_d0
    on_mtpa = (u_d0 * u_d0 + u_q0 * u_q0) <= U_max2 * (1.0 + 1e-12)

    id_lo = max(-I_max, flux_map.i_d_A[0])
    id_hi = flux_map.i_d_A[-1]
    iq_max_map = flux_map.i_q_A[-1]

    i_d_grid_1d = np.linspace(id_lo, id_hi, _ID_GRID_PTS)
    tau_grid, iq_grid = _tau_at_id_map(
        psi_d_interp, psi_q_interp, n_p, R_s, U_max2, I_max, iq_max_map,
        i_d_grid_1d[np.newaxis, :], w_e[:, np.newaxis],
    )

    tau_grid = np.where(np.isnan(tau_grid), -np.inf, tau_grid)
    k_best = np.argmax(tau_grid, axis=-1)
    sel = k_best[:, np.newaxis]
    best_tau = np.take_along_axis(tau_grid, sel, axis=-1)[:, 0]
    best_id = i_d_grid_1d[k_best]
    best_iq = np.take_along_axis(iq_grid, sel, axis=-1)[:, 0]

    no_peak = best_tau <= 0.0
    best_tau = np.where(no_peak, 0.0, best_tau)
    best_id = np.where(no_peak, 0.0, best_id)
    best_iq = np.where(no_peak, 0.0, best_iq)

    step = (id_hi - id_lo) / (_ID_GRID_PTS - 1)
    lo = np.maximum(id_lo, best_id - step)
    hi = np.minimum(id_hi, best_id + step)
    m1 = hi - _GOLDEN * (hi - lo)
    m2 = lo + _GOLDEN * (hi - lo)
    t1, _ = _tau_at_id_map(psi_d_interp, psi_q_interp, n_p, R_s, U_max2,
                           I_max, iq_max_map, m1, w_e)
    t2, _ = _tau_at_id_map(psi_d_interp, psi_q_interp, n_p, R_s, U_max2,
                           I_max, iq_max_map, m2, w_e)
    for _ in range(_GOLDEN_ITERS):
        shrink_lo = t1 < t2
        lo = np.where(shrink_lo, m1, lo)
        hi = np.where(shrink_lo, hi, m2)
        kept = np.where(shrink_lo, m2, m1)
        kept_t = np.where(shrink_lo, t2, t1)
        span = _GOLDEN * (hi - lo)
        probe = np.where(shrink_lo, lo + span, hi - span)
        probe_t, _ = _tau_at_id_map(psi_d_interp, psi_q_interp, n_p, R_s,
                                    U_max2, I_max, iq_max_map, probe, w_e)
        m1 = np.where(shrink_lo, kept, probe)
        m2 = np.where(shrink_lo, probe, kept)
        t1 = np.where(shrink_lo, kept_t, probe_t)
        t2 = np.where(shrink_lo, probe_t, kept_t)

    id_ref = 0.5 * (lo + hi)
    tau_ref, iq_ref = _tau_at_id_map(psi_d_interp, psi_q_interp, n_p, R_s,
                                     U_max2, I_max, iq_max_map, id_ref, w_e)
    keep_ref = tau_ref >= best_tau
    tau = np.where(keep_ref, tau_ref, best_tau)
    i_d_out = np.where(keep_ref, id_ref, best_id)
    i_q_out = np.where(keep_ref, iq_ref, best_iq)

    dead = no_peak
    tau = np.where(on_mtpa, tau_mtpa, np.where(dead, 0.0, tau))
    i_d_out = np.where(on_mtpa, i_d0, np.where(dead, 0.0, i_d_out))
    i_q_out = np.where(on_mtpa, i_q0, np.where(dead, 0.0, i_q_out))
    return tau, i_d_out, i_q_out


def envelope_curve_map(
        n_p: int, psi_f: float, R_s: float, L_d: float, L_q: float,
        I_max: float, U_max: float, flux_map: FluxMap,
        w_mech: NDArray) -> tuple[NDArray, NDArray, NDArray, str]:
    """Envelope on the nonlinear flux-linkage surface.

    Falls back to the analytical path when I_max exceeds the map grid.
    Returns (tau, i_d, i_q, envelope_source).
    """
    w_mech = np.asarray(w_mech, dtype=float)
    if (I_max > abs(flux_map.i_d_A[0])
            or I_max > flux_map.i_q_A[-1]
            or flux_map.i_q_A[0] > 0):
        tau, i_d, i_q = envelope_curve(n_p, psi_f, R_s, L_d, L_q,
                                       I_max, U_max, w_mech)
        return tau, i_d, i_q, "analytical"

    psi_d_interp, psi_q_interp = make_flux_map_interpolators(flux_map)
    mtpa = mtpa_numeric(n_p, flux_map, I_max)
    if mtpa is None:
        tau, i_d, i_q = envelope_curve(n_p, psi_f, R_s, L_d, L_q,
                                       I_max, U_max, w_mech)
        return tau, i_d, i_q, "analytical"
    tau_mtpa, i_d0, i_q0 = mtpa
    tau, i_d, i_q = _envelope_map_core(psi_d_interp, psi_q_interp, n_p,
                                       R_s, I_max, U_max, flux_map,
                                       tau_mtpa, i_d0, i_q0, w_mech)
    return tau, i_d, i_q, "flux_map"


def _base_speed_from_psi(n_p: int, R_s: float, I_max: float, U_max: float,
                         i_d: float, i_q: float,
                         psi_d: float, psi_q: float) -> float:
    """Base speed from d-q flux linkages at the MTPA operating point.

    |u|^2 at fixed MTPA currents is quadratic in w_e; the positive root
    is exact (R_s included). Same physics as base_speed(), parameterized
    on (psi_d, psi_q) instead of (psi_f, L_d, L_q).
    """
    A = psi_d * psi_d + psi_q * psi_q
    B = 2.0 * R_s * (i_q * psi_d - i_d * psi_q)
    C = R_s * R_s * I_max * I_max - U_max * U_max
    if C >= 0.0:
        return 0.0
    if A == 0.0:
        return float("inf")
    w_e = (-B + sqrt(B * B - 4.0 * A * C)) / (2.0 * A)
    return w_e / n_p


def base_speed(n_p: int, psi_f: float, R_s: float, L_d: float, L_q: float,
               I_max: float, U_max: float) -> float:
    """Highest mechanical speed (rad/s) where MTPA at I_max is voltage-feasible.

    |u|^2 at the fixed MTPA currents is quadratic in w_e; the positive
    root is exact (R_s included). Returns 0.0 when the resistive drop
    alone exceeds U_max (guarded against in TorqueSpeedParams).
    """
    g = mtpa_gamma(psi_f, L_d, L_q, I_max)
    i_d, i_q = -I_max * sin(g), I_max * cos(g)
    return _base_speed_from_psi(n_p, R_s, I_max, U_max, i_d, i_q,
                                psi_f + L_d * i_d, L_q * i_q)


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


def _max_speed_from_map(n_p: int, R_s: float, I_max: float, U_max: float,
                        flux_map: FluxMap, psi_d_interp) -> float | None:
    """Max speed using flux-map ψ_d(-I_max, 0).

    None when ψ_d(-I_max, 0) <= 0 (voltage-ellipse centre inside current
    circle). Requires i_q_A[0] == 0 and -I_max >= i_d_A[0].
    """
    if flux_map.i_q_A[0] > 0:
        return None
    i_d_eval = max(-I_max, flux_map.i_d_A[0])
    pt = np.array([[i_d_eval, 0.0]])
    psi_d_neg = float(psi_d_interp(pt)[0])
    if psi_d_neg <= 0:
        return None
    num = U_max * U_max - (R_s * I_max) ** 2
    if num <= 0.0:
        return 0.0
    w_e = sqrt(num) / psi_d_neg
    return w_e / n_p


def _envelope_scalars(n_p: int, psi_f: float, R_s: float, L_d: float,
                      L_q: float, I_max: float,
                      U_max: float) -> tuple[float, float | None]:
    return (
        base_speed(n_p, psi_f, R_s, L_d, L_q, I_max, U_max),
        max_speed(n_p, psi_f, R_s, L_d, L_q, I_max, U_max),
    )


def _speed_grid(p: TorqueSpeedParams) -> tuple[
        list[float], NDArray, tuple[float, float | None],
        tuple[float, float | None] | None]:
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
    return speeds, np.array(speeds, dtype=float), (w_base_pk, w_max_pk), \
        scalars_cont


Curve = tuple[list[float], list[float], list[float]]  # (tau, i_d, i_q)


def _as_curve(env: Sequence[Any]) -> Curve:
    """First three entries of an envelope_curve* return, as lists."""
    return env[0].tolist(), env[1].tolist(), env[2].tolist()


def _build_result(speeds: list[float], curve_pk: Curve,
                  w_base_pk: float, w_max_pk: float | None, I_peak: float,
                  U_max: float, curve_cont: Curve | None,
                  scalars_cont: tuple[float, float | None] | None,
                  I_cont: float | None) -> dict[str, Any]:
    tau_pk, i_d_pk, i_q_pk = curve_pk
    result: dict[str, Any] = {
        "speed_curve": speeds,
        "tau_curve_peak": tau_pk,
        "i_d_curve_peak": i_d_pk,
        "i_q_curve_peak": i_q_pk,
        "base_speed_peak": w_base_pk,
        "max_speed_peak": w_max_pk,
        "p_max_peak": max(t * w for t, w in zip(tau_pk, speeds)),
        "u_max": U_max,
        "I_env_peak": I_peak,
    }
    if curve_cont is not None and scalars_cont is not None:
        tau_cont, i_d_cont, i_q_cont = curve_cont
        result["tau_curve_cont"] = tau_cont
        result["i_d_curve_cont"] = i_d_cont
        result["i_q_curve_cont"] = i_q_cont
        result["base_speed_cont"] = scalars_cont[0]
        result["max_speed_cont"] = scalars_cont[1]
        result["p_max_cont"] = max(t * w for t, w in zip(tau_cont, speeds))
        result["I_env_cont"] = I_cont
    return result


def _run_map_envelope(motor: Motor, p: TorqueSpeedParams,
                      fm: FluxMap) -> dict[str, Any] | None:
    """Map-aware torque-speed envelope. Returns None to fall back."""
    if (p.I_peak > abs(fm.i_d_A[0])
            or p.I_peak > fm.i_q_A[-1]
            or fm.i_q_A[0] > 0):
        return None
    if p.I_cont is not None and (
            p.I_cont > abs(fm.i_d_A[0])
            or p.I_cont > fm.i_q_A[-1]):
        return None

    psi_d_interp, psi_q_interp = make_flux_map_interpolators(fm)

    def scalars_at(I_s):
        mtpa = mtpa_numeric(p.n_p, fm, I_s)
        if mtpa is None:
            return None
        _, id0, iq0 = mtpa
        pt = np.array([[id0, iq0]])
        pd0 = float(psi_d_interp(pt)[0])
        pq0 = float(psi_q_interp(pt)[0])
        wb = _base_speed_from_psi(p.n_p, p.R_s, I_s, p.U_max,
                                  id0, iq0, pd0, pq0)
        wm = _max_speed_from_map(p.n_p, p.R_s, I_s, p.U_max,
                                 fm, psi_d_interp)
        return wb, wm

    sp = scalars_at(p.I_peak)
    if sp is None:
        return None
    w_base_pk, w_max_pk = sp

    scalars_cont: tuple[float, float | None] | None = None
    if p.I_cont is not None:
        scalars_cont = scalars_at(p.I_cont)
        if scalars_cont is None:
            return None

    ends: list[float | None] = [w_max_pk]
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
    speed_arr = np.array(speeds, dtype=float)

    curve_pk = _as_curve(envelope_curve_map(
        p.n_p, p.psi_f, p.R_s, p.L_d, p.L_q, p.I_peak, p.U_max, fm, speed_arr))
    curve_cont = None
    if p.I_cont is not None:
        curve_cont = _as_curve(envelope_curve_map(
            p.n_p, p.psi_f, p.R_s, p.L_d, p.L_q, p.I_cont, p.U_max, fm,
            speed_arr))

    result = _build_result(speeds, curve_pk, w_base_pk, w_max_pk, p.I_peak,
                           p.U_max, curve_cont, scalars_cont, p.I_cont)
    result["envelope_source"] = "flux_map"
    return result


def run_torque_speed(config: RunConfig) -> dict[str, Any]:
    """Torque-speed envelope at the drive current limit (peak) and, when
    I_rated is set, at the continuous rating (cont). Speeds are mechanical
    rad/s; one shared speed grid spans the widest envelope."""

    p = prepare_torque_speed(config.motor)

    fm = derated_flux_map(config.motor)
    if fm is not None:
        map_result = _run_map_envelope(config.motor, p, fm)
        if map_result is not None:
            return map_result

    args = (p.n_p, p.psi_f, p.R_s, p.L_d, p.L_q)

    speeds, speed_arr, (w_base_pk, w_max_pk), scalars_cont = _speed_grid(p)
    curve_pk = _as_curve(envelope_curve(*args, p.I_peak, p.U_max, speed_arr))
    curve_cont = None
    if p.I_cont is not None:
        curve_cont = _as_curve(envelope_curve(*args, p.I_cont, p.U_max,
                                              speed_arr))

    result = _build_result(speeds, curve_pk, w_base_pk, w_max_pk, p.I_peak,
                           p.U_max, curve_cont, scalars_cont, p.I_cont)
    result["envelope_source"] = "analytical"
    return result


def run_torque_speed_batch(
        configs: Sequence[RunConfig]) -> list[dict[str, Any] | Exception]:
    """Batched torque-speed envelope: one broadcast kernel pass over all
    valid designs. Same order as input; per-design failures land in their
    slot as the raised Exception. Outputs are byte-identical to
    `run_torque_speed` per design.

    Map-bearing motors are routed through the serial path (FluxMap
    interpolators are per-motor, not vectorizable across designs).
    """

    results: list[Any] = [None] * len(configs)
    prepared = []
    for k, config in enumerate(configs):
        try:
            p = prepare_torque_speed(config.motor)
        except Exception as exc:
            results[k] = exc
            continue
        if config.motor.flux_map is not None:
            try:
                results[k] = run_torque_speed(config)
            except Exception as exc:
                results[k] = exc
            continue
        prepared.append((k, p, *_speed_grid(p)))

    if not prepared:
        return results

    params = [e[1] for e in prepared]
    w = np.stack([e[3] for e in prepared])

    def stack(get):
        return np.array([get(p) for p in params], dtype=float)

    n_p = stack(lambda p: p.n_p)
    psi_f = stack(lambda p: p.psi_f)
    R_s = stack(lambda p: p.R_s)
    L_d = stack(lambda p: p.L_d)
    L_q = stack(lambda p: p.L_q)
    I_peak = stack(lambda p: p.I_peak)
    U_max = stack(lambda p: p.U_max)
    pk_all = envelope_curve_batch(n_p, psi_f, R_s, L_d, L_q, I_peak,
                                  U_max, w)

    cont_idx = [j for j, p in enumerate(params) if p.I_cont is not None]
    cont_all = None
    if cont_idx:
        I_cont = np.array([params[j].I_cont for j in cont_idx], dtype=float)
        cont_all = envelope_curve_batch(
            n_p[cont_idx], psi_f[cont_idx], R_s[cont_idx], L_d[cont_idx],
            L_q[cont_idx], I_cont, U_max[cont_idx], w[cont_idx])
    cont_pos = {j: i for i, j in enumerate(cont_idx)}

    for j, (k, p, speeds, _, scalars_pk, scalars_cont) in enumerate(prepared):
        curve_pk = _as_curve(tuple(a[j] for a in pk_all))
        curve_cont = None
        if p.I_cont is not None and cont_all is not None:
            curve_cont = _as_curve(tuple(a[cont_pos[j]] for a in cont_all))
        result = _build_result(speeds, curve_pk, scalars_pk[0],
                               scalars_pk[1], p.I_peak, p.U_max,
                               curve_cont, scalars_cont, p.I_cont)
        result["envelope_source"] = "analytical"
        results[k] = result
    return results


@dataclass(frozen=True)
class TorqueSpeedParams:
    """Validated inputs for the torque-speed envelope computation.

    psi_f — PM flux linkage (Wb, peak); R_s — stator resistance (Ω, >= 0);
    L_d / L_q — inductances (H), both required (the voltage limit above
    base speed is inductance-dominated; omitting them would silently give
    a wildly wrong field-weakening range).
    I_peak — drive current limit, I_LIMIT or MAX_I_S (A, peak);
    I_cont — continuous rating I_rated (A, peak), None to skip the
    continuous envelope; U_max — phase peak voltage limit U_DC/sqrt(3).
    """

    n_p: int
    psi_f: float
    R_s: float
    L_d: float
    L_q: float
    I_peak: float
    U_max: float
    I_cont: float | None = None

    @property
    def is_reluctance(self) -> bool:
        return self.psi_f == 0.0

    def __post_init__(self) -> None:
        if self.n_p < 2:
            raise ValueError(f"n_p must be >= 2, got {self.n_p}")
        if self.psi_f < 0:
            raise ValueError(f"psi_f must be >= 0, got {self.psi_f}")
        if self.R_s < 0:
            raise ValueError(f"R_s must be >= 0, got {self.R_s}")
        if self.L_d <= 0:
            raise ValueError(f"L_d must be > 0, got {self.L_d}")
        if self.L_q <= 0:
            raise ValueError(f"L_q must be > 0, got {self.L_q}")
        if self.is_reluctance and self.L_d == self.L_q:
            raise ValueError(
                "psi_f = 0 with no saliency (L_d == L_q): "
                "machine cannot produce torque"
            )
        if self.I_peak <= 0:
            raise ValueError(f"I_peak must be > 0, got {self.I_peak}")
        if self.I_cont is not None and self.I_cont <= 0:
            raise ValueError(f"I_cont must be > 0, got {self.I_cont}")
        # Non-finite inputs: every check above is an inequality, and NaN
        # compares False against all of them, so a NaN sails past the lot.
        # The envelope optimizer then has no way to signal it — it would
        # return a plausible-looking 0 Nm envelope that persists to the
        # result store as if it were a real answer.
        non_finite = [
            name for name, v in (
                ("psi_f", self.psi_f), ("R_s", self.R_s), ("L_d", self.L_d),
                ("L_q", self.L_q), ("I_peak", self.I_peak),
                ("U_max", self.U_max), ("I_cont", self.I_cont),
            ) if v is not None and not isfinite(v)
        ]
        if non_finite:
            raise ValueError(
                f"torque_speed got non-finite {', '.join(non_finite)} — "
                f"NaN/inf inputs cannot produce a meaningful envelope"
            )
        if self.U_max <= self.R_s * self.I_peak:
            raise ValueError(
                f"U_max={self.U_max:.4g} V does not exceed the resistive "
                f"drop R_s*I_peak={self.R_s * self.I_peak:.4g} V — "
                f"drive voltage and motor are mismatched"
            )


def prepare_torque_speed(motor: Motor) -> TorqueSpeedParams:
    """Derive, validate, and extract torque-speed envelope params from a Motor.

    I_peak = I_LIMIT (or MAX_I_S when I_LIMIT is not set) — same convention
    as prepare_stall_torque. I_cont = I_rated when set. U_max = U_DC/sqrt(3)
    (SVPWM linear region, phase peak). psi_f and R_s are at the motor's
    magnet_temp / winding_temp_limit (solver_params.derate).
    """
    cold = cold_torque_speed_params(motor)
    R_s_op, psi_f_op = derate_motor(motor, cold.R_s, cold.psi_f)
    assert R_s_op is not None
    return replace(cold, psi_f=psi_f_op, R_s=R_s_op)


def cold_torque_speed_params(motor: Motor) -> TorqueSpeedParams:
    """prepare_torque_speed without the temperature derating (20 °C
    datasheet psi_f and R_s) — for callers that derate at solve time
    (operating_point)."""
    from math import sqrt

    reject_induction(motor, "torque_speed")
    if motor.psi_f is None and motor.flux_map is not None:
        from phasesweep.models.flux_map import cold_psi_f
        psi_f = cold_psi_f(motor.flux_map, motor.alpha_Br)
    else:
        psi_f = resolve_psi_f(motor)

    R_s, L_d, L_q = motor.R_s, motor.L_d, motor.L_q
    if R_s is None or L_d is None or L_q is None:
        fields = (("R_s", R_s), ("L_d", L_d), ("L_q", L_q))
        missing = [f for f, v in fields if v is None]
        raise ValueError(
            f"Motor '{motor.name}': torque_speed needs {', '.join(missing)}"
        )

    drive = motor.drive
    if drive.I_LIMIT is None and drive.MAX_I_S is None:
        raise ValueError(
            f"Motor '{motor.name}': torque_speed needs [drive] "
            f"MAX_I_S (or I_LIMIT)"
        )
    I_peak = drive.I_LIMIT if drive.I_LIMIT is not None else drive.MAX_I_S
    assert I_peak is not None  # narrowed above
    if drive.U_DC is None:
        raise ValueError(
            f"Motor '{motor.name}': torque_speed needs [drive] U_DC"
        )

    return TorqueSpeedParams(
        n_p=motor.n_p,
        psi_f=psi_f,
        R_s=R_s,
        L_d=L_d,
        L_q=L_q,
        I_peak=I_peak,
        U_max=drive.U_DC / sqrt(3),
        I_cont=motor.I_rated,
    )
