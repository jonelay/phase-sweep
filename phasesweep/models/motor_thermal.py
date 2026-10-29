"""Motor thermal network: lumped nodes coupled to the operating point.

Three nodes — winding W, iron Fe, magnet/rotor M — and a coolant sink at
`RunConfig.T_coolant`:

    C dT/dt = -G T + P + g_s T_s
    G = [[G_WFe,  -G_WFe,              0    ],
         [-G_WFe,  G_WFe+G_FeS+G_FeM, -G_FeM],
         [0,      -G_FeM,              G_FeM]]     g_s = [0, G_FeS, 0]

Sources P = [p_cu, p_fe, 0]: the magnet node has no loss model and heats
only through the iron. Steady state in closed form:
T_Fe = T_s + (p_cu + p_fe)/G_FeS, T_W = T_Fe + p_cu/G_WFe, T_M = T_Fe —
G_FeM and the capacitances are invisible at steady state; the magnet
rides the iron temperature.

The pure network functions are n-node generic (a 1x1 network is
thermal_duty's first-order model). Per segment of constant loss
T(t) = T_ss + expm(-C^-1 G t)(T0 - T_ss); the periodic start is solved
per eigenmode of C^-1 G (each mode is the scalar first-order cycle map
of thermal_duty.transient_peak, formed with expm1 so slow modes land on
their time-averaged forcing). Node traces are sums of decaying
exponentials and can peak inside a segment, so the peak is sampled at a
step no larger than a quarter of the fastest time constant.

The registered model `motor_thermal` is the coupled steady solve: the
operating point (copper loss, stator flux linkage) at the current node
temperatures, iron loss scaled by |psi_s|/psi_ref at the electrical
frequency, the steady network, relaxed to a fixed point. It raises
rather than caching a state that is not a fixed point, with a typed
outcome so a caller does not parse messages:

  InfeasibleDemand — the demand leaves the envelope at the iterate's
                     temperatures (cold at iteration 1, or as the magnet
                     heats);
  ThermalRunaway   — the residual grew over the last _RUNAWAY_K iterates
                     with every node heating, or an iterate (including a
                     converged or budget-final proposal) left the linear
                     derating range. A heuristic: where losses rise with
                     temperature the relaxed iteration from coolant
                     approaches the lowest fixed point from below, but
                     iron loss follows the PM flux and falls as the magnet
                     heats on either branch, so no branch guarantees it;
  NoConvergence    — the budget ran out without that signature.

The types are diagnostic, not separable physics. Only an InfeasibleDemand
with at_coolant (iteration 1) says the demand is infeasible at coolant
temperature. Any failure after the first iteration means no fixed point
was found on the climb, whatever its type, and the type can depend on
relax and max_iter (a slow runaway reports NoConvergence on a short
budget). A coolant temperature outside the derating range, a
non-finite coolant, max_iter < 1, tol_K not finite and > 0 or relax
outside (0, 1] is an input error: plain ValueError.

Over a limit is a result, not an error: each node limit (TempLimit, from
the motor or a per-run override) yields headroom_<node>_K = limit - T,
binding_limit names the node with the least headroom, and within_s1 is
True when no headroom is negative (None with no limits). A limit on a
different measurement basis from the node it bounds is rejected.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from math import ceil, hypot, isfinite, pi, sqrt
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.linalg import expm

from phasesweep.models.flux_map import cold_psi_f, map_covers
from phasesweep.models.iron_loss import _IRON_FIELDS, iron_loss_at
from phasesweep.models.operating_point import (
    OperatingPointParams,
    cold_operating_point_params,
    operating_point,
)
from phasesweep.solver_params import derate, reject_induction

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from phasesweep.machines.motor import Motor, TempLimit, ThermalNetwork
    from phasesweep.sweep_types import RunConfig

__all__ = [
    "LIMIT_NODES",
    "CoupledSolveError",
    "InfeasibleDemand",
    "MotorThermalParams",
    "NoConvergence",
    "ThermalRunaway",
    "assemble_network",
    "coupled_steady",
    "periodic_peak",
    "periodic_start",
    "prepare_motor_thermal",
    "run_motor_thermal",
    "segment_map",
    "steady_temps",
]

NODES = ("winding", "iron", "magnet")
_RELAX = 0.5
_TOL_K = 0.01
_MAX_ITER = 50
_SAMPLES_PER_TAU = 4
_RUNAWAY_K = 3
LIMIT_NODES = ("winding", "magnet")


# ---------------------------------------------------------------------------
# typed outcomes
# ---------------------------------------------------------------------------

class CoupledSolveError(ValueError):
    """No fixed point was reported. ValueError subclass: callers
    that catch ValueError keep working. T is the iterate (winding, iron,
    magnet degC) at failure, not a steady state."""

    def __init__(self, msg: str, *, iteration: int, T: Sequence[float],
                 residual_K: float | None = None) -> None:
        super().__init__(msg)
        self.iteration = iteration
        self.T = tuple(float(x) for x in T)
        self.residual_K = residual_K

    @property
    def at_coolant(self) -> bool:
        """Failed at iteration 1, every node at coolant temperature. Only
        an InfeasibleDemand with at_coolant is a statement about the
        demand itself; a later failure of any type means no fixed point
        was found on the climb."""
        return self.iteration == 1

    def __reduce__(self) -> tuple[Any, ...]:
        # Keyword-only attributes: the default Exception pickle passes args
        # only. The state dict keeps notes and any added attributes.
        return (_rebuild_error, (type(self), str(self), self.iteration, self.T,
                                 self.residual_K), self.__dict__)


def _rebuild_error(cls: type[CoupledSolveError], msg: str, iteration: int,
                   T: Sequence[float], residual_K: float | None) -> CoupledSolveError:
    return cls(msg, iteration=iteration, T=T, residual_K=residual_K)


class InfeasibleDemand(CoupledSolveError):
    """The demand is outside the envelope at the iterate's temperatures.
    at_coolant: infeasible at coolant temperature."""


class NoConvergence(CoupledSolveError):
    """Iteration budget exhausted without the runaway signature."""


class ThermalRunaway(CoupledSolveError):
    """Residual grew over the last _RUNAWAY_K iterates, every node heating."""


# ---------------------------------------------------------------------------
# pure network functions (n-node generic)
# ---------------------------------------------------------------------------

def assemble_network(tn: ThermalNetwork) -> tuple[NDArray, NDArray, NDArray]:
    """(G, g_s, C) of the three-node W/Fe/M network, node order NODES."""
    G = np.array([
        [tn.G_WFe, -tn.G_WFe, 0.0],
        [-tn.G_WFe, tn.G_WFe + tn.G_FeS + tn.G_FeM, -tn.G_FeM],
        [0.0, -tn.G_FeM, tn.G_FeM],
    ])
    g_s = np.array([0.0, tn.G_FeS, 0.0])
    C = np.array([tn.C_W, tn.C_Fe, tn.C_M])
    return G, g_s, C


def steady_temps(G: NDArray, g_s: NDArray, P: NDArray, T_s: float) -> NDArray:
    """Steady node temperatures: G T = P + g_s T_s."""
    return np.linalg.solve(G, np.asarray(P, dtype=float) + g_s * T_s)


def segment_map(G: NDArray, g_s: NDArray, C: NDArray, P: NDArray, T_s: float,
                dt: float) -> tuple[NDArray, NDArray]:
    """Affine map (M, v) with T(dt) = M T0 + v for constant loss P over dt."""
    if dt <= 0:
        raise ValueError(f"segment duration must be > 0, got {dt}")
    A = G / np.asarray(C, dtype=float)[:, None]
    M = expm(-A * dt)
    T_ss = steady_temps(G, g_s, P, T_s)
    # (I - M) T_ss, not T_ss - M T_ss: the same rounded (I - M) then
    # appears in the fixed point's numerator and denominator.
    return M, (np.eye(len(T_ss)) - M) @ T_ss


def _modes(G: NDArray, C: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """Eigen-decomposition of C^-1 G through the symmetric similarity
    C^-1/2 G C^-1/2: rates lam (1/s), and V, V_inv with T = V y."""
    c = np.asarray(C, dtype=float)
    s = 1.0 / np.sqrt(c)
    S = s[:, None] * G * s[None, :]
    if not np.allclose(S, S.T, rtol=1e-12, atol=0.0):
        raise ValueError("G must be symmetric (reciprocal conductances)")
    lam, U = np.linalg.eigh(S)
    return lam, s[:, None] * U, U.T * np.sqrt(c)[None, :]


def periodic_start(G: NDArray, g_s: NDArray, C: NDArray, T_s: float,
                   segments: Sequence[tuple[NDArray, float]]) -> NDArray:
    """Node temperatures at the start of the steady-periodic cycle.

    segments — ((P vector W, duration s), ...). Solved mode by mode: in
    the eigenbasis of C^-1 G each mode is thermal_duty.transient_peak's
    scalar first-order system, y' = -lam y + q, whose cycle map is
    y(T) = a y0 + b with a = exp(-lam T); the periodic start is
    b / (1 - a). Both b and 1 - a are built from -expm1(-lam dt), so a
    mode far slower than the cycle (a rounding to 1) lands on its
    time-averaged forcing q_avg / lam without a separate singular branch,
    and fast modes keep their resolved cycle whatever the slow ones do.
    """
    if not segments:
        raise ValueError("segments must have at least one entry")
    lam, V, V_inv = _modes(G, C)
    c = np.asarray(C, dtype=float)
    a_tot = np.ones_like(lam)
    b_tot = np.zeros_like(lam)
    q_avg = np.zeros_like(lam)
    total = 0.0
    for P, dt in segments:
        if dt <= 0:
            raise ValueError(f"segment duration must be > 0, got {dt}")
        q = V_inv @ ((np.asarray(P, dtype=float) + g_s * T_s) / c)
        rise = -np.expm1(-lam * dt)
        b_tot = (1.0 - rise) * b_tot + (q / lam) * rise
        a_tot *= 1.0 - rise
        q_avg += q * dt
        total += dt
    one_minus_a = -np.expm1(-lam * total)
    y0 = np.where(one_minus_a > 0.0, b_tot / np.where(one_minus_a > 0.0, one_minus_a, 1.0),
                  q_avg / (lam * total))
    return V @ y0


def periodic_peak(G: NDArray, g_s: NDArray, C: NDArray, T_s: float,
                  segments: Sequence[tuple[NDArray, float]]) -> tuple[NDArray, NDArray]:
    """(per-node peak temperature over the steady-periodic cycle, cycle
    start). Each segment is sampled at a step no larger than a quarter of
    the fastest time constant 1/max|eig(C^-1 G)|, so a peak inside a
    segment (a sum of exponentials is not monotone) is resolved."""
    T0 = periodic_start(G, g_s, C, T_s, segments)
    A = G / np.asarray(C, dtype=float)[:, None]
    tau_min = 1.0 / float(np.max(np.abs(np.linalg.eigvals(A))))
    T, peak = T0.copy(), T0.copy()
    for P, dt in segments:
        n = max(1, ceil(dt / (tau_min / _SAMPLES_PER_TAU)))
        M, v = segment_map(G, g_s, C, P, T_s, dt / n)
        for _ in range(n):
            T = M @ T + v
            peak = np.maximum(peak, T)
    return peak, T0


# ---------------------------------------------------------------------------
# coupled steady solve
# ---------------------------------------------------------------------------

def _check_nodes(nodes: Iterable[str]) -> None:
    for node in nodes:
        if node not in LIMIT_NODES:
            raise ValueError(
                f"temperature limit on unknown node {node!r}; nodes: {LIMIT_NODES}")


def _check_limits(limits: Mapping[str, TempLimit], tn: ThermalNetwork) -> None:
    """Known nodes only; a limit's basis must match its node's basis when
    both are declared (a hot-spot class against an ETD-fitted node would
    need an offset this model does not carry)."""
    _check_nodes(limits)
    for node, lim in limits.items():
        node_basis = getattr(tn, f"{node}_basis")
        if lim.basis is not None and node_basis is not None and lim.basis != node_basis:
            raise ValueError(
                f"{node} limit {lim.value_C} degC is on basis {lim.basis!r} but "
                f"the network's {node} node is {node_basis!r}; declare a limit "
                f"on the node's basis")


def _runaway(resid: Sequence[float], heating: Sequence[bool]) -> bool:
    k = _RUNAWAY_K
    if len(resid) < k:
        return False
    r = resid[-k:]
    return all(b > a for a, b in itertools.pairwise(r)) and all(heating[-k:])


@dataclass(frozen=True)
class MotorThermalParams:
    """Validated inputs for the coupled motor thermal solve.

    op — cold operating-point inputs (temperatures applied at solve time);
    k_h, k_e, alpha_fe, m_core, B_core — the [iron] lump; psi_ref — stator
    flux linkage (Wb, peak) at which the lump cycles at B_core, at the
    20 °C magnet reference on both paths: the map's psi_d(0, 0) referred
    to 20 °C (flux_map.cold_psi_f) when the solve runs on the map, else
    psi_f;
    network — the three-node conductances and capacitances;
    limits — node ("winding", "magnet") -> TempLimit, from the motor's
    winding_temp_limit / magnet_temp_max; checked against the network's
    node bases.
    """

    op: OperatingPointParams
    k_h: float
    k_e: float
    alpha_fe: float
    m_core: float
    B_core: float
    psi_ref: float
    network: ThermalNetwork
    limits: Mapping[str, TempLimit] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for f in ("k_h", "k_e", "alpha_fe", "m_core", "B_core", "psi_ref"):
            v = getattr(self, f)
            if not isfinite(v) or v <= 0:
                raise ValueError(f"{f} must be > 0 and finite, got {v}")
        _check_limits(self.limits, self.network)
        if self.op.alpha_Br is None:
            raise ValueError(
                "alpha_Br is required: the magnet node temperature derates "
                "the PM flux at every iteration")


def prepare_motor_thermal(motor: Motor) -> MotorThermalParams:
    """Derive, validate, and extract the coupled-solve params from a Motor."""
    reject_induction(motor, "motor_thermal")
    if motor.thermal_network is None:
        raise ValueError(
            f"Motor '{motor.name}': motor_thermal needs a [thermal_network] "
            f"section (G_WFe, G_FeS, G_FeM, C_W, C_Fe, C_M)")
    k_h, k_e, alpha_fe = motor.k_h, motor.k_e, motor.alpha_fe
    m_core, B_core = motor.m_core, motor.B_core
    if k_h is None or k_e is None or alpha_fe is None or m_core is None or B_core is None:
        missing = [f for f in _IRON_FIELDS if getattr(motor, f) is None]
        raise ValueError(
            f"Motor '{motor.name}': motor_thermal needs {', '.join(missing)} "
            f"([iron] TOML section)")
    # The motor's winding_temp_limit / magnet_temp are not operating
    # temperatures here (the solve sets both nodes), so they are not
    # range-checked as such; the limit reaches the headroom via temp_limits.
    op = cold_operating_point_params(motor, None, None)
    fm = op.flux_map
    if fm is not None and map_covers(fm, op.I_max):
        if not fm.has_origin:
            raise ValueError(
                f"Motor '{motor.name}': motor_thermal solves on the flux map "
                f"and needs its origin node (i_d = 0, i_q = 0) for the "
                f"magnet-temperature shift and psi_ref; i_d_A ends at "
                f"{fm.i_d_A[-1]}, i_q_A starts at {fm.i_q_A[0]}")
        psi_ref = cold_psi_f(fm, op.alpha_Br)
    else:
        psi_ref = op.psi_f
    try:
        return MotorThermalParams(
            op=op, k_h=k_h, k_e=k_e, alpha_fe=alpha_fe, m_core=m_core,
            B_core=B_core, psi_ref=psi_ref, network=motor.thermal_network,
            limits=motor.temp_limits,
        )
    except ValueError as e:
        raise ValueError(f"Motor '{motor.name}': {e}") from None


def coupled_steady(p: MotorThermalParams, w_mech: float, tau_ref: float,
                   T_coolant: float, *, U_DC: float | None = None,
                   limits: Mapping[str, TempLimit | None] | None = None,
                   relax: float = _RELAX, tol_K: float = _TOL_K,
                   max_iter: int = _MAX_ITER) -> dict[str, Any]:
    """Fixed point of operating point -> losses -> steady network -> temperatures.

    Starts every node at T_coolant. Each iteration solves the operating
    point with R_s at the winding node and the PM flux (map shift) at the
    magnet node, scales the iron lump by |psi_s|/psi_ref, and relaxes the
    node temperatures toward the network's steady response. Converged
    when the unrelaxed update moves no node by more than tol_K; the
    reported temperatures are the network's response to the reported
    losses, themselves checked in range and feasible. Raises a CoupledSolveError subclass (never a cached partial
    state): InfeasibleDemand, ThermalRunaway or NoConvergence (module
    docstring).

    U_DC overrides the drive bus for this solve (U_max follows as
    U_DC/sqrt(3)), so a caller coupling an inverter can pass the effective
    bus without forking the fixed point. None keeps the motor's bus and
    reproduces the registered model's outputs exactly.

    limits overrides p.limits per node for this solve (a scenario limit,
    e.g. an assumed i3 winding limit); a None value drops that node's
    limit. Limits change only the headroom outputs, never the solve.
    """
    op = p.op
    if U_DC is not None and not (isfinite(U_DC) and U_DC > 0):
        raise ValueError(f"motor_thermal: U_DC override must be finite and > 0, got {U_DC}")
    if not isfinite(T_coolant):
        raise ValueError(f"motor_thermal: T_coolant must be finite, got {T_coolant}")
    if max_iter < 1:
        raise ValueError(f"motor_thermal: max_iter must be >= 1, got {max_iter}")
    if not (isfinite(tol_K) and tol_K > 0):
        raise ValueError(f"motor_thermal: tol_K must be finite and > 0, got {tol_K}")
    if not (0 < relax <= 1):
        raise ValueError(f"motor_thermal: relax must be in (0, 1], got {relax}")
    # Input errors, not outcomes: the coolant itself outside the derating range.
    try:
        derate(op.R_s, op.psi_f, float(T_coolant), float(T_coolant), op.alpha_Br)
    except ValueError as e:
        raise ValueError(
            f"motor_thermal: T_coolant {T_coolant} degC is outside the "
            f"derating range: {e}") from e
    lims = dict(p.limits)
    _check_nodes(limits or {})
    for node, lim in (limits or {}).items():
        if lim is None:
            lims.pop(node, None)
        else:
            lims[node] = lim
    _check_limits(lims, p.network)
    U_DC_run = op.U_DC if U_DC is None else float(U_DC)
    U_max_run = op.U_max if U_DC is None else U_DC_run / sqrt(3.0)
    G, g_s, _ = assemble_network(p.network)
    T = np.full(3, float(T_coolant))
    resids: list[float] = []
    heating: list[bool] = []

    def check_range(T_it: NDArray, it: int, residual_K: float | None) -> None:
        # An iterate, a converged state or a budget-final proposal outside
        # the linear derating range: no fixed point inside it (heuristic).
        try:
            derate(op.R_s, op.psi_f, float(T_it[0]), float(T_it[2]), op.alpha_Br)
        except ValueError as e:
            raise ThermalRunaway(
                f"motor_thermal: thermal runaway at {tau_ref:.4g} N.m, "
                f"{w_mech:.4g} rad/s: iterate T_winding {T_it[0]:.1f}, T_magnet "
                f"{T_it[2]:.1f} degC (iteration {it}) left the derating range: {e}",
                iteration=it, T=T_it.tolist(), residual_K=residual_K) from e

    def solve_op(T_it: NDArray) -> Any:
        return operating_point(
            op.n_p, op.psi_f, op.R_s, op.L_d, op.L_q, op.I_max, U_max_run,
            U_DC_run, w_mech, tau_ref, flux_map=op.flux_map,
            winding_temp=float(T_it[0]), magnet_temp=float(T_it[2]),
            alpha_Br=op.alpha_Br,
        )

    for it in range(1, max_iter + 1):
        if it > 1:
            check_range(T, it, resids[-1] if resids else None)
        try:
            r = solve_op(T)
        except ValueError as e:
            state = (f"T_winding {T[0]:.1f} degC, T_magnet {T[2]:.1f} degC "
                     f"(iteration {it}, an iterate, not a steady state)")
            if _runaway(resids, heating):
                raise ThermalRunaway(
                    f"motor_thermal: thermal runaway at {tau_ref:.4g} N.m, "
                    f"{w_mech:.4g} rad/s: residual grew over the last "
                    f"{_RUNAWAY_K} iterates until the operating point failed at "
                    f"{state}: {e}", iteration=it, T=T.tolist(),
                    residual_K=resids[-1]) from e
            raise InfeasibleDemand(
                f"motor_thermal: demand {tau_ref:.4g} N.m at {w_mech:.4g} rad/s "
                f"infeasible at {state}: {e}", iteration=it, T=T.tolist(),
                residual_K=resids[-1] if resids else None) from e
        psi_s = hypot(r.psi_d, r.psi_q)
        psi_ratio = psi_s / p.psi_ref
        p_fe = iron_loss_at(op.n_p, p.k_h, p.k_e, p.alpha_fe, p.m_core,
                            p.B_core, w_mech, psi_ratio)
        T_new = steady_temps(G, g_s, np.array([r.p_cu, p_fe, 0.0]), T_coolant)
        resid = float(np.max(np.abs(T_new - T)))
        if resid <= tol_K:
            # The reported state is T_new, not the T the losses came from:
            # it must itself be in range and feasible.
            check_range(T_new, it + 1, resid)
            try:
                solve_op(T_new)
            except ValueError as e:
                raise InfeasibleDemand(
                    f"motor_thermal: demand {tau_ref:.4g} N.m at {w_mech:.4g} rad/s "
                    f"infeasible at the converged T_winding {T_new[0]:.1f}, "
                    f"T_magnet {T_new[2]:.1f} degC (iteration {it + 1}): {e}",
                    iteration=it + 1, T=T_new.tolist(), residual_K=resid) from e
            T = T_new
            break
        resids.append(resid)
        heating.append(bool(np.all(T_new > T)))
        T_next = T + relax * (T_new - T)
        if it == max_iter:
            # Report the last evaluated iterate T (the losses' temperatures),
            # not the unevaluated proposal; classify the proposal as the next
            # iteration's range check would.
            runaway = _runaway(resids, heating)
            if not runaway:
                check_range(T_next, it + 1, resid)
            what = ("thermal runaway: residual grew over the last "
                    f"{_RUNAWAY_K} iterates, every node heating"
                    if runaway else "no convergence")
            raise (ThermalRunaway if runaway else NoConvergence)(
                f"motor_thermal: {what} in {max_iter} iterations "
                f"(last evaluated T_winding {T[0]:.2f}, T_iron {T[1]:.2f}, "
                f"T_magnet {T[2]:.2f} degC, residual {resid:.3g} K > {tol_K} K)",
                iteration=max_iter, T=T.tolist(), residual_K=resid)
        T = T_next
    temps = {"winding": float(T[0]), "magnet": float(T[2])}
    headroom = {n: lims[n].value_C - temps[n] if n in lims else None
                for n in LIMIT_NODES}
    set_ = {n: h for n, h in headroom.items() if h is not None}
    return {
        "T_winding": float(T[0]),
        "T_iron": float(T[1]),
        "T_magnet": float(T[2]),
        "T_coolant": float(T_coolant),
        "p_cu": r.p_cu,
        "p_fe": p_fe,
        "p_shaft": r.p_shaft,
        "p_elec": r.p_shaft + r.p_cu + p_fe,
        "psi_s": psi_s,
        "psi_ratio": psi_ratio,
        "i_d": r.i_d,
        "i_q": r.i_q,
        "I_s_peak": r.I_s_peak,
        "u_mag": r.u_mag,
        "m_index": r.m_index,
        "regime": r.regime,
        "op_source": r.op_source,
        "R_s_op": r.R_s_op,
        "psi_f_op": r.psi_f_op,
        "f_e_Hz": op.n_p * w_mech / (2.0 * pi),
        "iterations": it,
        "residual_K": resid,
        "headroom_winding_K": headroom["winding"],
        "headroom_magnet_K": headroom["magnet"],
        "binding_limit": min(set_, key=lambda n: set_[n]) if set_ else None,
        "within_s1": all(h >= 0.0 for h in set_.values()) if set_ else None,
    }


def run_motor_thermal(config: RunConfig) -> dict[str, Any]:
    """Coupled steady solve at RunConfig.w_mech / tau_ref / T_coolant."""
    missing = [f for f in ("w_mech", "tau_ref", "T_coolant")
               if getattr(config, f) is None]
    if missing:
        raise ValueError(
            f"motor_thermal needs RunConfig.{' and RunConfig.'.join(missing)} "
            f"(mechanical rad/s, N.m demand, coolant degC)")
    assert (config.w_mech is not None and config.tau_ref is not None
            and config.T_coolant is not None)
    p = prepare_motor_thermal(config.motor)
    return coupled_steady(p, config.w_mech, config.tau_ref, config.T_coolant)
