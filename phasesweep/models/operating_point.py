"""Operating-point model: minimum stator current for a torque demand at a speed.

Answers the question torque_speed does not: given (w_mech, tau_ref), which
(i_d, i_q) produces tau_ref with the least current inside the current
circle |i| <= I_max and the voltage limit |u| <= U_max (SVPWM linear
range, U_max = U_DC/sqrt(3), phase peak — the torque_speed convention).
Same steady-state dq circuit as torque_speed:

    u_d = R_s i_d - w_e psi_q,   u_q = R_s i_q + w_e psi_d,   w_e = n_p w_mech

with psi_d = psi_f + L_d i_d, psi_q = L_q i_q on the constant-L path, or
the tabulated flux-linkage surface on the flux-map path.

Temperature is continuous on both paths: R_s is derated to the winding
operating temperature winding_temp and the PM flux to magnet_temp through the shared
solver_params.derate (the rule every circuit model applies), the map via
flux_map.flux_map_at_magnet_temp (first-order psi_d shift). Unlike the
other factories, prepare_operating_point carries the cold inputs and the
thermal fields separately so the solve can be repeated at new
temperatures without rebuilding the params.

Outputs are scalars: currents A peak (I_s_rms also given), voltages V
phase peak, m_index = 2 u_mag / U_DC (amplitude over half bus; <= 1 is
the SPWM linear range, <= 2/sqrt(3) the SVPWM range every feasible point
satisfies), and the dq flux linkages psi_d / psi_q (Wb peak) the voltages
were built from (additive outputs; rows written before them lack the
keys). p_elec = p_shaft + p_cu — no iron loss, the model emits no p_fe
(motor_thermal derives it from |psi_s|). pf is None at zero current
(undefined; consumers substitute their own default).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import atan2, cos, degrees, hypot, isfinite, pi, sin, sqrt
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from phasesweep.models.flux_map import (
    flux_map_at_magnet_temp,
    flux_map_psi_f,
    make_flux_map_interpolators,
    map_covers,
)
from phasesweep.models.rated_torque import mtpa_gamma
from phasesweep.models.thermal_duty import current_for_torque
from phasesweep.models.torque_speed import (
    _ID_GRID_PTS,
    TorqueSpeedParams,
    cold_torque_speed_params,
    dq_voltage,
    envelope_at_speed,
    envelope_curve_map,
)
from phasesweep.solver_params import derate

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from phasesweep.machines.motor import FluxMap, Motor
    from phasesweep.sweep_types import RunConfig

__all__ = [
    "OperatingPointParams",
    "OperatingPointResult",
    "cold_operating_point_params",
    "operating_point",
    "prepare_operating_point",
    "run_operating_point",
]

Regime = Literal["mtpa", "field_weakening"]
OpSource = Literal["flux_map", "analytical"]

_GOLDEN = (sqrt(5.0) - 1.0) / 2.0
_GOLDEN_ITERS = 80
_BISECT_ITERS = 60
_IQ_GRID_PTS = 201


def _derate(R_s: float, psi_f: float, winding_temp: float | None,
            magnet_temp: float | None, alpha_Br: float | None) -> tuple[float, float]:
    """solver_params.derate with R_s required (the shared rule and guard)."""
    R_s_op, psi_f_op = derate(R_s, psi_f, winding_temp, magnet_temp, alpha_Br)
    assert R_s_op is not None
    return R_s_op, psi_f_op


def _check_voltage(U_max: float, U_DC: float) -> None:
    if not isfinite(U_DC) or U_DC <= 0:
        raise ValueError(f"U_DC must be finite and > 0, got {U_DC}")
    if abs(U_max * sqrt(3.0) - U_DC) > 1e-9 * U_DC:
        raise ValueError(
            f"U_max {U_max:.6g} V is not U_DC/sqrt(3) = {U_DC / sqrt(3.0):.6g} V; "
            f"m_index would be inconsistent with the voltage limit")


def _check_map_temp(flux_map: FluxMap | None, I_max: float,
                    magnet_temp: float | None) -> None:
    """The PM shift needs the map's origin node; reject before the solve
    rather than fail inside flux_map_psi_f with a message about psi_f."""
    if (flux_map is None or magnet_temp is None
            or not map_covers(flux_map, I_max) or flux_map.has_origin):
        return
    raise ValueError(
        "magnet_temp needs the flux map's origin node (i_d = 0, i_q = 0) to "
        f"shift the PM flux; i_d_A ends at {flux_map.i_d_A[-1]}, i_q_A starts "
        f"at {flux_map.i_q_A[0]}: extend the map or drop magnet_temp")


@dataclass(frozen=True)
class OperatingPointResult:
    """One solved operating point (units in the module docstring)."""

    i_d: float
    i_q: float
    I_s_peak: float
    I_s_rms: float
    u_d: float
    u_q: float
    u_mag: float
    u_max: float
    m_index: float
    pf: float | None
    gamma_deg: float
    regime: Regime
    tau: float
    w_mech: float
    f_e_Hz: float
    p_cu: float
    p_shaft: float
    p_elec: float
    R_s_op: float
    psi_f_op: float
    op_source: OpSource
    psi_d: float
    psi_q: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# constant-L path
# ---------------------------------------------------------------------------

def _envelope_msg(n_p: int, psi_f: float, R_s: float, L_d: float, L_q: float,
                  I_max: float, U_max: float, w_mech: float) -> str:
    tau_env = envelope_at_speed(n_p, psi_f, R_s, L_d, L_q, I_max, U_max, w_mech)[0]
    return f"the envelope torque at w_mech {w_mech:.4g} rad/s is {tau_env:.4g} N.m"


def _zero_torque_analytical(n_p: int, psi_f: float, R_s: float, L_d: float,
                            I_max: float, U_max: float,
                            w_mech: float) -> tuple[float, Regime]:
    """i_d at i_q = 0: zero below the no-load base speed, else the
    least-negative root of |u(i_d, 0)| = U_max."""
    w_e = n_p * w_mech
    if w_e * psi_f <= U_max:
        return 0.0, "mtpa"
    a = R_s * R_s + (w_e * L_d) ** 2
    b = 2.0 * w_e * w_e * L_d * psi_f
    c = (w_e * psi_f) ** 2 - U_max * U_max
    disc = b * b - 4.0 * a * c
    i_d = (-b + sqrt(disc)) / (2.0 * a) if disc >= 0.0 else -float("inf")
    if i_d < -I_max:
        raise ValueError(
            f"no-load voltage {w_e * psi_f:.4g} V at w_mech {w_mech:.4g} rad/s "
            f"exceeds U_max {U_max:.4g} V and holding it needs |i_d| beyond "
            f"I_max {I_max:.4g} A"
        )
    return i_d, "field_weakening"


def _fw_analytical(n_p: int, psi_f: float, R_s: float, L_d: float, L_q: float,
                   I_max: float, U_max: float, w_mech: float, tau: float,
                   i_d_mtpa: float) -> tuple[float, float]:
    """Constant-torque field weakening: the i_d closest to MTPA on the
    torque contour whose voltage fits U_max.

    The contour i_q(i_d) = tau / (1.5 n_p (psi_f + (L_d - L_q) i_d)) is a
    hyperbola with a pole, so |u| along it is not monotone: (a) floor the
    bracket above the pole, (b) golden-section to the minimum-|u| point
    (MTPV), (c) bisect between MTPV and MTPA for |u| = U_max keeping the
    feasible side.
    """
    dL = L_d - L_q

    def point(i_d: float) -> tuple[float, float]:
        i_q = tau / (1.5 * n_p * (psi_f + dL * i_d))
        u = float(dq_voltage(n_p, psi_f, R_s, L_d, L_q, i_d, i_q, w_mech))
        return i_q, u

    floor = -I_max
    if dL != 0.0:
        pole = -psi_f / dL
        if pole < i_d_mtpa:
            floor = max(floor, pole + 1e-9 * (i_d_mtpa - pole))

    lo, hi = floor, i_d_mtpa
    for _ in range(_GOLDEN_ITERS):
        a = hi - _GOLDEN * (hi - lo)
        b = lo + _GOLDEN * (hi - lo)
        if point(a)[1] < point(b)[1]:
            hi = b
        else:
            lo = a
    i_d_mtpv = 0.5 * (lo + hi)
    u_min = point(i_d_mtpv)[1]
    if u_min > U_max:
        raise ValueError(
            f"torque {tau:.4g} N.m at w_mech {w_mech:.4g} rad/s exceeds the "
            f"MTPV limit: the minimum voltage along the constant-torque "
            f"contour is {u_min:.4g} V, above U_max {U_max:.4g} V; "
            + _envelope_msg(n_p, psi_f, R_s, L_d, L_q, I_max, U_max, w_mech)
        )

    lo, hi = i_d_mtpv, i_d_mtpa
    for _ in range(_BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        if point(mid)[1] <= U_max:
            lo = mid
        else:
            hi = mid
    return lo, point(lo)[0]


def _solve_analytical(n_p: int, psi_f: float, R_s: float, L_d: float,
                      L_q: float, I_max: float, U_max: float, w_mech: float,
                      tau_ref: float) -> tuple[float, float, Regime]:
    if tau_ref == 0.0:
        i_d, regime = _zero_torque_analytical(n_p, psi_f, R_s, L_d, I_max,
                                              U_max, w_mech)
        return i_d, 0.0, regime

    I_s = current_for_torque(n_p, psi_f, L_d, L_q, tau_ref)
    if I_s > I_max:
        raise ValueError(
            f"torque {tau_ref:.4g} N.m needs {I_s:.4g} A peak on the MTPA locus, "
            f"above I_max {I_max:.4g} A; "
            + _envelope_msg(n_p, psi_f, R_s, L_d, L_q, I_max, U_max, w_mech)
        )
    g = mtpa_gamma(psi_f, L_d, L_q, I_s)
    i_d, i_q = -I_s * sin(g), I_s * cos(g)
    if dq_voltage(n_p, psi_f, R_s, L_d, L_q, i_d, i_q, w_mech) <= U_max * (1.0 + 1e-12):
        return i_d, i_q, "mtpa"

    i_d, i_q = _fw_analytical(n_p, psi_f, R_s, L_d, L_q, I_max, U_max,
                              w_mech, tau_ref, i_d)
    if hypot(i_d, i_q) > I_max * (1.0 + 1e-12):
        raise ValueError(
            f"torque {tau_ref:.4g} N.m at w_mech {w_mech:.4g} rad/s needs "
            f"{hypot(i_d, i_q):.4g} A peak under the voltage limit, above "
            f"I_max {I_max:.4g} A; "
            + _envelope_msg(n_p, psi_f, R_s, L_d, L_q, I_max, U_max, w_mech)
        )
    return i_d, i_q, "field_weakening"


# ---------------------------------------------------------------------------
# flux-map path
# ---------------------------------------------------------------------------

class _MapSolver:
    """Contour search on the flux-linkage surface at one (w_mech, tau_ref).

    The map is only guaranteed finite and axis-ordered: torque need not
    be monotone in i_q nor |i| unimodal in i_d, so every torque root is
    bracketed by sign change on a 201-point i_q grid and bisected, and the
    i_d minimum is grid-based (151 samples) with golden-section refinement
    inside the winning cell. Resolution limits: a root pair closer than
    one i_q cell (or a tangency) shows no sign change and is missed, and
    global optimality in i_d is only to the grid; the refined point is
    within 1e-6 relative in |i| of the local minimum in its cell. tau_ref
    = 0 goes through the same search: on a map with psi_q(i_d, 0) != 0
    the zero-torque contour sits at a small positive i_q, not on the
    d-axis.
    """

    def __init__(self, n_p: int, R_s: float, I_max: float, U_max: float,
                 fm: FluxMap, w_mech: float, tau_ref: float) -> None:
        self.n_p, self.R_s, self.I_max, self.U_max = n_p, R_s, I_max, U_max
        self.w_e = n_p * w_mech
        self.tau_ref = tau_ref
        self.psi_d, self.psi_q = make_flux_map_interpolators(fm)
        self.id_lo = max(-I_max, fm.i_d_A[0])
        self.id_hi = fm.i_d_A[-1]
        self.iq_lo = fm.i_q_A[0]
        self.iq_map_hi = fm.i_q_A[-1]

    def flux(self, i_d: NDArray, i_q: NDArray) -> tuple[NDArray, NDArray]:
        pts = np.column_stack((i_d, i_q))
        return self.psi_d(pts), self.psi_q(pts)

    def torque_residual(self, i_d: NDArray, i_q: NDArray) -> NDArray:
        pd, pq = self.flux(i_d, i_q)
        return 1.5 * self.n_p * (pd * i_q - pq * i_d) - self.tau_ref

    def voltage(self, i_d: NDArray, i_q: NDArray) -> tuple[NDArray, NDArray]:
        pd, pq = self.flux(i_d, i_q)
        return self.R_s * i_d - self.w_e * pq, self.R_s * i_q + self.w_e * pd

    def roots(self, i_ds: NDArray) -> tuple[NDArray, NDArray]:
        """(row index into i_ds, i_q) of every torque-contour root.

        The i_q range at each i_d is the map bound intersected with the
        current circle; every sign change of tau - tau_ref on the 201-point
        grid is bisected, all brackets together.
        """
        iq_hi = np.minimum(self.iq_map_hi,
                           np.sqrt(np.maximum(0.0, self.I_max**2 - i_ds * i_ds)))
        valid = iq_hi >= self.iq_lo
        t = np.linspace(0.0, 1.0, _IQ_GRID_PTS)
        iq = self.iq_lo + (np.maximum(iq_hi, self.iq_lo) - self.iq_lo)[:, None] * t[None, :]
        i_d = np.broadcast_to(i_ds[:, None], iq.shape)
        f = self.torque_residual(i_d.ravel(), iq.ravel()).reshape(iq.shape)
        f[~valid] = np.nan
        rows_e, cols_e = np.nonzero(f == 0.0)
        rows, cols = np.nonzero(f[:, :-1] * f[:, 1:] < 0.0)
        lo, hi, f_lo = iq[rows, cols], iq[rows, cols + 1], f[rows, cols]
        i_d_b = i_ds[rows]
        for _ in range(_BISECT_ITERS):
            mid = 0.5 * (lo + hi)
            f_mid = self.torque_residual(i_d_b, mid)
            same = np.sign(f_mid) == np.sign(f_lo)
            lo = np.where(same, mid, lo)
            f_lo = np.where(same, f_mid, f_lo)
            hi = np.where(same, hi, mid)
        return (np.concatenate((rows_e, rows)),
                np.concatenate((iq[rows_e, cols_e], 0.5 * (lo + hi))))

    def candidates(self, i_ds: NDArray) -> tuple[NDArray, NDArray]:
        """Per i_d: (smallest root i_q, smallest voltage-feasible root
        i_q), NaN where none. At fixed i_d the smallest i_q is the
        smallest |i|."""
        n = len(i_ds)
        iq_any = np.full(n, np.nan)
        iq_ok = np.full(n, np.nan)
        rows, iq = self.roots(i_ds)
        if rows.size == 0:
            return iq_any, iq_ok
        u_d, u_q = self.voltage(i_ds[rows], iq)
        feas = np.hypot(u_d, u_q) <= self.U_max * (1.0 + 1e-12)
        order = np.lexsort((iq, rows))
        rows, iq, feas = rows[order], iq[order], feas[order]
        r, first = np.unique(rows, return_index=True)
        iq_any[r] = iq[first]
        r, first = np.unique(rows[feas], return_index=True)
        iq_ok[r] = iq[feas][first]
        return iq_any, iq_ok

    def feasible_current(self, i_d: float) -> float:
        _, iq = self.candidates(np.array([i_d]))
        return float("inf") if np.isnan(iq[0]) else hypot(i_d, float(iq[0]))

    def solve(self) -> tuple[float, float, Regime] | None:
        """(i_d, i_q, regime), or None when no feasible point exists."""
        grid = np.linspace(self.id_lo, self.id_hi, _ID_GRID_PTS)
        step = (self.id_hi - self.id_lo) / (_ID_GRID_PTS - 1)
        iq_any, iq_ok = self.candidates(grid)
        I_ok = np.hypot(grid, iq_ok)
        if np.isnan(I_ok).all():
            return None
        k = int(np.nanargmin(I_ok))
        k_any = int(np.nanargmin(np.hypot(grid, iq_any)))
        regime: Regime = ("mtpa" if iq_any[k_any] == iq_ok[k_any]
                          else "field_weakening")

        # Golden-section refinement inside the winning cell. The objective
        # jumps to +inf at the voltage boundary, so the minimum may sit on
        # a discontinuity: keep the best feasible probe seen rather than
        # the bracket midpoint, which can land on the infeasible side.
        best = (float(I_ok[k]), float(grid[k]))
        lo, hi = max(self.id_lo, best[1] - step), min(self.id_hi, best[1] + step)

        def probe(x: float) -> float:
            nonlocal best
            f = self.feasible_current(x)
            if f < best[0]:
                best = (f, x)
            return f

        a, b = hi - _GOLDEN * (hi - lo), lo + _GOLDEN * (hi - lo)
        fa, fb = probe(a), probe(b)
        for _ in range(_GOLDEN_ITERS):
            if fa < fb:
                hi, b, fb = b, a, fa
                a = hi - _GOLDEN * (hi - lo)
                fa = probe(a)
            else:
                lo, a, fa = a, b, fb
                b = lo + _GOLDEN * (hi - lo)
                fb = probe(b)
        i_d_best = best[1]
        _, iq_best = self.candidates(np.array([i_d_best]))
        return i_d_best, float(iq_best[0]), regime


def _solve_map(n_p: int, psi_f: float, R_s: float, L_d: float, L_q: float,
               I_max: float, U_max: float, fm: FluxMap, w_mech: float,
               tau_ref: float) -> tuple[float, float, Regime]:
    found = _MapSolver(n_p, R_s, I_max, U_max, fm, w_mech, tau_ref).solve()
    if found is None:
        tau_env = float(envelope_curve_map(
            n_p, psi_f, R_s, L_d, L_q, I_max, U_max, fm,
            np.array([w_mech]))[0][0])
        raise ValueError(
            f"torque {tau_ref:.4g} N.m at w_mech {w_mech:.4g} rad/s is not "
            f"reachable on the flux map inside I_max {I_max:.4g} A and U_max "
            f"{U_max:.4g} V; the map envelope torque at this speed is "
            f"{tau_env:.4g} N.m"
        )
    return found


# ---------------------------------------------------------------------------
# public solve
# ---------------------------------------------------------------------------

def operating_point(
    n_p: int, psi_f: float, R_s: float, L_d: float, L_q: float,
    I_max: float, U_max: float, U_DC: float, w_mech: float, tau_ref: float,
    *, flux_map: FluxMap | None = None, winding_temp: float | None = None,
    magnet_temp: float | None = None, alpha_Br: float | None = None,
) -> OperatingPointResult:
    """Minimum-current (i_d, i_q) for tau_ref at w_mech.

    Constant-L: MTPA point when voltage-feasible, else three-stage
    constant-torque field weakening (pole floor, MTPV by golden section,
    bisection to |u| = U_max). Flux-map (when the map covers I_max):
    bracketed contour search on the temperature-shifted surface; falls
    back to constant-L otherwise (op_source = "analytical"). Raises
    ValueError with the envelope torque when the demand is unreachable.

    winding_temp / magnet_temp are operating temperatures (degC), not
    limits: R_s and the PM flux are evaluated there. None leaves each
    at its 20 degC reference.
    """
    R_s_op, psi_f_op = _derate(R_s, psi_f, winding_temp, magnet_temp, alpha_Br)
    _check_voltage(U_max, U_DC)
    _check_map_temp(flux_map, I_max, magnet_temp)

    if flux_map is not None and map_covers(flux_map, I_max):
        fm = flux_map_at_magnet_temp(flux_map, magnet_temp, alpha_Br)
        if fm.has_origin:
            psi_f_op = flux_map_psi_f(fm)
        i_d, i_q, regime = _solve_map(n_p, psi_f_op, R_s_op, L_d, L_q, I_max,
                                      U_max, fm, w_mech, tau_ref)
        psi_d_i, psi_q_i = make_flux_map_interpolators(fm)
        pt = np.array([[i_d, i_q]])
        psi_d, psi_q = float(psi_d_i(pt)[0]), float(psi_q_i(pt)[0])
        source: OpSource = "flux_map"
    else:
        i_d, i_q, regime = _solve_analytical(n_p, psi_f_op, R_s_op, L_d, L_q,
                                             I_max, U_max, w_mech, tau_ref)
        psi_d, psi_q = psi_f_op + L_d * i_d, L_q * i_q
        source = "analytical"

    w_e = n_p * w_mech
    u_d = R_s_op * i_d - w_e * psi_q
    u_q = R_s_op * i_q + w_e * psi_d
    u_mag = hypot(u_d, u_q)
    I_s = hypot(i_d, i_q)
    pf = None if I_s == 0.0 or u_mag == 0.0 else (u_d * i_d + u_q * i_q) / (u_mag * I_s)
    p_cu = 1.5 * R_s_op * I_s * I_s
    p_shaft = tau_ref * w_mech
    return OperatingPointResult(
        i_d=i_d, i_q=i_q, I_s_peak=I_s, I_s_rms=I_s / sqrt(2.0),
        u_d=u_d, u_q=u_q, u_mag=u_mag, u_max=U_max,
        m_index=2.0 * u_mag / U_DC, pf=pf,
        gamma_deg=degrees(atan2(-i_d, i_q)), regime=regime,
        tau=tau_ref, w_mech=w_mech, f_e_Hz=w_e / (2.0 * pi),
        p_cu=p_cu, p_shaft=p_shaft, p_elec=p_shaft + p_cu,
        R_s_op=R_s_op, psi_f_op=psi_f_op, op_source=source,
        psi_d=psi_d, psi_q=psi_q,
    )


# ---------------------------------------------------------------------------
# params factory and runner
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OperatingPointParams:
    """Validated inputs for the operating-point solve.

    Shares TorqueSpeedParams' checks (I_max = I_LIMIT or MAX_I_S, non-finite
    and resistive-drop guards), requires U_max = U_DC/sqrt(3) (U_DC is kept
    for m_index), and range-checks the temperature derating and the map's
    origin node when magnet_temp is set. winding_temp is the winding
    operating temperature; prepare_operating_point sets it to the motor's
    winding_temp_limit (the worst-case S1 state every circuit model uses).
    """

    n_p: int
    psi_f: float
    R_s: float
    L_d: float
    L_q: float
    I_max: float
    U_max: float
    U_DC: float
    flux_map: FluxMap | None = None
    winding_temp: float | None = None
    magnet_temp: float | None = None
    alpha_Br: float | None = None

    def __post_init__(self) -> None:
        TorqueSpeedParams(n_p=self.n_p, psi_f=self.psi_f, R_s=self.R_s,
                          L_d=self.L_d, L_q=self.L_q, I_peak=self.I_max,
                          U_max=self.U_max)
        _check_voltage(self.U_max, self.U_DC)
        _derate(self.R_s, self.psi_f, self.winding_temp, self.magnet_temp,
                self.alpha_Br)
        _check_map_temp(self.flux_map, self.I_max, self.magnet_temp)


def prepare_operating_point(motor: Motor) -> OperatingPointParams:
    """Derive, validate, and extract operating-point params from a Motor.

    Takes the cold torque_speed inputs and carries the thermal fields
    explicitly — `operating_point` derates at solve time (the march
    re-solves the same point as the temperatures move)."""
    return cold_operating_point_params(motor, motor.winding_temp_limit,
                                       motor.magnet_temp)


def cold_operating_point_params(motor: Motor, winding_temp: float | None,
                                magnet_temp: float | None) -> OperatingPointParams:
    """prepare_operating_point with the operating temperatures given
    explicitly: motor_thermal passes None for both (it solves the nodes and
    ignores the motor's winding_temp_limit / magnet_temp as operating
    temperatures), so they are not range-checked as such."""
    ts = cold_torque_speed_params(motor)
    assert motor.drive.U_DC is not None  # narrowed by cold_torque_speed_params
    try:
        return OperatingPointParams(
            n_p=ts.n_p, psi_f=ts.psi_f, R_s=ts.R_s, L_d=ts.L_d, L_q=ts.L_q,
            I_max=ts.I_peak, U_max=ts.U_max, U_DC=motor.drive.U_DC,
            flux_map=motor.flux_map, winding_temp=winding_temp,
            magnet_temp=magnet_temp, alpha_Br=motor.alpha_Br,
        )
    except ValueError as e:
        raise ValueError(f"Motor '{motor.name}': {e}") from None


def run_operating_point(config: RunConfig) -> dict[str, Any]:
    """Operating point at RunConfig.w_mech / RunConfig.tau_ref."""
    missing = [f for f in ("w_mech", "tau_ref") if getattr(config, f) is None]
    if missing:
        raise ValueError(
            f"operating_point needs RunConfig.{' and RunConfig.'.join(missing)} "
            f"(mechanical rad/s and N.m demand)"
        )
    assert config.w_mech is not None and config.tau_ref is not None
    p = prepare_operating_point(config.motor)
    return operating_point(
        p.n_p, p.psi_f, p.R_s, p.L_d, p.L_q, p.I_max, p.U_max, p.U_DC,
        config.w_mech, config.tau_ref,
        flux_map=p.flux_map, winding_temp=p.winding_temp,
        magnet_temp=p.magnet_temp, alpha_Br=p.alpha_Br,
    ).to_dict()
