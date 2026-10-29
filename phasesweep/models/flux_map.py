"""Flux-linkage surface interpolation for saturated synchronous machines.

Wraps a Motor's FluxMap (ψ_d, ψ_q tabulated on an (i_d, i_q) grid, Wb
and A peak) in scipy RegularGridInterpolators and evaluates the d-q
torque cross product on the interpolated surface. Evaluation is limited
to the tabulated grid (bounds_error=True); callers clip or reject
out-of-range currents themselves.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import RegularGridInterpolator

if TYPE_CHECKING:
    from phasesweep.machines.motor import FluxMap, Motor

__all__ = [
    "DqMap",
    "cold_psi_f",
    "derated_flux_map",
    "flux_map_at_magnet_temp",
    "flux_map_psi_f",
    "flux_map_torque",
    "make_dq_maps",
    "make_flux_map_interpolators",
    "map_covers",
]

DqMap = Callable[[complex | np.ndarray], complex | np.ndarray]


def make_flux_map_interpolators(
    fm: FluxMap,
) -> tuple[RegularGridInterpolator, RegularGridInterpolator]:
    """(psi_d_interp, psi_q_interp) over axes (i_d_A, i_q_A), evaluated as
    interp((i_d, i_q)) in Wb. Cubic on the tabulated 7x7-class grids;
    the map layout [i_q_idx][i_d_idx] is transposed to (i_d, i_q) here.
    """
    axes = (np.asarray(fm.i_d_A), np.asarray(fm.i_q_A))
    psi_d = np.asarray(fm.psi_d_Wb).T
    psi_q = np.asarray(fm.psi_q_Wb).T
    kw = {"method": "cubic", "bounds_error": True}
    return (
        RegularGridInterpolator(axes, psi_d, **kw),
        RegularGridInterpolator(axes, psi_q, **kw),
    )


def flux_map_torque(
    n_p: int,
    psi_d_interp: RegularGridInterpolator,
    psi_q_interp: RegularGridInterpolator,
    i_d: float | np.ndarray,
    i_q: float | np.ndarray,
) -> float | np.ndarray:
    """τ = 1.5 · n_p · (ψ_d · i_q − ψ_q · i_d) on the interpolated surface
    (N·m; currents A peak). Accepts scalars or equal-length arrays."""
    pts = np.column_stack((np.atleast_1d(i_d), np.atleast_1d(i_q)))
    tau = 1.5 * n_p * (psi_d_interp(pts) * pts[:, 1] - psi_q_interp(pts) * pts[:, 0])
    return float(tau[0]) if np.isscalar(i_d) else tau


def flux_map_psi_f(fm: FluxMap) -> float:
    """PM flux linkage ψ_d(i_d=0, i_q=0) from the map (Wb, peak): first
    i_q row, last i_d column. Requires the origin to be a grid node."""
    if not fm.has_origin:
        raise ValueError(
            "[flux_map] psi_f needs (i_d=0, i_q=0) on the grid: i_d_A must "
            f"end at 0 and i_q_A start at 0 (got {fm.i_d_A[-1]}, {fm.i_q_A[0]})"
        )
    return fm.psi_d_Wb[0][-1]


def cold_psi_f(fm: FluxMap, alpha_Br: float | None) -> float:
    """ψ_d(0, 0) referred to the 20 °C scalar-derating reference (Wb, peak).

    The map's origin node is at fm.magnet_temp_ref_C; the scalar psi_f that
    solver_params.derate shifts is defined at 20 °C, so a map extracted hot
    is divided back by [1 + alpha_Br·(T_ref − 20)]. With the reference
    unset or 20 °C, or alpha_Br unset, the map value is returned as is
    (derate raises later if a magnet_temp needs the missing coefficient).
    """
    from phasesweep.solver_params import MAGNET_REF_TEMP_C

    psi_map = flux_map_psi_f(fm)
    t_ref = fm.magnet_temp_ref_C
    if t_ref is None or t_ref == MAGNET_REF_TEMP_C or alpha_Br is None:
        return psi_map
    return psi_map / (1.0 + alpha_Br * (t_ref - MAGNET_REF_TEMP_C))


def map_covers(fm: FluxMap, I_max: float) -> bool:
    """The grid holds the whole motoring quarter-circle of radius I_max:
    i_d reaches -I_max, i_q spans 0..I_max, and the i_d axis reaches into
    the circle. Outside it a grid-only search pins to the grid edge."""
    return not (I_max > abs(fm.i_d_A[0]) or I_max > fm.i_q_A[-1]
                or fm.i_q_A[0] > 0 or fm.i_d_A[-1] < -I_max)


def flux_map_at_magnet_temp(
    fm: FluxMap, magnet_temp: float | None, alpha_Br: float | None,
) -> FluxMap:
    """Copy of the map with the PM contribution shifted to magnet_temp.

    First-order model: the magnet contributes to ψ_d only and its
    remanence scales linearly with temperature, so every ψ_d node moves
    by ψ_f_map · alpha_Br · (T − T_ref) and ψ_q is unchanged. T_ref is
    fm.magnet_temp_ref_C, else the 20 °C scalar-derating reference. Exact
    on a linear (constant-L) surface, first order on a saturated one: the
    neglected term is the change of the magnet's own BH operating point as
    iron saturation shifts with temperature (second order). magnet_temp
    None returns fm unchanged; magnet_temp without alpha_Br raises (the
    coefficient is grade-specific, no safe default).
    """
    if magnet_temp is None:
        return fm
    if alpha_Br is None:
        raise ValueError(
            "magnet_temp is set but alpha_Br is not — the B_rem temperature "
            "coefficient is grade-specific (per K, e.g. NdFeB N42 -0.0012, "
            "SmCo -0.0003); set alpha_Br or drop magnet_temp"
        )
    from phasesweep.solver_params import MAGNET_REF_TEMP_C

    t_ref = fm.magnet_temp_ref_C if fm.magnet_temp_ref_C is not None else MAGNET_REF_TEMP_C
    shift = flux_map_psi_f(fm) * alpha_Br * (magnet_temp - t_ref)
    from dataclasses import replace

    return replace(
        fm,
        psi_d_Wb=tuple(tuple(x + shift for x in row) for row in fm.psi_d_Wb),
    )


def derated_flux_map(motor: Motor) -> FluxMap | None:
    """The motor's flux map at its magnet_temp (None when it has no map).

    The one map every circuit model runs on: rated_torque, stall_torque,
    torque_speed, operating_point and drive_sim all read the surface
    through this so a magnet_temp shifts them together.
    """
    if motor.flux_map is None:
        return None
    try:
        return flux_map_at_magnet_temp(motor.flux_map, motor.magnet_temp, motor.alpha_Br)
    except ValueError as e:
        raise ValueError(f"Motor '{motor.name}': {e}") from None


_FD_STEP_A = 1e-2


def _extended_surface(fm: FluxMap) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """(psi_d, psi_q) at (N, 2) current points: cubic inside the grid,
    continued linearly (edge incremental inductance) outside it so the
    surface stays monotone where the map has no data.

    When i_q starts at 0 (has_origin), the one-quadrant data is mirrored
    to i_q < 0 before building the spline (ψ_d even, ψ_q odd about its
    zero-current value) so the surface is C2 through i_q = 0."""
    i_d_ax = np.asarray(fm.i_d_A)
    i_q_pos = np.asarray(fm.i_q_A)
    psi_d_data = np.asarray(fm.psi_d_Wb).T
    psi_q_data = np.asarray(fm.psi_q_Wb).T

    if i_q_pos[0] == 0.0 and len(i_q_pos) > 1:
        i_q_full = np.concatenate((-i_q_pos[:0:-1], i_q_pos))
        psi_d_full = np.concatenate((psi_d_data[:, :0:-1], psi_d_data), axis=1)
        pq0 = psi_q_data[:, 0:1]
        psi_q_full = np.concatenate(
            (2 * pq0 - psi_q_data[:, :0:-1], psi_q_data), axis=1)
    else:
        i_q_full = i_q_pos
        psi_d_full = psi_d_data
        psi_q_full = psi_q_data

    axes = (i_d_ax, i_q_full)
    kw = {"method": "cubic", "bounds_error": False, "fill_value": None}
    psi_d = RegularGridInterpolator(axes, psi_d_full, **kw)
    psi_q = RegularGridInterpolator(axes, psi_q_full, **kw)
    lo = np.array([axes[0][0], axes[1][0]])
    hi = np.array([axes[0][-1], axes[1][-1]])
    e_d = np.array([_FD_STEP_A, 0.0])
    e_q = np.array([0.0, _FD_STEP_A])

    def evaluate(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        edge = np.clip(pts, lo, hi)
        delta = pts - edge
        pd, pq = psi_d(edge), psi_q(edge)
        out = (delta != 0).any(axis=1)
        if not out.any():
            return pd, pq
        c, dd, dq = edge[out], delta[out, 0], delta[out, 1]
        for arr, interp in ((pd, psi_d), (pq, psi_q)):
            g_d = (interp(c + e_d) - interp(c - e_d)) / (2 * _FD_STEP_A)
            g_q = (interp(c + e_q) - interp(c - e_q)) / (2 * _FD_STEP_A)
            arr[out] += g_d * dd + g_q * dq
        return pd, pq

    return evaluate


def _wrap_complex(fn: Callable[[np.ndarray], np.ndarray]) -> DqMap:
    def call(z: complex | np.ndarray) -> complex | np.ndarray:
        arr = np.asarray(z, dtype=complex)
        out = fn(arr.ravel())
        return complex(out[0]) if arr.ndim == 0 else out.reshape(arr.shape)
    return call


def _flux_of_current(fm: FluxMap) -> DqMap:
    surface = _extended_surface(fm)

    def psi(i: np.ndarray) -> np.ndarray:
        pd, pq = surface(np.column_stack((i.real, i.imag)))
        return pd + 1j * pq

    return _wrap_complex(psi)


def _invert_on_grid(psi: DqMap, target: np.ndarray, i0: np.ndarray,
                    step_max: float, max_iter: int = 100) -> np.ndarray:
    """Vectorized damped Newton: currents whose flux linkage equals `target`."""
    h = _FD_STEP_A
    i = i0.copy()
    for _ in range(max_iter):
        f = psi(i) - target
        if np.abs(f).max() < 1e-9:
            break
        j_d = (psi(i + h) - psi(i - h)) / (2 * h)
        j_q = (psi(i + 1j * h) - psi(i - 1j * h)) / (2 * h)
        det = j_d.real * j_q.imag - j_q.real * j_d.imag
        d_id = (j_q.imag * f.real - j_q.real * f.imag) / det
        d_iq = (j_d.real * f.imag - j_d.imag * f.real) / det
        step = d_id + 1j * d_iq
        scale = np.minimum(1.0, step_max / np.maximum(np.abs(step), 1e-30))
        i = i - step * scale
    residual = np.abs(psi(i) - target).max()
    if residual >= 1e-9:
        n_bad = int((np.abs(psi(i) - target) >= 1e-9).sum())
        raise ValueError(
            f"[flux_map] inverse Newton did not converge at {n_bad} of "
            f"{len(target)} flux-grid points (max residual {residual:.2e} Wb)"
        )
    return i


@lru_cache(maxsize=4)
def make_dq_maps(fm: FluxMap, n_grid: int = 161,
                 extension: float = 0.25) -> tuple[DqMap, DqMap]:
    """(psi_s_dq_fcn, i_s_dq_fcn) for a saturated d-q machine model —
    complex-valued callables (d + j·q; A and Wb peak) in motulator's
    calling convention, scalar or array in, same shape out.

    psi_s_dq_fcn is the tabulated surface: cubic inside the grid
    (one-quadrant data mirrored to i_q < 0 so the spline is C2 through
    zero), continued linearly beyond the grid edges. i_s_dq_fcn is its
    inverse, tabulated once by Newton on an n_grid² flux grid spanning the
    surface image over the grid extended by `extension` of its span per
    open side, and read back by linear interpolation (extrapolated outside).
    """
    psi = _flux_of_current(fm)
    i_d, i_q = np.asarray(fm.i_d_A), np.asarray(fm.i_q_A)
    span_d, span_q = i_d[-1] - i_d[0], i_q[-1] - i_q[0]
    d_axis = np.linspace(i_d[0] - extension * span_d, i_d[-1] + extension * span_d, 41)
    q_lim = i_q[-1] + extension * span_q
    q_axis = np.linspace(-q_lim, q_lim, 41)
    sample = psi(d_axis[:, None] + 1j * q_axis[None, :])
    assert isinstance(sample, np.ndarray)

    pd_axis = np.linspace(sample.real.min(), sample.real.max(), n_grid)
    pq_axis = np.linspace(sample.imag.min(), sample.imag.max(), n_grid)
    target = (pd_axis[:, None] + 1j * pq_axis[None, :]).ravel()

    h = _FD_STEP_A
    psi_f = complex(psi(0j)).real
    L_d0 = (complex(psi(h)) - complex(psi(-h))).real / (2 * h)
    L_q0 = (complex(psi(1j * h)) - complex(psi(-1j * h))).imag / (2 * h)
    i0 = (target.real - psi_f) / L_d0 + 1j * target.imag / L_q0
    grid = _invert_on_grid(psi, target, i0, 0.5 * max(span_d, span_q))

    current = RegularGridInterpolator(
        (pd_axis, pq_axis), grid.reshape(n_grid, n_grid),
        method="linear", bounds_error=False, fill_value=None,
    )
    i_of_psi = _wrap_complex(
        lambda p: current(np.column_stack((p.real, p.imag))))
    return psi, i_of_psi
