"""
2D magnetostatic FEM field solver for PMSM air-gap flux density.

Physics: A-formulation, H1 scalar vector potential A_z.
Equation: -∇·(ν ∇A_z) = ∇×(ν_PM B_rem)|_z  in PM region
Source:   ∫_PM ν_PM B_rem (sin(n_p θ) ∂v/∂x - cos(n_p θ) ∂v/∂y)

Uses NGSolve (LGPL) for FEM. CuPy (GPU) for batch harmonic decomposition.
"""

from __future__ import annotations

import sys
from math import pi
from typing import Any, Literal, overload

import numpy as np
from numpy.typing import NDArray
from ngsolve import (Mesh, H1, L2, BilinearForm, LinearForm, GridFunction,
                     sqrt, grad, dx, CoefficientFunction, x, y)
from netgen.occ import WorkPlane, OCCGeometry, Glue

try:
    import cupy as cp  # type: ignore[import-unresolved]
    _CUPY = True
except ImportError:
    cp = None  # type: ignore[assignment]
    _CUPY = False

# ---------------------------------------------------------------------------
# Physical constants & geometry (normalized, stator outer = 1 m)
# ---------------------------------------------------------------------------
_MU0: float = 4e-7 * pi

# Generic soft magnetic steel B-H curve (representative, not grade-specific).
# Source: composite of textbook soft iron data; suitable for design exploration.
_BH_B = np.array([
    0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.1, 1.2,
    1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2,
])
_BH_H = np.array([
    0., 25., 40., 60., 80., 110., 160., 210., 310.,
    480., 900., 1800., 3800., 7500., 15000., 30000., 60000., 120000., 200000.,
])

_MU_R_FE: float = 1000.0
_MU_R_PM: float = 1.05
_R_S: float = 1.00
_R_SI: float = 0.70
_R_RO: float = 0.64
_R_RI: float = 0.30
_R_AG: float = (_R_SI + _R_RO) / 2


def _bh_nu(B_mag: NDArray) -> NDArray:
    """Reluctivity ν(|B|) from generic soft iron B-H curve.

    Returns ν = H/B in SI units. At B→0, uses initial slope to avoid 0/0.
    """
    H = np.interp(B_mag, _BH_B, _BH_H)
    B_safe = np.maximum(B_mag, 1e-12)
    nu = H / B_safe
    nu_0 = _BH_H[1] / _BH_B[1]
    return np.clip(nu, nu_0, 1 / _MU0)


def _build_geometry() -> Any:
    """Concentric-circle motor cross-section using OCC (avoids SplineGeometry hang)."""
    f_s = WorkPlane().Circle(_R_S).Face()
    f_s.edges.name = "outer"
    f_si = WorkPlane().Circle(_R_SI).Face()
    f_ro = WorkPlane().Circle(_R_RO).Face()
    f_ri = WorkPlane().Circle(_R_RI).Face()

    stator = f_s - f_si
    stator.faces.name = "stator"
    airgap = f_si - f_ro
    airgap.faces.name = "airgap"
    pm_reg = f_ro - f_ri
    pm_reg.faces.name = "pm"
    shaft = f_ri
    shaft.faces.name = "shaft"

    return Glue([stator, airgap, pm_reg, shaft])


def _build_slotted_geometry(
    n_slots: int,
    slot_width_ratio: float = 0.6,
    slot_depth: float = 0.15,
) -> Any:
    """Motor cross-section with n_slots open rectangular slots cut into the stator."""
    from math import cos, sin, pi as _pi

    f_s = WorkPlane().Circle(_R_S).Face()
    f_s.edges.name = "outer"
    f_si = WorkPlane().Circle(_R_SI).Face()
    f_ro = WorkPlane().Circle(_R_RO).Face()
    f_ri = WorkPlane().Circle(_R_RI).Face()

    airgap = f_si - f_ro
    airgap.faces.name = "airgap"
    pm_reg = f_ro - f_ri
    pm_reg.faces.name = "pm"
    shaft = f_ri
    shaft.faces.name = "shaft"

    stator = f_s - f_si

    delta      = slot_width_ratio * _pi / n_slots
    R_slot_out = _R_SI + slot_depth
    R_big      = 2.0 * _R_S

    slot_faces = []
    for k in range(n_slots):
        theta_k = 2 * _pi * k / n_slots

        wp = WorkPlane()
        wp.MoveTo(0., 0.)
        wp.LineTo(R_big * cos(theta_k - delta), R_big * sin(theta_k - delta))
        wp.LineTo(R_big * cos(theta_k + delta), R_big * sin(theta_k + delta))
        wp.Close()
        wedge = wp.Face()

        f_outer = WorkPlane().Circle(R_slot_out).Face()
        f_inner = WorkPlane().Circle(_R_SI).Face()
        sf = (wedge * f_outer) - f_inner
        sf.faces.name = "slot"

        stator = stator - sf
        slot_faces.append(sf)

    stator.faces.name = "stator"
    f_s.edges.name = "outer"

    return Glue([stator, *slot_faces, airgap, pm_reg, shaft])


def _derive_B_rem(psi_f: float, n_p: int, N: int, k_w: float, L_stk: float) -> float:
    """Invert psi_f to magnet remanence B_rem via Zhu & Howe transfer ratio."""
    tr = zhu_howe_Br(np.array([0.0]), n_p, B_rem=1.0, r_eval=_R_SI)[0]
    B_ag_peak = psi_f * n_p / (2 * N * k_w * _R_SI * L_stk)
    return B_ag_peak / tr


def zhu_howe_Br(
    theta: NDArray[np.floating],
    n_p: int,
    B_rem: float,
    r_eval: float = _R_AG,
) -> NDArray[np.floating]:
    """Zhu & Howe (1993) closed-form air-gap radial flux density.

    Solves the concentric-cylinder magnetostatic problem for sinusoidal
    radial magnetization cos(n_p θ) with infinite iron permeability at
    R_RI (shaft) and R_SI (stator bore). Returns B_r(theta) in Tesla.

    Raises ValueError if n_p < 2 (n_p=1 has a singularity in the
    particular solution).
    """
    if n_p < 2:
        raise ValueError(f"n_p={n_p} not supported (singularity at n²-1=0)")

    n = n_p
    p, q, s = _R_RI, _R_RO, _R_SI
    mu = _MU_R_PM
    r = r_eval

    # Particular solution coefficient: A_p = [n·B_rem/(n²-1)]·r·sin(nθ)
    bp = n * B_rem / (n**2 - 1)

    # D₁ = C₁·p^{2n} + B_rem·p^{n+1}/(n²-1)
    # Interface conditions at r = q (PM outer / air-gap boundary):
    #   Condition 1 (A continuous):
    #     C₁·[q^n + p^{2n}·q^{-n}] + bp·[p^{n+1}·q^{-n}/n + q]
    #       = C₂·[q^n + s^{2n}·q^{-n}]
    #   Condition 2 (H_θ continuous, factor of n cancelled):
    #     (1/μ)·{C₁·[q^{n-1} - p^{2n}·q^{-(n+1)}]
    #            + bp/n·[1 - p^{n+1}·q^{-(n+1)}]}
    #       = C₂·[q^{n-1} - s^{2n}·q^{-(n+1)}]
    a11 = q**n + p**(2*n) * q**(-n)
    a12 = -(q**n + s**(2*n) * q**(-n))
    b1 = -bp * (p**(n+1) * q**(-n) / n + q)

    a21 = (1/mu) * (q**(n-1) - p**(2*n) * q**(-(n+1)))
    a22 = -(q**(n-1) - s**(2*n) * q**(-(n+1)))
    b2 = -(bp/n) * (1/mu) * (1 - p**(n+1) * q**(-(n+1)))

    det = a11 * a22 - a12 * a21
    C2 = (a11 * b2 - a21 * b1) / det

    # B_r in air gap: C₂·n·(r^{n-1} + s^{2n}·r^{-(n+1)})·cos(nθ)
    K = C2 * n * (r**(n-1) + s**(2*n) * r**(-(n+1)))
    return K * np.cos(n * theta)


def _demoivre(n_p: int) -> tuple[CoefficientFunction, CoefficientFunction]:
    """Return (cos(n_p θ), sin(n_p θ)) as NGSolve CFs via De Moivre — no atan2 needed."""
    r_cf = sqrt(x**2 + y**2)
    cos_t, sin_t = x / r_cf, y / r_cf
    cr, si = cos_t, sin_t
    for _ in range(n_p - 1):
        cr, si = cr * cos_t - si * sin_t, cr * sin_t + si * cos_t
    return cr, si


@overload
def solve_field_fem(
    n_p: int, psi_f: float, L_d: float, L_q: float,
    n_theta: int = ..., maxh: float = ..., *,
    return_full: Literal[False] = ...,
    n_slots: int = ..., j_s: float = ...,
    N: int = ..., k_w: float = ..., L_stk: float = ...,
    slot_width_ratio: float = ...,
    nonlinear: bool = ..., max_picard: int = ...,
    picard_tol: float = ..., picard_relax: float = ...,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]: ...

@overload
def solve_field_fem(
    n_p: int, psi_f: float, L_d: float, L_q: float,
    n_theta: int = ..., maxh: float = ..., *,
    return_full: Literal[True] = ...,
    n_slots: int = ..., j_s: float = ...,
    N: int = ..., k_w: float = ..., L_stk: float = ...,
    slot_width_ratio: float = ...,
    nonlinear: bool = ..., max_picard: int = ...,
    picard_tol: float = ..., picard_relax: float = ...,
) -> tuple[NDArray[np.floating], NDArray[np.floating], Any, Any]: ...

def solve_field_fem(
    n_p: int,
    psi_f: float,
    L_d: float,
    L_q: float,
    n_theta: int = 360,
    maxh: float = 0.04,
    return_full: bool = False,
    n_slots: int = 0,
    j_s: float = 0.0,
    N: int = 50,
    k_w: float = 0.966,
    L_stk: float = 0.10,
    slot_width_ratio: float = 0.6,
    nonlinear: bool = False,
    max_picard: int = 15,
    picard_tol: float = 0.01,
    picard_relax: float = 0.3,
) -> tuple[NDArray[np.floating], NDArray[np.floating]] | tuple[NDArray[np.floating], NDArray[np.floating], Any, Any]:
    """2D magnetostatic FEM -> B_r(theta) at air-gap centre."""
    # NOTE: real IPM saliency requires flux barrier geometry, not modeled here
    mu_r_rotor = _MU_R_FE

    if n_slots > 0:
        shape = _build_slotted_geometry(n_slots, slot_width_ratio=slot_width_ratio)
        maxh = min(maxh, 0.03)
    else:
        shape = _build_geometry()
    mesh = Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh=maxh))

    nu_cf: dict[str, float] = {
        "stator": 1 / (_MU0 * _MU_R_FE),
        "airgap": 1 / _MU0,
        "pm":     1 / (_MU0 * _MU_R_PM),
        "shaft":  1 / (_MU0 * mu_r_rotor),
    }
    if n_slots > 0:
        nu_cf["slot"] = 1 / _MU0
    nu = mesh.MaterialCF(nu_cf)

    nu_pm = 1 / (_MU0 * _MU_R_PM)
    B_rem = _derive_B_rem(psi_f, n_p, N, k_w, L_stk)
    cos_np, sin_np = _demoivre(n_p)

    # Radial magnetization: M = B_rem cos(n_p θ) r̂
    # In Cartesian: M_x = cos(n_p θ)·cos(θ), M_y = cos(n_p θ)·sin(θ)
    r_cf = sqrt(x**2 + y**2)
    cos_t = x / r_cf
    sin_t = y / r_cf
    M_r = nu_pm * B_rem * cos_np
    mx_cf: dict[str, CoefficientFunction | float] = {"pm": M_r * cos_t, "stator": 0., "airgap": 0., "shaft": 0.}
    my_cf: dict[str, CoefficientFunction | float] = {"pm": M_r * sin_t, "stator": 0., "airgap": 0., "shaft": 0.}
    if n_slots > 0:
        mx_cf["slot"] = 0.
        my_cf["slot"] = 0.
    Mx = mesh.MaterialCF(mx_cf)
    My = mesh.MaterialCF(my_cf)

    fes = H1(mesh, order=3, dirichlet="outer")
    u, v = fes.TnT()

    lf_expr = (Mx * grad(v)[1] - My * grad(v)[0]) * dx("pm")
    if n_slots > 0 and j_s != 0.0:
        lf_expr = lf_expr + (-j_s * k_w * cos_np) * v * dx("slot")
    lf = LinearForm(lf_expr)
    lf.Assemble()
    gfu = GridFunction(fes)

    if not nonlinear:
        a = BilinearForm(nu * grad(u) * grad(v) * dx)
        a.Assemble()
        gfu.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky") * lf.vec
    else:
        fes_nu = L2(mesh, order=0)
        gfu_nu = GridFunction(fes_nu)
        gfu_nu.Set(nu)

        is_iron_cf: dict[str, float] = {
            "stator": 1.0, "shaft": 1.0, "airgap": 0.0, "pm": 0.0,
        }
        if n_slots > 0:
            is_iron_cf["slot"] = 0.0
        is_iron = mesh.MaterialCF(is_iron_cf)
        iron_indicator = GridFunction(fes_nu)
        iron_indicator.Set(is_iron)
        iron_mask = iron_indicator.vec.FV().NumPy() > 0.5

        a = BilinearForm(gfu_nu * grad(u) * grad(v) * dx)

        delta_rel = float("inf")
        for _picard_i in range(max_picard):
            a.Assemble()
            gfu.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky") * lf.vec

            B_mag_cf = sqrt(grad(gfu)[0]**2 + grad(gfu)[1]**2)
            gfu_Bmag = GridFunction(fes_nu)
            gfu_Bmag.Set(B_mag_cf)
            B_vals = gfu_Bmag.vec.FV().NumPy().copy()

            nu_new = _bh_nu(B_vals)
            nu_old = gfu_nu.vec.FV().NumPy().copy()

            nu_update = nu_old.copy()
            nu_update[iron_mask] = (
                picard_relax * nu_new[iron_mask]
                + (1 - picard_relax) * nu_old[iron_mask]
            )

            delta = np.max(np.abs(nu_update[iron_mask] - nu_old[iron_mask]))
            scale = np.max(np.abs(nu_old[iron_mask]))
            delta_rel = delta / scale

            gfu_nu.vec.FV().NumPy()[:] = nu_update

            if delta_rel < picard_tol:
                break
        else:
            raise RuntimeError(
                f"Picard iteration did not converge in {max_picard} iterations "
                f"(residual: {delta_rel:.4e}, tol: {picard_tol})"
            )

    # Check peak |B| in stator iron
    B_mag_check = sqrt(grad(gfu)[0]**2 + grad(gfu)[1]**2)
    fes_L2_check = L2(mesh, order=0)
    gfu_Bcheck = GridFunction(fes_L2_check)
    gfu_Bcheck.Set(B_mag_check)
    iron_ind = GridFunction(fes_L2_check)
    iron_cf_check: dict[str, float] = {"stator": 1.0, "shaft": 0.0, "airgap": 0.0, "pm": 0.0}
    if n_slots > 0:
        iron_cf_check["slot"] = 0.0
    iron_ind.Set(mesh.MaterialCF(iron_cf_check))
    B_iron = gfu_Bcheck.vec.FV().NumPy()[iron_ind.vec.FV().NumPy() > 0.5]
    if len(B_iron) > 0 and np.max(B_iron) > 1.8:
        print(f"WARNING: peak |B| in stator iron = {np.max(B_iron):.2f} T "
              f"(> 1.8 T saturation threshold)", file=sys.stderr)

    r_cf = sqrt(x**2 + y**2)
    B_r_cf = (grad(gfu)[1] * x - grad(gfu)[0] * y) / r_cf

    fes_L2 = L2(mesh, order=3)
    gfu_Br = GridFunction(fes_L2)
    gfu_Br.Set(B_r_cf)

    theta = np.linspace(0, 2 * pi, n_theta, endpoint=False)
    cos_t_np = np.cos(theta)
    sin_t_np = np.sin(theta)
    B_r = np.array([
        gfu_Br(mesh(_R_AG * c, _R_AG * s))
        for c, s in zip(cos_t_np, sin_t_np)
    ])
    if return_full:
        return theta, B_r, mesh, gfu
    return theta, B_r


def rasterise_cross_section(
    mesh: Mesh,
    gfu: GridFunction,
    n_grid: int = 100,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Evaluate A_z and |B| on a regular Cartesian grid covering the motor disk."""
    B_mag_cf = sqrt(grad(gfu)[0]**2 + grad(gfu)[1]**2)
    fes_L2 = L2(mesh, order=3)
    gfu_Bmag = GridFunction(fes_L2)
    gfu_Bmag.Set(B_mag_cf)

    coords = np.linspace(-_R_S, _R_S, n_grid)
    xi, yi = np.meshgrid(coords, coords)

    Az   = np.full((n_grid, n_grid), np.nan)
    Bmag = np.full((n_grid, n_grid), np.nan)

    r2 = xi**2 + yi**2
    inside = r2 < _R_S**2

    failed = 0
    for i in range(n_grid):
        for j in range(n_grid):
            if not inside[i, j]:
                continue
            try:
                mp = mesh(xi[i, j], yi[i, j])
                Az[i, j]   = gfu(mp)
                Bmag[i, j] = gfu_Bmag(mp)
            except Exception:
                failed += 1

    if failed:
        print(f"  rasterise: {failed} points failed evaluation")

    return xi, yi, Az, Bmag


def harmonics_1sided(B_r: NDArray[np.floating]) -> NDArray[np.floating]:
    """One-sided FFT harmonic amplitudes for a single B_r waveform (NumPy)."""
    n_pts = len(B_r)
    amps = np.abs(np.fft.rfft(B_r)) / n_pts
    amps[1:-1] *= 2
    return amps


def batch_harmonics(B_r_dict: dict[str, NDArray[np.floating]]) -> dict[str, NDArray[np.floating]]:
    """One-sided FFT harmonic amplitudes. CuPy (GPU) if available, else NumPy."""
    names = list(B_r_dict.keys())
    arr = np.stack([B_r_dict[name] for name in names])
    n_pts = arr.shape[-1]

    if _CUPY:
        assert cp is not None
        arr_gpu = cp.asarray(arr)
        amps = cp.asnumpy(cp.abs(cp.fft.rfft(arr_gpu, axis=-1))) / n_pts
    else:
        amps = np.abs(np.fft.rfft(arr, axis=-1)) / n_pts

    amps[:, 1:-1] *= 2
    return {name: amps[i] for i, name in enumerate(names)}
