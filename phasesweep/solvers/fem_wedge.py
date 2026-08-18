"""3D wedge end-effect (k_end) FEM — scalar magnetic potential formulation.

k_end is intrinsically no-load: J_free = 0 ⟹ ∇×H = 0 ⟹ H = −∇φ valid in air,
iron, AND magnet (bound current ∇×M is in the constitutive law, not J_free). So
the scalar potential φ (H1, no gauge, no edge elements, no Simkin–Trowbridge
cancellation) is the natural 3D analog of the trusted 2D A_z solver.

Weak form (total potential, B = −μ∇φ + B_rem, ∇·B = 0):
    ∫ μ ∇φ·∇v dx = ∫_magnet B_rem·∇v dx     for all v with v = 0 on Dirichlet faces.
Natural (do-nothing) BC ⟺ B·n = 0 (flux-parallel). Dirichlet φ = const ⟺ ideal
iron pole face (equipotential).

Staged build:
  Stage 0 (here): iron-backed magnet + air-gap slab with an exact analytic answer,
                  validating source sign/magnitude + the Dirichlet iron interface.
  Stage 1: half-pole smooth-bore wedge, axial-Neumann caps ⟹ 2D ≈ analytic Zhu-Howe.
  Stage 2: short stack + end-air box, convergence (solve_wedge_endeffect).
  Stage 3: magnet/yoke overhang + one-shot k_end calculator (compute_k_end).

Uses NGSolve (LGPL). Analytical gate: phasesweep.analytical.zhu_howe_Br.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin
from typing import TYPE_CHECKING

import numpy as np

from phasesweep.solvers.analytical import zhu_howe_Br
from phasesweep.solvers.fem_field import _MU0

if TYPE_CHECKING:
    from ngsolve import CoefficientFunction, GridFunction, Mesh
    from ngsolve.comp import FESpace, LinearForm

    from phasesweep.machines.geometry import Geometry


def slab_analytic(B_rem: float, mu_r_pm: float, h_m: float, g: float) -> float:
    """Exact 1D iron-backed magnet + air-gap flux density.

    Closed magnetic circuit (ideal iron pole faces at both ends): Ampere's loop
    H_m·h_m + H_g·g = 0 with B continuous gives B_gap = B_rem / (1 + μ_r·g/h_m).
    """
    return B_rem / (1.0 + mu_r_pm * g / h_m)


@dataclass(frozen=True)
class SlabResult:
    B_gap_fem: float
    B_gap_analytic: float
    rel_error: float


def solve_slab(
    B_rem: float = 1.45,
    mu_r_pm: float = 1.05,
    h_m: float = 1.287e-3,
    g: float = 0.528e-3,
    *,
    width: float = 1.0e-3,
    maxh: float | None = None,
    order: int = 3,
) -> SlabResult:
    """Stage 0: scalar-φ solve of an iron-backed magnet + air-gap slab.

    Thin 3D box along x: magnet [0, h_m] (radial remanence +x), air gap [h_m, L].
    Dirichlet φ = 0 on the two x-end faces (ideal iron pole faces); lateral faces
    take the natural B·n = 0 BC. Compares B_x at the gap mid-plane against
    slab_analytic.
    """
    from netgen.occ import Box, Glue, OCCGeometry, Pnt, X
    from ngsolve import (
        H1,
        BilinearForm,
        GridFunction,
        LinearForm,
        Mesh,
        dx,
        grad,
    )

    L = h_m + g
    if maxh is None:
        maxh = min(h_m, g) / 4

    magnet = Box(Pnt(0, 0, 0), Pnt(h_m, width, width))
    magnet.mat("magnet")
    gap = Box(Pnt(h_m, 0, 0), Pnt(L, width, width))
    gap.mat("gap")

    shape = Glue([magnet, gap])
    shape.faces.Min(X).name = "left"
    shape.faces.Max(X).name = "right"

    mesh = Mesh(OCCGeometry(shape).GenerateMesh(maxh=maxh))

    mu = mesh.MaterialCF({"magnet": _MU0 * mu_r_pm, "gap": _MU0})
    Brem_x = mesh.MaterialCF({"magnet": B_rem, "gap": 0.0})

    fes = H1(mesh, order=order, dirichlet="left|right")
    u, v = fes.TnT()

    a = BilinearForm(mu * grad(u) * grad(v) * dx)
    a.Assemble()
    lf = LinearForm(Brem_x * grad(v)[0] * dx)
    lf.Assemble()

    gfu = GridFunction(fes)
    gfu.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky") * lf.vec

    # Physical B_x = −μ ∂φ/∂x + B_rem,x; B_rem,x = 0 in the gap.
    B_x = -_MU0 * grad(gfu)[0]
    B_gap_fem = B_x(mesh(h_m + g / 2, width / 2, width / 2))

    B_gap_analytic = slab_analytic(B_rem, mu_r_pm, h_m, g)
    rel_error = abs(B_gap_fem - B_gap_analytic) / B_gap_analytic
    return SlabResult(B_gap_fem, B_gap_analytic, rel_error)


# ---------------------------------------------------------------------------
# Stage 1 — half-pole smooth-bore wedge, axial-Neumann caps (2D-equivalent)
# ---------------------------------------------------------------------------

def _radial_bands(geo: Geometry) -> list[tuple[float, float, str, str]]:
    """Radial bands (r_in, r_out, material, role) inner→outer for a half-pole wedge.

    Generic over topology: iron / magnet / gap / iron in physical order. ``role``
    distinguishes the rotor-side iron (yoke, follows the magnet axially) from the
    stator-side iron (active stack length only).
    """
    if geo.topology == "inrunner":
        return [
            (geo.r_inner, geo.r_rotor, "iron", "rotor"),
            (geo.r_rotor, geo.r_magnet, "magnet", "magnet"),
            (geo.r_magnet, geo.r_stator, "gap", "gap"),
            (geo.r_stator, geo.r_outer, "iron", "stator"),
        ]
    return [
        (geo.r_inner, geo.r_stator, "iron", "stator"),
        (geo.r_stator, geo.r_magnet, "gap", "gap"),
        (geo.r_magnet, geo.r_rotor, "magnet", "magnet"),
        (geo.r_rotor, geo.r_outer, "iron", "rotor"),
    ]


def _sector_ring(alpha: float, r_in: float, r_out: float, r_big: float):
    """2D half-pole annular-sector face θ∈[0, alpha], radius [r_in, r_out]."""
    from netgen.occ import WorkPlane

    wp = WorkPlane()
    wp.MoveTo(0.0, 0.0)
    wp.LineTo(r_big, 0.0)
    wp.LineTo(r_big * cos(alpha), r_big * sin(alpha))
    wp.Close()
    face = wp.Face() * WorkPlane().Circle(r_out).Face()
    if r_in > 1e-9:
        face = face - WorkPlane().Circle(r_in).Face()
    return face


def _build_wedge(geo: Geometry, alpha: float, Lz: float):
    """Half-pole annular wedge θ∈[0, alpha], extruded along z by Lz.

    The pole-boundary plane θ=alpha is named "dpole" (Dirichlet φ=0). All other
    faces (pole-center θ=0, radial iron faces, axial caps) take the natural BC.
    """
    from netgen.occ import Glue, Pnt

    r_big = 2.0 * geo.r_outer

    solids = []
    for r_in, r_out, mat, _role in _radial_bands(geo):
        sol = _sector_ring(alpha, r_in, r_out, r_big).Extrude(Lz)
        sol.mat(mat)
        r_mid = 0.5 * (r_in + r_out)
        pole_pt = Pnt(r_mid * cos(alpha), r_mid * sin(alpha), Lz / 2)
        sol.faces.Nearest(pole_pt).name = "dpole"
        solids.append(sol)

    return Glue(solids)


def _fundamental_Br(
    mesh: Mesh, B_r_cf: CoefficientFunction, r_ag: float,
    n_p: int, alpha: float, z: float, n_sample: int,
) -> float:
    """Fundamental amplitude of B_r(θ) at radius r_ag, height z, over [0, alpha]."""
    thetas = np.linspace(0.0, alpha, n_sample)
    B_r = np.array([
        B_r_cf(mesh(r_ag * cos(t), r_ag * sin(t), z)) for t in thetas
    ])
    cos_np = np.cos(n_p * thetas)
    return np.trapezoid(B_r * cos_np, thetas) / np.trapezoid(cos_np**2, thetas)


def _halfpole_flux_Br(
    mesh: Mesh, B_r_cf: CoefficientFunction, r: float,
    alpha: float, z: float, n_sample: int,
) -> float:
    """Net (all-harmonic) radial flux per unit length through the half-pole arc."""
    thetas = np.linspace(0.0, alpha, n_sample)
    B_r = np.array([B_r_cf(mesh(r * cos(t), r * sin(t), z)) for t in thetas])
    return r * np.trapezoid(B_r, thetas)


def _picard_solve_wedge(
    mesh: Mesh, fes: FESpace, lf: LinearForm, mu_lin: CoefficientFunction,
    n_iter: int | None = None,
    tol: float = 1e-4, relax: float = 0.3,
) -> GridFunction:
    """Nonlinear iron Picard for the scalar-φ wedge: μ_iron = μ(|B|) from the BH
    curve. Returns the solved GridFunction. μ_lin is the linear MaterialCF (used
    for non-iron); iron μ is iterated on an L2 field. |B| = μ·|∇φ| in iron.

    Adaptive relaxation ported from the 2D ``_picard_solve``: stop on the
    raw (unrelaxed) step, damp relax on raw-step growth with the applied
    step capped at the previous applied step, recover
    toward the configured relax on decay. The fixed relax = 0.3 this had
    before oscillates on the saturating end-region (step pinned ~0.9 at
    the fast test settings) — and the old fall-through then returned that
    non-converged field silently.

    Raises:
        RuntimeError: If the iteration does not converge within ``n_iter``
            iterations (mirrors the 2D ``_picard_solve`` contract — a
            non-converged field must not be returned silently).
    """
    from ngsolve import L2, BilinearForm, GridFunction, dx, grad, sqrt

    from phasesweep.defaults import PICARD_MAX_ITERATIONS
    from phasesweep.solvers.fem_field import _MIN_PICARD_RELAX, _bh_nu

    if n_iter is None:
        n_iter = PICARD_MAX_ITERATIONS  # same budget as the 2D solver (200)

    u, v = fes.TnT()
    fes_mu = L2(mesh, order=0)
    gfu_mu = GridFunction(fes_mu)
    gfu_mu.Set(mu_lin)
    is_iron = GridFunction(fes_mu)
    is_iron.Set(mesh.MaterialCF({"iron": 1.0, "magnet": 0.0, "gap": 0.0}))
    iron = is_iron.vec.FV().NumPy() > 0.5

    gfu = GridFunction(fes)
    gradmag = GridFunction(fes_mu)
    relax_cfg = relax
    prev_raw = float("inf")
    prev_applied = float("inf")
    for _ in range(n_iter):
        a = BilinearForm(gfu_mu * grad(u) * grad(v) * dx)
        a.Assemble()
        gfu.vec.data = (
            a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky") * lf.vec
        )
        gradmag.Set(sqrt(grad(gfu)[0] ** 2 + grad(gfu)[1] ** 2 + grad(gfu)[2] ** 2))
        mu_old = gfu_mu.vec.FV().NumPy().copy()
        B_iron = mu_old[iron] * gradmag.vec.FV().NumPy()[iron]
        mu_new = 1.0 / _bh_nu(B_iron)
        step = np.max(np.abs(mu_new - mu_old[iron])) / np.max(np.abs(mu_old[iron]))
        if step < tol:
            break
        if step > prev_raw:
            relax = max(relax * 0.5, _MIN_PICARD_RELAX)
            while relax * step > prev_applied and relax > _MIN_PICARD_RELAX:
                relax = max(relax * 0.5, _MIN_PICARD_RELAX)
        else:
            relax = min(relax * 1.5, relax_cfg)
        upd = mu_old.copy()
        upd[iron] = relax * mu_new + (1 - relax) * mu_old[iron]
        gfu_mu.vec.FV().NumPy()[:] = upd
        prev_applied = relax * step
        prev_raw = step
    else:
        raise RuntimeError(
            f"Wedge Picard iteration did not converge in {n_iter} "
            f"iterations (step: {step:.4e}, tol: {tol})"
        )
    return gfu


@dataclass(frozen=True)
class WedgeResult:
    b1_fem: float
    b1_analytic: float
    rel_error: float


def solve_wedge_2d_equiv(
    geo: Geometry,
    n_p: int,
    B_rem: float = 1.45,
    *,
    mu_r_pm: float = 1.05,
    mu_r_fe: float = 1.0e4,
    Lz: float | None = None,
    maxh: float | None = None,
    order: int = 3,
    n_sample: int = 181,
) -> WedgeResult:
    """Stage 1: half-pole wedge with axial-Neumann caps → recovers the 2D field.

    Axial caps both take the natural B·n=0 BC, so with z-invariant geometry/source
    the solution is z-independent — i.e. the 2D smooth-bore field. Extracts the
    fundamental of B_r at r_ag (mid-plane) and gates it against zhu_howe_Br.
    """
    from netgen.occ import OCCGeometry
    from ngsolve import (
        H1,
        BilinearForm,
        GridFunction,
        LinearForm,
        Mesh,
        dx,
        grad,
        sqrt,
        x,
        y,
    )

    alpha = pi / (2 * n_p)
    gap = abs(geo.r_stator - geo.r_magnet)
    if Lz is None:
        Lz = 2.0 * gap
    if maxh is None:
        maxh = gap / 3.0

    shape = _build_wedge(geo, alpha, Lz)
    mesh = Mesh(OCCGeometry(shape).GenerateMesh(maxh=maxh))

    mu = mesh.MaterialCF(
        {"magnet": _MU0 * mu_r_pm, "gap": _MU0, "iron": _MU0 * mu_r_fe}
    )
    r_cf = sqrt(x * x + y * y)
    cos_t, sin_t = x / r_cf, y / r_cf
    # Single-sign radial remanence over the half-pole (B_rem outward, +r̂).
    Brem_r = mesh.MaterialCF({"magnet": B_rem, "gap": 0.0, "iron": 0.0})

    fes = H1(mesh, order=order, dirichlet="dpole")
    u, v = fes.TnT()

    a = BilinearForm(mu * grad(u) * grad(v) * dx)
    a.Assemble()
    lf = LinearForm(
        Brem_r * (cos_t * grad(v)[0] + sin_t * grad(v)[1]) * dx
    )
    lf.Assemble()

    gfu = GridFunction(fes)
    gfu.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky") * lf.vec

    # Sample B_r in the gap at r_ag, mid-plane; project onto cos(n_p θ).
    B_r_cf = -_MU0 * (cos_t * grad(gfu)[0] + sin_t * grad(gfu)[1])
    b1_fem = _fundamental_Br(mesh, B_r_cf, geo.r_ag, n_p, alpha, Lz / 2, n_sample)

    b1_analytic = float(zhu_howe_Br(
        np.array([0.0]), n_p, B_rem, r_eval=geo.r_ag,
        r_stator=geo.r_stator, r_magnet=geo.r_magnet, r_rotor=geo.r_rotor,
        mu_r_pm=mu_r_pm, alpha_p=1.0,
    )[0])

    rel_error = abs(abs(b1_fem) - abs(b1_analytic)) / abs(b1_analytic)
    return WedgeResult(b1_fem, b1_analytic, rel_error)


# ---------------------------------------------------------------------------
# Stage 2 — finite half-stack + end-air box (axial end effect)
# ---------------------------------------------------------------------------

def _build_wedge_endeffect(
    geo: Geometry, alpha: float, L_half: float, z_box: float,
    mag_overhang: float = 0.0, yoke_overhang: float = 0.0,
    maxh_fine: float | None = None,
):
    """Half-pole wedge: finite active stack + axial overhang, end-air to z_box.

    Each iron/magnet band occupies z∈[0, length] from the mid-plane; the airgap
    is an explicit thin solid (full z_box) and air fills the rest via boolean
    subtraction. Axial lengths by role: stator iron = L_half (active stack);
    magnet = L_half + mag_overhang; rotor iron (yoke) = L_half + yoke_overhang.
    The stator iron ending at L_half is what drives the end-effect droop; overhang
    lets the magnet/yoke extend past it.

    maxh_fine (if given) is applied locally to the airgap + magnet solids only, so
    the gap resolution that sets accuracy is decoupled from the coarse bulk mesh —
    otherwise a global gap-scaled maxh over-refines the whole domain on tight-gap
    motors (element count ∝ 1/gap³ → OOM).

    z=0 is the axial mid-plane (natural BC ⟹ B_z=0 by symmetry). Faces named:
    "dpole" (θ=alpha, Dirichlet φ=0), "outer" (r=r_outer) and "top" (z=z_box) —
    the latter two take a BC chosen by the caller.
    """
    from math import sqrt

    from netgen.occ import Glue

    z_max = L_half + max(mag_overhang, yoke_overhang)
    if z_max >= z_box:
        raise ValueError(
            f"magnet/yoke overhang reaches the end-air cap: L_half + "
            f"overhang = {z_max * 1e3:.3g} mm >= z_box = {z_box * 1e3:.3g} mm "
            f"— the end field cannot be represented; increase z_box_factor"
        )

    r_big = 2.0 * geo.r_outer
    r_out_max = geo.r_outer
    length = {
        "stator": L_half,
        "magnet": L_half + mag_overhang,
        "rotor": L_half + yoke_overhang,
    }

    solids = []
    for r_in, r_out, mat, role in _radial_bands(geo):
        if mat == "gap":
            continue  # the airgap is built as an explicit solid below
        sol = _sector_ring(alpha, r_in, r_out, r_big).Extrude(length[role])
        sol.mat(mat)
        if role == "magnet" and maxh_fine is not None:
            sol.maxh = maxh_fine
        solids.append(sol)

    # Explicit airgap solid (full z_box) so it can carry the fine local mesh that
    # resolves the radial field and the stack-end corner.
    g_lo, g_hi = sorted((geo.r_stator, geo.r_magnet))
    gap_sol = _sector_ring(alpha, g_lo, g_hi, r_big).Extrude(z_box)
    gap_sol.mat("gap")
    if maxh_fine is not None:
        gap_sol.maxh = maxh_fine

    air = _sector_ring(alpha, 0.0, r_out_max, r_big).Extrude(z_box)
    for sol in (*solids, gap_sol):
        air = air - sol
    air.mat("gap")
    shape = Glue([*solids, gap_sol, air])

    # Name boundary faces by geometry: θ=alpha plane (dpole), r=r_outer (outer),
    # z=z_box far-axial cap (top).
    for f in shape.faces:
        c = f.center
        r_c = sqrt(c[0] ** 2 + c[1] ** 2)
        if r_c > 1e-9 and abs(c[0] * sin(alpha) - c[1] * cos(alpha)) < 1e-6 * r_big:
            f.name = "dpole"
            continue
        if all(v.p[2] > z_box - 1e-9 for v in f.vertices):
            f.name = "top"
            continue
        vr = [sqrt(v.p[0] ** 2 + v.p[1] ** 2) for v in f.vertices]
        if vr and min(vr) > r_out_max - 1e-6:
            f.name = "outer"

    return shape


@dataclass(frozen=True)
class EndEffectResult:
    z: np.ndarray
    b3: np.ndarray
    b2_analytic: float
    k_end: float           # fundamental of B_r at r_ag ÷ analytic 2D fundamental
    k_end_bore: float = 0.0      # like k_end but at the stator bore (radius sensitivity)
    # NOTE: different baseline than k_end — all-harmonic half-pole flux,
    # stack-mean ÷ the FEM mid-plane (z=0), i.e. a pure axial-droop ratio against
    # the solve itself, not against the analytic 2D value. Coincides with k_end
    # only when the mid-plane already matches 2D (long/co-extensive stacks).
    flux_droop_ratio: float = 0.0


def solve_wedge_endeffect(
    geo: Geometry,
    n_p: int,
    L_stk: float,
    B_rem: float = 1.45,
    *,
    mag_overhang: float = 0.0,
    yoke_overhang: float = 0.0,
    mu_r_pm: float = 1.05,
    mu_r_fe: float = 1.0e4,
    nonlinear: bool = False,
    z_box_factor: float = 6.0,
    outer_bc: str = "neumann",
    top_bc: str = "neumann",
    maxh: float | None = None,
    maxh_bulk: float | None = None,
    order: int = 2,
    n_z: int = 97,
    n_theta: int = 31,
    picard_n_iter: int | None = None,
    picard_tol: float = 1e-4,
) -> EndEffectResult:
    """Stage 2/3: end-effect b3(z) profile and k_end for a finite stack.

    k_end = 2∫₀^{L_stk/2} b3(z)dz / (b2·L_stk), b2 = analytic Zhu-Howe fundamental.
    The integration limit is the active stack L_stk/2 (NOT z_box): fringing beyond
    the stack threads r_ag but does not link the winding (audit HIGH finding).

    mag_overhang / yoke_overhang: axial extension (per side) of the magnet and the
    rotor yoke beyond the stack end — both 0 for a co-extensive stack.
    nonlinear: solve the iron with the BH curve (Picard) instead of linear mu_r_fe.
    picard_n_iter / picard_tol: Picard budget and raw-step stop tolerance
    (defaults: the 2D solver's 200-iteration budget, 1e-4). Non-convergence
    raises — pass a looser tol for coarse studies rather than
    swallowing a bad field.
    maxh: fine mesh size local to the airgap + magnet (default gap/3, sets accuracy).
    maxh_bulk: coarse mesh size for iron/end-air (default r_outer/12) — decoupled
    from the gap so tight-gap motors don't over-refine the whole domain.
    outer_bc / top_bc: "neumann" (flux-parallel) or "dirichlet" (φ=0) at r_outer /
    the far axial cap z_box — exposed for the convergence/truncation studies.

    Also reports k_end_bore (fundamental evaluated at the stator bore instead of
    mid-gap — radius sensitivity) and flux_droop_ratio (all-harmonic half-pole flux,
    stack-mean ÷ FEM mid-plane). flux_droop_ratio uses a different denominator than
    k_end (FEM mid-plane, not analytic 2D), so the two coincide only when the
    mid-plane already matches the 2D value; do not treat them as interchangeable.
    """
    from netgen.occ import OCCGeometry
    from ngsolve import (
        H1,
        BilinearForm,
        GridFunction,
        LinearForm,
        Mesh,
        dx,
        grad,
        sqrt,
        x,
        y,
    )

    alpha = pi / (2 * n_p)
    L_half = L_stk / 2.0
    gap = abs(geo.r_stator - geo.r_magnet)
    z_box = L_half + z_box_factor * gap
    if maxh is None:
        maxh = gap / 3.0  # fine, local to the airgap + magnet
    if maxh_bulk is None:
        maxh_bulk = geo.r_outer / 12.0  # coarse bulk, decoupled from gap

    shape = _build_wedge_endeffect(
        geo, alpha, L_half, z_box, mag_overhang, yoke_overhang, maxh_fine=maxh
    )
    mesh = Mesh(OCCGeometry(shape).GenerateMesh(maxh=maxh_bulk))

    mu = mesh.MaterialCF(
        {"magnet": _MU0 * mu_r_pm, "gap": _MU0, "iron": _MU0 * mu_r_fe}
    )
    r_cf = sqrt(x * x + y * y)
    cos_t, sin_t = x / r_cf, y / r_cf
    Brem_r = mesh.MaterialCF({"magnet": B_rem, "gap": 0.0, "iron": 0.0})

    parts = ["dpole"]
    if outer_bc == "dirichlet":
        parts.append("outer")
    if top_bc == "dirichlet":
        parts.append("top")
    fes = H1(mesh, order=order, dirichlet="|".join(parts))
    u, v = fes.TnT()

    lf = LinearForm(Brem_r * (cos_t * grad(v)[0] + sin_t * grad(v)[1]) * dx)
    lf.Assemble()

    if nonlinear:
        gfu = _picard_solve_wedge(mesh, fes, lf, mu,
                                  n_iter=picard_n_iter, tol=picard_tol)
    else:
        a = BilinearForm(mu * grad(u) * grad(v) * dx)
        a.Assemble()
        gfu = GridFunction(fes)
        gfu.vec.data = (
            a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky") * lf.vec
        )

    # B in the gap is -mu0*grad(phi) regardless of iron model (air region).
    B_r_cf = -_MU0 * (cos_t * grad(gfu)[0] + sin_t * grad(gfu)[1])
    z = np.linspace(0.0, L_half, n_z)

    def b2_at(r):
        return abs(float(zhu_howe_Br(
            np.array([0.0]), n_p, B_rem, r_eval=r,
            r_stator=geo.r_stator, r_magnet=geo.r_magnet, r_rotor=geo.r_rotor,
            mu_r_pm=mu_r_pm, alpha_p=1.0,
        )[0]))

    b3 = np.array([
        abs(_fundamental_Br(mesh, B_r_cf, geo.r_ag, n_p, alpha, zi, n_theta))
        for zi in z
    ])
    b2_analytic = b2_at(geo.r_ag)
    k_end = float(np.trapezoid(b3, z) / (b2_analytic * L_half))

    # Radius sensitivity: fundamental at the stator bore (where flux enters iron).
    sgn = 1.0 if geo.topology == "outrunner" else -1.0
    r_bore = geo.r_stator + sgn * 0.05 * gap
    b3_bore = np.array([
        abs(_fundamental_Br(mesh, B_r_cf, r_bore, n_p, alpha, zi, n_theta))
        for zi in z
    ])
    k_end_bore = float(np.trapezoid(b3_bore, z) / (b2_at(r_bore) * L_half))

    # Axial-droop ratio: all-harmonic half-pole flux, stack-mean ÷ FEM mid-plane.
    # Different baseline than k_end (mid-plane, not analytic 2D) — see field note.
    phi = np.array([
        abs(_halfpole_flux_Br(mesh, B_r_cf, geo.r_ag, alpha, zi, n_theta))
        for zi in z
    ])
    flux_droop_ratio = float(np.trapezoid(phi, z) / (phi[0] * L_half))

    return EndEffectResult(z, b3, b2_analytic, k_end, k_end_bore, flux_droop_ratio)


# ---------------------------------------------------------------------------
# Stage 3 — one-shot k_end calculator
# ---------------------------------------------------------------------------

def compute_k_end(
    geo: Geometry,
    n_p: int,
    L_stk: float,
    *,
    overhang: float = 0.0,
    yoke_overhang: float | None = None,
    B_rem: float = 1.45,
    **kwargs,
) -> float:
    """3D end-effect factor k_end for the winding flux linkage (no-load).

    Range (0, ~1.05]: co-extensive magnets give k_end < 1, but magnet overhang
    concentrates flux and pushes k_end above 1 (empirically ~1.03 at saturation
    on the 14 mm outrunner). Not clamped — the >1 regime is real, not an artifact.

    One-shot calculator: run per motor, store the result. ``overhang`` is the
    magnet axial extension per side beyond the active stack; ``yoke_overhang``
    defaults to it (rotor cup follows the magnet) — pass 0 for a yoke co-extensive
    with the stack. B_rem cancels in the ratio, so its value does not matter. Pass
    ``nonlinear=True`` for BH iron (lowers k_end a uniform ~4.3–4.4 pt
    below linear across 0–1 mm overhang on the 14 mm outrunner, from converged
    runs — the earlier "~2%" came from silently non-converged solves;
    recommended for the headline).

    Idealizations: smooth-bore,
    full-pitch (alpha_p=1) magnet, distributed-winding fundamental, and flux linkage
    truncated at the active stack L_stk/2 (end-winding linkage excluded). Verified
    against the 2D analytic limit + convergence, NOT an independent 3D benchmark.
    """
    if yoke_overhang is None:
        yoke_overhang = overhang
    return solve_wedge_endeffect(
        geo, n_p, L_stk, B_rem,
        mag_overhang=overhang, yoke_overhang=yoke_overhang, **kwargs,
    ).k_end
