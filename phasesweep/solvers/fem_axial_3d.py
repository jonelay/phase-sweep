"""3D scalar-potential axial-flux edge-leakage study (Track B, NOT registered).

Research code: an offline validation study, not a registry model. It is
exercised by scripts and its (slow-marked) tests only; its API is not part
of the package's public contract and may change without a version bump.

Offline validation calculator in the ``fem_wedge`` pattern: it is the CV-4
arbiter for the radial-edge leakage factor ``k_sigma_edge`` used by the
shipped ``fem_axial`` model, and is never a registry dependency.

Physics — no-load (J_free = 0) ⟹ H = −∇φ everywhere; H1 scalar potential,
same weak form as fem_wedge (total potential, B = −μ∇φ + B_rem):

    ∫ μ ∇φ·∇v dx = ∫_magnet B_rem·∇v dx        v = 0 on Dirichlet faces

with axial magnetization M_z = B_rem/μ0 in the magnet, so the source is
``B_rem·∂v/∂z``. Linear iron only; slotless stator (plain iron slab).

Domain — half-pole annular sector of the ONE-GAP axial stack, θ ∈ [0, α],
α = π/(2·n_p) (mesh economy: same Dirichlet/Neumann half-pole symmetry as
fem_wedge). Axial stack bottom→top (meters): rotor yoke iron / PM (+z
magnetized, θ ∈ [0, alpha_p·α], interpole air beyond) / air gap (physical
``geo.airgap``, no Carter — slotless) / stator iron slab. Shared-yoke rule
(plan audit F2): the yoke shared between two gap circuits is HALVED in the
one-gap stack — rotor yoke for 1R2S, stator yoke for 2R1S-NN — with the cut
plane taking the natural B·n = 0 BC, i.e. the mid-yoke mirror plane.

Boundary conditions:
  θ = α (pole boundary, mid-way between N and S magnets): Dirichlet φ = 0 —
      the anti-periodicity substitute (φ odd across the plane, audit-verified
      dual of Track A's d-axis A_z = 0).
  θ = 0 (magnet center): natural ⟹ B_θ = 0, the mirror-symmetry plane.
  Everywhere else: natural (flux-parallel B·n = 0).

Two model flavors:
  finite  — [audit F1] air collars extend the domain radially to
      r_in − w and r_out + w with w = collar_factor·(airgap +
      magnet_thickness), collar_factor ≥ 4 by default, giving the radial
      leakage path around the magnet edges. Far-field BC at the collar
      outer radii (and all outer z faces) is the flux-parallel natural
      truncation B·n = 0; its error is checked by the domain-extent
      convergence on the leakage DEFICIT itself (not just on ψ) —
      see tests/solvers/test_fem_axial_3d.py.
  extruded — control model: the same axial stack with NO collars and
      natural (Neumann) walls at r = r_in and r = r_out, so no radial flux
      leaves the active band — the radially-unbounded ("extruded") limit.
      Reproduces the 2D plane result (CV-1 gate vs the layer series) and is
      the subtrahend for CV-4.

Outputs — B1(r): fundamental of B_z(θ) on the mid-gap surface, projected
onto cos(n_p·θ) per radius; ψ-equivalent geometric flux integral
ψ_geo = ∫ B1(r)·r dr over [r_in, r_out] (the winding aperture — matches the
multi-plane ψ sum in Track A up to the shared winding constants); net
half-pole flux Φ = ∫∫ B_z·r dr dθ. CV-4 deficit ratio = ψ_geo(finite) /
ψ_geo(extruded), compared against ``radial_edge_leakage_factor`` (the
Qu & Lipo magnet-to-rotor form evaluated with L_a and the physical gap —
computed here, independent of Track A).

Uses NGSolve (LGPL). Analytical gates: axial_layer_Bz_series (via
tests/crossval/axial_fem_contract.py slotless_reference_B1).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log, pi
from typing import TYPE_CHECKING

import numpy as np

from phasesweep.solvers.fem_materials import MU0
from phasesweep.solvers.fem_wedge import sector_ring

if TYPE_CHECKING:
    from ngsolve import CoefficientFunction, Mesh

    from phasesweep.machines.geometry import AxialFluxGeometry

# Minimum collar_factor accepted by the finite model — below this the
# flux-parallel truncation sits inside the leakage field and the deficit
# is not converged (plan audit F1: >= 4 gap+magnet lengths).
__all__ = [
    "MIN_COLLAR_FACTOR",
    "AxialSectorResult",
    "EdgeLeakageResult",
    "compute_edge_leakage",
    "radial_edge_leakage_factor",
    "solve_axial_sector",
]

MIN_COLLAR_FACTOR = 4.0


def radial_edge_leakage_factor(
    g: float, l_pm: float, active_length: float, mu_r_pm: float,
) -> float:
    """Qu & Lipo (2004) magnet-to-rotor leakage factor, radial-edge form.

    The plan's k_sigma_edge: the magnet-to-rotor term of the air-gap
    leakage (analytical_axial.pm_leakage_factor Eq. 2.15 lineage) with the
    magnet width replaced by the radial active length L_a = r_out − r_in,
    the PHYSICAL air gap g (no Carter), and the magnet-to-magnet / zigzag /
    k_add terms dropped (the in-plane FEM already captures those):

        k = 1 / (1 + μ_r·(g/l_pm)·(L_a/(L_a + 2g))
                     ·(2·l_pm/(π·μ_r·L_a))·ln(1 + π·g/l_pm))

    All lengths in meters; returns a dimensionless factor in (0, 1].
    """
    if g <= 0 or l_pm <= 0 or active_length <= 0:
        raise ValueError(
            f"g={g}, l_pm={l_pm}, active_length={active_length} "
            "must all be > 0"
        )
    w = active_length
    ln_term = (2.0 * l_pm / (pi * mu_r_pm * w)) * log(1.0 + pi * g / l_pm)
    return 1.0 / (
        1.0 + mu_r_pm * (g / l_pm) * (w / (w + 2.0 * g)) * ln_term
    )


def _one_gap_yokes(geo: AxialFluxGeometry) -> tuple[float, float]:
    """(rotor_yoke, stator_yoke) thicknesses (m) for the one-gap stack.

    Shared-yoke rule (plan audit F2): a yoke shared between the two gap
    circuits carries both gaps' flux, so the one-gap model gets HALF of it
    with the natural B·n = 0 cut at the mid-yoke mirror plane — rotor yoke
    for 1R2S, stator yoke for 2R1S-NN. 2R1S-NS flux passes axially through
    the stator (no mirror plane); kept full here as the v1 approximation.
    ψ at open circuit is insensitive to yoke thickness (near-∞ iron).
    """
    rotor_yoke, stator_yoke = geo.rotor_yoke, geo.stator_yoke
    if geo.variant == "1R2S":
        rotor_yoke /= 2.0
    elif geo.variant == "2R1S-NN":
        stator_yoke /= 2.0
    return rotor_yoke, stator_yoke


def _build_axial_sector(
    geo: AxialFluxGeometry,
    alpha: float,
    alpha_p: float,
    collar: float,
    maxh_fine: float,
):
    """Half-pole annular-sector solid stack; collar = radial air-collar
    width (m, 0 = extruded control model).

    z = 0 at the rotor-yoke bottom face; layers upward: rotor iron /
    magnet + interpole air / air gap / stator iron. The magnet and gap
    solids carry the local fine mesh size ``maxh_fine`` so gap resolution
    is decoupled from the bulk mesh (fem_wedge pattern). The θ = α faces
    are named "dpole"; every other face is unnamed (natural BC).
    """
    from math import cos, sin, sqrt

    from netgen.occ import Glue

    t_ry, t_sy = _one_gap_yokes(geo)
    l_pm, g = geo.magnet_thickness, geo.airgap
    z1 = t_ry
    z2 = z1 + l_pm
    z3 = z2 + g
    z4 = z3 + t_sy

    r_lo = geo.r_in - collar
    r_hi = geo.r_out + collar
    if r_lo <= 0:
        raise ValueError(
            f"inner air collar reaches the axis: r_in − collar = "
            f"{r_lo * 1e3:.3g} mm <= 0 — reduce collar_factor"
        )
    r_big = 2.0 * r_hi

    def ring(alpha_s: float, r_in: float, r_out: float, z_lo: float,
             z_hi: float, mat: str, fine: bool):
        sol = sector_ring(alpha_s, r_in, r_out, r_big).Extrude(z_hi - z_lo)
        if z_lo != 0.0:
            sol = sol.Move((0.0, 0.0, z_lo))
        sol.mat(mat)
        if fine:
            sol.maxh = maxh_fine
        return sol

    rotor = ring(alpha, geo.r_in, geo.r_out, 0.0, z1, "iron", False)
    magnet = ring(alpha_p * alpha, geo.r_in, geo.r_out, z1, z2,
                  "magnet", True)
    gap = ring(alpha, geo.r_in, geo.r_out, z2, z3, "air", True)
    stator = ring(alpha, geo.r_in, geo.r_out, z3, z4, "iron", False)
    solids = [rotor, magnet, gap, stator]

    # Remaining volume (interpole air + collars) by boolean subtraction.
    air = ring(alpha, r_lo, r_hi, 0.0, z4, "air", False)
    for sol in solids:
        air = air - sol
    air.mat("air")
    shape = Glue([*solids, air])

    # Name the θ = α pole-boundary plane "dpole" (Dirichlet φ = 0).
    for f in shape.faces:
        c = f.center
        r_c = sqrt(c[0] ** 2 + c[1] ** 2)
        if r_c > 1e-9 and abs(
            c[0] * sin(alpha) - c[1] * cos(alpha)
        ) < 1e-6 * r_big:
            f.name = "dpole"

    return shape, (z2 + z3) / 2.0


def _fundamental_Bz(
    mesh: Mesh, B_z_cf: CoefficientFunction, r: float,
    n_p: int, alpha: float, z: float, n_sample: int,
) -> float:
    """Fundamental amplitude of B_z(θ) at radius r, height z, over [0, α]."""
    from math import cos, sin

    thetas = np.linspace(0.0, alpha, n_sample)
    B_z = np.array([
        B_z_cf(mesh(r * cos(t), r * sin(t), z)) for t in thetas
    ])
    cos_np = np.cos(n_p * thetas)
    return float(
        np.trapezoid(B_z * cos_np, thetas) / np.trapezoid(cos_np**2, thetas)
    )


def _graded_radii(
    r_in: float, r_out: float, edge: float, inset: float,
    n_edge: int, n_mid: int,
) -> np.ndarray:
    """Radial sampling grid graded toward the band edges.

    The edge-leakage droop in B1(r) is confined to ~(airgap +
    magnet_thickness) of each edge; a uniform grid coarse enough to be
    cheap under-resolves it and biases ψ_geo. Dense linspace over
    3·edge at each end (n_edge points), coarse in the middle (n_mid).
    """
    d = min(3.0 * edge, 0.45 * (r_out - r_in))
    a, b = r_in + inset, r_out - inset
    return np.unique(np.concatenate([
        np.linspace(a, r_in + d, n_edge),
        np.linspace(r_in + d, r_out - d, n_mid),
        np.linspace(r_out - d, b, n_edge),
    ]))


@dataclass(frozen=True)
class AxialSectorResult:
    """One half-pole sector solve (finite or extruded control).

    r_list    — sampling radii on the mid-gap surface (m)
    B1_list   — fundamental B_z amplitude per radius (T)
    psi_geo   — ∫ B1(r)·r dr over [r_in, r_out] (T·m²) — the ψ-equivalent
                geometric flux integral (winding constants excluded)
    flux_net  — all-harmonic half-pole gap flux ∫∫ B_z·r dr dθ (Wb)
    finite    — True for the collared finite-annulus model
    """

    r_list: np.ndarray
    B1_list: np.ndarray
    psi_geo: float
    flux_net: float
    finite: bool


def solve_axial_sector(
    geo: AxialFluxGeometry,
    n_p: int,
    B_rem: float,
    *,
    alpha_p: float = 1.0,
    mu_r_pm: float = 1.05,
    mu_r_fe: float = 1.0e4,
    finite: bool = True,
    collar_factor: float = MIN_COLLAR_FACTOR,
    maxh_fine: float | None = None,
    maxh_bulk: float | None = None,
    order: int = 2,
    n_r_edge: int = 17,
    n_r_mid: int = 11,
    n_theta: int = 61,
) -> AxialSectorResult:
    """Solve the half-pole axial sector; see module docstring for the model.

    finite=True adds the radial air collars of width
    collar_factor·(airgap + magnet_thickness) (collar_factor ≥ 4, plan
    audit F1) with the flux-parallel natural far-field truncation;
    finite=False is the extruded control (Neumann walls at r_in/r_out).
    maxh_fine (m) is local to the magnet + gap solids (default
    (airgap + magnet_thickness)/4); maxh_bulk (m) meshes iron and collar
    air (default active_length/6). B1(r) is sampled on the graded radial
    grid of _graded_radii (n_r_edge points per edge zone, n_r_mid in the
    middle) × n_theta angles on the mid-gap surface, inset from the
    radial band edges by 1e-3·active_length so the extruded model's wall
    faces are never point-evaluated.
    """
    if finite and collar_factor < MIN_COLLAR_FACTOR:
        raise ValueError(
            f"collar_factor={collar_factor} < {MIN_COLLAR_FACTOR} — the "
            "far-field truncation would sit inside the leakage field "
            "(plan audit F1)"
        )
    if not (0 < alpha_p <= 1):
        raise ValueError(f"alpha_p must be in (0, 1], got {alpha_p}")
    if geo.variant == "coreless":
        raise ValueError("coreless variant has no stator iron slab")

    from netgen.occ import OCCGeometry
    from ngsolve import (
        H1,
        BilinearForm,
        GridFunction,
        LinearForm,
        Mesh,
        dx,
        grad,
    )

    alpha = pi / (2 * n_p)
    edge = geo.airgap + geo.magnet_thickness
    collar = collar_factor * edge if finite else 0.0
    if maxh_fine is None:
        maxh_fine = edge / 4.0
    if maxh_bulk is None:
        maxh_bulk = geo.active_length / 6.0

    shape, z_mid = _build_axial_sector(geo, alpha, alpha_p, collar, maxh_fine)
    mesh = Mesh(OCCGeometry(shape).GenerateMesh(maxh=maxh_bulk))

    mu = mesh.MaterialCF(
        {"magnet": MU0 * mu_r_pm, "air": MU0, "iron": MU0 * mu_r_fe}
    )
    Brem_z = mesh.MaterialCF({"magnet": B_rem, "air": 0.0, "iron": 0.0})

    fes = H1(mesh, order=order, dirichlet="dpole")
    u, v = fes.TnT()

    a = BilinearForm(mu * grad(u) * grad(v) * dx)
    a.Assemble()
    lf = LinearForm(Brem_z * grad(v)[2] * dx)
    lf.Assemble()

    gfu = GridFunction(fes)
    gfu.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky") * lf.vec

    # B_z in the gap (air): −μ0·∂φ/∂z.
    B_z_cf = -MU0 * grad(gfu)[2]

    inset = 1e-3 * geo.active_length
    r_list = _graded_radii(
        geo.r_in, geo.r_out, edge, inset, n_r_edge, n_r_mid,
    )
    B1_list = np.array([
        _fundamental_Bz(mesh, B_z_cf, r, n_p, alpha, z_mid, n_theta)
        for r in r_list
    ])
    psi_geo = float(np.trapezoid(B1_list * r_list, r_list))

    from math import cos, sin

    thetas = np.linspace(0.0, alpha, n_theta)
    Bz_grid = np.array([
        [B_z_cf(mesh(r * cos(t), r * sin(t), z_mid)) for t in thetas]
        for r in r_list
    ])
    flux_net = float(np.trapezoid(
        np.trapezoid(Bz_grid, thetas, axis=1) * r_list, r_list
    ))

    return AxialSectorResult(r_list, B1_list, psi_geo, flux_net, finite)


@dataclass(frozen=True)
class EdgeLeakageResult:
    """CV-4 comparison at one configuration.

    deficit_ratio    — ψ_geo(finite) / ψ_geo(extruded), the FEM edge factor
    k_sigma_edge     — radial_edge_leakage_factor (Qu & Lipo analytical)
    factor_error     — deficit_ratio − k_sigma_edge (abs vs CV4 tolerance)
    """

    finite: AxialSectorResult
    extruded: AxialSectorResult
    deficit_ratio: float
    k_sigma_edge: float
    factor_error: float


def compute_edge_leakage(
    geo: AxialFluxGeometry,
    n_p: int,
    B_rem: float,
    *,
    alpha_p: float = 1.0,
    mu_r_pm: float = 1.05,
    **kwargs,
) -> EdgeLeakageResult:
    """CV-4 matched-config pair: finite annulus vs extruded control.

    Both solves share every parameter except the collars/walls, so the
    ψ_geo ratio isolates the radial-edge leakage. Compared against the
    Qu & Lipo radial-edge factor (physical gap, L_a = active_length).
    kwargs pass through to solve_axial_sector (mesh sizes, order, ...).
    """
    fin = solve_axial_sector(
        geo, n_p, B_rem, alpha_p=alpha_p, mu_r_pm=mu_r_pm,
        finite=True, **kwargs,
    )
    ext = solve_axial_sector(
        geo, n_p, B_rem, alpha_p=alpha_p, mu_r_pm=mu_r_pm,
        finite=False, **kwargs,
    )
    ratio = fin.psi_geo / ext.psi_geo
    k_edge = radial_edge_leakage_factor(
        geo.airgap, geo.magnet_thickness, geo.active_length, mu_r_pm,
    )
    return EdgeLeakageResult(fin, ext, ratio, k_edge, ratio - k_edge)
