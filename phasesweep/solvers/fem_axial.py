"""Multi-plane 2D Cartesian FEM for axial-flux PM machines (Track A).

The machine is unrolled at K sampling radii into linear (Cartesian)
machines and each plane is solved as a 2D magnetostatic A_z problem —
the FEM counterpart of the quasi-3D layer model in
``analytical_axial.py``. Shared tolerances:
tests/crossval/axial_fem_contract.py.

Plane coordinates (SI, meters):
  x — circumferential position at radius r (x = r·θ)
  y — axial position, measured from the outer face of the rotor yoke

One-pole mirror domain: x ∈ [0, τ_p], τ_p = π·r/n_p (m), with the N and
S magnet CENTERS on the two vertical edges (half-magnets at each edge).
Mirror symmetry about a magnet center makes A_z odd ⟹ A_z = 0 on both
edges ("symm" Dirichlet); M_y is even there ⟹ (curl M)_z odd, so the
source is compatible. Validity precondition: slots-per-pole
n_slots/(2·n_p) must be a positive integer with a tooth centered on the
d-axis (enforced here; the registry guard arrives at merge). The top and
bottom horizontal edges ("outer") also take A_z = 0 — flux is confined
to the stack (no leakage above/below the yokes).

Periodic reference (CV-0): the same stack on a full pole-pair domain
x ∈ [0, 2·τ_p] with TRUE periodic identification of the left/right
edges — netgen OCC ``Edge.Identify(other, name,
IdentificationType.PERIODIC)`` per boundary segment (one per layer,
paired by y-center), then ``ngsolve.Periodic(H1(...))`` so left/right
DOFs are unified. No "symm" Dirichlet at all: the only assumption left
is 2·τ_p periodicity, which is exact. This machinery exists solely to
cross-check the mirror-BC geometry builder (a symmetry bug in the
builder would pass the theory proof but fail this comparison); it is
not used by the shipping solve path.

Axial stack (bottom → top): rotor_yoke / pm_n + pm_gap + pm_s / airgap
/ tooth + slot (slot_depth, if slotted) / stator_yoke (remaining back
iron). Per-variant shared-yoke rule (plan audit F2): NN-type shared
yokes carry both gaps' flux, so the one-gap stack halves them —
1R2S halves rotor_yoke; 2R1S-NN halves the stator back iron; 2R1S-NS
(flux passes axially through the stator) and DSSR are not halved.
Open-circuit ψ is insensitive to the rule (near-infinite iron); it
matters for b_iron_max / nonlinear mode, documented as per-gap
approximations for shared-yoke variants.

Slots are open (no tooth-tip lip), constant ABSOLUTE width
slot_width_ratio·2π·r_mean/n_slots across planes (plan §slots);
slot_opening_width is unused in v1.

B_z = −∂A_z/∂x (T) sampled on the mid-airgap line, mirror-extended to
the 2·τ_p electrical period. Multi-plane flux linkage: the solver
returns the geometric sum S = Σ_k B1(r_k)·r_k·Δr (T·m²) and the radial
edge-leakage factor k_σ_edge separately; the registry applies
GAP_LINKAGE·k_σ_edge·(2·N_eff·k_w/n_p)·S at merge (reduces to
``winding_transfer_axial`` at K = 1).

Nonlinear iron reuses the 2D Picard machinery from ``fem_field``
(geometry-agnostic: it iterates ν(|∇A|) on whatever regions are named
iron). Uses NGSolve (LGPL); all ngsolve imports are deferred so this
module imports without the [fem] extra.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log, pi
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from phasesweep.defaults import (
    PICARD_MAX_ITERATIONS,
    PICARD_RELAXATION,
    PICARD_TOLERANCE,
)
from phasesweep.solver_params import _reject_induction, _require_axial_geometry
from phasesweep.solvers.fem_field import _newton_solve, _picard_solve
from phasesweep.solvers.fem_materials import _MU0

if TYPE_CHECKING:
    from ngsolve import GridFunction, Mesh

    from phasesweep.machines.geometry import AxialFluxGeometry
    from phasesweep.machines.motor import Motor

# Regions treated as iron by the nonlinear (BH) solve.
IRON_REGIONS: frozenset[str] = frozenset({"rotor_yoke", "stator_yoke", "tooth"})

# Interpole gaps narrower than this (m) collapse to touching magnets
# (alpha_p treated as 1) — below OCC's reliable feature size.
_MIN_GAP_M = 1e-6

# Geometric coincidence tolerance (m) for edge naming/pairing on the
# glued shape — all breakpoints are exact sums of the same floats.
_EDGE_TOL = 1e-9

# In-memory mesh cache (no disk tier: axial plane meshes are cheap and
# keyed per sampling radius). Cleared with clear_mesh_cache().
_CacheKey = tuple[str, str, float, int, float, float]
_mesh_cache: dict[_CacheKey, Any] = {}


def clear_mesh_cache() -> None:
    """Drop all cached plane meshes (in-memory cache only)."""
    _mesh_cache.clear()


# ---------------------------------------------------------------------------
# Plane placement + stack
# ---------------------------------------------------------------------------

def _plane_radii(
    geo: AxialFluxGeometry, n_planes: int,
) -> tuple[NDArray[np.floating], float]:
    """Midpoint-rule sampling radii over the active annulus.

    Returns (r_k, Δr) in meters: Δr = active_length/n_planes and
    r_k = r_in + (k + ½)·Δr. Odd n_planes puts the middle plane exactly
    at r_mean.
    """
    if n_planes < 1:
        raise ValueError(f"n_planes must be >= 1, got {n_planes}")
    dr = geo.active_length / n_planes
    r = geo.r_in + (np.arange(n_planes) + 0.5) * dr
    return r, float(dr)


def _effective_yokes(geo: AxialFluxGeometry) -> tuple[float, float]:
    """(rotor_yoke, stator back iron) thickness (m) in the one-gap stack.

    Applies the per-variant shared-yoke rule (plan audit F2): the yoke
    shared by two NN-facing gaps carries both gaps' circumferential flux,
    so the one-gap model halves it — 1R2S halves the rotor yoke,
    2R1S-NN halves the stator back iron (the part above the slots).
    2R1S-NS flux passes axially through the stator (no circumferential
    sharing) and DSSR yokes are per-gap: both keep full thickness.
    Slot depth is carved out of stator_yoke first; only the remaining
    back iron is halved.
    """
    slot_depth = geo.slot_depth if geo.n_slots > 0 else 0.0
    stator_back = geo.stator_yoke - slot_depth
    rotor = geo.rotor_yoke
    if geo.variant == "1R2S":
        rotor /= 2.0
    elif geo.variant == "2R1S-NN":
        stator_back /= 2.0
    return rotor, stator_back


def _y_levels(geo: AxialFluxGeometry) -> dict[str, float]:
    """Axial breakpoints (m) of the plane stack, y = 0 at the rotor-yoke
    outer face. Keys: rotor_top, pm_top, gap_top, slot_top, top,
    gap_mid (the B_z sampling line)."""
    t_rotor, t_stator_back = _effective_yokes(geo)
    slot_depth = geo.slot_depth if geo.n_slots > 0 else 0.0
    rotor_top = t_rotor
    pm_top = rotor_top + geo.magnet_thickness
    gap_top = pm_top + geo.airgap
    slot_top = gap_top + slot_depth
    return {
        "rotor_top": rotor_top,
        "pm_top": pm_top,
        "gap_top": gap_top,
        "slot_top": slot_top,
        "top": slot_top + t_stator_back,
        "gap_mid": pm_top + geo.airgap / 2.0,
    }


# ---------------------------------------------------------------------------
# Geometry builder
# ---------------------------------------------------------------------------

def _magnet_segments(
    tau_p: float, alpha_p: float, n_pitch: int,
) -> list[tuple[float, float, str]]:
    """(x0, x1, name) spans of the magnet layer over n_pitch pole
    pitches (m). Pole k is centered at x = k·τ_p — pm_n at even k, pm_s
    at odd k — clipped to [0, n_pitch·τ_p]; pm_gap fills the interpole
    spans. Sub-micron interpole gaps collapse to touching magnets."""
    W = n_pitch * tau_p
    w = alpha_p * tau_p
    gapped = (tau_p - w) > _MIN_GAP_M
    if not gapped:
        w = tau_p
    segs: list[tuple[float, float, str]] = []
    prev_right = 0.0
    for k in range(n_pitch + 1):
        name = "pm_n" if k % 2 == 0 else "pm_s"
        x0 = max(k * tau_p - w / 2.0, 0.0)
        x1 = min(k * tau_p + w / 2.0, W)
        if gapped and x0 > prev_right + _MIN_GAP_M:
            segs.append((prev_right, x0, "pm_gap"))
        segs.append((x0, x1, name))
        prev_right = x1
    return segs


def _slot_segments(
    geo: AxialFluxGeometry, tau_p: float, n_p: int, n_pitch: int,
) -> list[tuple[float, float, str]]:
    """(x0, x1, name) spans of the tooth/slot layer over n_pitch pole
    pitches at the plane's radius (m).

    Tooth centers sit at x = j·τ_s (τ_s = τ_p/spp, spp = n_slots/(2·n_p)
    slots per pole) so a tooth is centered on each d-axis; slot width is
    the constant ABSOLUTE width slot_width_ratio·2π·r_mean/n_slots.
    Raises if spp is not a positive integer (mirror BC invalid) or the
    slot leaves no tooth at this plane's slot pitch.
    """
    if geo.n_slots % (2 * n_p) != 0:
        raise ValueError(
            f"n_slots={geo.n_slots} is not an integer multiple of "
            f"2*n_p={2 * n_p}: no tooth-centered-on-d-axis layout exists "
            "and the mirror Dirichlet BC is invalid"
        )
    spp = geo.n_slots // (2 * n_p)
    tau_s = tau_p / spp
    w_s = geo.slot_width_ratio * 2.0 * pi * geo.r_mean / geo.n_slots
    if w_s >= tau_s - _MIN_GAP_M:
        raise ValueError(
            f"slot width {w_s * 1e3:.3g} mm (constant, set at r_mean) >= "
            f"slot pitch {tau_s * 1e3:.3g} mm at this plane: no tooth left"
        )
    segs: list[tuple[float, float, str]] = []
    prev_right = 0.0
    W = n_pitch * tau_p
    for j in range(n_pitch * spp):
        c = (j + 0.5) * tau_s
        segs.append((prev_right, c - w_s / 2.0, "tooth"))
        segs.append((c - w_s / 2.0, c + w_s / 2.0, "slot"))
        prev_right = c + w_s / 2.0
    segs.append((prev_right, W, "tooth"))
    return segs


def _build_plane_geometry(
    geo: AxialFluxGeometry, r: float, n_p: int, alpha_p: float,
    *, n_pitch: int = 1,
) -> Any:
    """Netgen OCC 2D shape of the unrolled plane at radius r (m).

    n_pitch = 1: the one-pole mirror domain, vertical edges named
    "symm". n_pitch = 2: the pole-pair periodic domain, vertical edges
    named "period" (identified by the caller). Horizontal top/bottom
    edges are named "outer" in both. Per-face maxh: airgap/3 on the
    airgap band, interpole_width/2 on pm_gap faces.
    """
    from netgen.occ import Glue, WorkPlane

    if geo.variant == "coreless":
        raise ValueError(
            "coreless variant has no stator iron boundary — fem_axial "
            "requires an iron-backed stator"
        )
    yl = _y_levels(geo)
    tau_p = pi * r / n_p
    W = n_pitch * tau_p
    slotted = geo.n_slots > 0 and geo.slot_depth > 0

    def rect(x0: float, x1: float, y0: float, y1: float, name: str):
        f = WorkPlane().MoveTo(x0, y0).Rectangle(x1 - x0, y1 - y0).Face()
        f.name = name
        return f

    faces = [rect(0.0, W, 0.0, yl["rotor_top"], "rotor_yoke")]
    for x0, x1, name in _magnet_segments(tau_p, alpha_p, n_pitch):
        faces.append(rect(x0, x1, yl["rotor_top"], yl["pm_top"], name))
    faces.append(rect(0.0, W, yl["pm_top"], yl["gap_top"], "airgap"))
    if slotted:
        for x0, x1, name in _slot_segments(geo, tau_p, n_p, n_pitch):
            faces.append(rect(x0, x1, yl["gap_top"], yl["slot_top"], name))
    faces.append(rect(0.0, W, yl["slot_top"], yl["top"], "stator_yoke"))

    shape = Glue(faces)

    # Per-face refinement (names survive Glue; edge names do not, so
    # boundary edges are re-named by position below).
    shape.faces["airgap"].maxh = geo.airgap / 3.0
    interpole = tau_p - alpha_p * tau_p
    if interpole > _MIN_GAP_M:
        shape.faces["pm_gap"].maxh = interpole / 2.0

    side = "symm" if n_pitch == 1 else "period"
    for e in shape.edges:
        cx, cy = e.center[0], e.center[1]
        if abs(cx) < _EDGE_TOL or abs(cx - W) < _EDGE_TOL:
            e.name = side
        elif abs(cy) < _EDGE_TOL or abs(cy - yl["top"]) < _EDGE_TOL:
            e.name = "outer"
    return shape


def _identify_periodic_edges(shape: Any, W: float) -> None:
    """Identify each left-boundary edge with its right partner (same
    y-center) for periodic meshing — netgen OCC
    ``Edge.Identify(other, name, IdentificationType.PERIODIC)``. The
    identification makes the two boundary meshes congruent so
    ``ngsolve.Periodic`` can unify their DOFs."""
    from netgen.occ import IdentificationType

    left = [e for e in shape.edges if abs(e.center[0]) < _EDGE_TOL]
    right = [e for e in shape.edges if abs(e.center[0] - W) < _EDGE_TOL]
    right_by_y = sorted(right, key=lambda e: e.center[1])
    left_by_y = sorted(left, key=lambda e: e.center[1])
    if len(left_by_y) != len(right_by_y):
        raise RuntimeError(
            f"periodic edge mismatch: {len(left_by_y)} left vs "
            f"{len(right_by_y)} right boundary edges"
        )
    for i, (el, er) in enumerate(zip(left_by_y, right_by_y)):
        if abs(el.center[1] - er.center[1]) > _EDGE_TOL:
            raise RuntimeError(
                f"periodic edge pairing failed at y={el.center[1]:.6g}"
            )
        el.Identify(er, f"axial_periodic_{i}",
                    IdentificationType.PERIODIC)


def _get_or_build_plane_mesh(
    geo: AxialFluxGeometry, r: float, n_p: int, alpha_p: float,
    maxh_fraction: float, *, periodic: bool = False,
) -> Mesh:
    """Plane mesh at radius r (m): in-memory cache → build.

    Global maxh = maxh_fraction·τ_p(r); the airgap and pm_gap bands
    carry their own per-face maxh (see _build_plane_geometry).
    """
    from netgen.occ import OCCGeometry
    from ngsolve import Mesh

    kind = "periodic" if periodic else "mirror"
    key: _CacheKey = (
        geo.config_id, kind, round(r, 12), n_p, round(alpha_p, 12),
        maxh_fraction,
    )
    mesh = _mesh_cache.get(key)
    if mesh is None:
        tau_p = pi * r / n_p
        n_pitch = 2 if periodic else 1
        shape = _build_plane_geometry(geo, r, n_p, alpha_p,
                                      n_pitch=n_pitch)
        if periodic:
            _identify_periodic_edges(shape, n_pitch * tau_p)
        mesh = Mesh(OCCGeometry(shape, dim=2).GenerateMesh(
            maxh=maxh_fraction * tau_p))
        _mesh_cache[key] = mesh
    return mesh


# ---------------------------------------------------------------------------
# Plane solve
# ---------------------------------------------------------------------------

def _solve_on_mesh(
    mesh: Mesh,
    B_rem: float,
    mu_r_pm: float,
    mu_r_fe: float,
    *,
    periodic: bool = False,
    nonlinear: bool = False,
    nonlinear_solver: str = "newton",
) -> GridFunction:
    """Assemble and solve the A_z problem on a built plane mesh.

    Weak form: ∫ ν ∇A·∇v dx = ∫ −M_y·∂v/∂x dx over the magnets, with
    M_y = ±B_rem/(μ0·μr_pm) (A/m) in pm_n/pm_s — the Cartesian
    reduction of the fem_field curl-curl source. Mirror mesh: H1
    order 3, dirichlet="symm|outer". Periodic mesh: Periodic(H1),
    dirichlet="outer" only. Nonlinear iron runs the shared fem_field
    solvers on IRON_REGIONS — Newton by default, with the same
    automatic Picard fallback as ``solve_field_fem``.
    """
    from ngsolve import (
        H1,
        BilinearForm,
        GridFunction,
        LinearForm,
        Periodic,
        dx,
        grad,
    )

    region_names = list(mesh.GetMaterials())
    nu_cf: dict[str, float] = {}
    my_cf: dict[str, float] = {}
    nu_pm = 1.0 / (_MU0 * mu_r_pm)
    for name in region_names:
        if name in ("pm_n", "pm_s"):
            nu_cf[name] = nu_pm
            my_cf[name] = nu_pm * B_rem if name == "pm_n" else -nu_pm * B_rem
        elif name in IRON_REGIONS:
            nu_cf[name] = 1.0 / (_MU0 * mu_r_fe)
            my_cf[name] = 0.0
        else:  # airgap, pm_gap, slot
            nu_cf[name] = 1.0 / _MU0
            my_cf[name] = 0.0
    nu = mesh.MaterialCF(nu_cf)
    My = mesh.MaterialCF(my_cf)

    if periodic:
        fes = Periodic(H1(mesh, order=3, dirichlet="outer"))
    else:
        fes = H1(mesh, order=3, dirichlet="symm|outer")
    u, v = fes.TnT()

    lf = LinearForm(-My * grad(v)[0] * dx)
    lf.Assemble()
    gfu = GridFunction(fes)

    if nonlinear and nonlinear_solver == "newton":
        try:
            _newton_solve(mesh, fes, lf, gfu, nu, region_names,
                          set(IRON_REGIONS), PICARD_MAX_ITERATIONS, None)
        except RuntimeError as e:
            import logging
            logging.warning("Newton BH solve failed (%s); falling back "
                            "to Picard", e)
            _picard_solve(
                mesh, fes, lf, gfu, nu, region_names, set(IRON_REGIONS),
                PICARD_MAX_ITERATIONS, PICARD_TOLERANCE, PICARD_RELAXATION,
                None,
            )
    elif nonlinear:
        _picard_solve(
            mesh, fes, lf, gfu, nu, region_names, set(IRON_REGIONS),
            PICARD_MAX_ITERATIONS, PICARD_TOLERANCE, PICARD_RELAXATION,
            None,
        )
    else:
        a = BilinearForm(nu * grad(u) * grad(v) * dx)
        a.Assemble()
        gfu.vec.data = (
            a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky") * lf.vec
        )
    return gfu


def _sample_Bz(
    mesh: Mesh, gfu: GridFunction, x_pts: NDArray[np.floating], y: float,
) -> NDArray[np.floating]:
    """B_z = −∂A_z/∂x (T) at (x_pts, y) — vectorized point evaluation."""
    from ngsolve import grad

    Bz_cf = -grad(gfu)[0]
    mips = mesh(x_pts, np.full_like(x_pts, y))
    return np.asarray(Bz_cf(mips), dtype=float).reshape(-1)


def solve_plane(
    geo: AxialFluxGeometry,
    r: float,
    n_p: int,
    B_rem: float,
    mu_r_pm: float,
    mu_r_fe: float,
    alpha_p: float,
    *,
    maxh_fraction: float = 0.1,
    n_x: int = 256,
    nonlinear: bool = False,
    nonlinear_solver: str = "newton",
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Solve the mirror plane at radius r and return the full electrical
    period.

    Returns (x, B_z): 2·n_x midpoint samples (m, T) of mid-airgap B_z
    over [0, 2·τ_p), built by solving [0, τ_p] and mirror-extending —
    B_z is even about the magnet center (A_z odd), so
    B_z(2τ_p − x) = B_z(x). Midpoint sampling keeps the extended grid
    uniform (FFT-ready) with no duplicated boundary points.
    """
    tau_p = pi * r / n_p
    mesh = _get_or_build_plane_mesh(geo, r, n_p, alpha_p, maxh_fraction)
    gfu = _solve_on_mesh(mesh, B_rem, mu_r_pm, mu_r_fe,
                         nonlinear=nonlinear,
                         nonlinear_solver=nonlinear_solver)
    y_mid = _y_levels(geo)["gap_mid"]
    x_half = (np.arange(n_x) + 0.5) * tau_p / n_x
    Bz_half = _sample_Bz(mesh, gfu, x_half, y_mid)
    x_ext = np.concatenate([x_half, 2.0 * tau_p - x_half[::-1]])
    Bz_ext = np.concatenate([Bz_half, Bz_half[::-1]])
    return x_ext, Bz_ext


def solve_plane_periodic(
    geo: AxialFluxGeometry,
    r: float,
    n_p: int,
    B_rem: float,
    mu_r_pm: float,
    mu_r_fe: float,
    alpha_p: float,
    *,
    maxh_fraction: float = 0.1,
    n_x: int = 256,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """CV-0 reference: pole-pair domain with TRUE periodic identification.

    Same stack, x ∈ [0, 2·τ_p], left/right edges identified
    (netgen PERIODIC) and solved in Periodic(H1) with only the "outer"
    Dirichlet — no mirror-symmetry assumption. Returns (x, B_z) on the
    same 2·n_x midpoint grid as solve_plane for direct comparison.
    Linear iron only (a one-off verification path).
    """
    tau_p = pi * r / n_p
    mesh = _get_or_build_plane_mesh(geo, r, n_p, alpha_p, maxh_fraction,
                                    periodic=True)
    gfu = _solve_on_mesh(mesh, B_rem, mu_r_pm, mu_r_fe, periodic=True)
    y_mid = _y_levels(geo)["gap_mid"]
    x = (np.arange(2 * n_x) + 0.5) * tau_p / n_x
    return x, _sample_Bz(mesh, gfu, x, y_mid)


def fundamental_amplitude(Bz_period: NDArray[np.floating]) -> float:
    """Fundamental amplitude (T) of a uniformly sampled electrical
    period (one cycle per 2·τ_p): 2·|FFT bin 1|/N. Phase-independent, so
    midpoint sampling offsets do not matter."""
    n = len(Bz_period)
    return 2.0 * float(np.abs(np.fft.rfft(Bz_period)[1])) / n


# ---------------------------------------------------------------------------
# Radial edge leakage
# ---------------------------------------------------------------------------

def pm_edge_leakage_factor(geo: AxialFluxGeometry, mu_r_pm: float) -> float:
    """Radial-edge PM leakage factor k_σ_edge (dimensionless, ≤ 1).

    The in-plane FEM captures every circumferential leakage path
    (interpole magnet-to-magnet, zigzag, slot fringing) but the plane
    stack is invariant along the radius, so flux escaping across the
    magnets' RADIAL edges (at r_in / r_out, magnet edge → rotor yoke)
    is invisible to it. This factor re-applies exactly that one path,
    using the Qu & Lipo (2004) magnet-to-rotor form (the same form the
    layer model uses circumferentially in
    ``analytical_axial.pm_leakage_factor``) with the roles rotated 90°:
    the magnet span is the radial active length L_a = r_out − r_in and
    the leakage permeance is the two quarter-annulus air paths at the
    radial edges,

        k_σ_edge = 1 / (1 + μr·(g/l_pm)·(L_a/(L_a + 2g))
                        ·(2·l_pm/(π·μr·L_a))·ln(1 + π·g/l_pm))

    (g = physical air gap, l_pm = magnet thickness, all m).

    Deliberately dropped relative to the full Qu & Lipo expression:
      - magnet-to-magnet term — there is no adjacent magnet across the
        radial edge, only open air;
      - zigzag term — slots run radially, so there is no slotting
        across the radial edge;
      - the empirical k_add — it reconciles the CIRCUMFERENTIAL
        analytical factor with 3D FEA and does not describe this path;
      - Carter correction of g — Carter models circumferential slot
        openings, which the in-plane FEM already resolves; the radial
        edge sees the physical gap.
    Track B (3D finite annulus vs radially-extruded control) is the
    arbiter for this factor (CV-4).
    """
    g = geo.airgap
    l_pm = geo.magnet_thickness
    L_a = geo.active_length
    ln_term = (2.0 * l_pm / (pi * mu_r_pm * L_a)) * log(1.0 + pi * g / l_pm)
    return 1.0 / (
        1.0 + mu_r_pm * (g / l_pm) * (L_a / (L_a + 2.0 * g)) * ln_term
    )


# ---------------------------------------------------------------------------
# Multi-plane driver
# ---------------------------------------------------------------------------

def solve_axial_multiplane(
    geo: AxialFluxGeometry,
    n_p: int,
    B_rem: float,
    mu_r_pm: float,
    mu_r_fe: float,
    alpha_p: float,
    *,
    n_planes: int = 3,
    maxh_fraction: float = 0.1,
    n_theta: int = 720,
    nonlinear: bool = False,
    nonlinear_solver: str = "newton",
) -> dict[str, Any]:
    """Multi-plane quasi-3D solve: K mirror planes, midpoint rule.

    Returns a dict:
      plane_r_list  — sampling radii r_k (m)
      plane_B1_list — per-plane fundamental B1(r_k) (T)
      dr            — radial slice width Δr (m)
      theta_list    — n_theta mechanical angles over [0, 2π) (rad)
      B_z_list      — middle-plane mid-airgap B_z(θ) (T), the 2·τ_p
                      waveform tiled n_p times and linearly resampled
      fundamental   — middle-plane B1 (T) (order-n_p amplitude)
      peak          — max |B_z| of the middle-plane waveform (T)
      S             — geometric flux sum Σ_k B1(r_k)·r_k·Δr (T·m²),
                      WITHOUT winding transfer: the registry applies
                      GAP_LINKAGE·k_σ_edge·(2·N_eff·k_w/n_p)·S
      k_sigma_edge  — radial edge-leakage factor (dimensionless)

    Odd n_planes puts the middle plane at r_mean; even n_planes reports
    the waveform of plane index n_planes//2 (just above r_mean).
    """
    r_list, dr = _plane_radii(geo, n_planes)
    mid = n_planes // 2
    B1_list: list[float] = []
    x_mid: NDArray[np.floating] | None = None
    Bz_mid: NDArray[np.floating] | None = None
    for k, r in enumerate(r_list):
        x_ext, Bz = solve_plane(
            geo, float(r), n_p, B_rem, mu_r_pm, mu_r_fe, alpha_p,
            maxh_fraction=maxh_fraction, nonlinear=nonlinear,
            nonlinear_solver=nonlinear_solver,
        )
        B1_list.append(fundamental_amplitude(Bz))
        if k == mid:
            x_mid, Bz_mid = x_ext, Bz
    assert x_mid is not None and Bz_mid is not None

    # Tile the electrical period to the mechanical circle: θ = x/r, the
    # period is 2·τ_p/r = 2π/n_p, exact by construction.
    r_m = float(r_list[mid])
    period = 2.0 * pi / n_p
    theta_src = x_mid / r_m
    theta = np.linspace(0.0, 2.0 * pi, n_theta, endpoint=False)
    B_z = np.interp(theta % period, theta_src, Bz_mid, period=period)

    S = float(np.sum(np.asarray(B1_list) * r_list * dr))
    return {
        "plane_r_list": [float(r) for r in r_list],
        "plane_B1_list": B1_list,
        "dr": dr,
        "theta_list": theta,
        "B_z_list": B_z,
        "fundamental": B1_list[mid],
        "peak": float(np.max(np.abs(B_z))),
        "S": S,
        "k_sigma_edge": pm_edge_leakage_factor(geo, mu_r_pm),
    }


@dataclass(frozen=True)
class FemAxialParams:
    """Validated inputs for the multi-plane 2D Cartesian axial FEM solver.

    Same physical inputs as AnalyticalAxialParams (geometry in meters,
    B_rem in T, permeabilities dimensionless, alpha_p in (0, 1]), plus
    the mirror-domain preconditions the solver's boundary condition
    depends on.

    n_planes — quasi-3D computation planes (>= 1); the domain guards
    below are evaluated at the innermost sampling radius for this K,
    because the slot width is constant in absolute terms while the slot
    pitch shrinks with radius.
    """

    geometry: AxialFluxGeometry
    n_p: int
    B_rem: float
    mu_r_pm: float
    mu_r_fe: float
    alpha_p: float = 1.0
    n_planes: int = 3

    def __post_init__(self) -> None:
        if self.n_p < 1:
            raise ValueError(f"n_p must be >= 1, got {self.n_p}")
        if self.B_rem <= 0:
            raise ValueError(f"B_rem must be > 0, got {self.B_rem}")
        if self.mu_r_pm <= 0:
            raise ValueError(f"mu_r_pm must be > 0, got {self.mu_r_pm}")
        if self.mu_r_fe <= 0:
            raise ValueError(f"mu_r_fe must be > 0, got {self.mu_r_fe}")
        if not (0 < self.alpha_p <= 1):
            raise ValueError(f"alpha_p must be in (0, 1], got {self.alpha_p}")
        if self.n_planes < 1:
            raise ValueError(f"n_planes must be >= 1, got {self.n_planes}")
        geo = self.geometry
        if geo.variant == "coreless":
            raise ValueError(
                "fem_axial does not support the coreless variant — "
                "the mirror-domain stack needs stator iron behind the gap"
            )
        # Mirror-BC precondition: A_z = 0 on the magnet-center planes is
        # exact only when the slot pattern is mirror-symmetric about
        # them, i.e. an integer number of slots per pole with a tooth on
        # the d-axis. Fractional-slot windings need a pole-pair domain
        # with periodic identification (solve_plane_periodic) — not the
        # shipped path in v1.
        if geo.n_slots:
            if geo.n_slots % (2 * self.n_p):
                raise ValueError(
                    f"fem_axial needs an integer slots-per-pole for the "
                    f"mirror boundary: n_slots={geo.n_slots} is not "
                    f"divisible by 2*n_p={2 * self.n_p}"
                )
            # Constant absolute slot width (set at r_mean) against the
            # shrinking pitch at the innermost sampling plane.
            w_slot = geo.slot_width_ratio * 2 * pi * geo.r_mean / geo.n_slots
            dr = geo.active_length / self.n_planes
            r_inner = geo.r_in + 0.5 * dr
            pitch_inner = 2 * pi * r_inner / geo.n_slots
            if w_slot >= 0.95 * pitch_inner:
                raise ValueError(
                    f"fem_axial: slot width {w_slot * 1e3:.2f} mm leaves no "
                    f"tooth at the innermost plane r={r_inner * 1e3:.1f} mm "
                    f"(pitch {pitch_inner * 1e3:.2f} mm); reduce "
                    f"slot_width_ratio or n_planes"
                )


def prepare_fem_axial(motor: Motor, n_planes: int = 3,
                      j_s: float = 0.0) -> FemAxialParams:
    """Validate and extract multi-plane axial FEM params from a Motor.

    B_rem must be explicit for the same reason as analytical_axial: the
    psi_f -> B_rem inversion is radial-only. j_s must be zero — v1 is an
    open-circuit model and j_s is not in the model's hash_fields, so
    accepting it would alias distinct armature loadings to one run ID.
    """
    _reject_induction(motor, "fem_axial")
    geo = _require_axial_geometry(motor, "fem_axial")
    if motor.B_rem is None:
        raise ValueError(
            f"Motor '{motor.name}': fem_axial needs an explicit "
            f"[materials] B_rem (no psi_f derivation path for axial flux)"
        )
    if j_s:
        raise ValueError(
            f"Motor '{motor.name}': fem_axial is an open-circuit model "
            f"(j_s must be 0, got {j_s}); armature loading is not in its "
            f"hash_fields and would alias run IDs"
        )
    return FemAxialParams(
        geometry=geo,
        n_p=motor.n_p,
        B_rem=motor.B_rem,
        mu_r_pm=motor.mu_r_pm,
        mu_r_fe=motor.mu_r_fe,
        alpha_p=motor.alpha_p,
        n_planes=n_planes,
    )
