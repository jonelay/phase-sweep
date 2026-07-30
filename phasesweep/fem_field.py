"""
2D magnetostatic FEM field solver for PMSM air-gap flux density.

Physics: A-formulation, H1 scalar vector potential A_z.
Equation: -∇·(ν ∇A_z) = ∇×(ν_PM B_rem)|_z  in PM region
Source:   square-wave radial magnetization — the curl-curl source is the
          coercive field |H_c| = ν_PM·B_rem = B_rem/(μ0·μr_pm) (reduces to
          B_rem/μ0 only at μr_pm=1), per-pole sign alternation sign(cos(n_p θ))

Uses NGSolve (LGPL) for FEM. Analytical model: phasesweep.analytical;
harmonic decomposition: phasesweep.harmonics.
"""

from __future__ import annotations

import logging
import os
import tempfile
from math import pi
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ngsolve import CoefficientFunction, GridFunction, Mesh
    from ngsolve.comp import FESpace, LinearForm

from phasesweep.geometry import Geometry, default_inrunner

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
_MU0: float = 4e-7 * pi
_MIN_PICARD_RELAX: float = 0.02

# Interpole gaps narrower than this collapse to a full PM ring (α_p = 1.0):
# OCC boolean cuts on sub-0.5 mm arc slivers fail unreliably ("Could not
# divide Edge"). solve_field_fem warns and flags the collapse.
_MIN_ARC_GAP_M: float = 5e-4


def _interpole_gap_width(geo: Geometry, alpha_p: float, n_p: int) -> float:
    """Arc-length width of the interpole gap between adjacent magnet arcs.

    Evaluated at the *inner* PM-annulus radius (rotor iron behind the
    magnets: ``r_rotor`` inrunner, ``r_magnet`` outrunner). The gap is a
    radial wedge whose arc-length grows with radius, so the inner edge is
    its narrowest point — where the OCC boolean fails to cut the sliver
    (the ``_MIN_ARC_GAP_M`` collapse threshold) and where the pm_gap
    refinement must place its ≥2 elements. Using the outer radius here
    overestimated the gap on both topologies (fem v6 and earlier).
    """
    r_pm = geo.r_rotor if geo.topology == "inrunner" else geo.r_magnet
    return (1 - alpha_p) * pi / n_p * r_pm

# ---------------------------------------------------------------------------
# Mesh cache — keyed by (config_id, maxh_fraction, n_slots,
#                        slot_width_ratio, alpha_p, n_p, rotation)
# In-memory + optional disk persistence for subprocess workers.
# ---------------------------------------------------------------------------
_CacheKey = tuple[str, float, int, float, float, int, float]
_mesh_cache: dict[_CacheKey, Mesh] = {}
_disk_cache_dir: Path | None = None


def set_disk_cache_dir(path: Path | str | None) -> None:
    """Enable disk mesh cache. Set to None to disable."""
    global _disk_cache_dir
    if path is not None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        _disk_cache_dir = p
    else:
        _disk_cache_dir = None


def clear_mesh_cache() -> None:
    """Drop all cached meshes (for testing or memory management)."""
    _mesh_cache.clear()


def _cache_filename(key: _CacheKey) -> str:
    config_id, maxh_frac, n_slots, swr, alpha_p, n_p, _rotation = key
    # v5: injective float repr — the old 6-sig-fig %g could serve the wrong
    # mesh for params differing below 1e-6 relative. Rotated keys never
    # reach disk (see _save_to_disk), so no rotation suffix.
    # v6: interpole gap width measured at the inner radius (fem v7) changes
    # pm_gap and global maxh for every alpha_p<1 mesh — invalidate stale
    # .vol.gz.
    return (f"mesh_v6_{config_id}_{maxh_frac!r}_{n_slots}_{swr!r}"
            f"_{alpha_p!r}_{n_p}.vol.gz")


def _load_from_disk(key: _CacheKey) -> Mesh | None:
    if _disk_cache_dir is None or key[6] != 0.0:
        return None
    fpath = _disk_cache_dir / _cache_filename(key)
    if not fpath.exists():
        return None
    from netgen.meshing import Mesh as NetMesh
    from ngsolve import Mesh
    nm = NetMesh()
    try:
        nm.Load(str(fpath))
    except Exception:
        # Corrupt cache entry (e.g. interrupted write) — drop and rebuild.
        # Load raises from C++; the exception type is not stable across
        # netgen versions, so catch broadly.
        logging.warning("corrupt mesh cache %s, rebuilding", fpath.name)
        fpath.unlink(missing_ok=True)
        return None
    return Mesh(nm)


def _save_to_disk(key: _CacheKey, mesh: Mesh) -> None:
    # Rotated meshes (cogging sweeps: one per angle) stay in-memory only —
    # persisting them would grow the disk cache without bound.
    if _disk_cache_dir is None or key[6] != 0.0:
        return
    fpath = _disk_cache_dir / _cache_filename(key)
    # Unique temp name per writer — parallel workers on the same key must
    # not interleave; .vol.gz suffix keeps NGSolve from appending it again
    fd, tmp = tempfile.mkstemp(dir=_disk_cache_dir, suffix=".vol.gz")
    os.close(fd)
    mesh.ngmesh.Save(tmp)
    Path(tmp).replace(fpath)

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

# Default geometry for analytical model function defaults
_DEFAULT_GEO = default_inrunner()


def _bh_nu(B_mag: NDArray) -> NDArray:
    """Reluctivity ν(|B|) from generic soft iron B-H curve.

    Beyond the table end the curve continues with the vacuum differential
    slope, H = H_end + (B − B_end)/μ0, so the secant ν = H/B is monotone
    and approaches 1/μ0 from below. (np.interp alone would clamp H and
    make deeply saturated iron look magnetically softer with rising B.)
    """
    H = np.interp(B_mag, _BH_B, _BH_H)
    beyond = B_mag > _BH_B[-1]
    if np.any(beyond):
        H = np.where(beyond, _BH_H[-1] + (B_mag - _BH_B[-1]) / _MU0, H)
    B_safe = np.maximum(B_mag, 1e-12)
    nu = H / B_safe
    nu_0 = _BH_H[1] / _BH_B[1]
    return np.clip(nu, nu_0, 1 / _MU0)


# ---------------------------------------------------------------------------
# Region list from geometry
# ---------------------------------------------------------------------------

def _circle_spec(geo: Geometry) -> list[tuple[float, str]]:
    """Return (radius, material_name) pairs ordered outside-in.

    Inrunner: 4 regions (stator, airgap, pm, shaft). The shaft region is
    meshed as solid iron to r=0 — r_inner is NOT represented. Accurate for
    magnetic steel shafts (rotor yoke + shaft form one continuous iron
    disk); hollow or non-magnetic shafts are not modeled, so a thin rotor
    yoke over such a shaft cannot saturate realistically.
    Outrunner: 5 regions (stator, airgap, pm, yoke, air), or 6 when
    back_iron_thickness splits the yoke into back_iron (iron) + shell (mu_r=1).
    """
    if geo.topology == "inrunner":
        return [
            (geo.r_outer, "stator"),
            (geo.r_stator, "airgap"),
            (geo.r_magnet, "pm"),
            (geo.r_rotor, "shaft"),
        ]
    # outrunner: outside-in = rotor back-iron, pm, airgap, stator, air
    tail = [
        (geo.r_rotor, "pm"),
        (geo.r_magnet, "airgap"),
        (geo.r_stator, "stator"),
        (geo.r_inner, "air"),
    ]
    if geo.back_iron_thickness is not None:
        r_bi = geo.r_rotor + geo.back_iron_thickness
        return [(geo.r_outer, "shell"), (r_bi, "back_iron"), *tail]
    return [(geo.r_outer, "yoke"), *tail]


# ---------------------------------------------------------------------------
# Geometry builders — parameterized by Geometry
# ---------------------------------------------------------------------------

def _build_annular_regions(
    geo: Geometry,
) -> tuple[list[Any], Any | None, dict[str, Any]]:
    """Build annular ring faces from circle spec, outside-in.

    Returns (faces, innermost_or_None, named_faces_dict).
    named_faces_dict maps region name -> face for downstream operations.
    """
    from netgen.occ import WorkPlane

    specs = _circle_spec(geo)
    radii = [r for r, _ in specs]
    names = [n for _, n in specs]

    faces = []
    named: dict[str, Any] = {}
    for i in range(len(radii) - 1):
        f_big = WorkPlane().Circle(radii[i]).Face()
        f_small = WorkPlane().Circle(radii[i + 1]).Face()
        region = f_big - f_small
        region.faces.name = names[i]
        faces.append(region)
        named[names[i]] = region

    innermost = None
    if radii[-1] > 0:
        f_inner = WorkPlane().Circle(radii[-1]).Face()
        f_inner.faces.name = names[-1]
        faces.append(f_inner)
        named[names[-1]] = f_inner
        innermost = f_inner

    return faces, innermost, named


def _cut_slots(
    stator_face: Any,
    geo: Geometry,
    n_slots: int,
    slot_width_ratio: float,
    slot_depth: float,
    slot_opening_ratio: float = 0.0,
) -> tuple[Any, list[Any]]:
    """Cut slots from stator face. Returns (modified_stator, slot_faces).

    When slot_opening_ratio > 0 and differs from slot_width_ratio, cuts a
    trapezoidal slot (narrow opening at bore, wider body at yoke).
    """
    from math import cos, sin
    from math import pi as _pi

    from netgen.occ import WorkPlane

    delta_body = slot_width_ratio * _pi / n_slots
    delta_open = (slot_opening_ratio if slot_opening_ratio > 0
                  else slot_width_ratio) * _pi / n_slots

    tapered = abs(delta_open - delta_body) > 1e-9

    if geo.topology == "inrunner":
        r_bore = geo.r_stator
        r_bottom = geo.r_stator + slot_depth
    else:
        r_bore = geo.r_stator
        r_bottom = geo.r_stator - slot_depth

    # For stepped slots: opening depth as fraction of total slot depth
    from phasesweep.defaults import SLOT_OPENING_FRACTION
    if tapered:
        if geo.topology == "inrunner":
            r_step = r_bore + SLOT_OPENING_FRACTION * slot_depth
        else:
            r_step = r_bore - SLOT_OPENING_FRACTION * slot_depth

    R_big = 2.0 * geo.r_outer

    def _wedge_annulus(theta_k, delta, r_outer_clip, r_inner_clip):
        wp = WorkPlane()
        wp.MoveTo(0., 0.)
        wp.LineTo(R_big * cos(theta_k - delta),
                   R_big * sin(theta_k - delta))
        wp.LineTo(R_big * cos(theta_k + delta),
                   R_big * sin(theta_k + delta))
        wp.Close()
        f_o = WorkPlane().Circle(r_outer_clip).Face()
        f_i = WorkPlane().Circle(r_inner_clip).Face()
        return (wp.Face() * f_o) - f_i

    slot_faces = []
    for k in range(n_slots):
        theta_k = 2 * _pi * k / n_slots

        if not tapered:
            sf = _wedge_annulus(theta_k, delta_body,
                                max(r_bore, r_bottom), min(r_bore, r_bottom))
            sf.faces.name = "slot"
            stator_face = stator_face - sf
            slot_faces.append(sf)
        else:
            # Stepped slot: two separate cuts per slot
            if geo.topology == "inrunner":
                sf_open = _wedge_annulus(theta_k, delta_open, r_step, r_bore)
                sf_body = _wedge_annulus(theta_k, delta_body, r_bottom, r_step)
            else:
                sf_open = _wedge_annulus(theta_k, delta_open, r_bore, r_step)
                sf_body = _wedge_annulus(theta_k, delta_body, r_step, r_bottom)

            sf_open.faces.name = "slot"
            sf_body.faces.name = "slot"
            stator_face = stator_face - sf_open
            stator_face = stator_face - sf_body
            slot_faces.append(sf_open)
            slot_faces.append(sf_body)

    stator_face.faces.name = "stator"
    return stator_face, slot_faces


def slot_source_moment(
    geo: Geometry,
    n_slots: int | None = None,
    slot_width_ratio: float | None = None,
    n_p: int | None = None,
) -> float:
    """Radial second moment of the slot faces per unit angle (m²).

    S = Σ_parts w_part · (r_hi² − r_lo²) / 2 over the slot cross-section
    parts that ``_cut_slots`` builds (one annular sector, or opening +
    body when the slot is stepped), with w = angular duty (width / pitch).

    For the current-sheet source J_z = -j_s·k_w·cos(n_p·θ) over the slot
    faces, the fundamental of the ampere-turns-per-angle distribution is
    j_s·k_w·S: the θ-comb of uniformly spaced slots contributes its duty
    factor exactly (the cross terms cancel by symmetry unless 2·n_p is a
    multiple of n_slots), and behind high-μ teeth only a slot's total
    ampere-turns set the gap field, not their radial position. This is
    the geometry factor of the j_s ↔ phase-current mapping
    (``solver_params.j_s_from_phase_current``).

    Defaults mirror ``solve_field_fem``: n_slots / slot_width_ratio from
    the geometry, opening taper from ``geo.slot_opening_ratio``. Raises
    ValueError when the geometry builds no slot faces (n_slots == 0 or
    slot_depth == 0) — the same configurations on which j_s would raise —
    and, when ``n_p`` is given, when 2·n_p is a multiple of n_slots (the
    cross terms then add up to +2/π·duty-level error instead of
    cancelling; such combos put adjacent slots at electrical angle 0/π
    and cannot be wound as a balanced 3-phase machine anyway).
    """
    from phasesweep.defaults import SLOT_OPENING_FRACTION

    if n_slots is None:
        n_slots = geo.n_slots
    if slot_width_ratio is None:
        slot_width_ratio = geo.slot_width_ratio
    if n_slots == 0 or geo.slot_depth <= 0:
        raise ValueError(
            "slot_source_moment requires slot faces (n_slots > 0 and "
            "slot_depth > 0) — this geometry cannot carry a j_s source"
        )
    if n_p is not None and (2 * n_p) % n_slots == 0:
        raise ValueError(
            f"2*n_p={2 * n_p} is a multiple of n_slots={n_slots}: the "
            "slot-comb cross terms do not cancel and the j_s ↔ "
            "phase-current mapping would be off by tens of percent — "
            "this slot/pole combo is electrically degenerate (all slot "
            "EMF phasors collinear) and unsupported"
        )

    w_body = slot_width_ratio
    w_open = (geo.slot_opening_ratio if geo.slot_opening_ratio > 0
              else slot_width_ratio)
    tapered = abs(w_open - w_body) * pi / n_slots > 1e-9

    sign = 1.0 if geo.topology == "inrunner" else -1.0
    r_bore = geo.r_stator
    r_bottom = r_bore + sign * geo.slot_depth
    if not tapered:
        return w_body * abs(r_bottom**2 - r_bore**2) / 2
    r_step = r_bore + sign * SLOT_OPENING_FRACTION * geo.slot_depth
    return (w_open * abs(r_step**2 - r_bore**2) / 2
            + w_body * abs(r_bottom**2 - r_step**2) / 2)


def _cut_magnet_arcs(
    geo: Geometry,
    n_p: int,
    alpha_p: float,
    rotation: float = 0.0,
) -> tuple[list[Any], list[Any]]:
    """Cut PM annulus into discrete arcs + interpole gaps.

    Returns (arc_faces named "pm", gap_faces named "pm_gap").
    2*n_p arcs centered on pole axes at theta_k = k*pi/n_p + rotation.
    """
    from math import cos, sin
    from math import pi as _pi

    from netgen.occ import WorkPlane

    arc_faces = []
    gap_faces = []
    R_big = 2.0 * geo.r_outer

    # PM annulus radii for intersection
    if geo.topology == "inrunner":
        r_pm_outer = geo.r_magnet
        r_pm_inner = geo.r_rotor
    else:
        r_pm_outer = geo.r_rotor
        r_pm_inner = geo.r_magnet

    for k in range(2 * n_p):
        theta_k = _pi * k / n_p + rotation
        # Arc half-angle
        arc_half = alpha_p * _pi / (2 * n_p)
        # Gap half-angle (from arc edge to next arc edge)
        gap_half = (1 - alpha_p) * _pi / (2 * n_p)

        # Build arc wedge
        a_lo = theta_k - arc_half
        a_hi = theta_k + arc_half
        wp = WorkPlane()
        wp.MoveTo(0., 0.)
        wp.LineTo(R_big * cos(a_lo), R_big * sin(a_lo))
        wp.LineTo(R_big * cos(a_hi), R_big * sin(a_hi))
        wp.Close()
        wedge = wp.Face()

        f_outer = WorkPlane().Circle(r_pm_outer).Face()
        f_inner = WorkPlane().Circle(r_pm_inner).Face()
        arc = (wedge * f_outer) - f_inner
        arc.faces.name = "pm"
        arc_faces.append(arc)

        # Build gap wedge (between this arc and the next)
        if gap_half > 1e-10:
            g_lo = theta_k + arc_half
            g_hi = theta_k + arc_half + 2 * gap_half
            wp = WorkPlane()
            wp.MoveTo(0., 0.)
            wp.LineTo(R_big * cos(g_lo), R_big * sin(g_lo))
            wp.LineTo(R_big * cos(g_hi), R_big * sin(g_hi))
            wp.Close()
            wedge_g = wp.Face()

            gap = (wedge_g * f_outer) - f_inner
            gap.faces.name = "pm_gap"
            gap_faces.append(gap)

    return arc_faces, gap_faces


def _build_motor_geometry(
    geo: Geometry,
    *,
    n_slots: int = 0,
    slot_width_ratio: float = 0.6,
    slot_depth: float = 0.0,
    slot_opening_ratio: float = 0.0,
    alpha_p: float = 1.0,
    n_p: int = 0,
    rotation: float = 0.0,
) -> Any:
    """Composable motor geometry builder.

    1. Build annular regions (always)
    2. Cut magnet arcs (when alpha_p < 1), pole axes offset by ``rotation``
    3. Cut slots (when n_slots > 0), always stator-fixed
    """
    from netgen.occ import Glue

    faces, _innermost, named = _build_annular_regions(geo)

    # Cut magnet arcs if partial-pitch; sub-threshold gaps collapse to the
    # full ring (solve_field_fem warns and sets info["arcs_collapsed"])
    _do_arcs = (alpha_p < 1.0 and n_p > 0
                and _interpole_gap_width(geo, alpha_p, n_p) >= _MIN_ARC_GAP_M)
    if _do_arcs:
        pm_face = named["pm"]
        arc_faces, gap_faces = _cut_magnet_arcs(geo, n_p, alpha_p, rotation)
        faces = [f for f in faces if f is not pm_face]
        faces.extend(arc_faces)
        faces.extend(gap_faces)

    # Cut slots
    if n_slots > 0 and slot_depth > 0:
        stator_face = named["stator"]
        faces = [f for f in faces if f is not stator_face]
        modified_stator, slot_faces = _cut_slots(
            stator_face, geo, n_slots, slot_width_ratio, slot_depth,
            slot_opening_ratio,
        )
        faces.append(modified_stator)
        faces.extend(slot_faces)

    glued = Glue(faces)

    # Name outer boundary edges on the glued shape — edges created before
    # Glue lose their names, so match by circumference instead
    circumference = 2 * pi * _circle_spec(geo)[0][0]
    for e in glued.edges:
        if abs(e.mass - circumference) < 1e-6 * circumference:
            e.name = "outer"

    return glued


# Legacy wrappers (used by tests that call these directly)
def _build_geometry(geo: Geometry) -> Any:
    """Concentric-circle motor cross-section from Geometry."""
    return _build_motor_geometry(geo)


def _build_slotted_geometry(
    geo: Geometry,
    n_slots: int | None = None,
    slot_width_ratio: float | None = None,
    slot_depth: float | None = None,
) -> Any:
    """Motor cross-section with slots cut into the stator."""
    _n_slots = n_slots if n_slots is not None else geo.n_slots
    _swr = slot_width_ratio if slot_width_ratio is not None else geo.slot_width_ratio
    _sd = slot_depth if slot_depth is not None else geo.slot_depth
    return _build_motor_geometry(
        geo, n_slots=_n_slots, slot_width_ratio=_swr, slot_depth=_sd,
        slot_opening_ratio=geo.slot_opening_ratio,
    )


# ---------------------------------------------------------------------------
# FEM solver — new Geometry-aware signature
# ---------------------------------------------------------------------------

def _demoivre(n_p: int) -> tuple[CoefficientFunction, CoefficientFunction]:
    """Return (cos(n_p θ), sin(n_p θ)) as NGSolve CFs via De Moivre."""
    from ngsolve import sqrt, x, y
    r_cf = sqrt(x**2 + y**2)
    cos_t, sin_t = x / r_cf, y / r_cf
    cr, si = cos_t, sin_t
    for _ in range(n_p - 1):
        cr, si = cr * cos_t - si * sin_t, cr * sin_t + si * cos_t
    return cr, si


def _get_or_build_mesh(
    geo: Geometry,
    maxh_fraction: float,
    n_slots: int,
    slot_width_ratio: float,
    alpha_p: float,
    n_p: int,
    has_arcs: bool,
    rotation: float = 0.0,
) -> Mesh:
    """Motor cross-section mesh: in-process cache → disk cache → build."""
    from netgen.occ import OCCGeometry
    from ngsolve import Mesh

    cache_key: _CacheKey = (geo.config_id, maxh_fraction, n_slots,
                            slot_width_ratio, alpha_p, n_p, rotation)
    mesh = _mesh_cache.get(cache_key)
    if mesh is None:
        mesh = _load_from_disk(cache_key)
    if mesh is None:
        maxh = maxh_fraction * geo.r_outer
        slot_depth = geo.slot_depth if n_slots > 0 else 0.0
        shape = _build_motor_geometry(
            geo,
            n_slots=n_slots,
            slot_width_ratio=slot_width_ratio,
            slot_depth=slot_depth,
            slot_opening_ratio=geo.slot_opening_ratio,
            alpha_p=alpha_p,
            n_p=n_p,
            rotation=rotation,
        )
        if n_slots > 0:
            maxh = min(maxh, 0.03 * geo.r_outer)
        # Refine for interpole gaps — at least 2 elements across.
        # The global maxh keeps its 0.5 mm floor; the per-face maxh
        # below enforces the rule locally for 0.5–1.0 mm gaps where the
        # floor would otherwise under-resolve (a no-op for wider gaps).
        if has_arcs:
            gap_width = _interpole_gap_width(geo, alpha_p, n_p)
            clamp = max(gap_width * 0.5, _MIN_ARC_GAP_M)
            if clamp < maxh:
                # The §3.8 arc clamp binds: for mm-scale arc motors the
                # gap/2 term drives the global maxh below maxh_fraction·
                # r_outer, which is then inert.
                logging.info(
                    "interpole arc clamp: global maxh %.4g mm -> %.4g mm "
                    "(gap/2=%.4g mm, alpha_p=%.4g); maxh_fraction now inert",
                    maxh * 1e3, clamp * 1e3, gap_width * 0.5 * 1e3, alpha_p,
                )
            maxh = min(maxh, clamp)
            shape.faces["pm_gap"].maxh = gap_width * 0.5
        # Tie airgap element size to the gap width via per-face maxh so the
        # refinement stays local: thin-gap motors otherwise get a single
        # element across the gap (actuator: 0.26 mm gap vs 0.29 mm global
        # maxh), under-resolving B_r at the r_ag sampling circle.
        shape.faces["airgap"].maxh = abs(geo.r_stator - geo.r_magnet) / 3
        mesh = Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh=maxh))
        _save_to_disk(cache_key, mesh)
    _mesh_cache[cache_key] = mesh
    return mesh


def _picard_solve(
    mesh: Mesh,
    fes: FESpace,
    lf: LinearForm,
    gfu: GridFunction,
    nu: CoefficientFunction,
    region_names: list[str],
    iron_regions: set[str],
    max_picard: int,
    picard_tol: float,
    picard_relax: float,
    info: dict[str, float] | None,
) -> None:
    """Nonlinear BH solve: Picard fixed-point with adaptive relaxation.

    Solves in place into ``gfu``. Stops on the raw (unrelaxed) nu step;
    damps relax on raw-step growth and recovers it on decay.

    Raises:
        RuntimeError: If the iteration does not converge within
            ``max_picard`` iterations.
    """
    from ngsolve import L2, BilinearForm, GridFunction, dx, grad, sqrt

    u, v = fes.TnT()
    fes_nu = L2(mesh, order=0)
    gfu_nu = GridFunction(fes_nu)
    gfu_nu.Set(nu)

    is_iron_cf: dict[str, float] = {
        name: (1.0 if name in iron_regions else 0.0)
        for name in region_names
    }
    is_iron = mesh.MaterialCF(is_iron_cf)
    iron_indicator = GridFunction(fes_nu)
    iron_indicator.Set(is_iron)
    iron_mask = iron_indicator.vec.FV().NumPy() > 0.5

    a = BilinearForm(gfu_nu * grad(u) * grad(v) * dx)

    delta_raw_rel = float("inf")
    prev_raw_rel = float("inf")
    prev_applied_rel = float("inf")
    relax = picard_relax
    for _picard_i in range(max_picard):
        a.Assemble()
        gfu.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky") * lf.vec

        B_mag_cf = sqrt(grad(gfu)[0]**2 + grad(gfu)[1]**2)
        gfu_Bmag = GridFunction(fes_nu)
        gfu_Bmag.Set(B_mag_cf)
        B_vals = gfu_Bmag.vec.FV().NumPy().copy()

        nu_new = _bh_nu(B_vals)
        nu_old = gfu_nu.vec.FV().NumPy().copy()

        # Stop criterion on the raw (unrelaxed) step: relax-independent,
        # so the effective tolerance cannot loosen as relax shrinks
        # (the old relax-scaled criterion accepted raw residual 0.40 at
        # the relax floor).
        scale = np.max(np.abs(nu_old[iron_mask]))
        delta_raw_rel = (
            np.max(np.abs(nu_new[iron_mask] - nu_old[iron_mask])) / scale
        )

        if delta_raw_rel < picard_tol:
            # gfu was solved with nu_old and nu_new ~ nu_old: the pair
            # is a fixed point within tol — stop without updating.
            break

        if delta_raw_rel > prev_raw_rel:
            # Diverging: damp, and cap the applied step at the previous
            # applied step. This damping must stay — relax >= 0.35
            # diverges without it on saturating geometries.
            relax = max(relax * 0.5, _MIN_PICARD_RELAX)
            while (relax * delta_raw_rel > prev_applied_rel
                   and relax > _MIN_PICARD_RELAX):
                relax = max(relax * 0.5, _MIN_PICARD_RELAX)
        else:
            # Converging: recover toward the configured relaxation so a
            # saturation transient cannot pin relax at the floor for
            # the rest of the solve.
            relax = min(relax * 1.5, picard_relax)

        nu_update = nu_old.copy()
        nu_update[iron_mask] = (
            relax * nu_new[iron_mask]
            + (1 - relax) * nu_old[iron_mask]
        )
        gfu_nu.vec.FV().NumPy()[:] = nu_update
        prev_applied_rel = relax * delta_raw_rel
        prev_raw_rel = delta_raw_rel
    else:
        raise RuntimeError(
            f"Picard iteration did not converge in {max_picard} iterations "
            f"(residual: {delta_raw_rel:.4e}, tol: {picard_tol})"
        )
    if info is not None:
        info["picard_iterations"] = float(_picard_i + 1)
        info["picard_relax_final"] = float(relax)
        info["picard_residual"] = float(delta_raw_rel)


def _check_iron_saturation(
    mesh: Mesh,
    gfu: GridFunction,
    region_names: list[str],
    iron_regions: set[str],
    info: dict[str, float] | None,
) -> None:
    """Record peak |B| in iron into ``info``; warn above 1.8 T."""
    from ngsolve import L2, GridFunction, grad, sqrt

    B_mag_check = sqrt(grad(gfu)[0]**2 + grad(gfu)[1]**2)
    fes_L2_check = L2(mesh, order=0)
    gfu_Bcheck = GridFunction(fes_L2_check)
    gfu_Bcheck.Set(B_mag_check)
    iron_ind = GridFunction(fes_L2_check)
    iron_cf_check: dict[str, float] = {
        name: (1.0 if name in iron_regions else 0.0)
        for name in region_names
    }
    iron_ind.Set(mesh.MaterialCF(iron_cf_check))
    B_iron = gfu_Bcheck.vec.FV().NumPy()[iron_ind.vec.FV().NumPy() > 0.5]
    b_iron_max = float(np.max(B_iron)) if len(B_iron) > 0 else 0.0
    if info is not None:
        info["b_iron_max"] = b_iron_max
    if b_iron_max > 1.8:
        logging.warning(
            "peak |B| in stator iron = %.2f T (> 1.8 T saturation threshold)",
            b_iron_max,
        )


@overload
def solve_field_fem(
    geo: Geometry, n_p: int, B_rem: float, mu_r_pm: float, mu_r_fe: float,
    n_theta: int = ..., maxh_fraction: float = ...,
    return_full: Literal[False] = ...,
    n_slots: int | None = ..., j_s: float = ..., k_w: float = ...,
    slot_width_ratio: float | None = ..., nonlinear: bool = ...,
    max_picard: int = ..., picard_tol: float = ..., picard_relax: float = ...,
    alpha_p: float = ..., rotation: float = ..., sheet_phase: float = ...,
    info: dict[str, float] | None = ...,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]: ...

@overload
def solve_field_fem(
    geo: Geometry, n_p: int, B_rem: float, mu_r_pm: float, mu_r_fe: float,
    n_theta: int = ..., maxh_fraction: float = ...,
    return_full: Literal[True] = ...,
    n_slots: int | None = ..., j_s: float = ..., k_w: float = ...,
    slot_width_ratio: float | None = ..., nonlinear: bool = ...,
    max_picard: int = ..., picard_tol: float = ..., picard_relax: float = ...,
    alpha_p: float = ..., rotation: float = ..., sheet_phase: float = ...,
    info: dict[str, float] | None = ...,
) -> tuple[NDArray[np.floating], NDArray[np.floating], Any, Any]: ...

def solve_field_fem(
    geo: Geometry,
    n_p: int,
    B_rem: float,
    mu_r_pm: float,
    mu_r_fe: float,
    n_theta: int = 360,
    maxh_fraction: float = 0.05,
    return_full: bool = False,
    n_slots: int | None = None,
    j_s: float = 0.0,
    k_w: float = 0.966,
    slot_width_ratio: float | None = None,
    nonlinear: bool = False,
    max_picard: int | None = None,
    picard_tol: float | None = None,
    picard_relax: float | None = None,
    alpha_p: float = 1.0,
    rotation: float = 0.0,
    sheet_phase: float = 0.0,
    info: dict[str, float] | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating]] | tuple[NDArray[np.floating], NDArray[np.floating], Any, Any]:
    """Geometry-aware FEM solver with discrete magnet arcs.

    Inrunner idealization: everything inside ``r_rotor`` is solid iron —
    ``geo.r_inner`` does not enter the mesh (valid for magnetic steel
    shafts; hollow or non-magnetic shafts are not represented).

    ``mu_r_fe`` sets the iron reluctivity of the linear solve only; with
    ``nonlinear=True`` iron follows the generic B-H table
    and ``mu_r_fe`` merely seeds Picard iteration 0 — the converged
    result is insensitive to its value.

    When ``alpha_p < 1.0``, builds discrete magnet arcs with interpole gaps
    named ``"pm_gap"`` (air reluctivity, no source). The square-wave source
    term (uniform magnitude, per-pole sign) integrates only over ``"pm"``
    regions. Pass physical remanence as ``B_rem`` (no sin(πα_p/2) or 4/π
    pre-scaling needed). Interpole gaps narrower than 0.5 mm
    (``_MIN_ARC_GAP_M``) cannot be cut reliably by OCC: the arcs collapse
    to a full PM ring and the solve is effectively ``alpha_p = 1.0`` — a
    warning is logged and ``info["arcs_collapsed"]`` is set. This fires
    for ``alpha_p >= 1 - _MIN_ARC_GAP_M * n_p / (pi * r_pm_inner)``, where
    ``r_pm_inner`` is the inner PM-annulus radius (``r_rotor`` inrunner,
    ``r_magnet`` outrunner) — the gap wedge is narrowest there.

    ``n_slots`` and ``slot_width_ratio`` default to the geometry's values;
    pass an explicit value (including 0) to override, e.g. ``n_slots=0``
    for a smooth-bore solve of a slotted geometry.

    Armature reaction (``j_s != 0``) is an equivalent current sheet
    ``J_z = -j_s * k_w * cos(n_p * θ - sheet_phase)`` integrated over the
    slot regions: ``k_w`` scales the sheet to the effective (distributed)
    winding, and ``sheet_phase`` (electrical radians) sets the current
    angle. At ``sheet_phase = 0`` the MMF lies purely on the q-axis (the
    magnet d-axis sits at θ=0) — the torque-producing sheet. At
    ``sheet_phase = ±π/2`` it lies on the d-axis and produces no
    order-n_p interaction torque; ``+π/2`` opposes the magnet flux
    (demagnetizing / flux-weakening) and ``-π/2`` aids it, on both
    topologies (sign pinned numerically, see tests). The j_s ↔
    phase-current mapping is phase-invariant: the uniform slot comb
    rotates the sampled fundamental phasor without changing its
    amplitude, so the same ``j_s`` represents the same current magnitude
    at any angle. Requires slots: ``j_s != 0`` with ``n_slots == 0``
    raises ValueError.

    ``rotation`` rotates the magnet pattern (arc positions and the
    per-pole magnetization sign, i.e. the whole rotor) by a mechanical
    angle in radians; the stator (slots, current sheet) stays fixed.
    Sweeping ``rotation`` over one cogging period 2π/lcm(n_slots, 2·n_p)
    with ``j_s = 0`` and ``maxwell_stress_torque`` per position yields
    the cogging waveform. A combined ``j_s != 0`` + ``rotation != 0``
    solve is a fixed-current locked-rotor snapshot, not a synchronous
    operating point; setting ``sheet_phase = n_p * rotation + γ`` holds
    the sheet MMF at a fixed electrical angle γ from the rotated magnet
    axis. Each distinct rotation builds and caches its own mesh.

    ``info``, when provided, is populated with solve diagnostics:
    ``b_iron_max`` (peak |B| in iron regions, T), ``arcs_collapsed``
    (only when a sub-0.5 mm interpole gap collapsed the magnet arcs, see
    above) and — nonlinear only — ``picard_iterations``,
    ``picard_relax_final`` and ``picard_residual`` (raw, i.e. unrelaxed,
    step — the stop criterion).

    Contract:
        Preconditions:
            - ``n_p`` >= 1.
            - ``maxh_fraction`` > 0.
            - ``picard_relax`` > 0 when ``nonlinear=True``.

        Raises:
            RuntimeError: If ``nonlinear=True`` and Picard iteration
                does not converge within ``max_picard`` iterations.
    """
    from ngsolve import (
        H1,
        BilinearForm,
        GridFunction,
        IfPos,
        LinearForm,
        dx,
        grad,
        sqrt,
        x,
        y,
    )

    from phasesweep.defaults import (
        PICARD_MAX_ITERATIONS,
        PICARD_RELAXATION,
        PICARD_TOLERANCE,
    )
    if max_picard is None:
        max_picard = PICARD_MAX_ITERATIONS
    if picard_tol is None:
        picard_tol = PICARD_TOLERANCE
    if picard_relax is None:
        picard_relax = PICARD_RELAXATION
    if n_slots is None:
        n_slots = geo.n_slots
    if slot_width_ratio is None:
        slot_width_ratio = geo.slot_width_ratio
    if j_s != 0.0 and n_slots == 0:
        raise ValueError(
            "j_s != 0 requires n_slots > 0: the armature current sheet "
            "integrates over slot regions and would be silently ignored "
            "on a smooth-bore geometry"
        )

    # Determine if discrete arcs will actually be built (same threshold as
    # _build_motor_geometry). A sub-threshold gap collapses the arcs to a
    # full PM ring — the FEM then solves α_p = 1.0, not the requested α_p.
    _wants_arcs = alpha_p < 1.0 and n_p > 0
    _has_arcs = (_wants_arcs
                 and _interpole_gap_width(geo, alpha_p, n_p) >= _MIN_ARC_GAP_M)
    if _wants_arcs and not _has_arcs:
        logging.warning(
            "interpole gap %.3g mm < %.3g mm OCC limit: magnet arcs "
            "collapsed, FEM solves alpha_p=1.0 instead of requested "
            "alpha_p=%.4g (fundamental error ~1/sin(pi*alpha_p/2) - 1)",
            _interpole_gap_width(geo, alpha_p, n_p) * 1e3,
            _MIN_ARC_GAP_M * 1e3, alpha_p,
        )
        if info is not None:
            info["arcs_collapsed"] = 1.0

    mesh = _get_or_build_mesh(geo, maxh_fraction, n_slots, slot_width_ratio,
                              alpha_p, n_p, _has_arcs, rotation)

    if j_s != 0.0 and "slot" not in set(mesh.GetMaterials()):
        # n_slots > 0 alone does not guarantee slot faces — a zero
        # slot_depth builds none, and the current sheet would integrate
        # over an empty region (this silently ignored j_s before).
        raise ValueError(
            "j_s != 0 but the mesh has no slot regions (slot_depth == 0?): "
            "the armature current sheet would be silently ignored"
        )

    # Build material property dicts from circle spec regions
    region_names = [name for _, name in _circle_spec(geo)]
    if n_slots > 0:
        region_names.append("slot")
    if _has_arcs:
        region_names.append("pm_gap")

    iron_regions = {"stator", "shaft", "yoke", "back_iron"}

    nu_cf: dict[str, float] = {}
    for name in region_names:
        if name == "pm":
            nu_cf[name] = 1 / (_MU0 * mu_r_pm)
        elif name in iron_regions:
            nu_cf[name] = 1 / (_MU0 * mu_r_fe)
        else:
            nu_cf[name] = 1 / _MU0
    nu = mesh.MaterialCF(nu_cf)

    nu_pm = 1 / (_MU0 * mu_r_pm)
    cos_np, sin_np = _demoivre(n_p)

    r_cf = sqrt(x**2 + y**2)
    cos_t = x / r_cf
    sin_t = y / r_cf
    # Square-wave radial magnetization: the curl-curl source is the coercive
    # field H_c = B_rem/(μ0·μr_pm) = nu_pm·B_rem (reduces to B_rem/μ0 only at
    # μr_pm=1), uniform within each pole, sign alternating per pole. Arcs are
    # centered at θ_k = kπ/n_p
    # where cos(n_p·θ) = ±1, so the sign is constant within each arc; at
    # α_p = 1 the sign flips inside the full ring at pole boundaries.
    # Rotate the sign pattern with the magnet arcs: cos(n_p·(θ − rotation)).
    # The rotation == 0 branch keeps the CF expression bit-identical to the
    # pre-rotation solver.
    if rotation != 0.0:
        cos_np_rot = (cos_np * float(np.cos(n_p * rotation))
                      + sin_np * float(np.sin(n_p * rotation)))
    else:
        cos_np_rot = cos_np
    M_r = nu_pm * B_rem * IfPos(cos_np_rot, 1.0, -1.0)
    mx_cf: dict[str, CoefficientFunction | float] = {}
    my_cf: dict[str, CoefficientFunction | float] = {}
    for name in region_names:
        if name == "pm":
            mx_cf[name] = M_r * cos_t
            my_cf[name] = M_r * sin_t
        else:
            mx_cf[name] = 0.
            my_cf[name] = 0.
    Mx = mesh.MaterialCF(mx_cf)
    My = mesh.MaterialCF(my_cf)

    fes = H1(mesh, order=3, dirichlet="outer")
    u, v = fes.TnT()

    lf_expr = (Mx * grad(v)[1] - My * grad(v)[0]) * dx("pm")
    if n_slots > 0 and j_s != 0.0:
        # Shift the sheet wave by the electrical angle sheet_phase:
        # cos(n_p·θ − β) via De Moivre, same pattern as rotation above.
        # The sheet_phase == 0 branch keeps the CF bit-identical.
        if sheet_phase != 0.0:
            cos_np_sheet = (cos_np * float(np.cos(sheet_phase))
                            + sin_np * float(np.sin(sheet_phase)))
        else:
            cos_np_sheet = cos_np
        lf_expr = lf_expr + (-j_s * k_w * cos_np_sheet) * v * dx("slot")
    lf = LinearForm(lf_expr)
    lf.Assemble()
    gfu = GridFunction(fes)

    if not nonlinear:
        a = BilinearForm(nu * grad(u) * grad(v) * dx)
        a.Assemble()
        gfu.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky") * lf.vec
    else:
        _picard_solve(mesh, fes, lf, gfu, nu, region_names, iron_regions,
                      max_picard, picard_tol, picard_relax, info)

    _check_iron_saturation(mesh, gfu, region_names, iron_regions, info)

    theta, B_r = sample_Br(mesh, gfu, geo.r_ag, n_theta)
    if return_full:
        return theta, B_r, mesh, gfu
    return theta, B_r


def sample_Br(
    mesh: Mesh, gfu: GridFunction, radius: float, n_theta: int = 360,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Sample B_r = (1/r) ∂A_z/∂θ on a circle of the given radius from a
    solved A_z GridFunction (as returned by solve_field_fem(return_full=True)).

    Evaluates grad(A_z) directly at the sample points (vectorized) — no
    global L2 projection.
    """
    from ngsolve import grad, sqrt, x, y

    r_cf = sqrt(x**2 + y**2)
    B_r_cf = (grad(gfu)[1] * x - grad(gfu)[0] * y) / r_cf

    theta = np.linspace(0, 2 * pi, n_theta, endpoint=False)
    mips = mesh(radius * np.cos(theta), radius * np.sin(theta))
    B_r = np.asarray(B_r_cf(mips), dtype=float).reshape(-1)
    return theta, B_r


def maxwell_stress_torque(
    mesh: Mesh, gfu: GridFunction, radius: float, L_stk: float = 1.0,
    n_theta: int = 720,
) -> float:
    """Torque (N·m about z) on the material enclosed by a circle of the
    given radius, via the Maxwell stress tensor evaluated on that circle.

        tau = (L_stk · r² / μ0) · ∮ B_r · B_θ dθ

    from the solved A_z (B_r = (1/r)·∂A/∂θ, B_θ = −∂A/∂r). The circle
    must lie in a source-free air region (the air gap): there the
    integral is contour-independent, which doubles as the correctness
    check (see tests). Uniform grid over a periodic integrand — the
    rectangle rule is spectrally accurate. L_stk = 1.0 gives torque per
    metre of stack.

    Sign: torque on the enclosed region — the rotor for an inrunner; for
    an outrunner (rotor outside the sampling circle) the rotor torque is
    the negative.
    """
    B_r, B_t = _sample_B_polar(mesh, gfu, radius, n_theta)
    integral = float(np.sum(B_r * B_t)) * (2 * pi / n_theta)
    return L_stk * radius * radius / _MU0 * integral


def _sample_B_polar(
    mesh: Mesh, gfu: GridFunction, radius: float, n_theta: int,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """(B_r, B_θ) on a uniform circle of the given radius from solved A_z."""
    from ngsolve import grad, sqrt, x, y

    r_cf = sqrt(x**2 + y**2)
    B_r_cf = (grad(gfu)[1] * x - grad(gfu)[0] * y) / r_cf
    B_t_cf = -(grad(gfu)[0] * x + grad(gfu)[1] * y) / r_cf
    theta = np.linspace(0, 2 * pi, n_theta, endpoint=False)
    mips = mesh(radius * np.cos(theta), radius * np.sin(theta))
    return (np.asarray(B_r_cf(mips), dtype=float).reshape(-1),
            np.asarray(B_t_cf(mips), dtype=float).reshape(-1))


def maxwell_interaction_torque_order(
    mesh: Mesh,
    gfu_a: GridFunction,
    gfu_b: GridFunction,
    radius: float,
    order: int,
    L_stk: float = 1.0,
    n_theta: int = 720,
) -> float:
    """Single-space-harmonic interaction torque (N·m about z) between two
    solved fields on the same mesh, from the Maxwell-stress cross terms:

        tau_n = (2π·L_stk·r²/μ0) · 2·Re[ĉ_r^a(n)·ĉ_θ^b(n)* + ĉ_r^b(n)·ĉ_θ^a(n)*]

    with ĉ the complex Fourier coefficients of B_r/B_θ on the circle.
    At order = n_p between a magnet-only and an armature-only solve this
    is the synchronous mean torque — the piece that closes to the circuit
    tier's 1.5·n_p·ψf·i_q. The full static ``maxwell_stress_torque``
    additionally carries position-locked slot-harmonic ripple (armature
    comb sidebands at m·n_slots ± n_p meeting magnet space harmonics of
    the same order), which averages out under rotation.

    The circle must lie in the source-free gap annulus; both solves must
    share the mesh (same geometry/maxh — the mesh cache guarantees this).
    Sign convention matches maxwell_stress_torque.
    """
    B_r_a, B_t_a = _sample_B_polar(mesh, gfu_a, radius, n_theta)
    B_r_b, B_t_b = _sample_B_polar(mesh, gfu_b, radius, n_theta)
    c_r_a = np.fft.rfft(B_r_a)[order] / n_theta
    c_t_a = np.fft.rfft(B_t_a)[order] / n_theta
    c_r_b = np.fft.rfft(B_r_b)[order] / n_theta
    c_t_b = np.fft.rfft(B_t_b)[order] / n_theta
    cross = 2.0 * np.real(c_r_a * np.conj(c_t_b) + c_r_b * np.conj(c_t_a))
    return float(2 * pi * L_stk * radius * radius / _MU0 * cross)


def sample_magnet_Bm(
    mesh: Mesh,
    gfu: GridFunction,
    geo: Geometry,
    n_p: int,
    alpha_p: float = 1.0,
    rotation: float = 0.0,
    n_r: int = 8,
    n_arc: int = 24,
) -> tuple[NDArray[np.floating], NDArray[np.floating],
           NDArray[np.floating], NDArray[np.floating]]:
    """Flux density along the magnetization direction inside the magnets.

    Samples B_m = sign(M_r)·B_r on a cell-centered polar grid over the PM
    annulus restricted to the magnet arcs (n_r radii × n_arc angles per
    pole, 2·n_p poles). B_m is the magnet operating point in its own
    magnetization direction — the quantity a demag criterion compares
    against the knee B_knee(T): irreversible demagnetization onsets where
    B_m drops below it. Returns flat arrays ``(r, theta, B_m, w)`` with
    ``w`` the fractional area weights (∝ r, summing to 1) for
    volume-fraction statistics.

    Pass the same ``n_p`` / ``alpha_p`` / ``rotation`` the field was
    solved with — the grid mirrors the solver's arc placement (arcs
    centered at θ_k = k·π/n_p + rotation, half-width α_p·π/(2·n_p), sign
    (−1)^k) and cell-centering keeps every point strictly inside the
    magnet material, away from the pm/pm_gap and annulus interfaces.

    α_p = 1 caveat: with a full-ring square-wave magnet, adjacent
    opposite poles touch and the ideal (infinitely sharp) magnetization
    transition drives a locally huge opposing self-field at the
    pole-boundary corners — the sampled minimum sits there and barely
    responds to armature current. Real magnets have finite transition
    zones, so a min-based demag criterion is over-conservative at
    α_p = 1; with discrete arcs (α_p < 1) the interpole gap removes the
    corner and the minimum tracks the armature d-axis field as expected.
    """
    from ngsolve import grad, sqrt, x, y

    if n_p < 1:
        raise ValueError("n_p must be >= 1")
    if not 0.0 < alpha_p <= 1.0:
        raise ValueError("alpha_p must be in (0, 1]")

    if geo.topology == "inrunner":
        r_lo, r_hi = geo.r_rotor, geo.r_magnet
    else:
        r_lo, r_hi = geo.r_magnet, geo.r_rotor

    r_grid = r_lo + (np.arange(n_r) + 0.5) * (r_hi - r_lo) / n_r
    arc_half = alpha_p * pi / (2 * n_p)
    t_local = -arc_half + (np.arange(n_arc) + 0.5) * (2 * arc_half) / n_arc

    r_all, t_all, s_all = [], [], []
    for k in range(2 * n_p):
        theta_k = pi * k / n_p + rotation
        tt, rr = np.meshgrid(theta_k + t_local, r_grid)
        r_all.append(rr.ravel())
        t_all.append(tt.ravel())
        s_all.append(np.full(rr.size, 1.0 if k % 2 == 0 else -1.0))
    r = np.concatenate(r_all)
    theta = np.concatenate(t_all)
    sign = np.concatenate(s_all)

    r_cf = sqrt(x**2 + y**2)
    B_r_cf = (grad(gfu)[1] * x - grad(gfu)[0] * y) / r_cf
    mips = mesh(r * np.cos(theta), r * np.sin(theta))
    B_r = np.asarray(B_r_cf(mips), dtype=float).reshape(-1)

    w = r / np.sum(r)
    return r, theta, sign * B_r, w


def demag_margin(
    mesh: Mesh,
    gfu: GridFunction,
    geo: Geometry,
    n_p: int,
    B_knee: float,
    alpha_p: float = 1.0,
    rotation: float = 0.0,
    n_r: int = 8,
    n_arc: int = 24,
) -> dict[str, float]:
    """Worst-point demag margin of the solved field against a knee B_knee.

    Thin statistics layer over :func:`sample_magnet_Bm`: ``B_m_min`` (T,
    worst magnet operating point), ``margin`` (B_m_min − B_knee, negative
    means demag onset somewhere), ``frac_below_knee`` (area fraction of
    magnet cross-section with B_m < B_knee) and the worst point's
    ``r_min`` / ``theta_min``. The caller supplies B_knee at the magnet
    operating temperature (grade data — not derived here).
    """
    r, theta, B_m, w = sample_magnet_Bm(mesh, gfu, geo, n_p,
                                        alpha_p=alpha_p, rotation=rotation,
                                        n_r=n_r, n_arc=n_arc)
    i = int(np.argmin(B_m))
    return {
        "B_m_min": float(B_m[i]),
        "margin": float(B_m[i] - B_knee),
        "frac_below_knee": float(np.sum(w[B_m < B_knee])),
        "r_min": float(r[i]),
        "theta_min": float(theta[i]),
    }


# ---------------------------------------------------------------------------
# 3D geometry export
# ---------------------------------------------------------------------------

def build_cross_section(
    geo: Geometry,
    slotted: bool = True,
    alpha_p: float = 1.0,
    n_p: int = 0,
) -> Any:
    """Build the 2D OCC cross-section shape for a motor geometry."""
    n_slots = geo.n_slots if slotted else 0
    return _build_motor_geometry(
        geo,
        n_slots=n_slots,
        slot_width_ratio=geo.slot_width_ratio,
        slot_depth=geo.slot_depth,
        slot_opening_ratio=geo.slot_opening_ratio,
        alpha_p=alpha_p,
        n_p=n_p,
    )


def geometry_to_step(
    geo: Geometry,
    path: str | Path,
    L_stk: float | None = None,
    slotted: bool = True,
    alpha_p: float = 1.0,
    n_p: int = 0,
) -> Path:
    """Export motor geometry as a 3D STEP file (extruded cross-section)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    face = build_cross_section(geo, slotted=slotted, alpha_p=alpha_p, n_p=n_p)
    if L_stk is not None and L_stk > 0:
        solid = face.Extrude(L_stk)
    else:
        solid = face
    solid.WriteStep(str(out))
    return out


# ---------------------------------------------------------------------------
# Rasterisation
# ---------------------------------------------------------------------------

def rasterise_cross_section(
    mesh: Mesh,
    gfu: GridFunction,
    n_grid: int = 100,
    r_bound: float = _DEFAULT_GEO.r_outer,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Evaluate A_z and |B| on a regular Cartesian grid covering the motor disk."""
    from netgen.libngpy._meshing import NgException
    from ngsolve import L2, GridFunction, grad, sqrt

    B_mag_cf = sqrt(grad(gfu)[0]**2 + grad(gfu)[1]**2)
    fes_L2 = L2(mesh, order=3)
    gfu_Bmag = GridFunction(fes_L2)
    gfu_Bmag.Set(B_mag_cf)

    coords = np.linspace(-r_bound, r_bound, n_grid)
    xi, yi = np.meshgrid(coords, coords)

    Az   = np.full((n_grid, n_grid), np.nan)
    Bmag = np.full((n_grid, n_grid), np.nan)

    r2 = xi**2 + yi**2
    inside = r2 < r_bound**2

    failed = 0
    for i in range(n_grid):
        for j in range(n_grid):
            if not inside[i, j]:
                continue
            try:
                mp = mesh(xi[i, j], yi[i, j])
                Az[i, j]   = gfu(mp)
                Bmag[i, j] = gfu_Bmag(mp)
            except NgException:
                failed += 1

    if failed:
        logging.warning("rasterise: %d points failed evaluation", failed)

    return xi, yi, Az, Bmag
