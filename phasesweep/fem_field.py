"""
2D magnetostatic FEM field solver for PMSM air-gap flux density.

Physics: A-formulation, H1 scalar vector potential A_z.
Equation: -∇·(ν ∇A_z) = ∇×(ν_PM B_rem)|_z  in PM region
Source:   square-wave radial magnetization — uniform |M_r| = B_rem/μ0 with
          per-pole sign alternation, sign(cos(n_p θ))

Uses NGSolve (LGPL) for FEM. CuPy (GPU) for batch harmonic decomposition.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import warnings
from math import pi
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ngsolve import CoefficientFunction, GridFunction, Mesh

from phasesweep.geometry import Geometry, default_inrunner

try:
    import cupy as cp  # type: ignore[import-unresolved]
    _CUPY = True
except ImportError:
    cp = None  # type: ignore[assignment]
    _CUPY = False

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
_MU0: float = 4e-7 * pi
_MIN_PICARD_RELAX: float = 0.02

# ---------------------------------------------------------------------------
# Mesh cache — keyed by (config_id, maxh_fraction, n_slots,
#                        slot_width_ratio, alpha_p, n_p)
# In-memory + optional disk persistence for subprocess workers.
# ---------------------------------------------------------------------------
_CacheKey = tuple[str, float, int, float, float, int]
_mesh_cache: dict[_CacheKey, Any] = {}
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
    config_id, maxh_frac, n_slots, swr, alpha_p, n_p = key
    return f"mesh_v2_{config_id}_{maxh_frac:.6g}_{n_slots}_{swr:.6g}_{alpha_p:.6g}_{n_p}.vol.gz"


def _load_from_disk(key: _CacheKey) -> Mesh | None:
    if _disk_cache_dir is None:
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
    if _disk_cache_dir is None:
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
    """Reluctivity ν(|B|) from generic soft iron B-H curve."""
    H = np.interp(B_mag, _BH_B, _BH_H)
    B_safe = np.maximum(B_mag, 1e-12)
    nu = H / B_safe
    nu_0 = _BH_H[1] / _BH_B[1]
    return np.clip(nu, nu_0, 1 / _MU0)


# ---------------------------------------------------------------------------
# Region list from geometry
# ---------------------------------------------------------------------------

def _circle_spec(geo: Geometry) -> list[tuple[float, str]]:
    """Return (radius, material_name) pairs ordered outside-in.

    Inrunner: 4 regions (stator, airgap, pm, shaft).
    Outrunner: 5 regions (stator, airgap, pm, yoke, air).
    """
    if geo.topology == "inrunner":
        return [
            (geo.r_outer, "stator"),
            (geo.r_stator, "airgap"),
            (geo.r_magnet, "pm"),
            (geo.r_rotor, "shaft"),
        ]
    # outrunner: outside-in = yoke, pm, airgap, stator, air
    return [
        (geo.r_outer, "yoke"),
        (geo.r_rotor, "pm"),
        (geo.r_magnet, "airgap"),
        (geo.r_stator, "stator"),
        (geo.r_inner, "air"),
    ]


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


def _cut_magnet_arcs(
    pm_face: Any,
    geo: Geometry,
    n_p: int,
    alpha_p: float,
) -> tuple[list[Any], list[Any]]:
    """Cut PM annulus into discrete arcs + interpole gaps.

    Returns (arc_faces named "pm", gap_faces named "pm_gap").
    2*n_p arcs centered on pole axes at theta_k = k*pi/n_p.
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
        theta_k = _pi * k / n_p
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
) -> Any:
    """Composable motor geometry builder.

    1. Build annular regions (always)
    2. Cut magnet arcs (when alpha_p < 1)
    3. Cut slots (when n_slots > 0)
    """
    from netgen.occ import Glue

    faces, _innermost, named = _build_annular_regions(geo)

    # Cut magnet arcs if partial-pitch (skip if gap is negligible)
    _do_arcs = alpha_p < 1.0 and n_p > 0
    if _do_arcs:
        r_pm = geo.r_magnet if geo.topology == "inrunner" else geo.r_rotor
        gap_width = (1 - alpha_p) * pi / n_p * r_pm
        if gap_width < 5e-4:  # < 0.5 mm — too thin for reliable OCC booleans
            _do_arcs = False
    if _do_arcs:
        pm_face = named["pm"]
        arc_faces, gap_faces = _cut_magnet_arcs(pm_face, geo, n_p, alpha_p)
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


def _derive_B_rem(
    psi_f: float, n_p: int, N: int, k_w: float, L_stk: float,
    *,
    r_stator: float = _DEFAULT_GEO.r_stator,
    r_magnet: float = _DEFAULT_GEO.r_magnet,
    r_rotor: float = _DEFAULT_GEO.r_rotor,
    mu_r_pm: float = 1.05,
    alpha_p: float = 1.0,
    r_stator_c: float | None = None,
    r_magnet_c: float | None = None,
) -> float:
    """Invert psi_f to magnet remanence B_rem via Zhu & Howe transfer ratio.

    Mirrors psi_f_carter: Carter-adjusted radii (r_stator_c, r_magnet_c)
    for the field solution when given, original bore for the evaluation
    point and the winding formula.
    """
    tr = zhu_howe_Br(
        np.array([0.0]), n_p, B_rem=1.0, r_eval=r_stator,
        r_stator=r_stator_c if r_stator_c is not None else r_stator,
        r_magnet=r_magnet_c if r_magnet_c is not None else r_magnet,
        r_rotor=r_rotor, mu_r_pm=mu_r_pm,
        alpha_p=alpha_p,
    )[0]
    B_ag_peak = psi_f * n_p / (2 * N * k_w * r_stator * L_stk)
    return B_ag_peak / tr


def carter_factor(
    r_stator: float,
    n_slots: int,
    slot_opening_ratio: float,
    g_prime: float,
) -> float:
    """Carter coefficient for PM machine per Zhu & Howe 1993 III Eq 16.

    slot_opening_ratio is the slot OPENING over the slot pitch at the bore —
    not the slot body width ratio, which over-corrects.
    g_prime is the effective airgap: g + h_m / mu_r_pm.
    Returns 1.0 when n_slots == 0.
    """
    if n_slots == 0:
        return 1.0
    from math import atan, log
    from math import sqrt as _sqrt
    tau_s = 2 * pi * r_stator / n_slots
    b_o = slot_opening_ratio * tau_s
    u = b_o / (2 * g_prime)
    gamma = (4 / pi) * (u * atan(u) - log(_sqrt(1 + u**2)))
    return tau_s / (tau_s - gamma * g_prime)


def carter_adjusted_radii(geo: Geometry, mu_r_pm: float) -> tuple[float, float, float]:
    """Carter modified-geometry correction: (r_stator, r_magnet, k_c).

    Widens the air gap by delta_g = (k_c - 1)·g' by moving the stator bore
    away from the magnets; r_magnet is deliberately unchanged. Returns the
    original radii with k_c = 1.0 for smooth bores. Uses the slot opening
    ratio at the bore, falling back to slot_width_ratio when unset.
    """
    r_stator = geo.r_stator
    r_magnet = geo.r_magnet
    sor = geo.slot_opening_ratio if geo.slot_opening_ratio > 0 else geo.slot_width_ratio
    if not (geo.n_slots > 0 and sor > 0 and geo.slot_depth > 0):
        return r_stator, r_magnet, 1.0
    if geo.slot_opening_ratio <= 0:
        warnings.warn(
            "slot_opening_ratio unset; Carter factor falling back to "
            f"slot_width_ratio={geo.slot_width_ratio:.3g} (slot body width), "
            "which over-corrects — set slot_opening_ratio to the bore "
            "opening fraction",
            stacklevel=2,
        )
    g = abs(r_stator - r_magnet)
    h_m = abs(r_magnet - geo.r_rotor)
    g_prime = g + h_m / mu_r_pm
    k_c = carter_factor(r_stator, geo.n_slots, sor, g_prime)
    delta_g = (k_c - 1) * g_prime
    if geo.topology == "inrunner":
        return r_stator + delta_g, r_magnet, k_c
    return r_stator - delta_g, r_magnet, k_c


def end_effect_factor(L_stk: float, g_eff: float) -> float:
    """Russell-Norsworthy end-effect correction for short-stack machines.

    Returns k_end in (0, 1] — multiply B_g1 by k_end to account for axial
    flux leakage at stack ends.  k_end → 1 for L_stk >> g_eff.

    g_eff is the effective airgap: g + h_m / mu_r_pm.
    """
    if L_stk <= 0 or g_eff <= 0:
        return 1.0
    ratio = pi * L_stk / (2 * g_eff)
    from math import exp
    return 1.0 - (2 * g_eff) / (pi * L_stk) * (1.0 - exp(-ratio))


def zhu_howe_Br(
    theta: NDArray[np.floating],
    n_p: int,
    B_rem: float,
    r_eval: float | None = None,
    *,
    r_stator: float = _DEFAULT_GEO.r_stator,
    r_magnet: float = _DEFAULT_GEO.r_magnet,
    r_rotor: float = _DEFAULT_GEO.r_rotor,
    mu_r_pm: float = 1.05,
    alpha_p: float = 1.0,
) -> NDArray[np.floating]:
    """Zhu, Howe & Chan (2002) improved-model air-gap radial flux density.

    Convention: B_rem is physical remanence; the source is the fundamental
    of square-wave radial magnetization, M_1 = (4/π)(B_rem/μ0)·sin(πα_p/2)
    (paper Eq 6a M_rn at n=1). Matches the FEM solver's uniform-magnitude
    per-pole-alternating arc source exactly at the fundamental.
    """
    if n_p < 2:
        raise ValueError(f"n_p={n_p} not supported (singularity at n²-1=0)")

    if r_eval is None:
        r_eval = (r_stator + r_magnet) / 2

    n = n_p
    p, q, s = r_rotor, r_magnet, r_stator
    mu = mu_r_pm
    r = r_eval

    # Square-wave magnetization fundamental: (4/π)·sin(πα_p/2) is the
    # Fourier coefficient of a ±1 square wave with pole-arc ratio α_p
    # (paper Eq 6a). k_mp = 1.0 when α_p = 1 (full-pitch magnets).
    from math import sin as _sin
    k_mp = _sin(pi * alpha_p / 2)
    bp = n * (4 / pi) * B_rem * k_mp / (n**2 - 1)

    a11 = q**n + p**(2*n) * q**(-n)
    a12 = -(q**n + s**(2*n) * q**(-n))
    b1 = -bp * (p**(n+1) * q**(-n) / n + q)

    a21 = (1/mu) * (q**(n-1) - p**(2*n) * q**(-(n+1)))
    a22 = -(q**(n-1) - s**(2*n) * q**(-(n+1)))
    b2 = -(bp/n) * (1/mu) * (1 - p**(n+1) * q**(-(n+1)))

    det = a11 * a22 - a12 * a21
    C2 = (a11 * b2 - a21 * b1) / det

    K = C2 * n * (r**(n-1) + s**(2*n) * r**(-(n+1)))
    return K * np.cos(n * theta)


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


@overload
def solve_field_fem(
    geo: Geometry, n_p: int, B_rem: float, mu_r_pm: float, mu_r_fe: float,
    n_theta: int = ..., maxh_fraction: float = ...,
    return_full: Literal[False] = ...,
    n_slots: int | None = ..., j_s: float = ..., k_w: float = ...,
    slot_width_ratio: float | None = ..., nonlinear: bool = ...,
    max_picard: int = ..., picard_tol: float = ..., picard_relax: float = ...,
    alpha_p: float = ..., info: dict[str, float] | None = ...,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]: ...

@overload
def solve_field_fem(
    geo: Geometry, n_p: int, B_rem: float, mu_r_pm: float, mu_r_fe: float,
    n_theta: int = ..., maxh_fraction: float = ...,
    return_full: Literal[True] = ...,
    n_slots: int | None = ..., j_s: float = ..., k_w: float = ...,
    slot_width_ratio: float | None = ..., nonlinear: bool = ...,
    max_picard: int = ..., picard_tol: float = ..., picard_relax: float = ...,
    alpha_p: float = ..., info: dict[str, float] | None = ...,
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
    info: dict[str, float] | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating]] | tuple[NDArray[np.floating], NDArray[np.floating], Any, Any]:
    """Geometry-aware FEM solver with discrete magnet arcs.

    When ``alpha_p < 1.0``, builds discrete magnet arcs with interpole gaps
    named ``"pm_gap"`` (air reluctivity, no source). The square-wave source
    term (uniform magnitude, per-pole sign) integrates only over ``"pm"``
    regions. Pass physical remanence as ``B_rem`` (no sin(πα_p/2) or 4/π
    pre-scaling needed).

    ``n_slots`` and ``slot_width_ratio`` default to the geometry's values;
    pass an explicit value (including 0) to override, e.g. ``n_slots=0``
    for a smooth-bore solve of a slotted geometry.

    ``info``, when provided, is populated with solve diagnostics:
    ``b_iron_max`` (peak |B| in iron regions, T) and — nonlinear only —
    ``picard_iterations``, ``picard_relax_final``, ``picard_residual``
    (relaxed step, the stop criterion) and ``picard_residual_raw``
    (unrelaxed step; large values with small relax_final indicate shallow
    convergence).

    Contract:
        Preconditions:
            - ``n_p`` >= 1.
            - ``maxh_fraction`` > 0.
            - ``picard_relax`` > 0 when ``nonlinear=True``.

        Raises:
            RuntimeError: If ``nonlinear=True`` and Picard iteration
                does not converge within ``max_picard`` iterations.
    """
    from netgen.occ import OCCGeometry
    from ngsolve import (
        H1,
        L2,
        BilinearForm,
        GridFunction,
        IfPos,
        LinearForm,
        Mesh,
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

    maxh = maxh_fraction * geo.r_outer

    # Determine if discrete arcs will actually be built (same threshold as _build_motor_geometry)
    _r_pm_ref = geo.r_magnet if geo.topology == "inrunner" else geo.r_rotor
    _has_arcs = (alpha_p < 1.0 and n_p > 0
                 and (1 - alpha_p) * pi / n_p * _r_pm_ref >= 5e-4)

    cache_key: _CacheKey = (geo.config_id, maxh_fraction, n_slots, slot_width_ratio, alpha_p, n_p)
    mesh = _mesh_cache.get(cache_key)
    if mesh is None:
        mesh = _load_from_disk(cache_key)
    if mesh is None:
        slot_depth = geo.slot_depth if n_slots > 0 else 0.0
        shape = _build_motor_geometry(
            geo,
            n_slots=n_slots,
            slot_width_ratio=slot_width_ratio,
            slot_depth=slot_depth,
            slot_opening_ratio=geo.slot_opening_ratio,
            alpha_p=alpha_p,
            n_p=n_p,
        )
        if n_slots > 0:
            maxh = min(maxh, 0.03 * geo.r_outer)
        # Refine mesh for interpole gaps — at least 2 elements across,
        # but don't go below 0.5mm to keep mesh size reasonable
        if _has_arcs:
            gap_width = (1 - alpha_p) * pi / n_p * _r_pm_ref
            maxh = min(maxh, max(gap_width * 0.5, 5e-4))
        mesh = Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh=maxh))
        _save_to_disk(cache_key, mesh)
    _mesh_cache[cache_key] = mesh

    # Build material property dicts from circle spec regions
    region_names = [name for _, name in _circle_spec(geo)]
    if n_slots > 0:
        region_names.append("slot")
    if _has_arcs:
        region_names.append("pm_gap")

    iron_regions = {"stator", "shaft", "yoke"}

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
    # Square-wave radial magnetization: uniform |M| = B_rem/μ0 within each
    # pole, sign alternating per pole. Arcs are centered at θ_k = kπ/n_p
    # where cos(n_p·θ) = ±1, so the sign is constant within each arc; at
    # α_p = 1 the sign flips inside the full ring at pole boundaries.
    M_r = nu_pm * B_rem * IfPos(cos_np, 1.0, -1.0)
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
            name: (1.0 if name in iron_regions else 0.0)
            for name in region_names
        }
        is_iron = mesh.MaterialCF(is_iron_cf)
        iron_indicator = GridFunction(fes_nu)
        iron_indicator.Set(is_iron)
        iron_mask = iron_indicator.vec.FV().NumPy() > 0.5

        a = BilinearForm(gfu_nu * grad(u) * grad(v) * dx)

        delta_rel = float("inf")
        prev_delta_rel = float("inf")
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

            nu_update = nu_old.copy()
            nu_update[iron_mask] = (
                relax * nu_new[iron_mask]
                + (1 - relax) * nu_old[iron_mask]
            )

            delta = np.max(np.abs(nu_update[iron_mask] - nu_old[iron_mask]))
            scale = np.max(np.abs(nu_old[iron_mask]))
            delta_rel = delta / scale

            # Backtracking: halve relaxation until residual stops growing
            while delta_rel > prev_delta_rel and relax > _MIN_PICARD_RELAX:
                relax = max(relax * 0.5, _MIN_PICARD_RELAX)
                nu_update[iron_mask] = (
                    relax * nu_new[iron_mask]
                    + (1 - relax) * nu_old[iron_mask]
                )
                delta = np.max(np.abs(nu_update[iron_mask] - nu_old[iron_mask]))
                delta_rel = delta / scale

            prev_delta_rel = delta_rel
            gfu_nu.vec.FV().NumPy()[:] = nu_update

            if delta_rel < picard_tol:
                break
        else:
            raise RuntimeError(
                f"Picard iteration did not converge in {max_picard} iterations "
                f"(residual: {delta_rel:.4e}, tol: {picard_tol})"
            )
        if info is not None:
            # The stop criterion is on the relaxed step (relax × raw), so a
            # small relax_final permits a large raw residual — record both.
            info["picard_iterations"] = float(_picard_i + 1)
            info["picard_relax_final"] = float(relax)
            info["picard_residual"] = float(delta_rel)
            info["picard_residual_raw"] = float(
                np.max(np.abs(nu_new[iron_mask] - nu_old[iron_mask])) / scale
            )

    # Check peak |B| in stator iron
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
        print(f"WARNING: peak |B| in stator iron = {b_iron_max:.2f} T "
              f"(> 1.8 T saturation threshold)", file=sys.stderr)

    r_cf = sqrt(x**2 + y**2)
    B_r_cf = (grad(gfu)[1] * x - grad(gfu)[0] * y) / r_cf

    fes_L2 = L2(mesh, order=3)
    gfu_Br = GridFunction(fes_L2)
    gfu_Br.Set(B_r_cf)

    r_ag = geo.r_ag
    theta = np.linspace(0, 2 * pi, n_theta, endpoint=False)
    cos_t_np = np.cos(theta)
    sin_t_np = np.sin(theta)
    B_r = np.array([
        gfu_Br(mesh(r_ag * c, r_ag * s))
        for c, s in zip(cos_t_np, sin_t_np)
    ])
    if return_full:
        return theta, B_r, mesh, gfu
    return theta, B_r


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
        print(f"  rasterise: {failed} points failed evaluation")

    return xi, yi, Az, Bmag


# ---------------------------------------------------------------------------
# Harmonics
# ---------------------------------------------------------------------------

def _double_onesided(amps: NDArray[np.floating], n_pts: int) -> NDArray[np.floating]:
    """Double all bins except DC and (for even n) the Nyquist bin."""
    end = -1 if n_pts % 2 == 0 else None
    amps[..., 1:end] *= 2
    return amps


def harmonics_1sided(B_r: NDArray[np.floating]) -> NDArray[np.floating]:
    """One-sided FFT harmonic amplitudes for a single B_r waveform (NumPy)."""
    n_pts = len(B_r)
    amps = np.abs(np.fft.rfft(B_r)) / n_pts
    return _double_onesided(amps, n_pts)


def compute_thd(amps: NDArray[np.floating], fund_idx: int) -> float:
    """THD as percentage of fundamental from one-sided harmonic amplitudes."""
    fundamental = float(amps[fund_idx])
    if fundamental == 0.0:
        return float("nan")
    return float(
        np.sqrt(max(np.sum(amps[1:] ** 2) - amps[fund_idx] ** 2, 0.0))
        / fundamental * 100
    )


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

    amps = _double_onesided(amps, n_pts)
    return {name: amps[i] for i, name in enumerate(names)}
