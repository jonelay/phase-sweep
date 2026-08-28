"""OCC geometry construction and mesh caching for the radial FEM solver.

Builds the annular motor cross-section (yoke / PM arcs / airgap /
slotted stator, Halbach segments, inset bridge ring), meshes it via
netgen, and caches meshes in memory plus an optional LRU-pruned disk
store for subprocess workers. Also the 3D STEP
export of the extruded cross-section.
"""

from __future__ import annotations

import contextlib
import logging
import os
import tempfile
import time
from math import pi
from pathlib import Path
from typing import TYPE_CHECKING, Any

from phasesweep.machines.geometry import Geometry, default_inrunner

if TYPE_CHECKING:
    from ngsolve import Mesh

# Interpole gaps narrower than this collapse to a full PM ring (α_p = 1.0):
# OCC boolean cuts on sub-0.5 mm arc slivers fail unreliably ("Could not
# divide Edge"). solve_field_fem warns and flags the collapse.
_MIN_ARC_GAP_M: float = 5e-4


def _pm_radii(geo: Geometry) -> tuple[float, float]:
    """(r_pm_outer, r_pm_inner) of the magnet annulus, net of the bridge.

    A bridge ring (inset rotors) occupies the airgap side of the PM
    annulus, so the magnet arcs stop bridge_thickness short of it:
    inrunner outer = r_magnet - t_b, outrunner inner = r_magnet + t_b.
    Without a bridge this is the plain PM annulus.
    """
    t_b = geo.bridge_thickness or 0.0
    if geo.topology == "inrunner":
        return geo.r_magnet - t_b, geo.r_rotor
    return geo.r_rotor, geo.r_magnet + t_b


def _interpole_gap_width(geo: Geometry, alpha_p: float, n_p: int) -> float:
    """Arc-length width of the interpole gap between adjacent magnet arcs.

    Evaluated at the *inner* PM-annulus radius (rotor iron behind the
    magnets: ``r_rotor`` inrunner, ``r_magnet`` outrunner — plus the
    bridge offset when set). The gap is a
    radial wedge whose arc-length grows with radius, so the inner edge is
    its narrowest point — where the OCC boolean fails to cut the sliver
    (the ``_MIN_ARC_GAP_M`` collapse threshold) and where the pm_gap
    refinement must place its ≥2 elements. Using the outer radius here
    overestimated the gap on both topologies (fem v6 and earlier).
    """
    _, r_pm = _pm_radii(geo)
    return (1 - alpha_p) * pi / n_p * r_pm


def _arcs_active(geo: Geometry, alpha_p: float, n_p: int) -> bool:
    """Whether discrete magnet arcs are actually cut into the PM annulus.

    The single source of truth for three decisions that must agree:
    _build_motor_geometry cutting the arcs, solve_field_fem reporting the
    collapse, and _get_or_build_mesh deciding whether ``rotation`` reaches
    the geometry at all. If these ever disagreed, a rotation-independent
    cache key could serve a mesh built at the wrong angle.
    """
    return (alpha_p < 1.0 and n_p > 0
            and _interpole_gap_width(geo, alpha_p, n_p) >= _MIN_ARC_GAP_M)


# ---------------------------------------------------------------------------
# Mesh cache — keyed by (config_id, maxh_fraction, n_slots,
#                        slot_width_ratio, alpha_p, n_p, rotation)
# In-memory + optional disk persistence for subprocess workers.
# ---------------------------------------------------------------------------
_CacheKey = tuple[str, float, int, float, float, int, float, str]
_mesh_cache: dict[_CacheKey, Mesh] = {}
_disk_cache_dir: Path | None = None

# Disk-cache byte budget. Rotated meshes (one per cogging angle) are
# persisted, so the directory grows with the number of distinct
# (geometry, angle) pairs rather than geometries alone; the LRU prune below
# bounds it. ~400 kB/mesh at maxh_fraction=0.05, so the default holds
# ~1300 meshes (~100 twelve-angle cogging sweeps).
ENV_CACHE_DIR = "PHASESWEEP_MESH_CACHE"
ENV_CACHE_BUDGET_MB = "PHASESWEEP_MESH_CACHE_MB"


def _read_budget_mb() -> float:
    # Read at import, but never let a typo in the environment take the whole
    # module down — an unusable budget falls back to the default.
    raw = os.environ.get(ENV_CACHE_BUDGET_MB)
    if raw is None:
        return 512.0
    try:
        return float(raw)
    except ValueError:
        logging.warning("ignoring non-numeric %s=%r, using 512 MB",
                        ENV_CACHE_BUDGET_MB, raw)
        return 512.0


_DISK_CACHE_BUDGET_MB: float = _read_budget_mb()

# Age past which a stranded mkstemp file is assumed abandoned rather than
# mid-write. A single mesh save is sub-second; an hour is pure headroom.
_TMP_REAP_AGE_S: float = 3600.0


def set_disk_cache_dir(path: Path | str | None) -> None:
    """Enable disk mesh cache. Set to None to disable."""
    global _disk_cache_dir
    if path is not None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        _disk_cache_dir = p
    else:
        _disk_cache_dir = None


def disk_cache_dir() -> Path | None:
    """Currently configured disk mesh cache directory, if any."""
    return _disk_cache_dir


def clear_mesh_cache() -> None:
    """Drop all cached meshes (for testing or memory management)."""
    _mesh_cache.clear()


def _cache_filename(key: _CacheKey) -> str:
    config_id, maxh_frac, n_slots, swr, alpha_p, n_p, rotation, hspec = key
    # v5: injective float repr — the old 6-sig-fig %g could serve the wrong
    # mesh for params differing below 1e-6 relative.
    # v6: interpole gap width measured at the inner radius (fem v7) changes
    # pm_gap and global maxh for every alpha_p<1 mesh — invalidate stale
    # .vol.gz.
    rot = "" if rotation == 0.0 else f"_rot{float(rotation)!r}"
    hsuf = f"_{hspec}" if hspec else ""
    return (f"mesh_v6_{config_id}_{maxh_frac!r}_{n_slots}_{swr!r}"
            f"_{alpha_p!r}_{n_p}{rot}{hsuf}.vol.gz")


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
        if not fpath.exists():
            # A concurrent prune evicted it between the exists() check and
            # the load. Not corruption — just a miss.
            return None
        logging.warning("corrupt mesh cache %s, rebuilding", fpath.name)
        fpath.unlink(missing_ok=True)
        return None
    # Mark as recently used so the LRU prune keeps hot meshes.
    with contextlib.suppress(OSError):
        os.utime(fpath, None)
    return Mesh(nm)


def _prune_disk_cache() -> None:
    """Evict least-recently-used mesh files over the byte budget.

    Pool workers share one cache dir, so the whole scan-and-evict runs under
    an exclusive flock: two pruners that each measured the same over-budget
    total would otherwise both evict down to it and empty far more than the
    budget requires. The lock is taken non-blocking — a worker that finds it
    held skips pruning entirely rather than queueing behind a scan that is
    already fixing the same overage.

    Files are only ever evicted oldest-mtime first, and _save_to_disk /
    _load_from_disk both refresh mtime, so the entry a peer just wrote or
    read is the last candidate rather than the first. A reader can still
    lose its file between the exists() check and the load; _load_from_disk
    treats that as a miss and rebuilds.
    """
    if _disk_cache_dir is None or _DISK_CACHE_BUDGET_MB <= 0:
        return
    import fcntl

    budget = int(_DISK_CACHE_BUDGET_MB * 1024 * 1024)
    lock_path = _disk_cache_dir / ".prune.lock"
    try:
        lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    except OSError:
        return
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return  # another worker is already pruning
        # A writer killed between mkstemp and replace strands a tmp* file
        # that no glob below counts and nothing else ever removes. Sweep
        # ones old enough that no live writer could still hold them —
        # otherwise the budget silently leaks.
        _stale = time.time() - _TMP_REAP_AGE_S
        for f in _disk_cache_dir.glob("tmp*.vol.gz"):
            try:
                if f.stat().st_mtime < _stale:
                    f.unlink()
            except OSError:
                continue

        # glob excludes the mkstemp temp files: they are named tmp*, and a
        # half-written mesh must not be counted or evicted.
        entries = []
        total = 0
        for f in _disk_cache_dir.glob("mesh_*.vol.gz"):
            try:
                st = f.stat()
            except OSError:
                continue
            entries.append((st.st_mtime, st.st_size, f))
            total += st.st_size
        if total <= budget:
            return
        entries.sort()  # oldest mtime first
        for _mtime, size, f in entries:
            if total <= budget:
                break
            try:
                f.unlink()
            except FileNotFoundError:
                # Already gone — those bytes are freed either way, so still
                # discount them. Not doing so is what makes a pruner keep
                # deleting past the budget.
                total -= size
            except OSError:
                continue
            else:
                total -= size
        logging.info("mesh cache pruned to %.0f MB", total / 1024 / 1024)
    finally:
        os.close(lock_fd)


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
    _prune_disk_cache()



# Default geometry for analytical model function defaults
_DEFAULT_GEO = default_inrunner()


# ---------------------------------------------------------------------------
# Circle spec — region list from geometry
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
    A bridge ring (inset rotors) inserts a "bridge" annulus between the pm
    region and the airgap, shrinking the pm annulus by bridge_thickness.
    """
    t_b = geo.bridge_thickness
    if geo.topology == "inrunner":
        spec = [
            (geo.r_outer, "stator"),
            (geo.r_stator, "airgap"),
            (geo.r_magnet, "pm"),
            (geo.r_rotor, "shaft"),
        ]
        if t_b is not None:
            spec[2:3] = [(geo.r_magnet, "bridge"),
                         (geo.r_magnet - t_b, "pm")]
        return spec
    # outrunner: outside-in = rotor back-iron, pm, airgap, stator, air
    tail = [
        (geo.r_rotor, "pm"),
        (geo.r_magnet, "airgap"),
        (geo.r_stator, "stator"),
        (geo.r_inner, "air"),
    ]
    if t_b is not None:
        tail[0:1] = [(geo.r_rotor, "pm"),
                     (geo.r_magnet + t_b, "bridge")]
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

    # PM annulus radii for intersection (net of the bridge ring)
    r_pm_outer, r_pm_inner = _pm_radii(geo)

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


def _cut_halbach_segments(
    geo: Geometry,
    n_p: int,
    kappa_m: int,
    rotation: float = 0.0,
) -> list[Any]:
    """Cut PM annulus into 2*n_p*kappa_m Halbach segments.

    Each segment is a wedge-annulus arc named "pmh{m}". Segment m is
    centered at theta_c(m) = rotation + m*pi/(n_p*kappa_m).
    Requires alpha_p = 1.0 (full magnet ring, no interpole gaps).

    Returns list of named segment faces.
    """
    from math import cos, sin
    from math import pi as _pi

    from netgen.occ import WorkPlane

    n_seg = 2 * n_p * kappa_m
    if n_p * kappa_m < 2:
        raise ValueError(
            f"n_p * halbach_segments must be >= 2 (got {n_p} * {kappa_m} = "
            f"{n_p * kappa_m}): a single segment pair spans pi radians and "
            f"the OCC wedge construction degenerates"
        )

    r_pm_outer, r_pm_inner = _pm_radii(geo)

    seg_arc = _pi / (n_p * kappa_m)
    min_width = seg_arc * r_pm_inner
    if min_width < 2 * _MIN_ARC_GAP_M:
        raise ValueError(
            f"Halbach segment arc width {min_width * 1e3:.2f} mm "
            f"(at r_pm_inner={r_pm_inner * 1e3:.1f} mm) is below the "
            f"OCC minimum {2 * _MIN_ARC_GAP_M * 1e3:.1f} mm — "
            f"reduce halbach_segments or increase rotor radius"
        )

    R_big = 2.0 * geo.r_outer
    seg_faces = []
    for m in range(n_seg):
        theta_c = rotation + m * _pi / (n_p * kappa_m)
        a_lo = theta_c - seg_arc / 2
        a_hi = theta_c + seg_arc / 2

        wp = WorkPlane()
        wp.MoveTo(0., 0.)
        wp.LineTo(R_big * cos(a_lo), R_big * sin(a_lo))
        wp.LineTo(R_big * cos(a_hi), R_big * sin(a_hi))
        wp.Close()
        wedge = wp.Face()

        f_outer = WorkPlane().Circle(r_pm_outer).Face()
        f_inner = WorkPlane().Circle(r_pm_inner).Face()
        seg = (wedge * f_outer) - f_inner
        seg.faces.name = f"pmh{m}"
        seg_faces.append(seg)

    return seg_faces


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
    magnetization: str = "radial",
    halbach_segments: int | None = None,
) -> Any:
    """Composable motor geometry builder.

    1. Build annular regions (always)
    2. Cut magnet arcs (when alpha_p < 1) OR Halbach segments, pole axes
       offset by ``rotation``
    3. Cut slots (when n_slots > 0), always stator-fixed
    """
    from netgen.occ import Glue

    faces, _innermost, named = _build_annular_regions(geo)

    if magnetization == "halbach" and halbach_segments is not None:
        pm_face = named["pm"]
        seg_faces = _cut_halbach_segments(
            geo, n_p, halbach_segments, rotation,
        )
        faces = [f for f in faces if f is not pm_face]
        faces.extend(seg_faces)
    else:
        _do_arcs = _arcs_active(geo, alpha_p, n_p)
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


def _get_or_build_mesh(
    geo: Geometry,
    maxh_fraction: float,
    n_slots: int,
    slot_width_ratio: float,
    alpha_p: float,
    n_p: int,
    has_arcs: bool,
    rotation: float = 0.0,
    magnetization: str = "radial",
    halbach_segments: int | None = None,
) -> Mesh:
    """Motor cross-section mesh: in-process cache → disk cache → build."""
    from netgen.occ import OCCGeometry
    from ngsolve import Mesh

    _is_halbach = magnetization == "halbach" and halbach_segments is not None
    hspec = f"h{halbach_segments}" if _is_halbach else ""
    # rotation reaches the geometry through _cut_magnet_arcs (radial) or
    # _cut_halbach_segments (Halbach). Without either, the mesh is provably
    # rotation-independent, so normalizing the key lets a full-ring cogging
    # sweep build one mesh instead of one per angle.
    _mesh_rotates = _arcs_active(geo, alpha_p, n_p) or _is_halbach
    key_rotation = rotation if _mesh_rotates else 0.0
    cache_key: _CacheKey = (geo.config_id, maxh_fraction, n_slots,
                            slot_width_ratio, alpha_p, n_p, key_rotation,
                            hspec)
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
            magnetization=magnetization,
            halbach_segments=halbach_segments,
        )
        if n_slots > 0:
            maxh = min(maxh, 0.03 * geo.r_outer)
        # Refine for interpole gaps — at least 2 elements across. The
        # global maxh keeps its 0.5 mm floor; the per-face maxh
        # below enforces the rule locally for 0.5–1.0 mm gaps where the
        # floor would otherwise under-resolve (a no-op for wider gaps).
        if has_arcs:
            gap_width = _interpole_gap_width(geo, alpha_p, n_p)
            clamp = max(gap_width * 0.5, _MIN_ARC_GAP_M)
            if clamp < maxh:
                logging.info(
                    "interpole arc clamp: global maxh %.4g mm -> %.4g mm "
                    "(gap/2=%.4g mm, alpha_p=%.4g); maxh_fraction now inert",
                    maxh * 1e3, clamp * 1e3, gap_width * 0.5 * 1e3, alpha_p,
                )
            maxh = min(maxh, clamp)
            shape.faces["pm_gap"].maxh = gap_width * 0.5
        # A thin bridge ring needs the same >= 2 elements across as the
        # interpole gaps — its saturation level is the whole point of
        # modeling it.
        if geo.bridge_thickness is not None:
            shape.faces["bridge"].maxh = geo.bridge_thickness / 2
        # Tie airgap element size to the gap width via per-face maxh so the
        # refinement stays local: thin-gap motors otherwise get a single
        # element across the gap (14 mm outrunner: 0.26 mm gap vs 0.29 mm global
        # maxh), under-resolving B_r at the r_ag sampling circle.
        shape.faces["airgap"].maxh = abs(geo.r_stator - geo.r_magnet) / 3
        mesh = Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh=maxh))
        _save_to_disk(cache_key, mesh)
    _mesh_cache[cache_key] = mesh
    return mesh


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
