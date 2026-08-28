"""Tests for fem_mesh.py — geometry building, mesh caching, Halbach cuts."""

import dataclasses as _dc
import os
from importlib.util import find_spec
from unittest import mock

import pytest

if not find_spec("ngsolve"):
    pytest.skip("requires phasesweep[fem] (ngsolve not installed)", allow_module_level=True)
try:
    from ngsolve import H1, Mesh
except ImportError:
    pytest.skip("requires phasesweep[fem] (ngsolve not installed)", allow_module_level=True)

import numpy as np
from netgen.occ import OCCGeometry

from phasesweep.machines.geometry import default_inrunner, inrunner, outrunner
from phasesweep.solvers import fem_mesh
from phasesweep.solvers.fem_mesh import (
    _build_annular_regions,
    _build_geometry,
    _build_motor_geometry,
    _build_slotted_geometry,
    _circle_spec,
    _mesh_cache,
    clear_mesh_cache,
    set_disk_cache_dir,
)
from tests.conftest import requires_fem
from tests.solvers.fem_shared import (
    _GEO,
    _GEO_OUT,
    _GEO_SMALL,
    _SLOTTED_GEO_KW,
    _SLOTTED_OUT_GEO_KW,
    FAST,
    _solve,
)

# Module-level guard covers missing ngsolve; the marker makes the module
# selectable by lane.
pytestmark = requires_fem


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def test_smooth_geometry_materials():
    shape = _build_geometry(_GEO)
    mesh = Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh=0.08))
    mats = set(mesh.GetMaterials())
    assert {"stator", "airgap", "pm", "shaft"} <= mats
    assert "slot" not in mats


def test_circle_spec_inrunner_regions():
    geo = default_inrunner()
    specs = _circle_spec(geo)
    assert len(specs) == 4
    names = [n for _, n in specs]
    assert set(names) == {"stator", "airgap", "pm", "shaft"}
    radii = [r for r, _ in specs]
    assert radii == sorted(radii, reverse=True)


def test_circle_spec_outrunner_regions():
    geo = outrunner(r_outer=0.10, r_stator=0.04, r_magnet=0.06,
                    r_rotor=0.08, r_inner=0.02)
    specs = _circle_spec(geo)
    assert len(specs) == 5
    names = [n for _, n in specs]
    assert set(names) == {"stator", "airgap", "pm", "yoke", "air"}
    radii = [r for r, _ in specs]
    assert radii == sorted(radii, reverse=True)
    # Verify name-to-radius mapping (not just the set)
    spec_dict = dict(specs)
    assert spec_dict[geo.r_outer] == "yoke"
    assert spec_dict[geo.r_rotor] == "pm"
    assert spec_dict[geo.r_magnet] == "airgap"
    assert spec_dict[geo.r_stator] == "stator"
    assert spec_dict[geo.r_inner] == "air"


def test_build_geometry_from_geo_materials():
    geo = default_inrunner()
    shape = _build_geometry(geo)
    mesh = Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh=0.08))
    mats = set(mesh.GetMaterials())
    expected = {n for _, n in _circle_spec(geo)}
    assert expected <= mats


def test_slotted_geometry_materials():
    geo = inrunner(r_outer=1.0, r_stator=0.70, r_magnet=0.64, r_rotor=0.30,
                   r_inner=0.0, n_slots=12, slot_depth=0.05)
    shape = _build_slotted_geometry(geo, n_slots=12)
    mesh = Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh=0.08))
    mats = set(mesh.GetMaterials())
    assert {"stator", "slot", "airgap", "pm", "shaft"} <= mats


@pytest.mark.parametrize("n_slots", [6, 12, 24])
def test_slotted_geometry_slot_count(n_slots):
    geo = inrunner(r_outer=1.0, r_stator=0.70, r_magnet=0.64, r_rotor=0.30,
                   r_inner=0.0, n_slots=n_slots, slot_depth=0.05)
    shape = _build_slotted_geometry(geo, n_slots=n_slots)
    mesh = Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh=0.08))
    slot_count = sum(1 for m in mesh.GetMaterials() if m == "slot")
    assert slot_count == n_slots


@pytest.mark.parametrize("topology,slotted", [
    ("inrunner", False), ("inrunner", True),
    ("outrunner", False), ("outrunner", True),
])
def test_outer_boundary_named_and_constrained(topology, slotted):
    """Glued shape keeps a named outer boundary that constrains DOFs."""
    if topology == "inrunner":
        geo = inrunner(**_SLOTTED_GEO_KW)
    else:
        geo = outrunner(**_SLOTTED_OUT_GEO_KW)
    shape = _build_motor_geometry(
        geo,
        n_slots=geo.n_slots if slotted else 0,
        slot_width_ratio=geo.slot_width_ratio,
        slot_depth=geo.slot_depth if slotted else 0.0,
        slot_opening_ratio=geo.slot_opening_ratio,
    )
    mesh = Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh=0.08 * geo.r_outer))
    assert "outer" in set(mesh.GetBoundaries())
    fes = H1(mesh, order=1, dirichlet="outer")
    n_constrained = fes.ndof - sum(fes.FreeDofs())
    assert n_constrained > 0


def test_inset_mesh_materials():
    shape = _build_motor_geometry(
        _dc.replace(_GEO_SMALL, magnet_mount="inset", bridge_thickness=2e-3),
        alpha_p=0.7, n_p=2,
    )
    mesh = Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh=0.005))
    mats = set(mesh.GetMaterials())
    assert {"bridge", "pm", "pm_gap", "shaft", "airgap", "stator"} <= mats


def test_surface_bridge_mesh_materials():
    shape = _build_motor_geometry(
        _dc.replace(_GEO_SMALL, bridge_thickness=2e-3),
        alpha_p=0.7, n_p=2,
    )
    mesh = Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh=0.005))
    mats = set(mesh.GetMaterials())
    assert {"bridge", "pm", "pm_gap", "shaft", "airgap", "stator"} <= mats


def test_outrunner_smooth_geometry_materials():
    shape = _build_geometry(_GEO_OUT)
    mesh = Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh=0.01))
    mats = set(mesh.GetMaterials())
    assert {"stator", "airgap", "pm", "yoke", "air"} <= mats
    assert "shaft" not in mats


def test_outrunner_back_iron_split_materials():
    """back_iron_thickness splits the yoke into back_iron + shell regions."""
    geo = outrunner(r_outer=0.10, r_rotor=0.08, r_magnet=0.06,
                    r_stator=0.04, r_inner=0.02, back_iron_thickness=0.01)
    shape = _build_geometry(geo)
    mesh = Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh=0.01))
    mats = set(mesh.GetMaterials())
    assert {"stator", "airgap", "pm", "back_iron", "shell", "air"} <= mats
    assert "yoke" not in mats


def test_outrunner_slotted_geometry_materials():
    geo = outrunner(r_outer=0.10, r_rotor=0.08, r_magnet=0.06,
                    r_stator=0.04, r_inner=0.02, n_slots=6, slot_depth=0.005)
    shape = _build_slotted_geometry(geo, n_slots=6)
    mesh = Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh=0.005))
    mats = set(mesh.GetMaterials())
    assert {"stator", "slot", "airgap", "pm", "yoke", "air"} <= mats
    assert "shaft" not in mats


# ---------------------------------------------------------------------------
# Mesh caching
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=False)
def _clean_mesh_cache():
    """Ensure mesh cache is empty before and after cache tests."""
    clear_mesh_cache()
    yield
    clear_mesh_cache()


def test_mesh_cache_hit_same_geometry(_clean_mesh_cache):
    """Same geometry + maxh_fraction reuses cached mesh."""
    geo = default_inrunner()
    _solve(geo=geo, **FAST)
    assert len(_mesh_cache) == 1

    _solve(geo=geo, psi_f=0.2, **FAST)
    assert len(_mesh_cache) == 1  # no new entry


def test_mesh_cache_miss_different_geometry(_clean_mesh_cache):
    """Different geometry produces a new cache entry."""
    geo1 = inrunner(r_outer=0.05, r_stator=0.035, r_magnet=0.030,
                    r_rotor=0.015)
    geo2 = inrunner(r_outer=0.06, r_stator=0.040, r_magnet=0.035,
                    r_rotor=0.018)
    _solve(geo=geo1, **FAST)
    _solve(geo=geo2, **FAST)
    assert len(_mesh_cache) == 2


def test_mesh_cache_miss_different_maxh(_clean_mesh_cache):
    """Different maxh_fraction on same geometry creates separate entry."""
    geo = default_inrunner()
    _solve(geo=geo, maxh_fraction=0.08)
    _solve(geo=geo, maxh_fraction=0.04)
    assert len(_mesh_cache) == 2


def test_mesh_cache_miss_different_alpha_p(_clean_mesh_cache):
    """Same geometry, different alpha_p creates separate mesh-cache entries."""
    _solve(alpha_p=1.0)
    _solve(alpha_p=0.75)
    assert len(_mesh_cache) == 2


def test_clear_mesh_cache():
    """clear_mesh_cache() empties the cache."""
    _solve(**FAST)
    assert len(_mesh_cache) >= 1
    clear_mesh_cache()
    assert len(_mesh_cache) == 0


def test_disk_cache_save_and_load(_clean_mesh_cache, tmp_path):
    """Mesh saved to disk is reloaded on cache miss."""
    set_disk_cache_dir(tmp_path)
    try:
        geo = default_inrunner()
        _solve(geo=geo, **FAST)
        assert len(_mesh_cache) == 1

        # Clear in-memory, disk file should exist
        clear_mesh_cache()
        assert len(_mesh_cache) == 0
        disk_files = list(tmp_path.glob("mesh_*.vol.gz"))
        assert len(disk_files) == 1

        # Re-solve: should load from disk, not regenerate
        _solve(geo=geo, **FAST)
        assert len(_mesh_cache) == 1
    finally:
        set_disk_cache_dir(None)


def test_disk_cache_persists_rotated_mesh(_clean_mesh_cache, tmp_path):
    """Rotated meshes (one per cogging angle) persist to disk under their own
    filename, so a repeated sweep reloads instead of remeshing."""
    set_disk_cache_dir(tmp_path)
    try:
        geo = default_inrunner()
        _solve(geo=geo, n_p=4, alpha_p=0.75, rotation=0.1, **FAST)
        rotated = list(tmp_path.glob("mesh_*_rot*.vol.gz"))
        assert len(rotated) == 1
        _solve(geo=geo, n_p=4, alpha_p=0.75, **FAST)
        files = list(tmp_path.glob("mesh_*.vol.gz"))
        assert len(files) == 2
        # The unrotated filename carries no suffix — v6 caches stay valid.
        assert len([f for f in files if "_rot" not in f.name]) == 1

        # A second process (in-memory cache cleared) must load from disk.
        # Asserting on the file count alone would pass even if the disk read
        # always missed — the rebuild writes the same filename back. So make
        # remeshing impossible and require the solve to succeed anyway.
        clear_mesh_cache()
        import phasesweep.solvers.fem_mesh as _ff
        with mock.patch.object(
            _ff, "_build_motor_geometry",
            side_effect=AssertionError("remeshed instead of loading disk"),
        ):
            _solve(geo=geo, n_p=4, alpha_p=0.75, rotation=0.1, **FAST)
    finally:
        set_disk_cache_dir(None)


def test_disk_cache_prune_respects_budget(_clean_mesh_cache, tmp_path,
                                          monkeypatch):
    """Over-budget caches evict oldest-first, and never the temp files a
    concurrent writer is still filling."""
    import phasesweep.solvers.fem_mesh as _ff

    set_disk_cache_dir(tmp_path)
    try:
        # Three stand-in entries with distinct ages, plus a mkstemp-style
        # partial write that pruning must ignore.
        for i, name in enumerate(["mesh_a.vol.gz", "mesh_b.vol.gz",
                                  "mesh_c.vol.gz"]):
            p = tmp_path / name
            p.write_bytes(b"x" * 1000)
            os.utime(p, (1000 + i, 1000 + i))
        partial = tmp_path / "tmp_halfwritten.vol.gz"
        partial.write_bytes(b"x" * 1000)

        # Budget fits two of the three mesh entries.
        monkeypatch.setattr(_ff, "_DISK_CACHE_BUDGET_MB", 2000 / 1024 / 1024)
        _ff._prune_disk_cache()

        remaining = sorted(p.name for p in tmp_path.glob("mesh_*.vol.gz"))
        assert remaining == ["mesh_b.vol.gz", "mesh_c.vol.gz"]  # oldest went
        assert partial.exists()  # temp file untouched
    finally:
        set_disk_cache_dir(None)


def test_disk_cache_prune_is_serialised(_clean_mesh_cache, tmp_path,
                                        monkeypatch):
    """A pruner that finds the lock held does nothing — two concurrent
    pruners must not each evict down to the budget and empty the cache."""
    import fcntl

    import phasesweep.solvers.fem_mesh as _ff

    set_disk_cache_dir(tmp_path)
    try:
        for i, name in enumerate(["mesh_a.vol.gz", "mesh_b.vol.gz"]):
            p = tmp_path / name
            p.write_bytes(b"x" * 1000)
            os.utime(p, (1000 + i, 1000 + i))
        monkeypatch.setattr(_ff, "_DISK_CACHE_BUDGET_MB", 1000 / 1024 / 1024)

        # Stand in for the peer worker holding the lock.
        fd = os.open(str(tmp_path / ".prune.lock"), os.O_CREAT | os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            _ff._prune_disk_cache()
            assert len(list(tmp_path.glob("mesh_*.vol.gz"))) == 2
        finally:
            os.close(fd)

        # Lock released: the same call now evicts.
        _ff._prune_disk_cache()
        assert len(list(tmp_path.glob("mesh_*.vol.gz"))) == 1
    finally:
        set_disk_cache_dir(None)


def test_rotation_shares_mesh_without_arcs(_clean_mesh_cache, tmp_path):
    """A full-ring rotor has no rotation-dependent geometry, so every angle
    shares one mesh instead of building one each."""
    set_disk_cache_dir(tmp_path)
    try:
        geo = default_inrunner()
        _solve(geo=geo, n_p=4, alpha_p=1.0, rotation=0.0, **FAST)
        _solve(geo=geo, n_p=4, alpha_p=1.0, rotation=0.1, **FAST)
        _solve(geo=geo, n_p=4, alpha_p=1.0, rotation=0.2, **FAST)
        assert len(_mesh_cache) == 1
        assert list(tmp_path.glob("mesh_*_rot*.vol.gz")) == []
        assert len(list(tmp_path.glob("mesh_*.vol.gz"))) == 1
    finally:
        set_disk_cache_dir(None)


def test_shared_mesh_still_rotates_the_field(_clean_mesh_cache):
    """Sharing one mesh across angles must not flatten the physics — the
    magnetisation rotates in the coefficient function, so B_r still moves.
    Without this, a normalisation bug that also dropped the CF rotation
    would pass test_rotation_shares_mesh_without_arcs unnoticed."""
    geo = default_inrunner()
    _, br0, *_ = _solve(geo=geo, n_p=4, alpha_p=1.0, rotation=0.0, **FAST)
    _, br1, *_ = _solve(geo=geo, n_p=4, alpha_p=1.0, rotation=0.3, **FAST)
    assert len(_mesh_cache) == 1  # same mesh
    br0, br1 = np.asarray(br0), np.asarray(br1)
    assert np.max(np.abs(br0 - br1)) > 0.05 * np.max(np.abs(br0))


def test_arcs_active_legs():
    """_arcs_active is the single gate deciding both whether rotation reaches
    the geometry and whether the cache key keeps it. Pin all three legs.
    (n_p == 0 is unreachable through solve_field_fem — the analytical
    derivation rejects n_p < 2 — so it is checked at this level only.)"""
    geo = default_inrunner()
    assert fem_mesh._arcs_active(geo, 0.75, 4)        # arcs cut
    assert not fem_mesh._arcs_active(geo, 1.0, 4)     # full ring
    assert not fem_mesh._arcs_active(geo, 0.75, 0)    # no pole axes
    # Gap exactly at the OCC threshold counts as active; just under does not.
    r_pm, n_p = geo.r_rotor, 4
    at = 1.0 - fem_mesh._MIN_ARC_GAP_M * n_p / (np.pi * r_pm)
    assert fem_mesh._arcs_active(geo, at, n_p)
    assert not fem_mesh._arcs_active(geo, at + 1e-9, n_p)
    # Outrunner leg: the gap wedge is narrowest at the PM-annulus inner
    # radius, which is r_magnet here, not r_rotor — pin the _pm_radii
    # topology branch through the same boundary check.
    geo_o = outrunner(r_outer=0.10, r_rotor=0.08, r_magnet=0.06,
                      r_stator=0.04, r_inner=0.02)
    at_o = 1.0 - fem_mesh._MIN_ARC_GAP_M * n_p / (np.pi * geo_o.r_magnet)
    assert fem_mesh._arcs_active(geo_o, at_o, n_p)
    assert not fem_mesh._arcs_active(geo_o, at_o + 1e-9, n_p)
    # A bridge ring moves the outrunner magnet inner edge outward to
    # r_magnet + t_b, widening the wedge there: the alpha_p that was just
    # under threshold flips back to active.
    geo_b = _dc.replace(geo_o, bridge_thickness=0.005)
    assert fem_mesh._arcs_active(geo_b, at_o + 1e-9, n_p)


def test_rotation_key_normalised_when_arcs_collapse(_clean_mesh_cache):
    """The riskiest leg of _arcs_active: alpha_p < 1 requests arcs, but the
    interpole gap is under the OCC cut threshold so none are built. The key
    must follow the geometry that was actually meshed, not the request."""
    geo = default_inrunner()
    # Gap just under _MIN_ARC_GAP_M, so the arcs collapse to a full ring.
    n_p = 4
    r_pm = geo.r_rotor
    alpha_p = 1.0 - 0.9 * fem_mesh._MIN_ARC_GAP_M * n_p / (np.pi * r_pm)
    assert not fem_mesh._arcs_active(geo, alpha_p, n_p)

    _solve(geo=geo, n_p=n_p, alpha_p=alpha_p, rotation=0.0, **FAST)
    _solve(geo=geo, n_p=n_p, alpha_p=alpha_p, rotation=0.12, **FAST)
    assert len(_mesh_cache) == 1


def test_disk_cache_disabled_by_default(_clean_mesh_cache, tmp_path):
    """No disk files created when disk cache is not enabled."""
    set_disk_cache_dir(None)
    _solve(**FAST)
    assert len(list(tmp_path.glob("mesh_*"))) == 0


# ---------------------------------------------------------------------------
# Composable geometry builder tests
# ---------------------------------------------------------------------------

class TestBuildAnnularRegions:
    """Unit tests for _build_annular_regions."""

    def test_inrunner_face_count(self):
        geo = default_inrunner()
        faces, innermost, named = _build_annular_regions(geo)
        assert len(faces) == 4  # stator, airgap, pm, shaft
        assert innermost is not None
        assert set(named.keys()) == {"stator", "airgap", "pm", "shaft"}

    def test_outrunner_face_count(self):
        geo = outrunner(r_outer=0.050, r_rotor=0.045, r_magnet=0.040,
                        r_stator=0.035, r_inner=0.015)
        faces, innermost, named = _build_annular_regions(geo)
        assert len(faces) == 5  # yoke, pm, airgap, stator, air
        assert innermost is not None
        assert set(named.keys()) == {"yoke", "pm", "airgap", "stator", "air"}


class TestBuildMotorGeometry:
    """Verify _build_motor_geometry produces meshable shapes."""

    def test_smooth_bore_meshable(self):
        geo = default_inrunner()
        shape = _build_motor_geometry(geo)
        mesh = Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh=0.01))
        assert mesh.ne > 0

    def test_slotted_meshable(self):
        geo = inrunner(r_outer=0.050, r_stator=0.030, r_magnet=0.025,
                       r_rotor=0.015, n_slots=6, slot_depth=0.005,
                       slot_width_ratio=0.5)
        shape = _build_motor_geometry(
            geo, n_slots=6, slot_width_ratio=0.5, slot_depth=0.005)
        mesh = Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh=0.005))
        assert mesh.ne > 0

    @pytest.mark.slow
    def test_magnet_arcs_meshable(self):
        geo = default_inrunner()
        shape = _build_motor_geometry(geo, alpha_p=0.75, n_p=4)
        mesh = Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh=0.005))
        assert mesh.ne > 0

    def test_full_pitch_no_arcs(self):
        """alpha_p=1.0 should produce same result as smooth bore."""
        geo = default_inrunner()
        shape_plain = _build_motor_geometry(geo)
        shape_full = _build_motor_geometry(geo, alpha_p=1.0, n_p=4)
        m1 = Mesh(OCCGeometry(shape_plain, dim=2).GenerateMesh(maxh=0.01))
        m2 = Mesh(OCCGeometry(shape_full, dim=2).GenerateMesh(maxh=0.01))
        assert abs(m1.ne - m2.ne) < 5


# ---------------------------------------------------------------------------
# Halbach Phase 2 — mesh segment cuts + cache key
# ---------------------------------------------------------------------------

class TestHalbachMesh:

    _HALBACH_GEO = inrunner(
        r_outer=0.057, r_stator=0.047, r_magnet=0.046, r_rotor=0.043,
    )

    def test_halbach_mesh_materials(self):
        """Halbach mesh has pmh0..pmh{2*n_p*kappa_m-1} and no 'pm'."""
        n_p, kappa_m = 10, 3
        shape = _build_motor_geometry(
            self._HALBACH_GEO, n_p=n_p,
            magnetization="halbach", halbach_segments=kappa_m,
        )
        mesh = Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh=0.005))
        mats = set(mesh.GetMaterials())
        expected_segs = {f"pmh{m}" for m in range(2 * n_p * kappa_m)}
        assert expected_segs <= mats
        assert "pm" not in mats

    def test_halbach_segment_count(self):
        """Number of distinct pmhN materials equals 2*n_p*kappa_m."""
        n_p, kappa_m = 4, 4
        shape = _build_motor_geometry(
            self._HALBACH_GEO, n_p=n_p,
            magnetization="halbach", halbach_segments=kappa_m,
        )
        mesh = Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh=0.005))
        pmh_count = sum(1 for m in mesh.GetMaterials() if m.startswith("pmh"))
        assert pmh_count == 2 * n_p * kappa_m

    def test_radial_mesh_materials_unchanged(self):
        """Radial mesh still has 'pm' and no 'pmh' regions."""
        shape = _build_motor_geometry(self._HALBACH_GEO, n_p=4)
        mesh = Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh=0.005))
        mats = set(mesh.GetMaterials())
        assert "pm" in mats
        assert not any(m.startswith("pmh") for m in mats)

    def test_halbach_sliver_guard(self):
        """Absurd kappa_m that would create sub-mm segments raises."""
        from phasesweep.solvers.fem_mesh import _cut_halbach_segments
        geo = inrunner(
            r_outer=0.057, r_stator=0.047, r_magnet=0.046, r_rotor=0.043,
        )
        with pytest.raises(ValueError, match="arc width"):
            _cut_halbach_segments(geo, n_p=10, kappa_m=100)

    def test_halbach_degenerate_n_p_kappa_guard(self):
        """n_p * kappa_m == 1 would make a pi-radian wedge; guard raises."""
        from phasesweep.solvers.fem_mesh import _cut_halbach_segments
        geo = inrunner(
            r_outer=0.057, r_stator=0.047, r_magnet=0.046, r_rotor=0.043,
        )
        with pytest.raises(ValueError, match="n_p .* halbach_segments must be >= 2"):
            _cut_halbach_segments(geo, n_p=1, kappa_m=1)

    def test_halbach_bridge_mesh_materials(self):
        """Halbach + bridge_thickness builds pmh segments and a bridge ring."""
        geo = inrunner(
            r_outer=0.057, r_stator=0.047, r_magnet=0.046, r_rotor=0.043,
            bridge_thickness=0.5e-3,
        )
        n_p, kappa_m = 4, 2
        shape = _build_motor_geometry(
            geo, n_p=n_p,
            magnetization="halbach", halbach_segments=kappa_m,
        )
        mesh = Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh=0.005))
        mats = set(mesh.GetMaterials())
        assert "bridge" in mats
        assert "pm" not in mats
        pmh_count = sum(1 for m in mats if m.startswith("pmh"))
        assert pmh_count == 2 * n_p * kappa_m

    def test_halbach_outrunner_materials(self):
        """Outrunner Halbach mesh has pmh segments, no 'pm'."""
        geo = outrunner(
            r_outer=0.10, r_rotor=0.08, r_magnet=0.06,
            r_stator=0.04, r_inner=0.02,
        )
        n_p, kappa_m = 7, 2
        shape = _build_motor_geometry(
            geo, n_p=n_p,
            magnetization="halbach", halbach_segments=kappa_m,
        )
        mesh = Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh=0.005))
        mats = set(mesh.GetMaterials())
        assert "pm" not in mats
        pmh_count = sum(1 for m in mats if m.startswith("pmh"))
        assert pmh_count == 2 * n_p * kappa_m


class TestHalbachCacheKey:

    def test_radial_cache_filename_unchanged(self):
        """Radial cache filename is byte-identical to pre-halbach format."""
        from phasesweep.solvers.fem_mesh import _cache_filename
        key = ("abc123", 0.05, 12, 0.6, 1.0, 4, 0.0, "")
        fname = _cache_filename(key)
        assert fname == (
            f"mesh_v6_abc123_{0.05!r}_12_{0.6!r}_{1.0!r}_4.vol.gz"
        )

    def test_halbach_cache_filename_has_suffix(self):
        """Halbach cache filename appends _h{kappa_m}."""
        from phasesweep.solvers.fem_mesh import _cache_filename
        key = ("abc123", 0.05, 0, 0.6, 1.0, 10, 0.0, "h3")
        fname = _cache_filename(key)
        assert fname.endswith("_h3.vol.gz")
        assert "mesh_v6_" in fname

    def test_different_kappa_m_different_cache_key(self):
        """Different halbach_segments produce different cache keys."""
        from phasesweep.solvers.fem_mesh import _cache_filename
        k3 = ("abc123", 0.05, 0, 0.6, 1.0, 10, 0.0, "h3")
        k4 = ("abc123", 0.05, 0, 0.6, 1.0, 10, 0.0, "h4")
        assert _cache_filename(k3) != _cache_filename(k4)

    def test_rotation_in_cache_key_position(self):
        """Rotation at index 6 distinguishes both keys and filenames."""
        from phasesweep.solvers.fem_mesh import _cache_filename
        r0 = ("abc123", 0.05, 0, 0.6, 1.0, 10, 0.0, "h3")
        r1 = ("abc123", 0.05, 0, 0.6, 1.0, 10, 0.1, "h3")
        assert r0 != r1
        assert _cache_filename(r0) != _cache_filename(r1)
