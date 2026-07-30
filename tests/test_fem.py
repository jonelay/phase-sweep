"""Integration tests for fem_field.py — use coarse meshes to keep runtime low."""

import numpy as np
import pytest
from netgen.occ import OCCGeometry
from ngsolve import H1, L2, GridFunction, Mesh, grad, sqrt

from phasesweep import fem_field
from phasesweep.analytical import (
    _derive_B_rem,
    carter_adjusted_radii,
    carter_factor,
    end_effect_factor,
    end_effect_factor_pole_pitch,
    zhu_howe_Br,
)
from phasesweep.fem_field import (
    _BH_B,
    _BH_H,
    _DEFAULT_GEO,
    _MU0,
    _bh_nu,
    _build_annular_regions,
    _build_geometry,
    _build_motor_geometry,
    _build_slotted_geometry,
    _circle_spec,
    _mesh_cache,
    clear_mesh_cache,
    set_disk_cache_dir,
    solve_field_fem,
)
from phasesweep.geometry import default_inrunner, inrunner, outrunner
from phasesweep.harmonics import batch_harmonics, harmonics_1sided

FAST = dict(n_theta=60, maxh_fraction=0.08)
_GEO = default_inrunner()
# zhu_howe_Br radii are required keywords
_RADII = dict(r_stator=_GEO.r_stator, r_magnet=_GEO.r_magnet,
              r_rotor=_GEO.r_rotor)


def _solve(n_p=2, psi_f=0.1, N=50, k_w=0.966, L_stk=0.10,
           n_slots=0, j_s=0.0, nonlinear=False, geo=None, **kw):
    """Helper: legacy-style call via new Geometry-based solve_field_fem."""
    _geo = geo or _GEO
    B_rem = _derive_B_rem(psi_f, n_p, N, k_w, L_stk,
                          r_stator=_geo.r_stator, r_magnet=_geo.r_magnet,
                          r_rotor=_geo.r_rotor)
    merged = {**FAST, **kw}
    return solve_field_fem(
        geo=_geo, n_p=n_p, B_rem=B_rem,
        mu_r_pm=1.05, mu_r_fe=1000.0,
        n_slots=n_slots, j_s=j_s, nonlinear=nonlinear,
        **merged,
    )


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


_SLOTTED_GEO_KW = dict(r_outer=1.0, r_stator=0.70, r_magnet=0.64, r_rotor=0.30,
                       r_inner=0.0, n_slots=12, slot_depth=0.05)
_SLOTTED_OUT_GEO_KW = dict(r_outer=0.10, r_rotor=0.08, r_magnet=0.06,
                           r_stator=0.04, r_inner=0.02, n_slots=6,
                           slot_depth=0.005)


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


def _spy_build_geometry(monkeypatch):
    """Capture kwargs passed to _build_motor_geometry, bypassing caches."""
    captured = {}
    orig = fem_field._build_motor_geometry

    def spy(g, **kw):
        captured.update(kw)
        return orig(g, **kw)

    monkeypatch.setattr(fem_field, "_build_motor_geometry", spy)
    monkeypatch.setattr(fem_field, "_disk_cache_dir", None)
    clear_mesh_cache()
    return captured


def test_solve_field_fem_slot_params_default_from_geometry(monkeypatch):
    """Omitted n_slots/slot_width_ratio fall back to the geometry's values."""
    geo = inrunner(**_SLOTTED_GEO_KW, slot_width_ratio=0.42)
    captured = _spy_build_geometry(monkeypatch)
    solve_field_fem(geo=geo, n_p=2, B_rem=1.0, mu_r_pm=1.05, mu_r_fe=1000.0,
                    **FAST)
    assert captured["n_slots"] == 12
    assert captured["slot_width_ratio"] == 0.42


def test_solve_field_fem_explicit_zero_overrides_geometry(monkeypatch):
    """Explicit n_slots=0 forces smooth bore despite slotted geometry."""
    geo = inrunner(**_SLOTTED_GEO_KW, slot_width_ratio=0.42)
    captured = _spy_build_geometry(monkeypatch)
    solve_field_fem(geo=geo, n_p=2, B_rem=1.0, mu_r_pm=1.05, mu_r_fe=1000.0,
                    n_slots=0, **FAST)
    assert captured["n_slots"] == 0


def _fake_raster(mesh, gfu, n_grid, r_bound):
    grid = np.full((n_grid, n_grid), np.nan)
    return grid, grid, grid, grid


def test_fem_runner_defers_slot_params_to_geometry(monkeypatch):
    """Production FEM path passes no slot kwargs — geometry governs."""
    from phasesweep.fem_runner import _run_fem_impl
    from phasesweep.sweep_types import RunConfig
    from tests.conftest import make_motor

    captured = {}

    def fake_solve(*args, **kw):
        captured.update(kw)
        th = np.linspace(0, 2 * np.pi, 60, endpoint=False)
        # runner always requests return_full=True for the raster
        return th, np.cos(2 * th), None, None

    monkeypatch.setattr(fem_field, "solve_field_fem", fake_solve)
    monkeypatch.setattr(fem_field, "rasterise_cross_section", _fake_raster)
    motor = make_motor(B_rem=1.2, psi_f=None)
    _run_fem_impl(RunConfig(motor=motor, model="fem"))
    assert "n_slots" not in captured
    assert "slot_width_ratio" not in captured


def test_fem_runner_j_s_requires_k_w():
    from phasesweep.fem_runner import _run_fem_impl
    from phasesweep.sweep_types import RunConfig
    from tests.conftest import make_motor

    motor = make_motor(B_rem=1.2, psi_f=None, k_w=None)
    with pytest.raises(ValueError, match="k_w"):
        _run_fem_impl(RunConfig(motor=motor, model="fem", j_s=1.0))


def test_fem_runner_reports_both_slot_harmonic_sidebands(monkeypatch):
    """Slot harmonics come in pairs Q ± n_p; both sidebands are reported."""
    from phasesweep.fem_runner import _run_fem_impl
    from phasesweep.geometry import inrunner
    from phasesweep.sweep_types import RunConfig
    from tests.conftest import make_motor

    def fake_solve(*args, **kw):
        th = np.linspace(0, 2 * np.pi, 120, endpoint=False)
        # fundamental at n_p=2; sidebands at Q-n_p=10 and Q+n_p=14
        return th, (np.cos(2 * th) + 0.20 * np.cos(10 * th)
                    + 0.10 * np.cos(14 * th)), None, None

    monkeypatch.setattr(fem_field, "solve_field_fem", fake_solve)
    monkeypatch.setattr(fem_field, "rasterise_cross_section", _fake_raster)
    motor = make_motor(geometry=inrunner(**_SLOTTED_GEO_KW), B_rem=1.2, psi_f=None)
    metrics = _run_fem_impl(RunConfig(motor=motor, model="fem"))
    assert metrics["sh_pct"] == pytest.approx(20.0, rel=1e-6)
    assert metrics["sh_upper_pct"] == pytest.approx(10.0, rel=1e-6)


def test_fem_runner_emits_cross_section_raster():
    """Dashboard heatmap raster: |B| on a shared-axis grid,
    None outside the motor domain, additive alongside existing metrics."""
    from phasesweep.fem_runner import _RASTER_N_GRID, _run_fem_impl
    from phasesweep.sweep_types import RunConfig
    from tests.conftest import make_motor

    motor = make_motor(B_rem=1.2, psi_f=None)
    metrics = _run_fem_impl(RunConfig(motor=motor, model="fem", **FAST))
    grid = metrics["B_mag_grid"]
    coords = metrics["grid_coords_list"]
    assert len(grid) == _RASTER_N_GRID
    assert all(len(row) == _RASTER_N_GRID for row in grid)
    assert len(coords) == _RASTER_N_GRID
    assert coords[-1] == pytest.approx(motor.geometry.r_outer)
    assert coords[0] == pytest.approx(-motor.geometry.r_outer)
    # corners of the square are outside the motor disk
    assert grid[0][0] is None and grid[-1][-1] is None
    vals = [v for row in grid for v in row if v is not None]
    assert vals and max(vals) > 0.1
    assert "fundamental" in metrics and "B_r_list" in metrics


# ---------------------------------------------------------------------------
# Solver — smooth bore
# ---------------------------------------------------------------------------

def test_smooth_bore_returns_correct_shape():
    theta, B_r = _solve()
    assert theta.shape == (FAST["n_theta"],)
    assert B_r.shape == (FAST["n_theta"],)


def test_smooth_bore_has_dominant_harmonic():
    n_p = 2
    theta, B_r = _solve(n_p=n_p)
    amps = harmonics_1sided(B_r)
    assert np.max(amps[1:]) > 0.01


def test_smooth_bore_field_nonzero():
    theta, B_r = _solve()
    assert np.nanmax(np.abs(B_r)) > 1e-4


def test_smooth_bore_field_sign_matches_analytical():
    n_p = 2
    theta, B_fem = _solve(n_p=n_p)
    B_rem = _derive_B_rem(0.1, n_p, 50, 0.966, 0.10,
                          r_stator=_DEFAULT_GEO.r_stator,
                          r_magnet=_DEFAULT_GEO.r_magnet,
                          r_rotor=_DEFAULT_GEO.r_rotor)
    B_an = zhu_howe_Br(theta, n_p, B_rem, **_RADII)
    # FEM waveform is flat-topped (square-wave source) vs the analytical
    # pure cosine; correlation ~0.95 proves same sign/phase, not shape
    assert np.corrcoef(B_fem, B_an)[0, 1] > 0.9


def test_smooth_bore_scales_with_psi_f():
    _, B1 = _solve(psi_f=0.1)
    _, B2 = _solve(psi_f=0.2)
    ratio = np.nanmax(np.abs(B2)) / np.nanmax(np.abs(B1))
    assert ratio == pytest.approx(2.0, rel=0.05)


# ---------------------------------------------------------------------------
# Solver — slotted + winding
# ---------------------------------------------------------------------------

def test_slotted_returns_correct_shape():
    # Real slot faces required by the j_s guard (the default geometry
    # has slot_depth = 0, where j_s != 0 now raises instead of being
    # silently ignored).
    theta, B_r = _solve(n_slots=12, j_s=0.1, geo=inrunner(**_SLOTTED_GEO_KW))
    assert B_r.shape == (FAST["n_theta"],)


def test_armature_without_slots_raises():
    with pytest.raises(ValueError, match="n_slots > 0"):
        _solve(n_slots=0, j_s=0.1)


def test_backward_compat_n_slots_zero():
    _, B1 = _solve()
    _, B2 = _solve(n_slots=0, j_s=0.0)
    np.testing.assert_allclose(B1, B2, rtol=1e-10)


def test_slotted_solve_completes():
    n_p, Q = 2, 12
    theta, B_r = _solve(n_p=n_p, n_slots=Q, j_s=0.0)
    assert B_r.shape == (FAST["n_theta"],)
    assert np.all(np.isfinite(B_r))
    assert np.nanmax(np.abs(B_r)) > 1e-4


def test_winding_current_solve_completes():
    # Real slot faces required — this test used to run on the default
    # geometry (slot_depth = 0), where the j_s source integrated over an
    # empty region and the "winding current" solve was actually
    # open-circuit. Now it also asserts the current changes the
    # field.
    geo = inrunner(**_SLOTTED_GEO_KW)
    theta, B_r = _solve(n_slots=12, j_s=1e5, geo=geo)
    assert B_r.shape == (FAST["n_theta"],)
    assert np.all(np.isfinite(B_r))
    assert np.nanmax(np.abs(B_r)) > 1e-4
    _, B_r0 = _solve(n_slots=12, j_s=0.0, geo=geo)
    assert np.max(np.abs(B_r - B_r0)) > 1e-3


# ---------------------------------------------------------------------------
# Zhu & Howe analytical model
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# End-effect factor tests
# ---------------------------------------------------------------------------

def test_end_effect_factor_long_stack():
    """k_end → 1 for stacks much longer than the air gap."""
    assert end_effect_factor(L_stk=1.0, g_eff=0.001) == pytest.approx(1.0, abs=0.001)


def test_end_effect_factor_short_stack():
    """k_end < 1 for short stacks."""
    k = end_effect_factor(L_stk=0.007, g_eff=0.00175)
    assert 0.8 < k < 0.95


def test_end_effect_factor_monotonic():
    """k_end increases monotonically with L_stk."""
    g_eff = 0.002
    prev = 0.0
    for L in [0.005, 0.01, 0.02, 0.05, 0.1]:
        k = end_effect_factor(L_stk=L, g_eff=g_eff)
        assert k > prev
        prev = k


def test_end_effect_factor_zero_returns_one():
    """Edge cases: zero or negative inputs return 1.0."""
    assert end_effect_factor(L_stk=0, g_eff=0.001) == 1.0
    assert end_effect_factor(L_stk=0.01, g_eff=0) == 1.0


def test_end_effect_factor_actuator_range():
    """Actuator-like geometry: L_stk=7mm, g_eff≈1.75mm → k_end ≈ 0.84."""
    k = end_effect_factor(L_stk=0.007, g_eff=0.00175)
    assert 0.80 < k < 0.90


def test_end_effect_factor_pole_pitch_limits():
    """Pole-pitch form: → 1 for long stacks, 1.0 at degenerate inputs."""
    assert end_effect_factor_pole_pitch(L_stk=1.0, tau_p=0.004) == pytest.approx(1.0, abs=0.01)
    assert end_effect_factor_pole_pitch(L_stk=0, tau_p=0.004) == 1.0
    assert end_effect_factor_pole_pitch(L_stk=0.01, tau_p=0) == 1.0


def test_end_effect_factor_pole_pitch_monotonic():
    """Pole-pitch form increases monotonically with L_stk."""
    tau_p = 0.004
    prev = 0.0
    for L in [0.005, 0.01, 0.02, 0.05, 0.1]:
        k = end_effect_factor_pole_pitch(L_stk=L, tau_p=tau_p)
        assert k > prev
        prev = k


def test_end_effect_factor_pole_pitch_actuator_bracket():
    """Actuator-like geometry: pole-pitch form sits below the gap-scale
    form (aggressive edge of the bracket)."""
    k_pp = end_effect_factor_pole_pitch(L_stk=0.007, tau_p=0.00378)
    k_gap = end_effect_factor(L_stk=0.007, g_eff=0.00175)
    assert 0.60 < k_pp < 0.72
    assert k_pp < k_gap


def test_zhu_howe_rejects_np1():
    with pytest.raises(ValueError, match="n_p=1"):
        zhu_howe_Br(np.linspace(0, 2 * np.pi, 60), n_p=1, B_rem=1.0,
                    **_RADII)


def test_zhu_howe_is_pure_cosine():
    theta = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    Br = zhu_howe_Br(theta, n_p=2, B_rem=1.0, **_RADII)
    amps = harmonics_1sided(Br)
    assert amps[2] > 0.01
    assert amps[2] > 100 * np.max(np.delete(amps[1:], 1))


@pytest.mark.parametrize("n_p,psi_f,L_d,L_q,N,k_w,L_stk", [
    (2, 0.1, 4e-3, 4e-3, 50, 0.966, 0.10),   # Config A
    (4, 0.08, 2e-3, 2e-3, 40, 0.966, 0.08),   # Config C
    (8, 0.05, 1e-3, 1e-3, 50, 0.966, 0.10),   # high pole count
    (2, 0.1144, 0.2055, 0.3320, 328, 0.866, 0.0301),  # Creator Case PMSM
])
def test_fem_matches_zhu_howe(n_p, psi_f, L_d, L_q, N, k_w, L_stk):
    """FEM smooth-bore fundamental should match Zhu & Howe within 2%."""
    B_rem = _derive_B_rem(psi_f, n_p, N, k_w, L_stk,
                          r_stator=_DEFAULT_GEO.r_stator,
                          r_magnet=_DEFAULT_GEO.r_magnet,
                          r_rotor=_DEFAULT_GEO.r_rotor)
    n_theta = 180

    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    Br_an = zhu_howe_Br(theta, n_p, B_rem, **_RADII)
    an_fund = harmonics_1sided(Br_an)[n_p]

    _, Br_fem = solve_field_fem(
        geo=_GEO, n_p=n_p, B_rem=B_rem,
        mu_r_pm=1.05, mu_r_fe=1000.0,
        n_theta=n_theta, maxh_fraction=0.04,
        n_slots=0, j_s=0.0,
    )
    fem_fund = harmonics_1sided(Br_fem)[n_p]

    assert fem_fund == pytest.approx(an_fund, rel=0.02)


# ---------------------------------------------------------------------------
# batch_harmonics
# ---------------------------------------------------------------------------

def test_batch_harmonics_shape():
    n_theta = 360
    B = {"A": np.sin(2 * np.linspace(0, 2 * np.pi, n_theta, endpoint=False))}
    result = batch_harmonics(B)
    assert "A" in result
    assert result["A"].shape[0] == n_theta // 2 + 1


def test_batch_harmonics_fundamental():
    n_theta = 360
    k = 3
    B_r = np.sin(k * np.linspace(0, 2 * np.pi, n_theta, endpoint=False))
    result = batch_harmonics({"test": B_r})
    amps = result["test"]
    assert amps[k] == pytest.approx(1.0, rel=0.01)
    assert amps[k] > 10 * amps[k + 1]


def test_batch_harmonics_multi():
    n_theta = 360
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    data = {"A": np.cos(2 * theta), "B": np.cos(4 * theta)}
    result = batch_harmonics(data)
    assert set(result.keys()) == {"A", "B"}
    assert result["A"][2] > result["A"][4]
    assert result["B"][4] > result["B"][2]


def test_harmonics_odd_length_last_bin_doubled():
    n = 9
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    k = (n - 1) // 2  # last rfft bin is a regular harmonic for odd n
    amps = harmonics_1sided(0.25 * np.cos(k * theta))
    assert amps[k] == pytest.approx(0.25)
    batch = batch_harmonics({"A": 0.25 * np.cos(k * theta)})
    assert batch["A"][k] == pytest.approx(0.25)


def test_harmonics_even_length_nyquist_not_doubled():
    n = 8
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    amps = harmonics_1sided(np.cos((n // 2) * theta))
    assert amps[n // 2] == pytest.approx(1.0)


def test_double_onesided_does_not_mutate_input():
    from phasesweep.harmonics import _double_onesided
    amps = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    original = amps.copy()
    _double_onesided(amps, 8)
    np.testing.assert_array_equal(amps, original)


def test_compute_thd_nan_on_zero_fundamental():
    from phasesweep.harmonics import compute_thd
    assert np.isnan(compute_thd(np.array([0.0, 0.0, 0.5]), 1))


def test_compute_thd_pins_value_from_amplitudes():
    """Exact value from hand-built amplitudes: sqrt(0.3^2 + 0.4^2)/1.0 = 50%.

    The only numeric THD assertions elsewhere are one-sided bounds on measured
    harmonics, so a pure scale error in compute_thd was invisible suite-wide
    (a 1.10x scale passed the whole suite).
    """
    from phasesweep.harmonics import compute_thd
    # index:      0(DC)  1    2(fund)  3    4    5
    amps = np.array([0.7, 0.0, 1.0, 0.3, 0.0, 0.4])
    assert compute_thd(amps, 2) == pytest.approx(50.0)
    # DC is excluded: a large DC bin must not move the result.
    assert compute_thd(amps * np.array([100.0, 1, 1, 1, 1, 1]), 2) == pytest.approx(50.0)
    # THD is scale-invariant in the waveform amplitude.
    assert compute_thd(2.5 * amps, 2) == pytest.approx(50.0)


def test_compute_thd_square_wave_textbook_value():
    """End-to-end through the FFT: an ideal square wave has THD = 48.34%
    (sqrt(pi^2/8 - 1)); 720 samples leaves ~0.001 of series truncation."""
    from phasesweep.harmonics import compute_thd
    theta = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    amps = harmonics_1sided(np.sign(np.cos(theta)))
    assert compute_thd(amps, 1) == pytest.approx(
        np.sqrt(np.pi**2 / 8 - 1) * 100, abs=0.01
    )


# ---------------------------------------------------------------------------
# B-H curve and _bh_nu (Issue 1)
# ---------------------------------------------------------------------------

def test_bh_table_monotonic():
    assert np.all(np.diff(_BH_H) > 0)


def test_bh_nu_at_zero():
    nu = _bh_nu(np.array([0.0]))
    assert np.isfinite(nu[0])
    assert nu[0] > 0


def test_bh_nu_at_saturation():
    nu = _bh_nu(np.array([2.5]))
    nu_linear = _BH_H[1] / _BH_B[1]
    assert nu[0] > 10 * nu_linear


def test_bh_nu_linear_regime():
    nu = _bh_nu(np.array([0.1]))
    mu_r_initial = _BH_B[1] / (_MU0 * _BH_H[1])
    expected_nu = 1 / (_MU0 * mu_r_initial)
    np.testing.assert_allclose(nu[0], expected_nu, rtol=0.10)


# ---------------------------------------------------------------------------
# Nonlinear Picard iteration (Issue 2)
# ---------------------------------------------------------------------------

def test_nonlinear_reproduces_linear_at_low_flux():
    _, B_lin = _solve(psi_f=0.05)
    _, B_nl = _solve(psi_f=0.05, nonlinear=True)
    peak_lin = np.nanmax(np.abs(B_lin))
    peak_nl = np.nanmax(np.abs(B_nl))
    assert peak_nl == pytest.approx(peak_lin, rel=0.05)


def test_nonlinear_converges():
    theta, B_r = _solve(nonlinear=True)
    assert np.all(np.isfinite(B_r))


def test_nonlinear_changes_field_at_high_flux():
    small = inrunner(r_outer=0.05, r_stator=0.035, r_magnet=0.032, r_rotor=0.015, r_inner=0.0)
    _, B_lin = _solve(psi_f=0.3, geo=small, maxh_fraction=0.04)
    _, B_nl = _solve(psi_f=0.3, geo=small, nonlinear=True,
                     max_picard=40, picard_relax=0.15, picard_tol=0.02, maxh_fraction=0.04)
    assert not np.allclose(B_lin, B_nl, rtol=1e-3)


def test_nonlinear_raises_on_non_convergence():
    with pytest.raises(RuntimeError, match="Picard iteration did not converge"):
        _solve(nonlinear=True, max_picard=1, picard_tol=1e-15)


# ---------------------------------------------------------------------------
# Nonlinear API threading (Issue 3)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Outrunner FEM
# ---------------------------------------------------------------------------

_GEO_OUT = outrunner(r_outer=0.10, r_rotor=0.08, r_magnet=0.06,
                     r_stator=0.04, r_inner=0.02)


def test_outrunner_smooth_geometry_materials():
    shape = _build_geometry(_GEO_OUT)
    mesh = Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh=0.01))
    mats = set(mesh.GetMaterials())
    assert {"stator", "airgap", "pm", "yoke", "air"} <= mats
    assert "shaft" not in mats


def test_outrunner_smooth_bore_solve():
    """Outrunner FEM solve completes with nonzero field."""
    theta, B_r = solve_field_fem(
        geo=_GEO_OUT, n_p=4, B_rem=1.0,
        mu_r_pm=1.05, mu_r_fe=1000.0, **FAST,
    )
    assert B_r.shape == (FAST["n_theta"],)
    assert np.all(np.isfinite(B_r))
    assert np.nanmax(np.abs(B_r)) > 1e-4


def test_outrunner_back_iron_split_materials():
    """back_iron_thickness splits the yoke into back_iron + shell regions."""
    geo = outrunner(r_outer=0.10, r_rotor=0.08, r_magnet=0.06,
                    r_stator=0.04, r_inner=0.02, back_iron_thickness=0.01)
    shape = _build_geometry(geo)
    mesh = Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh=0.01))
    mats = set(mesh.GetMaterials())
    assert {"stator", "airgap", "pm", "back_iron", "shell", "air"} <= mats
    assert "yoke" not in mats


def test_outrunner_back_iron_split_matches_solid_when_linear():
    """A linear back-iron ring carries the return flux, so the split (iron ring
    + non-magnetic shell) reproduces the solid yoke's airgap fundamental. The
    physical difference only appears once the thin ring saturates (nonlinear)."""
    kw = dict(r_outer=0.10, r_rotor=0.08, r_magnet=0.06,
              r_stator=0.04, r_inner=0.02)
    solid = outrunner(**kw)
    split = outrunner(**kw, back_iron_thickness=0.01)
    common = dict(n_p=4, B_rem=1.0, mu_r_pm=1.05, mu_r_fe=1000.0,
                  n_theta=180, maxh_fraction=0.04)
    fund_solid = harmonics_1sided(solve_field_fem(geo=solid, **common)[1])[4]
    fund_split = harmonics_1sided(solve_field_fem(geo=split, **common)[1])[4]
    assert fund_split == pytest.approx(fund_solid, rel=0.02)


def test_outrunner_fem_matches_analytical():
    """Outrunner FEM fundamental should match Zhu & Howe within 2%."""
    n_p = 4
    B_rem = 1.0
    n_theta = 180

    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    Br_an = zhu_howe_Br(
        theta, n_p, B_rem, r_eval=_GEO_OUT.r_ag,
        r_stator=_GEO_OUT.r_stator, r_magnet=_GEO_OUT.r_magnet,
        r_rotor=_GEO_OUT.r_rotor, mu_r_pm=1.05,
    )
    an_fund = harmonics_1sided(Br_an)[n_p]

    _, Br_fem = solve_field_fem(
        geo=_GEO_OUT, n_p=n_p, B_rem=B_rem,
        mu_r_pm=1.05, mu_r_fe=1000.0,
        n_theta=n_theta, maxh_fraction=0.04,
    )
    fem_fund = harmonics_1sided(Br_fem)[n_p]

    assert fem_fund == pytest.approx(an_fund, rel=0.02)


def test_outrunner_slotted_geometry_materials():
    geo = outrunner(r_outer=0.10, r_rotor=0.08, r_magnet=0.06,
                    r_stator=0.04, r_inner=0.02, n_slots=6, slot_depth=0.005)
    shape = _build_slotted_geometry(geo, n_slots=6)
    mesh = Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh=0.005))
    mats = set(mesh.GetMaterials())
    assert {"stator", "slot", "airgap", "pm", "yoke", "air"} <= mats
    assert "shaft" not in mats


def test_outrunner_slotted_solve():
    """Outrunner slotted FEM solve completes."""
    geo = outrunner(r_outer=0.10, r_rotor=0.08, r_magnet=0.06,
                    r_stator=0.04, r_inner=0.02, n_slots=6, slot_depth=0.005)
    theta, B_r = solve_field_fem(
        geo=geo, n_p=4, B_rem=1.0,
        mu_r_pm=1.05, mu_r_fe=1000.0,
        n_slots=6, **FAST,
    )
    assert B_r.shape == (FAST["n_theta"],)
    assert np.all(np.isfinite(B_r))
    assert np.nanmax(np.abs(B_r)) > 1e-4


def test_outrunner_nonlinear_solve():
    """Outrunner nonlinear Picard converges."""
    theta, B_r = solve_field_fem(
        geo=_GEO_OUT, n_p=4, B_rem=1.0,
        mu_r_pm=1.05, mu_r_fe=1000.0,
        nonlinear=True, **FAST,
    )
    assert np.all(np.isfinite(B_r))


# ---------------------------------------------------------------------------
# Paper validation: FEM vs analytical cross-validation
# ---------------------------------------------------------------------------

def test_paper_inrunner_fem_vs_analytical(paper_geometry_8pole):
    """Paper inrunner (2p=8, Zhu & Howe dims): FEM vs analytical < 1% on fundamental."""
    n_p = 4
    B_rem = 1.2
    mu_r_pm = 1.05
    n_theta = 360
    geo = paper_geometry_8pole

    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    Br_an = zhu_howe_Br(
        theta, n_p, B_rem, r_eval=geo.r_ag,
        r_stator=geo.r_stator, r_magnet=geo.r_magnet,
        r_rotor=geo.r_rotor, mu_r_pm=mu_r_pm,
    )
    an_fund = harmonics_1sided(Br_an)[n_p]

    _, Br_fem = solve_field_fem(
        geo=geo, n_p=n_p, B_rem=B_rem,
        mu_r_pm=mu_r_pm, mu_r_fe=1000.0,
        n_theta=n_theta, maxh_fraction=0.03,
    )
    fem_fund = harmonics_1sided(Br_fem)[n_p]

    assert fem_fund == pytest.approx(an_fund, rel=0.01), (
        f"FEM fundamental {fem_fund:.6f} vs analytical {an_fund:.6f}"
    )


def test_paper_outrunner_fem_vs_analytical(paper_geometry_8pole_outrunner):
    """Paper outrunner (rearranged dims): FEM vs analytical < 1% on fundamental."""
    n_p = 4
    B_rem = 1.2
    mu_r_pm = 1.05
    n_theta = 360
    geo = paper_geometry_8pole_outrunner

    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    Br_an = zhu_howe_Br(
        theta, n_p, B_rem, r_eval=geo.r_ag,
        r_stator=geo.r_stator, r_magnet=geo.r_magnet,
        r_rotor=geo.r_rotor, mu_r_pm=mu_r_pm,
    )
    an_fund = harmonics_1sided(Br_an)[n_p]

    _, Br_fem = solve_field_fem(
        geo=geo, n_p=n_p, B_rem=B_rem,
        mu_r_pm=mu_r_pm, mu_r_fe=1000.0,
        n_theta=n_theta, maxh_fraction=0.03,
    )
    fem_fund = harmonics_1sided(Br_fem)[n_p]

    assert fem_fund == pytest.approx(an_fund, rel=0.01), (
        f"FEM fundamental {fem_fund:.6f} vs analytical {an_fund:.6f}"
    )


def test_config_id_differs_for_nonlinear():
    from phasesweep.motor import Motor
    from phasesweep.sweep_types import RunConfig, compute_run_id
    motor = Motor(name="test", geometry=_GEO, n_p=2, R_s=3.6,
                  L_d=4e-3, L_q=4e-3, psi_f=0.1, J=1.5e-3)
    rc1 = RunConfig(motor=motor, model="fem", nonlinear=False)
    rc2 = RunConfig(motor=motor, model="fem", nonlinear=True)
    assert compute_run_id(rc1) != compute_run_id(rc2)


def test_config_roundtrip_nonlinear():
    from phasesweep.motor import Motor
    from phasesweep.sweep_types import RunConfig, compute_run_id
    motor = Motor(name="test", geometry=_GEO, n_p=2, R_s=3.6,
                  L_d=4e-3, L_q=4e-3, psi_f=0.1, J=1.5e-3)
    rc = RunConfig(motor=motor, model="fem", nonlinear=True)
    rc2 = RunConfig.from_dict(rc.to_dict())
    assert rc2.nonlinear is True
    assert compute_run_id(rc2) == compute_run_id(rc)


# ---------------------------------------------------------------------------
# P0: _derive_B_rem roundtrip (Issue audit)
# ---------------------------------------------------------------------------

def test_derive_B_rem_scaled_radii():
    psi_f, n_p, N, k_w, L_stk = 0.1, 2, 50, 0.966, 0.10
    B_rem_default = _derive_B_rem(psi_f, n_p, N, k_w, L_stk,
                                  r_stator=_DEFAULT_GEO.r_stator,
                                  r_magnet=_DEFAULT_GEO.r_magnet,
                                  r_rotor=_DEFAULT_GEO.r_rotor)
    B_rem_scaled = _derive_B_rem(
        psi_f, n_p, N, k_w, L_stk,
        r_stator=0.35, r_magnet=0.32, r_rotor=0.15,
    )
    assert B_rem_scaled != pytest.approx(B_rem_default, rel=0.01)
    B_peak = zhu_howe_Br(
        np.array([0.0]), n_p, B_rem_scaled, r_eval=0.35,
        r_stator=0.35, r_magnet=0.32, r_rotor=0.15,
    )[0]
    psi_f_recovered = B_peak * 2 * N * k_w * 0.35 * L_stk / n_p
    assert psi_f_recovered == pytest.approx(psi_f, rel=1e-10)


def test_derive_B_rem_roundtrip():
    psi_f, n_p, N, k_w, L_stk = 0.1, 2, 50, 0.966, 0.10
    B_rem = _derive_B_rem(psi_f, n_p, N, k_w, L_stk,
                          r_stator=_DEFAULT_GEO.r_stator,
                          r_magnet=_DEFAULT_GEO.r_magnet,
                          r_rotor=_DEFAULT_GEO.r_rotor)
    assert 0.01 < B_rem < 1.0, f"B_rem={B_rem} outside physically reasonable range"
    r_si = _DEFAULT_GEO.r_stator
    B_peak = zhu_howe_Br(np.array([0.0]), n_p, B_rem, r_eval=r_si,
                         **_RADII)[0]
    psi_f_recovered = B_peak * 2 * N * k_w * r_si * L_stk / n_p
    assert psi_f_recovered == pytest.approx(psi_f, rel=1e-10)


# ---------------------------------------------------------------------------
# P1: _demoivre direct evaluation (Issue audit)
# ---------------------------------------------------------------------------

def test_demoivre_matches_trig():
    from math import cos, pi, sin

    from netgen.occ import OCCGeometry, WorkPlane

    from phasesweep.fem_field import _demoivre

    face = WorkPlane().Circle(1.0).Face()
    geo = OCCGeometry(face, dim=2)
    ngmesh = Mesh(geo.GenerateMesh(maxh=0.5))

    for n_p in (2, 8):
        cos_np, sin_np = _demoivre(n_p)
        for angle in (0.0, pi / (2 * n_p)):
            r = 0.5
            mp = ngmesh(r * cos(angle), r * sin(angle))
            assert cos_np(mp) == pytest.approx(cos(n_p * angle), abs=1e-12)
            assert sin_np(mp) == pytest.approx(sin(n_p * angle), abs=1e-12)


# ---------------------------------------------------------------------------
# P1: Nonlinear saturation validation (Issue audit)
# ---------------------------------------------------------------------------

def test_nonlinear_scaling_sublinear_at_high_flux():
    kw_base = dict(n_theta=60, maxh_fraction=0.04,
                   max_picard=40, picard_relax=0.15, picard_tol=0.02)
    _, B_lo = _solve(psi_f=0.05, nonlinear=True, **kw_base)
    _, B_hi = _solve(psi_f=0.10, nonlinear=True, **kw_base)
    ratio_low = np.nanmax(np.abs(B_hi)) / np.nanmax(np.abs(B_lo))

    # Square-wave source has a lower crest than cosine at
    # equal fundamental, so saturation onset moved up: psi_f=1→2 stays
    # linear; 2→4 is in the transition regime (ratio ≈ 1.94)
    _, B_lo2 = _solve(psi_f=2.0, nonlinear=True, **kw_base)
    _, B_hi2 = _solve(psi_f=4.0, nonlinear=True, **kw_base)
    ratio_high = np.nanmax(np.abs(B_hi2)) / np.nanmax(np.abs(B_lo2))

    assert ratio_low == pytest.approx(2.0, abs=0.01), (
        f"Low-flux ratio should be ~2.0 (got {ratio_low:.4f})"
    )
    assert ratio_high < ratio_low, (
        f"High-flux ratio ({ratio_high:.4f}) should be < low-flux ({ratio_low:.4f}) "
        f"due to iron saturation"
    )


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


def test_disk_cache_skips_rotated_mesh(_clean_mesh_cache, tmp_path):
    """Rotated meshes (cogging sweeps) are cached in-memory only — one disk
    file per angle would grow the cache without bound."""
    set_disk_cache_dir(tmp_path)
    try:
        geo = default_inrunner()
        _solve(geo=geo, n_p=4, alpha_p=0.75, rotation=0.1, **FAST)
        assert len(_mesh_cache) == 1
        assert list(tmp_path.glob("mesh_*.vol.gz")) == []
        _solve(geo=geo, n_p=4, alpha_p=0.75, **FAST)
        assert len(list(tmp_path.glob("mesh_*.vol.gz"))) == 1
    finally:
        set_disk_cache_dir(None)


def test_disk_cache_disabled_by_default(_clean_mesh_cache, tmp_path):
    """No disk files created when disk cache is not enabled."""
    set_disk_cache_dir(None)
    _solve(**FAST)
    assert len(list(tmp_path.glob("mesh_*"))) == 0


# ---------------------------------------------------------------------------
# Physics validation
# ---------------------------------------------------------------------------

def _peak_B_iron(mesh_obj, gfu):
    """Extract peak |B| in iron regions from a FEM solution."""
    B_mag = sqrt(grad(gfu)[0]**2 + grad(gfu)[1]**2)
    fes_L2 = L2(mesh_obj, order=0)
    gfu_B = GridFunction(fes_L2)
    gfu_B.Set(B_mag)

    iron_names = {"stator", "shaft", "yoke"}
    iron_cf = {
        name: (1.0 if name in iron_names else 0.0)
        for name in mesh_obj.GetMaterials()
    }
    iron_ind = GridFunction(fes_L2)
    iron_ind.Set(mesh_obj.MaterialCF(iron_cf))

    B_vals = gfu_B.vec.FV().NumPy()
    mask = iron_ind.vec.FV().NumPy() > 0.5
    if not np.any(mask):
        return 0.0
    return float(np.max(B_vals[mask]))


def test_mesh_convergence():
    """FEM fundamental converges as mesh is refined; FAST mesh is adequate.

    Solves at 4 mesh densities and checks that the coarse test mesh
    (maxh_fraction=0.08) is within 2% of the finest mesh, proving the
    FAST mesh is in the asymptotic convergence regime.
    """
    maxh_fracs = [0.10, 0.06, 0.04, 0.02]
    funds = []
    for mf in maxh_fracs:
        _, B_r = solve_field_fem(
            geo=_GEO, n_p=2, B_rem=1.0,
            mu_r_pm=1.05, mu_r_fe=1000.0,
            n_theta=360, maxh_fraction=mf,
        )
        funds.append(harmonics_1sided(B_r)[2])

    # Successive differences should decrease (convergence)
    diffs = [abs(funds[i+1] - funds[i]) for i in range(len(funds) - 1)]
    for i in range(len(diffs) - 1):
        assert diffs[i+1] < diffs[i], (
            f"Convergence stalled: delta[{i}]={diffs[i]:.6e}, "
            f"delta[{i+1}]={diffs[i+1]:.6e}"
        )

    # FAST mesh (0.08, between first two points) within 2% of finest
    _, B_r_fast = solve_field_fem(
        geo=_GEO, n_p=2, B_rem=1.0,
        mu_r_pm=1.05, mu_r_fe=1000.0,
        n_theta=360, maxh_fraction=0.08,
    )
    fund_fast = harmonics_1sided(B_r_fast)[2]
    fund_finest = funds[-1]
    assert fund_fast == pytest.approx(fund_finest, rel=0.02), (
        f"FAST mesh fundamental {fund_fast:.6f} vs finest {fund_finest:.6f}"
    )


def test_fem_analytical_gap_decreases_with_mu_r_fe():
    """FEM-analytical error decreases as mu_r_fe increases.

    The analytical model assumes infinite iron permeability. FEM uses
    finite mu_r_fe. The gap between them should shrink as mu_r_fe → ∞,
    confirming the 2% tolerance is physically motivated.
    Uses fine mesh (maxh_fraction=0.02) to isolate the mu_r_fe effect.
    """
    n_p = 2
    B_rem = 1.0
    n_theta = 360
    maxh = 0.02

    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    Br_an = zhu_howe_Br(theta, n_p, B_rem, **_RADII)
    an_fund = harmonics_1sided(Br_an)[n_p]

    mu_r_fe_values = [100, 1000, 10000]
    errors = []
    for mu_r_fe in mu_r_fe_values:
        _, Br_fem = solve_field_fem(
            geo=_GEO, n_p=n_p, B_rem=B_rem,
            mu_r_pm=1.05, mu_r_fe=mu_r_fe,
            n_theta=n_theta, maxh_fraction=maxh,
        )
        fem_fund = harmonics_1sided(Br_fem)[n_p]
        errors.append(abs(fem_fund - an_fund) / an_fund)

    # Errors must decrease monotonically
    for i in range(len(errors) - 1):
        assert errors[i] > errors[i+1], (
            f"Error at mu_r_fe={mu_r_fe_values[i]} ({errors[i]:.6f}) "
            f"should exceed error at mu_r_fe={mu_r_fe_values[i+1]} "
            f"({errors[i+1]:.6f})"
        )

    # At mu_r_fe=10000, error should be well under the 2% tolerance
    assert errors[-1] < 0.005, (
        f"Error at mu_r_fe=10000 = {errors[-1]:.6f}, expected < 0.005"
    )


def test_outrunner_inner_bc_effect():
    """Outrunner FEM matches analytical within 2% for both small and large bore.

    The FEM applies Neumann BC at the inner boundary (hollow center),
    while the analytical model assumes Dirichlet (infinite iron). For
    practical geometries the effect should be small.
    """
    n_p = 4
    B_rem = 1.2
    mu_r_pm = 1.05
    n_theta = 360

    geo_small = outrunner(r_outer=0.10, r_rotor=0.08, r_magnet=0.06,
                          r_stator=0.04, r_inner=0.005)
    geo_large = outrunner(r_outer=0.10, r_rotor=0.08, r_magnet=0.06,
                          r_stator=0.04, r_inner=0.025)

    errors = []
    for geo in [geo_small, geo_large]:
        theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
        Br_an = zhu_howe_Br(
            theta, n_p, B_rem, r_eval=geo.r_ag,
            r_stator=geo.r_stator, r_magnet=geo.r_magnet,
            r_rotor=geo.r_rotor, mu_r_pm=mu_r_pm,
        )
        an_fund = harmonics_1sided(Br_an)[n_p]

        _, Br_fem = solve_field_fem(
            geo=geo, n_p=n_p, B_rem=B_rem,
            mu_r_pm=mu_r_pm, mu_r_fe=1000.0,
            n_theta=n_theta, maxh_fraction=0.03,
        )
        fem_fund = harmonics_1sided(Br_fem)[n_p]
        errors.append(abs(fem_fund - an_fund) / an_fund)

    # Both within 2%
    for i, (geo, err) in enumerate(zip([geo_small, geo_large], errors)):
        assert err < 0.02, (
            f"{'Small' if i == 0 else 'Large'} bore: FEM-analytical "
            f"error {err:.4f} exceeds 2%"
        )


def test_picard_relax_recovery_and_raw_criterion():
    """Tier-1 Picard guards: relax recovers after divergence damping, and
    the stop criterion is honored on the raw (unrelaxed) step.

    Saturating thin-yoke geometry with deliberately high picard_relax=0.5
    (diverges without damping). Pre-Tier-1, damping ratcheted relax to its
    floor (0.02-0.025) with no recovery, and the relax-scaled criterion
    accepted raw residuals ~0.4.
    """
    geo = inrunner(r_outer=0.038, r_stator=0.035, r_magnet=0.030,
                   r_rotor=0.020)
    info: dict[str, float] = {}
    solve_field_fem(
        geo=geo, n_p=2, B_rem=1.5, mu_r_pm=1.05, mu_r_fe=1000.0,
        n_theta=180, maxh_fraction=0.03,
        nonlinear=True, max_picard=60, picard_tol=0.005, picard_relax=0.5,
        info=info,
    )
    assert info["picard_residual"] < 0.005, (
        f"Stop criterion not honored on raw step: "
        f"residual {info['picard_residual']:.4e}"
    )
    # Damping re-engages repeatedly on this deeply saturated case, so full
    # recovery to 0.5 is not expected — observed ~0.075. Pre-Tier-1 pinning
    # left relax at 0.02-0.025 with no growth path.
    assert info["picard_relax_final"] >= 0.05, (
        f"relax pinned near floor ({info['picard_relax_final']:.4f}); "
        f"recovery after divergence damping is broken"
    )


@pytest.mark.slow
def test_picard_relaxation_independence():
    """Converged nonlinear solution is independent of relaxation parameter.

    Uses a purpose-built thin-yoke geometry (3mm stator yoke, n_p=2) to
    guarantee iron saturation at B_rem=1.5T. Asserts peak |B_iron| > 1.2T
    as a precondition so the test fails loudly if saturation isn't reached.
    """
    geo = inrunner(r_outer=0.038, r_stator=0.035, r_magnet=0.030,
                   r_rotor=0.020)
    n_p = 2
    B_rem = 1.5
    common = dict(
        geo=geo, n_p=n_p, B_rem=B_rem,
        mu_r_pm=1.05, mu_r_fe=1000.0,
        n_theta=180, maxh_fraction=0.03,
        nonlinear=True, max_picard=60, picard_tol=0.005,
        return_full=True,
    )

    theta_a, B_r_a, mesh_a, gfu_a = solve_field_fem(**common, picard_relax=0.10)
    theta_b, B_r_b, mesh_b, gfu_b = solve_field_fem(**common, picard_relax=0.25)
    # relax=0.35 and 0.50 previously diverged; adaptive backtracking fixes this
    theta_c, B_r_c, mesh_c, gfu_c = solve_field_fem(**common, picard_relax=0.35)
    theta_d, B_r_d, mesh_d, gfu_d = solve_field_fem(**common, picard_relax=0.50)

    # Precondition: iron must be saturated for this test to be meaningful
    B_iron_a = _peak_B_iron(mesh_a, gfu_a)
    assert B_iron_a > 1.2, (
        f"Test requires saturated iron but peak |B| = {B_iron_a:.2f} T; "
        f"adjust geometry or B_rem"
    )

    fund_a = harmonics_1sided(B_r_a)[n_p]
    fund_b = harmonics_1sided(B_r_b)[n_p]
    fund_c = harmonics_1sided(B_r_c)[n_p]
    fund_d = harmonics_1sided(B_r_d)[n_p]
    for label, fund in [("0.25", fund_b), ("0.35", fund_c), ("0.50", fund_d)]:
        assert fund_a == pytest.approx(fund, rel=0.01), (
            f"Relaxation 0.10 fundamental {fund_a:.6f} vs "
            f"relaxation {label} fundamental {fund:.6f}"
        )


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
# Discrete magnet arcs tests
# ---------------------------------------------------------------------------

class TestDiscreteArcs:
    """Tests for FEM with discrete magnet arcs."""

    @pytest.mark.timeout(60)
    def test_alpha_p_1_invariance(self):
        """alpha_p=1.0 should give same B_r as no alpha_p."""
        geo = default_inrunner()
        _, B_r_base = solve_field_fem(
            geo, n_p=4, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=5000,
            **FAST)
        _, B_r_full = solve_field_fem(
            geo, n_p=4, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=5000,
            alpha_p=1.0, **FAST)
        np.testing.assert_allclose(B_r_base, B_r_full, atol=1e-12)

    @pytest.mark.timeout(60)
    def test_alpha_p_half_reduces_fundamental(self):
        """alpha_p=0.5 should significantly reduce B₁."""
        geo = default_inrunner()
        _, B_r_full = solve_field_fem(
            geo, n_p=4, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=5000,
            **FAST)
        _, B_r_half = solve_field_fem(
            geo, n_p=4, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=5000,
            alpha_p=0.5, **FAST)
        B1_full = harmonics_1sided(B_r_full)[4]
        B1_half = harmonics_1sided(B_r_half)[4]
        assert B1_half < B1_full
        assert B1_half > 0.5 * B1_full  # not more than 50% drop

    @pytest.mark.timeout(60)
    def test_outrunner_arcs_meshable(self):
        """Outrunner with discrete arcs should solve without error."""
        geo = outrunner(r_outer=0.050, r_rotor=0.045, r_magnet=0.040,
                        r_stator=0.035, r_inner=0.015)
        theta, B_r = solve_field_fem(
            geo, n_p=4, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=5000,
            alpha_p=0.8, **FAST)
        assert len(B_r) == FAST["n_theta"]
        assert np.max(np.abs(B_r)) > 0

    @pytest.mark.timeout(120)
    def test_near_full_pitch(self):
        """alpha_p=0.95 should solve and give result close to full-pitch."""
        geo = inrunner(r_outer=0.050, r_stator=0.030, r_magnet=0.025,
                       r_rotor=0.015)
        _, B_r_full = solve_field_fem(
            geo, n_p=4, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=5000, **FAST)
        _, B_r_95 = solve_field_fem(
            geo, n_p=4, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=5000,
            alpha_p=0.95, **FAST)
        B1_full = harmonics_1sided(B_r_full)[4]
        B1_95 = harmonics_1sided(B_r_95)[4]
        assert B1_95 > 0
        assert B1_95 / B1_full > 0.95  # near full pitch → small reduction

    @pytest.mark.timeout(120)
    def test_thin_arcs(self):
        """alpha_p=0.3 (thin arcs) should solve."""
        geo = inrunner(r_outer=0.050, r_stator=0.030, r_magnet=0.025,
                       r_rotor=0.015)
        theta, B_r = solve_field_fem(
            geo, n_p=4, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=5000,
            alpha_p=0.3, **FAST)
        B1 = harmonics_1sided(B_r)[4]
        assert B1 > 0

    @pytest.mark.timeout(120)
    def test_slotted_plus_arcs(self):
        """Both slots and discrete arcs together."""
        geo = inrunner(r_outer=0.050, r_stator=0.030, r_magnet=0.025,
                       r_rotor=0.015, n_slots=6, slot_depth=0.005,
                       slot_width_ratio=0.5)
        theta, B_r = solve_field_fem(
            geo, n_p=4, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=5000,
            n_slots=6, slot_width_ratio=0.5, alpha_p=0.75,
            **FAST)
        assert np.max(np.abs(B_r)) > 0

    @pytest.mark.timeout(120)
    def test_sub_threshold_gap_collapses_with_warning(self, caplog):
        """Interpole gap < 0.5 mm: arcs collapse to alpha_p=1.0, loudly.

        The collapse itself is an OCC limitation and stays; this pins
        the warning, the info
        flag, and that the solve equals the alpha_p=1.0 solve.
        """
        import logging as _logging

        # gap at inner radius (fem v7) = (1-0.93)·π/7·0.010 = 0.314 mm
        # < 0.5 mm threshold (was 0.471 mm at the outer radius r_magnet)
        geo = inrunner(r_outer=0.030, r_stator=0.017, r_magnet=0.015,
                       r_rotor=0.010)
        info: dict[str, float] = {}
        with caplog.at_level(_logging.WARNING):
            _, B_r_sub = solve_field_fem(
                geo, n_p=7, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=5000,
                alpha_p=0.93, info=info, **FAST)
        assert info.get("arcs_collapsed") == 1.0
        assert any("arcs" in r.message and "alpha_p=1.0" in r.message
                   for r in caplog.records)
        _, B_r_full = solve_field_fem(
            geo, n_p=7, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=5000,
            alpha_p=1.0, **FAST)
        np.testing.assert_allclose(B_r_sub, B_r_full, atol=1e-10)

    @pytest.mark.timeout(120)
    def test_supra_threshold_gap_no_collapse_flag(self, caplog):
        """A comfortably wide gap must not warn or set the flag."""
        import logging as _logging

        geo = default_inrunner()
        info: dict[str, float] = {}
        with caplog.at_level(_logging.WARNING):
            solve_field_fem(
                geo, n_p=4, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=5000,
                alpha_p=0.75, info=info, **FAST)
        assert "arcs_collapsed" not in info
        assert not any("arcs" in r.message and "collapsed" in r.message
                       for r in caplog.records)

    @pytest.mark.timeout(120)
    def test_gap_band_honors_per_face_refinement(self):
        """0.5–1.0 mm gaps honor requested maxh ≤ gap/2 via per-face maxh.

        The global maxh floor (0.5 mm) used to mesh
        these gaps at the same requested resolution as the whole domain,
        violating the maxh ≤ gap × 0.5 rule. Pins the mean
        element density in the pm_gap regions against the equilateral-
        triangle area at the requested maxh (netgen honors maxh only
        approximately, so worst-edge metrics are too noisy to pin).
        """
        from ngsolve import VOL

        # gap at inner radius (fem v7) = (1-0.953)·π/4·0.015 = 0.554 mm —
        # inside the floored band [0.5, 1.0] mm
        alpha_p = 0.953
        geo = inrunner(r_outer=0.050, r_stator=0.030, r_magnet=0.025,
                       r_rotor=0.015)
        gap = (1 - alpha_p) * np.pi / 4 * geo.r_rotor
        _, _, mesh, _ = solve_field_fem(
            geo, n_p=4, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=5000,
            alpha_p=alpha_p, return_full=True, **FAST)
        n_gap_els = sum(1 for el in mesh.Elements(VOL)
                        if el.mat == "pm_gap")
        assert n_gap_els > 0
        gap_area = (1 - alpha_p) * np.pi * (geo.r_magnet**2 - geo.r_rotor**2)
        eq_area = np.sqrt(3) / 4 * (gap * 0.5) ** 2
        # measured ~1.4 with the per-face maxh, ~3.4 under the old
        # global floor
        assert gap_area / n_gap_els <= 2.5 * eq_area


# ---------------------------------------------------------------------------
# Carter factor tests
# ---------------------------------------------------------------------------

class TestCarterFactor:

    def test_no_slots(self):
        assert carter_factor(0.030, 0, 0.6, 0.001) == 1.0

    def test_increases_with_slot_opening(self):
        k1 = carter_factor(0.030, 12, 0.3, 0.005)
        k2 = carter_factor(0.030, 12, 0.6, 0.005)
        k3 = carter_factor(0.030, 12, 0.9, 0.005)
        assert 1.0 < k1 < k2 < k3

    def test_approaches_unity_large_g_prime(self):
        """Very thick magnets (large g') → slot openings negligible → k_c → 1."""
        k = carter_factor(0.030, 12, 0.6, 0.100)  # 100mm g'
        assert k == pytest.approx(1.0, abs=0.01)

    def test_known_value(self):
        """Verify against hand-calculated value for CREATOR-like geometry."""
        # CREATOR: r_stator=0.0239, n_slots=6, swr≈0.6, g'≈4.4mm
        k = carter_factor(0.0239, 6, 0.6, 0.0044)
        assert 1.2 < k < 1.5

class TestCarterAdjustedRadii:

    def test_smooth_bore_identity(self):
        geo = default_inrunner()
        r_s, r_m, k_c = carter_adjusted_radii(geo, 1.05)
        assert (r_s, r_m, k_c) == (geo.r_stator, geo.r_magnet, 1.0)

    def test_inrunner_widens_gap_keeps_magnet(self):
        geo = inrunner(**_SLOTTED_GEO_KW, slot_opening_width=0.055)  # ratio ~0.15
        r_s, r_m, k_c = carter_adjusted_radii(geo, 1.05)
        assert k_c > 1.0
        assert r_s > geo.r_stator
        assert r_m == geo.r_magnet

    def test_outrunner_widens_gap_keeps_magnet(self):
        geo = outrunner(**_SLOTTED_OUT_GEO_KW, slot_opening_width=0.0063)  # ratio ~0.15
        r_s, r_m, k_c = carter_adjusted_radii(geo, 1.05)
        assert k_c > 1.0
        assert r_s < geo.r_stator
        assert r_m == geo.r_magnet


# ---------------------------------------------------------------------------
# Maxwell-stress torque + j_s mapping closure
# ---------------------------------------------------------------------------
# First FEM torque output: tau = (L_stk r^2/mu0) * closed-circle integral of
# B_r*B_theta from the solved A_z, on a circle in the source-free air gap.
# Contour independence there is the correctness check. The j_s -> phase
# current mapping (solver_params.j_s_from_phase_current) closes the
# loop: the order-n_p interaction torque between magnet-only and
# armature-only solves reproduces the winding formula 3*k_w*N_eff*I*B1*r*L
# (equivalently 1.5*n_p*psi_f*i_q); the CREATOR anchor version lives in
# test_creator_model.py.

_TORQUE_FAST = dict(n_p=2, B_rem=1.0, mu_r_pm=1.05, mu_r_fe=1000.0,
                    k_w=0.966, n_theta=60, maxh_fraction=0.08,
                    return_full=True)


def _torque_geo():
    return inrunner(**_SLOTTED_GEO_KW)


def _gap_radii(geo, fractions=(0.25, 0.5, 0.75)):
    g = geo.r_stator - geo.r_magnet
    return [geo.r_magnet + f * g for f in fractions]


def test_maxwell_torque_contour_independent_and_above_noise():
    """The stress integral is contour-independent in the source-free gap
    (< 5% spread across three radii at this mesh), and the zero-current
    solve's residual torque (cogging + discretization noise) sits well
    below the interaction torque."""
    from phasesweep.fem_field import maxwell_stress_torque
    geo = _torque_geo()
    radii = _gap_radii(geo)
    _, _, mesh, gfu = solve_field_fem(geo=geo, j_s=1e5, **_TORQUE_FAST)
    taus = [maxwell_stress_torque(mesh, gfu, r) for r in radii]
    mean = sum(taus) / len(taus)
    assert (max(taus) - min(taus)) / abs(mean) < 0.05
    _, _, mesh0, gfu0 = solve_field_fem(geo=geo, j_s=0.0, **_TORQUE_FAST)
    tau0 = maxwell_stress_torque(mesh0, gfu0, radii[1])
    assert abs(tau0) < 0.05 * abs(mean)


def test_maxwell_torque_affine_in_j_s():
    """Linear solve -> torque is affine in j_s: doubling the sheet doubles
    the interaction torque; flipping its sign flips it."""
    from phasesweep.fem_field import maxwell_stress_torque
    geo = _torque_geo()
    r = _gap_radii(geo)[1]
    taus = {}
    for j in (0.0, 1e5, 2e5, -1e5):
        _, _, mesh, gfu = solve_field_fem(geo=geo, j_s=j, **_TORQUE_FAST)
        taus[j] = maxwell_stress_torque(mesh, gfu, r)
    base = taus[1e5] - taus[0.0]
    assert (taus[2e5] - taus[0.0]) / base == pytest.approx(2.0, rel=1e-4)
    assert (taus[-1e5] - taus[0.0]) / base == pytest.approx(-1.0, rel=1e-4)


def test_j_s_without_slot_faces_raises():
    """n_slots > 0 with slot_depth = 0 builds no slot faces; j_s must not
    be silently ignored (this passed quietly before)."""
    with pytest.raises(ValueError, match="no slot regions"):
        solve_field_fem(geo=_GEO, n_slots=12, j_s=1e5, **_TORQUE_FAST)


def test_maxwell_interaction_closes_to_winding_formula():
    """The order-n_p interaction torque between magnet-only and
    armature-only solves matches tau = pi*r_bore^2*K_eff*B1 with
    K_eff = j_s*k_w*S/r_bore from the slot source moment — the identity
    behind j_s_from_phase_current. The FEM lands a few percent BELOW the
    ideal-sheet value (slot openings Carter-widen the armature gap;
    finite mu_r_fe), so the band is one-sided.

    The full static maxwell_stress_torque is NOT the comparison target:
    it adds position-locked ripple from armature comb sidebands
    (m*n_slots ± n_p) meeting magnet space harmonics of the same order.
    """
    from phasesweep.fem_field import (
        maxwell_interaction_torque_order,
        slot_source_moment,
    )
    geo = _torque_geo()
    kw = dict(geo=geo, n_p=2, mu_r_pm=1.05, mu_r_fe=1000.0,
              k_w=0.966, n_theta=360, maxh_fraction=0.05, return_full=True)
    j_s = 1e5
    _, Br_mag, mesh, gfu_mag = solve_field_fem(j_s=0.0, B_rem=1.0, **kw)
    _, _, _, gfu_arm = solve_field_fem(j_s=j_s, B_rem=1e-30, **kw)
    B1 = float(harmonics_1sided(Br_mag)[2])
    K_eff = j_s * 0.966 * slot_source_moment(geo) / geo.r_stator
    r_mid = _gap_radii(geo)[1]
    tau_fem = maxwell_interaction_torque_order(mesh, gfu_mag, gfu_arm,
                                               r_mid, order=2)
    tau_winding = np.pi * geo.r_stator**2 * K_eff * B1
    assert 0.90 < tau_fem / tau_winding < 1.02, (
        f"order-2 interaction {tau_fem:.1f} vs winding formula "
        f"{tau_winding:.1f} (ratio {tau_fem/tau_winding:.3f})"
    )


def test_fem_runner_reports_maxwell_torque():
    """_run_fem_impl emits tau_maxwell_per_m (and tau_maxwell with L_stk)
    when j_s != 0; no torque keys on open-circuit runs."""
    from phasesweep.fem_runner import _run_fem_impl
    from phasesweep.motor import Motor
    from phasesweep.sweep_types import RunConfig
    motor = Motor(name="torque-test", geometry=_torque_geo(), n_p=2,
                  B_rem=1.0, k_w=0.966, L_stk=0.10)
    rc = RunConfig(motor=motor, model="fem", j_s=1e5,
                   n_theta=60, maxh_fraction=0.08)
    m = _run_fem_impl(rc)
    assert m["tau_maxwell"] == pytest.approx(
        0.10 * m["tau_maxwell_per_m"], rel=1e-12)
    assert m["tau_maxwell_per_m"] > 0.0  # motoring sign for the q-axis sheet
    rc0 = RunConfig(motor=motor, model="fem",
                    n_theta=60, maxh_fraction=0.08)
    m0 = _run_fem_impl(rc0)
    assert "tau_maxwell_per_m" not in m0
    assert "tau_maxwell" not in m0


# ---------------------------------------------------------------------------
# Magnet-pattern rotation — cogging-sweep support
# ---------------------------------------------------------------------------

class TestMagnetRotation:
    """rotation rotates arcs + magnetization sign; the stator stays fixed."""

    @pytest.mark.timeout(60)
    def test_rotation_zero_identity(self):
        """Explicit rotation=0.0 is bit-identical to the default."""
        geo = default_inrunner()
        _, B_r_base = solve_field_fem(
            geo, n_p=4, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=5000,
            alpha_p=0.5, **FAST)
        _, B_r_rot0 = solve_field_fem(
            geo, n_p=4, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=5000,
            alpha_p=0.5, rotation=0.0, **FAST)
        np.testing.assert_allclose(B_r_base, B_r_rot0, atol=1e-12)

    @pytest.mark.timeout(120)
    def test_rotation_shifts_pattern_preserves_amplitude(self):
        """Rotating the magnet pattern by phi shifts every magnet space
        harmonic's phase by -nu*phi and leaves amplitudes unchanged (up to
        mesh scatter — the rotated arcs build a different mesh)."""
        geo = default_inrunner()
        phi = np.deg2rad(10.0)
        kw = dict(n_p=4, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=5000,
                  alpha_p=0.5, n_theta=360, maxh_fraction=0.08)
        _, Br0 = solve_field_fem(geo, **kw)
        _, Br1 = solve_field_fem(geo, rotation=float(phi), **kw)
        c0 = np.fft.rfft(Br0)
        c1 = np.fft.rfft(Br1)
        for nu in (4, 12):  # fundamental + 3rd magnet harmonic
            assert abs(c1[nu]) == pytest.approx(abs(c0[nu]), rel=0.02)
            phase_err = np.angle(
                (c1[nu] / c0[nu]) * np.exp(1j * nu * phi))
            assert abs(phase_err) < 0.02, (
                f"order {nu}: phase shift off by {phase_err:.4f} rad"
            )


# ---------------------------------------------------------------------------
# Current-sheet phase — d-axis MMF for the demag screen
# ---------------------------------------------------------------------------
# J_z = -j_s*k_w*cos(n_p*theta - sheet_phase): beta = 0 is the q-axis sheet
# (the original behavior), beta = +/-pi/2 a pure d-axis MMF.

class TestSheetPhase:

    def test_sheet_phase_zero_identity(self):
        """Explicit sheet_phase=0.0 is bit-identical to the default."""
        geo = _torque_geo()
        _, base, _, _ = solve_field_fem(geo=geo, j_s=1e5, **_TORQUE_FAST)
        _, zero, _, _ = solve_field_fem(geo=geo, j_s=1e5, sheet_phase=0.0,
                                        **_TORQUE_FAST)
        np.testing.assert_allclose(zero, base, atol=1e-12)

    def test_d_axis_sheet_no_interaction_torque(self):
        """At sheet_phase = +/-pi/2 the order-n_p interaction torque with
        the magnet field vanishes (measured ~1e-4 of the q-axis torque —
        pure d-axis MMF; the sign/axis invariant behind the demag screen)."""
        from phasesweep.fem_field import maxwell_interaction_torque_order
        geo = _torque_geo()
        kw = dict(geo=geo, n_p=2, mu_r_pm=1.05, mu_r_fe=1000.0, k_w=0.966,
                  n_theta=60, maxh_fraction=0.08, return_full=True)
        _, _, mesh, gfu_mag = solve_field_fem(j_s=0.0, B_rem=1.0, **kw)
        _, _, _, gfu_q = solve_field_fem(j_s=1e5, B_rem=1e-30, **kw)
        r = _gap_radii(geo)[1]
        tau_q = maxwell_interaction_torque_order(mesh, gfu_mag, gfu_q, r,
                                                 order=2)
        for beta in (np.pi / 2, -np.pi / 2):
            _, _, _, gfu_d = solve_field_fem(j_s=1e5, B_rem=1e-30,
                                             sheet_phase=beta, **kw)
            tau_d = maxwell_interaction_torque_order(mesh, gfu_mag, gfu_d,
                                                     r, order=2)
            assert abs(tau_d) < 0.01 * abs(tau_q), (
                f"beta={beta:+.3f}: tau_d {tau_d:.2f} vs tau_q {tau_q:.1f}"
            )

    def test_d_axis_sign_and_superposition(self):
        """sheet_phase = +pi/2 demagnetizes (gap B1 down), -pi/2 aids;
        the two shifts are symmetric (exact superposition in the linear
        solve). Pins the demag sign convention documented in the solver."""
        geo = _torque_geo()
        kw = dict(geo=geo, **_TORQUE_FAST)
        b1 = {}
        for beta in (None, np.pi / 2, -np.pi / 2):
            j = 0.0 if beta is None else 1e5
            _, Br, _, _ = solve_field_fem(
                j_s=j, sheet_phase=(beta or 0.0), **kw)
            b1[beta] = float(harmonics_1sided(Br)[2])
        drop = b1[None] - b1[np.pi / 2]
        gain = b1[-np.pi / 2] - b1[None]
        assert drop > 1e-3 * b1[None]  # demag shift well above noise
        assert gain == pytest.approx(drop, rel=1e-3)


# ---------------------------------------------------------------------------
# Magnet operating-point sampler — demag screen step 2
# ---------------------------------------------------------------------------
# B_m = sign(M_r)*B_r on a cell-centered polar grid inside the magnet arcs;
# demag_margin is the statistics layer (min, margin vs B_knee, area fraction
# below knee). The knee schema + registry model are step 3.

_BM_FAST = dict(n_p=2, B_rem=1.0, mu_r_pm=1.05, mu_r_fe=1000.0,
                n_theta=60, maxh_fraction=0.08, alpha_p=0.5,
                return_full=True)


class TestMagnetBmSampler:

    def test_open_circuit_positive_and_weights(self):
        """Open-circuit operating point is positive throughout the magnet
        on both topologies; area weights sum to 1."""
        from phasesweep.fem_field import sample_magnet_Bm
        geo = _torque_geo()
        _, _, mesh, gfu = solve_field_fem(geo=geo, **_BM_FAST)
        _, _, B_m, w = sample_magnet_Bm(mesh, gfu, geo, n_p=2, alpha_p=0.5)
        assert np.all(B_m > 0)
        assert np.sum(w) == pytest.approx(1.0, abs=1e-12)
        geo_out = outrunner(**_SLOTTED_OUT_GEO_KW)
        _, _, mesh_o, gfu_o = solve_field_fem(
            geo=geo_out, n_p=2, B_rem=1.0, mu_r_pm=1.05, mu_r_fe=1000.0,
            n_theta=60, maxh_fraction=0.08, return_full=True)
        _, _, B_m_o, _ = sample_magnet_Bm(mesh_o, gfu_o, geo_out, n_p=2)
        assert np.all(B_m_o > 0)

    def test_demag_sheet_lowers_operating_point(self):
        """At fault-level current the demag sheet (+pi/2) lowers both the
        worst-point and mean operating point; the magnetizing sheet
        raises them; the mean shifts are exactly symmetric (linear
        superposition, sheet CFs at +/-pi/2 are exact negations)."""
        from phasesweep.fem_field import sample_magnet_Bm
        geo = _torque_geo()
        stats = {}
        for tag, j, beta in [("open", 0.0, 0.0), ("demag", 1e6, np.pi / 2),
                             ("magn", 1e6, -np.pi / 2)]:
            _, _, mesh, gfu = solve_field_fem(
                geo=geo, j_s=j, sheet_phase=beta, **_BM_FAST)
            _, _, B_m, _ = sample_magnet_Bm(mesh, gfu, geo, n_p=2,
                                            alpha_p=0.5)
            stats[tag] = (float(B_m.min()), float(B_m.mean()))
        assert stats["demag"][0] < stats["open"][0] < stats["magn"][0]
        assert stats["demag"][1] < stats["open"][1] < stats["magn"][1]
        drop = stats["open"][1] - stats["demag"][1]
        gain = stats["magn"][1] - stats["open"][1]
        assert gain == pytest.approx(drop, rel=1e-6)

    def test_rotation_equivalence(self):
        """Sampling a rotated solve with the matching rotation reproduces
        the unrotated B_m distribution (rotationally equivalent problem;
        rotated arcs build a different mesh -> mesh-scatter tolerance)."""
        from phasesweep.fem_field import sample_magnet_Bm
        geo = _torque_geo()
        phi = float(np.deg2rad(7.0))
        _, _, m0, g0 = solve_field_fem(geo=geo, **_BM_FAST)
        _, _, B0, _ = sample_magnet_Bm(m0, g0, geo, n_p=2, alpha_p=0.5)
        _, _, m1, g1 = solve_field_fem(geo=geo, rotation=phi, **_BM_FAST)
        _, _, B1, _ = sample_magnet_Bm(m1, g1, geo, n_p=2, alpha_p=0.5,
                                       rotation=phi)
        assert B1.min() == pytest.approx(B0.min(), rel=0.02)
        assert B1.mean() == pytest.approx(B0.mean(), rel=0.02)

    def test_demag_margin_dict(self):
        """margin = B_m_min - B_knee; frac_below_knee hits 0/1 for a knee
        below the min / above the max; worst point lies in the magnet
        annulus."""
        from phasesweep.fem_field import demag_margin, sample_magnet_Bm
        geo = _torque_geo()
        _, _, mesh, gfu = solve_field_fem(
            geo=geo, j_s=1e6, sheet_phase=np.pi / 2, **_BM_FAST)
        _, _, B_m, _ = sample_magnet_Bm(mesh, gfu, geo, n_p=2, alpha_p=0.5)
        lo = demag_margin(mesh, gfu, geo, n_p=2, B_knee=float(B_m.min()) - 0.1,
                          alpha_p=0.5)
        hi = demag_margin(mesh, gfu, geo, n_p=2, B_knee=float(B_m.max()) + 0.1,
                          alpha_p=0.5)
        assert lo["B_m_min"] == pytest.approx(float(B_m.min()))
        assert lo["margin"] == pytest.approx(0.1)
        assert lo["frac_below_knee"] == 0.0
        assert hi["frac_below_knee"] == pytest.approx(1.0)
        assert geo.r_rotor < lo["r_min"] < geo.r_magnet
