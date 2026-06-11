"""Integration tests for fem_field.py — use coarse meshes to keep runtime low."""

import numpy as np
import pytest
from netgen.occ import OCCGeometry
from ngsolve import H1, L2, GridFunction, Mesh, grad, sqrt

from phasesweep import fem_field
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
    _derive_B_rem,
    _mesh_cache,
    batch_harmonics,
    carter_adjusted_radii,
    carter_factor,
    clear_mesh_cache,
    end_effect_factor,
    harmonics_1sided,
    set_disk_cache_dir,
    solve_field_fem,
    zhu_howe_Br,
)
from phasesweep.geometry import default_inrunner, inrunner, outrunner

FAST = dict(n_theta=60, maxh_fraction=0.08)
_GEO = default_inrunner()


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


def test_fem_runner_defers_slot_params_to_geometry(monkeypatch):
    """Production FEM path passes no slot kwargs — geometry governs."""
    from phasesweep.fem_runner import _run_fem_impl
    from phasesweep.sweep_types import RunConfig
    from tests.conftest import make_motor

    captured = {}

    def fake_solve(*args, **kw):
        captured.update(kw)
        th = np.linspace(0, 2 * np.pi, 60, endpoint=False)
        return th, np.cos(2 * th)

    monkeypatch.setattr(fem_field, "solve_field_fem", fake_solve)
    motor = make_motor(B_rem=1.2, psi_f=None)
    _run_fem_impl(RunConfig(motor=motor, model="fem"))
    assert "n_slots" not in captured
    assert "slot_width_ratio" not in captured


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
    B_rem = _derive_B_rem(0.1, n_p, 50, 0.966, 0.10)
    B_an = zhu_howe_Br(theta, n_p, B_rem)
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
    theta, B_r = _solve(n_slots=12, j_s=0.1)
    assert B_r.shape == (FAST["n_theta"],)


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
    theta, B_r = _solve(n_slots=12, j_s=0.1)
    assert B_r.shape == (FAST["n_theta"],)
    assert np.all(np.isfinite(B_r))
    assert np.nanmax(np.abs(B_r)) > 1e-4


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


def test_zhu_howe_rejects_np1():
    with pytest.raises(ValueError, match="n_p=1"):
        zhu_howe_Br(np.linspace(0, 2 * np.pi, 60), n_p=1, B_rem=1.0)


def test_zhu_howe_is_pure_cosine():
    theta = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    Br = zhu_howe_Br(theta, n_p=2, B_rem=1.0)
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
    B_rem = _derive_B_rem(psi_f, n_p, N, k_w, L_stk)
    n_theta = 180

    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    Br_an = zhu_howe_Br(theta, n_p, B_rem)
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


def test_compute_thd_nan_on_zero_fundamental():
    from phasesweep.fem_field import compute_thd
    assert np.isnan(compute_thd(np.array([0.0, 0.0, 0.5]), 1))


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
    B_rem_default = _derive_B_rem(psi_f, n_p, N, k_w, L_stk)
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
    B_rem = _derive_B_rem(psi_f, n_p, N, k_w, L_stk)
    assert 0.01 < B_rem < 1.0, f"B_rem={B_rem} outside physically reasonable range"
    r_si = _DEFAULT_GEO.r_stator
    B_peak = zhu_howe_Br(np.array([0.0]), n_p, B_rem, r_eval=r_si)[0]
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

    # Square-wave source (S110) has a lower crest than the old cosine at
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
    Br_an = zhu_howe_Br(theta, n_p, B_rem)
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
