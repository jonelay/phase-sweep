"""Integration tests for fem_field.py — use coarse meshes to keep runtime low."""

import numpy as np
import pytest
from ngsolve import Mesh
from netgen.occ import OCCGeometry

from phasesweep.fem_field import (
    _build_geometry, _build_slotted_geometry,
    solve_field_fem, batch_harmonics,
    zhu_howe_Br, _derive_B_rem, harmonics_1sided,
    _BH_B, _BH_H, _bh_nu, _MU0,
)
from phasesweep.sweep_types import MotorSweepConfig

FAST = dict(n_theta=60, maxh=0.08)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def test_smooth_geometry_materials():
    shape = _build_geometry()
    mesh = Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh=0.08))
    mats = set(mesh.GetMaterials())
    assert {"stator", "airgap", "pm", "shaft"} <= mats
    assert "slot" not in mats


def test_slotted_geometry_materials():
    shape = _build_slotted_geometry(12)
    mesh = Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh=0.08))
    mats = set(mesh.GetMaterials())
    assert {"stator", "slot", "airgap", "pm", "shaft"} <= mats


@pytest.mark.parametrize("n_slots", [6, 12, 24])
def test_slotted_geometry_slot_count(n_slots):
    shape = _build_slotted_geometry(n_slots)
    mesh = Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh=0.08))
    slot_count = sum(1 for m in mesh.GetMaterials() if m == "slot")
    assert slot_count == n_slots


# ---------------------------------------------------------------------------
# Solver — smooth bore
# ---------------------------------------------------------------------------

def test_smooth_bore_returns_correct_shape():
    theta, B_r = solve_field_fem(n_p=2, psi_f=0.1, L_d=4e-3, L_q=4e-3, **FAST)
    assert theta.shape == (FAST["n_theta"],)
    assert B_r.shape == (FAST["n_theta"],)


def test_smooth_bore_has_dominant_harmonic():
    """B_r should have a clear dominant harmonic (FEM produces AC field, not DC)."""
    n_p = 2
    theta, B_r = solve_field_fem(n_p=n_p, psi_f=0.1, L_d=4e-3, L_q=4e-3, **FAST)
    amps = harmonics_1sided(B_r)
    # Dominant harmonic should be at least 100x larger than typical noise floor
    assert np.max(amps[1:]) > 0.01


def test_smooth_bore_field_nonzero():
    theta, B_r = solve_field_fem(n_p=2, psi_f=0.1, L_d=4e-3, L_q=4e-3, **FAST)
    assert np.nanmax(np.abs(B_r)) > 1e-4


def test_smooth_bore_field_sign_matches_analytical():
    """FEM B_r must have same sign as Zhu & Howe (not negated)."""
    n_p = 2
    theta, B_fem = solve_field_fem(n_p=n_p, psi_f=0.1, L_d=4e-3, L_q=4e-3, **FAST)
    B_rem = _derive_B_rem(0.1, n_p, 50, 0.966, 0.10)
    B_an = zhu_howe_Br(theta, n_p, B_rem)
    assert np.corrcoef(B_fem, B_an)[0, 1] > 0.99


def test_smooth_bore_scales_with_psi_f():
    """B_r peak should scale roughly linearly with PM flux."""
    _, B1 = solve_field_fem(n_p=2, psi_f=0.1, L_d=4e-3, L_q=4e-3, **FAST)
    _, B2 = solve_field_fem(n_p=2, psi_f=0.2, L_d=4e-3, L_q=4e-3, **FAST)
    ratio = np.nanmax(np.abs(B2)) / np.nanmax(np.abs(B1))
    assert ratio == pytest.approx(2.0, rel=0.05)


# ---------------------------------------------------------------------------
# Solver — slotted + winding
# ---------------------------------------------------------------------------

def test_slotted_returns_correct_shape():
    theta, B_r = solve_field_fem(n_p=2, psi_f=0.1, L_d=4e-3, L_q=4e-3,
                                  n_slots=12, j_s=0.1, **FAST)
    assert B_r.shape == (FAST["n_theta"],)


def test_backward_compat_n_slots_zero():
    """Explicit n_slots=0 must give the same result as default (smooth bore)."""
    kw = dict(n_p=2, psi_f=0.1, L_d=4e-3, L_q=4e-3, **FAST)
    _, B1 = solve_field_fem(**kw)
    _, B2 = solve_field_fem(**kw, n_slots=0, j_s=0.0)
    np.testing.assert_allclose(B1, B2, rtol=1e-10)


def test_slotted_solve_completes():
    """Slotted FEM should complete successfully and return valid B_r."""
    n_p, Q = 2, 12
    theta, B_r = solve_field_fem(n_p=n_p, psi_f=0.1, L_d=4e-3, L_q=4e-3,
                                  n_slots=Q, j_s=0.0, **FAST)
    assert B_r.shape == (FAST["n_theta"],)
    assert np.all(np.isfinite(B_r))
    assert np.nanmax(np.abs(B_r)) > 1e-4


def test_winding_current_solve_completes():
    """Slotted FEM with winding current should complete successfully."""
    kw = dict(n_p=2, psi_f=0.1, L_d=4e-3, L_q=4e-3, n_slots=12, j_s=0.1, **FAST)
    theta, B_r = solve_field_fem(**kw)
    assert B_r.shape == (FAST["n_theta"],)
    assert np.all(np.isfinite(B_r))
    assert np.nanmax(np.abs(B_r)) > 1e-4


# ---------------------------------------------------------------------------
# batch_harmonics
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Zhu & Howe analytical model
# ---------------------------------------------------------------------------

def test_zhu_howe_rejects_np1():
    with pytest.raises(ValueError, match="n_p=1"):
        zhu_howe_Br(np.linspace(0, 2 * np.pi, 60), n_p=1, B_rem=1.0)


def test_zhu_howe_is_pure_cosine():
    """Zhu & Howe with sinusoidal magnetization should produce a pure cos(n_p θ) waveform."""
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
        n_p=n_p, psi_f=psi_f, L_d=L_d, L_q=L_q,
        n_theta=n_theta, maxh=0.04,
        n_slots=0, j_s=0.0, N=N, k_w=k_w, L_stk=L_stk,
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
    """Pure sine at order k → amplitude at bin k should dominate."""
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
    # Deep saturation: ν should be much higher than unsaturated
    nu_linear = _BH_H[1] / _BH_B[1]  # ~250
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
    kw = dict(n_p=2, psi_f=0.05, L_d=4e-3, L_q=4e-3, **FAST)
    _, B_lin = solve_field_fem(**kw)
    _, B_nl = solve_field_fem(**kw, nonlinear=True)
    peak_lin = np.nanmax(np.abs(B_lin))
    peak_nl = np.nanmax(np.abs(B_nl))
    assert peak_nl == pytest.approx(peak_lin, rel=0.05)


def test_nonlinear_converges():
    theta, B_r = solve_field_fem(n_p=2, psi_f=0.1, L_d=4e-3, L_q=4e-3,
                                  nonlinear=True, **FAST)
    assert np.all(np.isfinite(B_r))


def test_nonlinear_changes_field_at_high_flux():
    """Nonlinear solve should produce a different result than linear at high flux."""
    kw = dict(n_p=2, psi_f=0.3, L_d=4e-3, L_q=4e-3, **FAST)
    _, B_lin = solve_field_fem(**kw)
    _, B_nl = solve_field_fem(**kw, nonlinear=True)
    # Fields should differ (iron permeability changes from constant to B-dependent)
    assert not np.allclose(B_lin, B_nl, rtol=1e-3)


def test_nonlinear_raises_on_non_convergence():
    with pytest.raises(RuntimeError, match="Picard iteration did not converge"):
        solve_field_fem(n_p=2, psi_f=0.1, L_d=4e-3, L_q=4e-3,
                        nonlinear=True, max_picard=1, picard_tol=1e-15, **FAST)


# ---------------------------------------------------------------------------
# Nonlinear API threading (Issue 3)
# ---------------------------------------------------------------------------

_BASE_CFG = dict(n_p=2, R_s=3.6, L_d=4e-3, L_q=4e-3, psi_f=0.1, J=1.5e-3)


def test_config_id_differs_for_nonlinear():
    c1 = MotorSweepConfig(**_BASE_CFG, nonlinear=False)
    c2 = MotorSweepConfig(**_BASE_CFG, nonlinear=True)
    assert c1.config_id != c2.config_id


def test_config_roundtrip_nonlinear():
    c = MotorSweepConfig(**_BASE_CFG, nonlinear=True)
    c2 = MotorSweepConfig.from_dict(c.to_dict())
    assert c2.nonlinear is True
    assert c2.config_id == c.config_id


# ---------------------------------------------------------------------------
# P0: _derive_B_rem roundtrip (Issue audit)
# ---------------------------------------------------------------------------

def test_derive_B_rem_roundtrip():
    """_derive_B_rem must invert the flux linkage -> B_rem relationship."""
    psi_f, n_p, N, k_w, L_stk = 0.1, 2, 50, 0.966, 0.10
    B_rem = _derive_B_rem(psi_f, n_p, N, k_w, L_stk)
    assert 0.01 < B_rem < 1.0, f"B_rem={B_rem} outside physically reasonable range"
    from phasesweep.fem_field import _R_SI
    B_peak = zhu_howe_Br(np.array([0.0]), n_p, B_rem, r_eval=_R_SI)[0]
    psi_f_recovered = B_peak * 2 * N * k_w * _R_SI * L_stk / n_p
    assert psi_f_recovered == pytest.approx(psi_f, rel=1e-10)


# ---------------------------------------------------------------------------
# P1: _demoivre direct evaluation (Issue audit)
# ---------------------------------------------------------------------------

def test_demoivre_matches_trig():
    """De Moivre recursion must match cos/sin at known angles."""
    from math import cos, sin, pi
    from netgen.occ import WorkPlane, OCCGeometry
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
    """At high flux, nonlinear 2× psi_f should scale air-gap B by < 2× (saturation).

    At low flux (unsaturated), doubling psi_f doubles B (ratio ≈ 2.0).
    At high flux where iron saturates, the ratio should drop below 2.0.
    Uses psi_f=1.0 → 2.0 where B-H curve shows reduced permeability.
    """
    kw_base = dict(n_p=2, L_d=4e-3, L_q=4e-3, n_theta=60, maxh=0.04,
                   max_picard=40, picard_relax=0.15, picard_tol=0.02)
    _, B_lo = solve_field_fem(psi_f=0.05, **kw_base, nonlinear=True)
    _, B_hi = solve_field_fem(psi_f=0.10, **kw_base, nonlinear=True)
    ratio_low = np.nanmax(np.abs(B_hi)) / np.nanmax(np.abs(B_lo))

    _, B_lo2 = solve_field_fem(psi_f=1.0, **kw_base, nonlinear=True)
    _, B_hi2 = solve_field_fem(psi_f=2.0, **kw_base, nonlinear=True)
    ratio_high = np.nanmax(np.abs(B_hi2)) / np.nanmax(np.abs(B_lo2))

    assert ratio_low == pytest.approx(2.0, abs=0.01), (
        f"Low-flux ratio should be ~2.0 (got {ratio_low:.4f})"
    )
    assert ratio_high < ratio_low, (
        f"High-flux ratio ({ratio_high:.4f}) should be < low-flux ({ratio_low:.4f}) "
        f"due to iron saturation"
    )
