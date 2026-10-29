"""Tests for the multi-plane 2D Cartesian axial-flux FEM (fem_axial).

Cross-validation gates and tolerances come from
tests/crossval/axial_fem_contract.py (shared with Tracks B/C). CV-3 is a
reported diagnostic, not a gate. Registry-level guard tests
(coreless/j_s/inner-plane slot overflow) arrive with the W2 wiring.
"""

from dataclasses import replace
from itertools import pairwise
from math import pi

import numpy as np
import pytest

from phasesweep.machines.geometry import axial_flux
from phasesweep.solvers.analytical_axial import (
    GAP_LINKAGE,
    axial_layer_Bz_series,
    carter_airgap_axial,
    winding_transfer_axial,
)
from phasesweep.solvers.fem_axial import (
    _effective_yokes,
    _plane_radii,
    _slot_segments,
    clear_mesh_cache,
    fundamental_amplitude,
    pm_edge_leakage_factor,
    solve_axial_multiplane,
    solve_plane,
    solve_plane_periodic,
)
from tests.conftest import requires_fem
from tests.crossval.axial_fem_contract import (
    CV0_REL_TOL,
    CV1_ANALYTICAL_REL_TOL,
    CV2_REL_TOL,
    circuit_limit_Bz,
    load_parviainen_motor,
    slotless_reference_B1,
)


@pytest.fixture(scope="module")
def parviainen():
    return load_parviainen_motor()


@pytest.fixture(scope="module")
def geo_slotted(parviainen):
    return parviainen.geometry


@pytest.fixture(scope="module")
def geo_slotless(geo_slotted):
    return replace(geo_slotted, n_slots=0, slot_depth=0.0)


# ---------------------------------------------------------------------------
# Pure-geometry helpers (no FEM)
# ---------------------------------------------------------------------------

class TestPlaneRadii:

    def test_midpoint_rule(self, geo_slotless):
        r, dr = _plane_radii(geo_slotless, 3)
        assert dr == pytest.approx(geo_slotless.active_length / 3)
        assert r[1] == pytest.approx(geo_slotless.r_mean)
        assert r[0] == pytest.approx(geo_slotless.r_in + dr / 2)
        assert r[-1] == pytest.approx(geo_slotless.r_out - dr / 2)

    def test_single_plane_is_r_mean(self, geo_slotless):
        r, dr = _plane_radii(geo_slotless, 1)
        assert dr == pytest.approx(geo_slotless.active_length)
        assert r[0] == pytest.approx(geo_slotless.r_mean)

    def test_rejects_zero_planes(self, geo_slotless):
        with pytest.raises(ValueError, match="n_planes"):
            _plane_radii(geo_slotless, 0)


class TestEffectiveYokes:
    """Per-variant shared-yoke rule (plan audit F2)."""

    def test_1r2s_halves_rotor_yoke(self, geo_slotted):
        rotor, stator_back = _effective_yokes(geo_slotted)
        assert rotor == pytest.approx(geo_slotted.rotor_yoke / 2)
        assert stator_back == pytest.approx(
            geo_slotted.stator_yoke - geo_slotted.slot_depth)

    def test_2r1s_nn_halves_stator_back_iron(self):
        geo = axial_flux(
            variant="2R1S-NN", r_out=0.164, r_in=0.0985, airgap=0.0015,
            magnet_thickness=0.004, rotor_yoke=0.020, stator_yoke=0.020,
        )
        rotor, stator_back = _effective_yokes(geo)
        assert rotor == pytest.approx(0.020)
        assert stator_back == pytest.approx(0.010)

    def test_ns_and_dssr_unhalved(self):
        for variant in ("2R1S-NS", "DSSR"):
            geo = axial_flux(
                variant=variant, r_out=0.164, r_in=0.0985, airgap=0.0015,
                magnet_thickness=0.004, rotor_yoke=0.020, stator_yoke=0.020,
            )
            assert _effective_yokes(geo) == (
                pytest.approx(0.020), pytest.approx(0.020))


class TestSlotSegments:

    def test_non_integer_slots_per_pole_rejected(self, geo_slotted):
        tau_p = pi * geo_slotted.r_mean / 5  # 36/(2*5) = 3.6
        with pytest.raises(ValueError, match="mirror Dirichlet"):
            _slot_segments(geo_slotted, tau_p, 5, 1)

    def test_segments_tile_the_pitch(self, geo_slotted):
        tau_p = pi * geo_slotted.r_mean / 6
        segs = _slot_segments(geo_slotted, tau_p, 6, 1)
        assert segs[0][0] == 0.0
        assert segs[-1][1] == pytest.approx(tau_p)
        for (_, x1, _n), (x0, _, _m) in pairwise(segs):
            assert x1 == pytest.approx(x0)
        assert sum(1 for s in segs if s[2] == "slot") == 3  # spp = 3


class TestEdgeLeakageFactor:

    def test_parviainen_band(self, geo_slotted):
        k = pm_edge_leakage_factor(geo_slotted, 1.05)
        # Radial-edge leakage is a ~1% effect on the Parviainen annulus.
        assert 0.97 < k < 1.0
        print(f"\nk_sigma_edge (Parviainen) = {k:.6f}")

    def test_wider_gap_leaks_more(self, geo_slotted):
        k1 = pm_edge_leakage_factor(geo_slotted, 1.05)
        k2 = pm_edge_leakage_factor(
            replace(geo_slotted, airgap=2 * geo_slotted.airgap), 1.05)
        assert k2 < k1

    def test_longer_annulus_leaks_less(self, geo_slotted):
        k1 = pm_edge_leakage_factor(geo_slotted, 1.05)
        k2 = pm_edge_leakage_factor(
            replace(geo_slotted, r_in=geo_slotted.r_in / 2), 1.05)
        assert k2 > k1


# ---------------------------------------------------------------------------
# CV-0: mirror domain vs true periodic-identification pole-pair solve
# ---------------------------------------------------------------------------

@requires_fem
class TestCV0PeriodicReference:

    def _compare(self, geo, mot, maxh_fraction, n_x=256):
        args = (geo, geo.r_mean, mot.n_p, mot.B_rem, mot.mu_r_pm,
                1.0e4, mot.alpha_p)
        _, Bz_m = solve_plane(*args, maxh_fraction=maxh_fraction, n_x=n_x)
        _, Bz_p = solve_plane_periodic(
            *args, maxh_fraction=maxh_fraction, n_x=n_x)
        b1_m = fundamental_amplitude(Bz_m)
        b1_p = fundamental_amplitude(Bz_p)
        rel_b1 = abs(b1_p - b1_m) / b1_m
        rel_l2 = float(
            np.linalg.norm(Bz_p - Bz_m) / np.linalg.norm(Bz_m))
        return b1_m, rel_b1, rel_l2

    def test_slotless(self, geo_slotless, parviainen):
        b1, rel_b1, rel_l2 = self._compare(geo_slotless, parviainen, 0.1)
        print(f"\nCV-0 slotless: B1={b1:.6f} T, "
              f"rel_B1={rel_b1:.2e}, field_rel_L2={rel_l2:.2e} "
              f"(tol {CV0_REL_TOL})")
        assert rel_b1 < CV0_REL_TOL
        assert rel_l2 < CV0_REL_TOL

    def test_slotted(self, geo_slotted, parviainen):
        # The pointwise metric is floored at ~9.1e-4 by slot-corner
        # ripple that the two non-nested meshes resolve differently
        # (stable across maxh_fraction 0.05 -> 0.0125, set by the fixed
        # airgap/3 per-face band, not the global mesh); n_x=512 samples
        # that ripple densely enough for a stable L2 estimate. B1
        # agrees at ~3e-5 at every mesh level tried.
        b1, rel_b1, rel_l2 = self._compare(
            geo_slotted, parviainen, 0.05, n_x=512)
        print(f"\nCV-0 slotted: B1={b1:.6f} T, "
              f"rel_B1={rel_b1:.2e}, field_rel_L2={rel_l2:.2e} "
              f"(tol {CV0_REL_TOL})")
        assert rel_b1 < CV0_REL_TOL
        assert rel_l2 < CV0_REL_TOL


# ---------------------------------------------------------------------------
# CV-1: slotless plane at r_mean vs the analytical layer series
# ---------------------------------------------------------------------------

@requires_fem
def test_cv1_slotless_b1_matches_layer_series(geo_slotless, parviainen):
    mot = parviainen
    _, Bz = solve_plane(
        geo_slotless, geo_slotless.r_mean, mot.n_p, mot.B_rem,
        mot.mu_r_pm, 1.0e4, mot.alpha_p,
    )
    b1 = fundamental_amplitude(Bz)
    ref = slotless_reference_B1(
        geo_slotless, mot.n_p, mot.B_rem, mot.mu_r_pm, mot.alpha_p)
    rel = abs(b1 - ref) / ref
    print(f"\nCV-1: FEM B1={b1:.6f} T, analytical={ref:.6f} T, "
          f"rel={rel:.2e} (tol {CV1_ANALYTICAL_REL_TOL})")
    assert rel < CV1_ANALYTICAL_REL_TOL


# ---------------------------------------------------------------------------
# CV-2: 1D magnetic-circuit limit (alpha_p = 1, tau_p >> gap stack)
# ---------------------------------------------------------------------------

@requires_fem
@pytest.mark.parametrize("n_p", [1, 2])
def test_cv2_circuit_limit(geo_slotless, n_p):
    """Scaled-down n_p at fixed r stretches tau_p to ~150-300x the gap
    stack; mu_r_fe = 1e6 removes the (real, geometry-stretched) yoke
    MMF drop so the plateau isolates the magnet + gap divider."""
    geo = geo_slotless
    _, Bz = solve_plane(
        geo, geo.r_mean, n_p, 1.05, 1.05, 1.0e6, 1.0)
    half = Bz[: len(Bz) // 2]           # first pole [0, tau_p)
    plateau = float(np.median(np.abs(half[: len(half) // 4])))
    ref = circuit_limit_Bz(1.05, 1.05, geo.magnet_thickness, geo.airgap)
    rel = abs(plateau - ref) / ref
    print(f"\nCV-2 (n_p={n_p}): plateau={plateau:.6f} T, "
          f"circuit={ref:.6f} T, rel={rel:.2e} (tol {CV2_REL_TOL})")
    assert rel < CV2_REL_TOL


# ---------------------------------------------------------------------------
# CV-3: slotted vs Carter-corrected analytical — REPORTED DIAGNOSTIC
# ---------------------------------------------------------------------------

@requires_fem
def test_cv3_slotted_vs_carter_diagnostic(geo_slotted, parviainen):
    """No assert on the gap (audit F4: both sides share the
    slot_width_ratio Carter fallback, so the comparison measures
    Carter's error, not the FEM's). Recorded for the Gate 1 review."""
    mot = parviainen
    _, Bz = solve_plane(
        geo_slotted, geo_slotted.r_mean, mot.n_p, mot.B_rem,
        mot.mu_r_pm, 1.0e4, mot.alpha_p,
    )
    b1_fem = fundamental_amplitude(Bz)
    with pytest.warns(UserWarning, match="slot_opening_ratio unset"):
        g_prime, k_c = carter_airgap_axial(geo_slotted, mot.mu_r_pm)
    theta = np.linspace(0, 2 * pi, 3600, endpoint=False)
    Bz_a = axial_layer_Bz_series(
        theta, mot.n_p, mot.B_rem,
        tau_p=pi * geo_slotted.r_mean / mot.n_p,
        g_prime=g_prime, l_pm=geo_slotted.magnet_thickness,
        mu_r_pm=mot.mu_r_pm, alpha_p=mot.alpha_p, max_order=1800,
    )
    b1_carter = 2 * float(np.abs(
        np.mean(Bz_a * np.exp(-1j * mot.n_p * theta))))
    gap = (b1_fem - b1_carter) / b1_carter
    print(f"\nCV-3 diagnostic: FEM B1={b1_fem:.6f} T, "
          f"Carter analytical={b1_carter:.6f} T (k_c={k_c:.4f}), "
          f"FEM/Carter gap={gap:+.2%} — reported, not gated")
    assert np.isfinite(b1_fem) and b1_fem > 0


# ---------------------------------------------------------------------------
# K = 1 identity vs the analytical winding-transfer chain
# ---------------------------------------------------------------------------

@requires_fem
def test_k1_identity_matches_winding_transfer(geo_slotted, parviainen):
    """At K = 1 the geometric sum S = B1·r_mean·L_a, so the registry
    chain GAP_LINKAGE·(2·N_eff·k_w/n_p)·S must equal
    GAP_LINKAGE·winding_transfer_axial·B1 exactly (pure algebra —
    same B1 on both sides)."""
    mot = parviainen
    res = solve_axial_multiplane(
        geo_slotted, mot.n_p, mot.B_rem, mot.mu_r_pm, mot.mu_r_fe,
        mot.alpha_p, n_planes=1,
    )
    b1 = res["plane_B1_list"][0]
    n_eff = mot.N * mot.coils_series
    assert res["plane_r_list"][0] == pytest.approx(geo_slotted.r_mean)
    assert res["S"] == pytest.approx(
        b1 * geo_slotted.r_mean * geo_slotted.active_length, rel=1e-12)
    chain = (2 * n_eff * mot.k_w / mot.n_p) * res["S"]
    analytical = winding_transfer_axial(
        geo_slotted, n_eff, mot.k_w, mot.n_p) * b1
    assert chain == pytest.approx(analytical, rel=1e-12)
    assert GAP_LINKAGE[geo_slotted.variant] == 1  # 1R2S: one gap/circuit


# ---------------------------------------------------------------------------
# Mesh convergence
# ---------------------------------------------------------------------------

@requires_fem
def test_mesh_convergence_on_maxh_halving(geo_slotted, parviainen):
    mot = parviainen
    args = (geo_slotted, geo_slotted.r_mean, mot.n_p, mot.B_rem,
            mot.mu_r_pm, mot.mu_r_fe, mot.alpha_p)
    _, Bz_c = solve_plane(*args, maxh_fraction=0.1)
    _, Bz_f = solve_plane(*args, maxh_fraction=0.05)
    b1_c = fundamental_amplitude(Bz_c)
    b1_f = fundamental_amplitude(Bz_f)
    shift = abs(b1_c - b1_f) / b1_f
    print(f"\nmesh convergence: B1(maxh)={b1_c:.6f}, "
          f"B1(maxh/2)={b1_f:.6f}, shift={shift:.2e} (gate 5e-3)")
    assert shift < 5e-3


# ---------------------------------------------------------------------------
# Multi-plane output structure + tiled-waveform symmetry
# ---------------------------------------------------------------------------

@requires_fem
class TestMultiplane:

    @pytest.fixture(scope="class")
    def result(self, geo_slotted, parviainen):
        mot = parviainen
        return solve_axial_multiplane(
            geo_slotted, mot.n_p, mot.B_rem, mot.mu_r_pm, mot.mu_r_fe,
            mot.alpha_p, n_planes=3, n_theta=720,
        )

    def test_structure(self, result, geo_slotted):
        assert len(result["plane_r_list"]) == 3
        assert len(result["plane_B1_list"]) == 3
        assert result["plane_r_list"][1] == pytest.approx(
            geo_slotted.r_mean)
        assert result["fundamental"] == result["plane_B1_list"][1]
        assert len(result["theta_list"]) == 720
        assert len(result["B_z_list"]) == 720
        assert result["peak"] > result["fundamental"] * 0.5
        assert 0 < result["k_sigma_edge"] < 1

    def test_b1_grows_with_radius(self, result):
        # Wider pole pitch at larger radius -> less interpole leakage.
        b1 = result["plane_B1_list"]
        assert b1[0] < b1[1] < b1[2]

    def test_geometric_sum(self, result):
        b1 = np.array(result["plane_B1_list"])
        r = np.array(result["plane_r_list"])
        assert result["S"] == pytest.approx(
            float(np.sum(b1 * r * result["dr"])), rel=1e-12)

    def test_tiled_waveform_symmetry(self, result, parviainen):
        """Zero mean and odd harmonics of n_p only: the mean vanishes
        because net gap flux per pole pitch is A(0) − A(tau_p) = 0
        (both Dirichlet); bins off multiples of n_p vanish because the
        waveform is an exact n_p-fold tiling; even multiples of n_p are
        suppressed by the half-wave antisymmetry of the mirror
        extension (approximate — the mesh is not symmetric)."""
        n_p = parviainen.n_p
        B = np.asarray(result["B_z_list"])
        spec = np.abs(np.fft.rfft(B)) / len(B) * 2
        b1 = spec[n_p]
        assert abs(np.mean(B)) < 1e-3 * b1
        off = [k for k in range(1, 4 * n_p) if k % n_p != 0]
        assert max(spec[off]) < 1e-9 * b1
        assert spec[2 * n_p] < 1e-2 * b1
        assert spec[4 * n_p] < 1e-2 * b1
        assert spec[3 * n_p] > 1e-2 * b1  # real slot/shape harmonic


# ---------------------------------------------------------------------------
# Nonlinear iron (shared fem_field solvers)
# ---------------------------------------------------------------------------

@requires_fem
def test_nonlinear_iron_lowers_b1(geo_slotted, parviainen):
    """The halved 1R2S rotor yoke runs saturated at open circuit, so BH
    iron must land below the linear mu_r_fe=1000 solve — but within the
    same physical ballpark (per-gap approximation, plan audit F2)."""
    mot = parviainen
    args = (geo_slotted, geo_slotted.r_mean, mot.n_p, mot.B_rem,
            mot.mu_r_pm, mot.mu_r_fe, mot.alpha_p)
    _, Bz_lin = solve_plane(*args)
    _, Bz_nl = solve_plane(*args, nonlinear=True)
    b1_lin = fundamental_amplitude(Bz_lin)
    b1_nl = fundamental_amplitude(Bz_nl)
    print(f"\nnonlinear: B1_lin={b1_lin:.6f} T, B1_BH={b1_nl:.6f} T "
          f"({(b1_nl / b1_lin - 1):+.2%})")
    assert 0 < b1_nl < b1_lin
    assert b1_nl > 0.8 * b1_lin


@requires_fem
def test_nonlinear_newton_matches_picard(geo_slotted, parviainen, monkeypatch):
    """Newton and Picard converge to the same axial plane solution at
    matched tolerance — closes the three-solver equivalence panel
    (2D: test_newton_matches_picard_at_high_flux, wedge:
    test_endeffect_newton_matches_picard_fixed_point). Picard's shipping
    tolerance is tightened so the comparison measures solver agreement
    at the shared fixed point, not Picard's stopping error."""
    from phasesweep.solvers import fem_axial as fa

    mot = parviainen
    args = (geo_slotted, geo_slotted.r_mean, mot.n_p, mot.B_rem,
            mot.mu_r_pm, mot.mu_r_fe, mot.alpha_p)
    monkeypatch.setattr(fa, "PICARD_TOLERANCE", 1e-4)
    _, Bz_p = solve_plane(*args, nonlinear=True, nonlinear_solver="picard")

    # The axial dispatch has no info plumbing, so a silent Newton->Picard
    # fallback would make this comparison vacuous (Picard vs Picard).
    # Trap the fallback path instead.
    def _fallback_trap(*a, **k):
        raise AssertionError("axial Newton silently fell back to Picard")

    monkeypatch.setattr(fa, "picard_solve", _fallback_trap)
    _, Bz_n = solve_plane(*args, nonlinear=True, nonlinear_solver="newton")
    assert np.max(np.abs(Bz_n - Bz_p)) < 1e-3 * np.max(np.abs(Bz_p))


# ---------------------------------------------------------------------------
# Mesh cache
# ---------------------------------------------------------------------------

@requires_fem
def test_mesh_cache_reuse_and_clear(geo_slotless, parviainen):
    from phasesweep.solvers import fem_axial

    mot = parviainen
    clear_mesh_cache()
    assert len(fem_axial._mesh_cache) == 0
    args = (geo_slotless, geo_slotless.r_mean, mot.n_p, mot.B_rem,
            mot.mu_r_pm, mot.mu_r_fe, mot.alpha_p)
    solve_plane(*args)
    assert len(fem_axial._mesh_cache) == 1
    solve_plane(*args)  # cache hit — no second entry
    assert len(fem_axial._mesh_cache) == 1
    solve_plane_periodic(*args)  # distinct key for the periodic domain
    assert len(fem_axial._mesh_cache) == 2
    clear_mesh_cache()
    assert len(fem_axial._mesh_cache) == 0
