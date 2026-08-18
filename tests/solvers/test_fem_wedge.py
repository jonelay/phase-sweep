"""Tests for the 3D wedge k_end FEM (phasesweep.fem_wedge).

Stage 0: iron-backed magnet + air-gap slab against the exact 1D circuit answer.
Stage 1: half-pole smooth-bore wedge (axial-Neumann) against zhu_howe_Br.
Stage 2: finite stack + end-air box — end-effect droop and k_end.
"""

import pytest

from phasesweep.machines.geometry import default_inrunner, outrunner
from phasesweep.solvers.fem_wedge import (
    compute_k_end,
    slab_analytic,
    solve_slab,
    solve_wedge_2d_equiv,
    solve_wedge_endeffect,
)
from tests.conftest import requires_fem


def _outrunner_14mm():
    return outrunner(
        r_outer=0.0097, r_rotor=0.00877, r_magnet=0.00748,
        r_stator=0.00695, r_inner=0.003,
    )


def test_slab_analytic_matches_circuit_formula():
    # B_gap = B_rem / (1 + mu_r*g/h_m); h_m = g, mu_r = 1 -> exactly B_rem/2.
    assert slab_analytic(1.0, 1.0, 1e-3, 1e-3) == pytest.approx(0.5)


@requires_fem
def test_slab_fem_matches_analytic():
    r = solve_slab()
    # Exact solution is piecewise-linear in phi -> H1 order-3 is exact (ULP scatter).
    assert r.rel_error < 1e-10
    assert r.B_gap_fem == pytest.approx(r.B_gap_analytic, rel=1e-10)


@requires_fem
def test_slab_fem_sign_and_magnitude():
    r = solve_slab(B_rem=1.45, mu_r_pm=1.05, h_m=1.287e-3, g=0.528e-3)
    # Flux is driven the right way (positive) and bounded above by B_rem.
    assert 0.0 < r.B_gap_fem < 1.45


@requires_fem
def test_slab_larger_gap_lowers_field():
    tight = solve_slab(g=0.3e-3)
    wide = solve_slab(g=1.5e-3)
    assert wide.B_gap_fem < tight.B_gap_fem


@requires_fem
def test_slab_mesh_independent():
    coarse = solve_slab(maxh=0.5e-3)
    fine = solve_slab(maxh=0.1e-3)
    assert coarse.B_gap_fem == pytest.approx(fine.B_gap_fem, rel=1e-9)


# --- Stage 1: half-pole wedge vs analytic Zhu-Howe -----------------------

@requires_fem
def test_wedge_inrunner_matches_zhu_howe():
    r = solve_wedge_2d_equiv(default_inrunner(), n_p=4)
    # Axial-Neumann caps recover the 2D field; residual is finite-mu iron + mesh.
    assert r.rel_error < 5e-3
    # Sign agrees (both outward-radial fundamental).
    assert r.b1_fem * r.b1_analytic > 0


@requires_fem
def test_wedge_outrunner_matches_zhu_howe():
    r = solve_wedge_2d_equiv(_outrunner_14mm(), n_p=6, B_rem=1.45)
    assert r.rel_error < 5e-3


@requires_fem
def test_wedge_converges_with_mesh():
    geo = default_inrunner()
    gap = geo.r_stator - geo.r_magnet
    coarse = solve_wedge_2d_equiv(geo, n_p=4, maxh=gap / 3)
    fine = solve_wedge_2d_equiv(geo, n_p=4, maxh=gap / 6)
    assert fine.rel_error < coarse.rel_error


# --- Stage 2: finite stack + end-air, k_end ------------------------------
# Lighter params than the production defaults (smaller end-air box, fewer axial
# samples), so these pin the fast-config k_end, not the published production
# value (0.9694 linear / 0.9252 nonlinear).
_FAST_END = dict(n_p=6, L_stk=0.007, z_box_factor=3.0, n_z=49)

# Measured at _FAST_END; repeatable to ~1e-15 across runs. The band is
# mesh/solver headroom, not physics uncertainty. It replaces `0.5 < k_end < 1.0`,
# which accepted a 0.75x scale error — the same blind spot that let the
# silently non-converged nonlinear column pass review.
_K_END_FAST = 0.9428
_K_END_BAND = 0.02


@requires_fem
def test_endeffect_profile_and_k_end():
    import numpy as np

    r = solve_wedge_endeffect(_outrunner_14mm(), **_FAST_END)
    n = len(r.b3)
    # Mid-plane (z=0) of a finite stack matches the 2D analytic fundamental.
    assert float(r.b3[0]) == pytest.approx(r.b2_analytic, rel=5e-3)
    # Field falls substantially from mid-plane to the stack end; the outer third
    # is well below the inner third (robust to point-sampling wobble).
    assert float(r.b3[-1]) < 0.7 * float(r.b3[0])
    assert np.mean(r.b3[2 * n // 3:]) < np.mean(r.b3[:n // 3])
    # Integrating to L_stk/2 keeps k_end < 1 (audit HIGH finding).
    assert r.k_end < 1.0
    assert r.k_end == pytest.approx(_K_END_FAST, abs=_K_END_BAND)


@requires_fem
def test_endeffect_outer_bc_insensitive():
    geo = _outrunner_14mm()
    neu = solve_wedge_endeffect(geo, outer_bc="neumann", **_FAST_END)
    dir_ = solve_wedge_endeffect(geo, outer_bc="dirichlet", **_FAST_END)
    # High-mu yoke contains the flux: k_end is insensitive to the r_outer BC.
    assert neu.k_end == pytest.approx(dir_.k_end, abs=5e-3)


@requires_fem
def test_endeffect_flux_measures_agree():
    r = solve_wedge_endeffect(_outrunner_14mm(), **_FAST_END)
    # Radius (mid-gap vs bore) and baseline (analytic-2D vs FEM mid-plane) choices
    # move k_end by < ~1% on this co-extensive stack, where the mid-plane already
    # matches 2D — flux_droop_ratio and k_end only coincide in that regime.
    assert r.k_end_bore == pytest.approx(r.k_end, abs=1e-2)
    assert r.flux_droop_ratio == pytest.approx(r.k_end, abs=1e-2)


@requires_fem
@pytest.mark.timeout(900)
def test_endeffect_nonlinear_iron_lowers_k_end():
    # picard_tol 1e-3: the saturating end-region converges slowly under
    # the adaptive relaxation at this coarse fast config (the old
    # fixed relax = 0.3 never converged here at all and the fall-through
    # returned the oscillating field silently). A 1e-3 mu step bounds the
    # k_end error well below the ~2% saturation effect asserted.
    geo = _outrunner_14mm()
    lin = solve_wedge_endeffect(geo, **_FAST_END)
    nl = solve_wedge_endeffect(geo, nonlinear=True, picard_tol=1e-3,
                               **_FAST_END)
    # End-region iron saturates -> more droop -> k_end a few % below linear.
    # Pinned two-sided: the ordering alone passed the non-converged solves.
    assert nl.k_end < lin.k_end
    assert nl.k_end == pytest.approx(0.8986, abs=0.03)


@requires_fem
def test_endeffect_top_bc_runs():
    r = solve_wedge_endeffect(_outrunner_14mm(), top_bc="dirichlet", **_FAST_END)
    # Grounding the top of the air box barely moves k_end (0.9412 vs 0.9428).
    assert r.k_end == pytest.approx(_K_END_FAST, abs=_K_END_BAND)


@requires_fem
def test_endeffect_tight_gap_solves_at_defaults():
    # 0.262 mm gap (tight-gap variant): local gap refinement keeps the bulk coarse so
    # the default mesh stays tractable — global gap/3 would over-refine and OOM.
    geo = outrunner(
        r_outer=0.009645, r_rotor=0.008515, r_magnet=0.007212,
        r_stator=0.006950, r_inner=0.003,
    )
    r = solve_wedge_endeffect(geo, n_p=6, L_stk=0.007, z_box_factor=3.0, n_z=49)
    # Tighter gap -> less end leakage relative to the gap flux -> k_end above
    # the reference geometry's 0.9428.
    assert r.k_end == pytest.approx(0.9626, abs=_K_END_BAND)


# --- Stage 3: overhang + one-shot k_end calculator -----------------------

@requires_fem
def test_compute_k_end_overhang_raises_k_end():
    geo = _outrunner_14mm()
    kw = dict(n_p=6, L_stk=0.007, z_box_factor=3.0, n_z=49)
    flush = compute_k_end(geo, overhang=0.0, **kw)
    over = compute_k_end(geo, overhang=0.0005, **kw)
    # Magnet extending past the stack props up the end field -> less droop.
    assert flush == pytest.approx(_K_END_FAST, abs=_K_END_BAND)
    assert over == pytest.approx(0.9866, abs=_K_END_BAND)
    assert over > flush


@requires_fem
def test_endeffect_overhang_exceeding_air_box_raises():
    """Overhang reaching the z_box cap would silently clamp the fringe field
    (magnet flush with / past the far boundary) — must raise instead."""
    geo = _outrunner_14mm()
    with pytest.raises(ValueError, match="z_box"):
        solve_wedge_endeffect(geo, mag_overhang=0.005, **_FAST_END)


@requires_fem
def test_wedge_picard_nonconvergence_raises(monkeypatch):
    """F6b: a non-converged wedge Picard solve must raise, mirroring the 2D
    _picard_solve contract — never silently return a bad field."""
    import phasesweep.solvers.fem_wedge as fw
    real = fw._picard_solve_wedge

    def strict(*args, **kwargs):
        kwargs.update(n_iter=1, tol=0.0)
        return real(*args, **kwargs)

    monkeypatch.setattr(fw, "_picard_solve_wedge", strict)
    with pytest.raises(RuntimeError, match="did not converge"):
        solve_wedge_endeffect(_outrunner_14mm(), nonlinear=True,
                              **_FAST_END)
