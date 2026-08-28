"""Tests for fem_sampling.py — Maxwell-stress torque and the magnet
operating-point sampler."""

from importlib.util import find_spec

import pytest

if not find_spec("ngsolve"):
    pytest.skip("requires phasesweep[fem] (ngsolve not installed)", allow_module_level=True)

import numpy as np

from phasesweep.machines.geometry import outrunner
from phasesweep.solvers.fem_field import solve_field_fem
from phasesweep.solvers.harmonics import harmonics_1sided
from tests.conftest import requires_fem
from tests.solvers.fem_shared import (
    _SLOTTED_OUT_GEO_KW,
    _TORQUE_FAST,
    _gap_radii,
    _torque_geo,
)

# Module-level guard covers missing ngsolve; the marker makes the module
# selectable by lane.
pytestmark = requires_fem


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


def test_maxwell_torque_contour_independent_and_above_noise():
    """The stress integral is contour-independent in the source-free gap
    (< 5% spread across three radii at this mesh), and the zero-current
    solve's residual torque (cogging + discretization noise) sits well
    below the interaction torque."""
    from phasesweep.solvers.fem_sampling import maxwell_stress_torque
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
    from phasesweep.solvers.fem_sampling import maxwell_stress_torque
    geo = _torque_geo()
    r = _gap_radii(geo)[1]
    taus = {}
    for j in (0.0, 1e5, 2e5, -1e5):
        _, _, mesh, gfu = solve_field_fem(geo=geo, j_s=j, **_TORQUE_FAST)
        taus[j] = maxwell_stress_torque(mesh, gfu, r)
    base = taus[1e5] - taus[0.0]
    assert (taus[2e5] - taus[0.0]) / base == pytest.approx(2.0, rel=1e-4)
    assert (taus[-1e5] - taus[0.0]) / base == pytest.approx(-1.0, rel=1e-4)


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
    from phasesweep.solvers.fem_mesh import slot_source_moment
    from phasesweep.solvers.fem_sampling import maxwell_interaction_torque_order
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
        from phasesweep.solvers.fem_sampling import sample_magnet_Bm
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
        from phasesweep.solvers.fem_sampling import sample_magnet_Bm
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

    def test_halbach_requires_segments(self):
        """sample_magnet_Bm requires halbach_segments for Halbach."""
        from phasesweep.solvers.fem_sampling import sample_magnet_Bm
        geo = _torque_geo()
        _, _, m, g = solve_field_fem(geo=geo, **_BM_FAST)
        with pytest.raises(ValueError, match="halbach_segments"):
            sample_magnet_Bm(m, g, geo, n_p=2, magnetization="halbach")

    def test_halbach_open_circuit_positive(self):
        """No-load Halbach B_m is positive throughout and weights sum to 1."""
        from phasesweep.solvers.fem_sampling import sample_magnet_Bm
        geo = _torque_geo()
        _, _, mesh, gfu = solve_field_fem(
            geo=geo, n_p=2, B_rem=1.0, mu_r_pm=1.05, mu_r_fe=1000.0,
            n_theta=60, maxh_fraction=0.08, alpha_p=1.0,
            magnetization="halbach", halbach_segments=2,
            return_full=True)
        _, _, B_m, w = sample_magnet_Bm(
            mesh, gfu, geo, n_p=2,
            magnetization="halbach", halbach_segments=2)
        assert np.all(B_m > 0), f"min B_m={B_m.min():.4f}"
        assert np.sum(w) == pytest.approx(1.0, abs=1e-12)

    def test_halbach_projection_differs_from_radial(self):
        """Halbach per-segment projection gives different B_m statistics
        than the radial sign(M_r)*B_r on the same field.  With kappa_m=2
        the tangential segments carry flux mostly tangentially — the
        per-segment projection captures this (B_m > 0), while radial
        projection gives near-zero B_r there."""
        from phasesweep.solvers.fem_sampling import sample_magnet_Bm
        geo = _torque_geo()
        _, _, mesh, gfu = solve_field_fem(
            geo=geo, n_p=2, B_rem=1.0, mu_r_pm=1.05, mu_r_fe=1000.0,
            n_theta=60, maxh_fraction=0.08, alpha_p=1.0,
            magnetization="halbach", halbach_segments=2,
            return_full=True)
        _, _, B_m_h, w_h = sample_magnet_Bm(
            mesh, gfu, geo, n_p=2,
            magnetization="halbach", halbach_segments=2)
        _, _, B_m_r, w_r = sample_magnet_Bm(
            mesh, gfu, geo, n_p=2, magnetization="radial")
        mean_h = float(np.average(B_m_h, weights=w_h))
        mean_r = float(np.average(B_m_r, weights=w_r))
        assert not np.isclose(mean_h, mean_r, rtol=0.05), (
            f"mean_h={mean_h:.4f}, mean_r={mean_r:.4f} — projections "
            f"should differ for kappa_m=2 (tangential segments)"
        )

    def test_rotation_equivalence(self):
        """Sampling a rotated solve with the matching rotation reproduces
        the unrotated B_m distribution (rotationally equivalent problem;
        rotated arcs build a different mesh -> mesh-scatter tolerance)."""
        from phasesweep.solvers.fem_sampling import sample_magnet_Bm
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
        from phasesweep.solvers.fem_sampling import demag_margin, sample_magnet_Bm
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
