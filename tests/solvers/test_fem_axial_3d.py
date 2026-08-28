"""Tests for the Track B 3D axial edge-leakage study (fem_axial_3d).

Gates (plan rev 2, Track B):
  - extruded control vs the analytical layer series (CV-1 tolerance) —
    the formulation cross-check;
  - mesh convergence AND domain-extent convergence, both evaluated on
    the leakage deficit itself (plan audit F1), gate values pinned from
    measured convergence runs (see the constants below);
  - CV-4 deficit-vs-k_sigma_edge numbers are REPORTED diagnostics here
    (printed table); the CV-4 arbitration verdict happens in W3.

Tolerances imported from tests/crossval/axial_fem_contract.py; solve
results shared through module-scoped fixtures for runtime economy.
"""

import dataclasses

import numpy as np
import pytest

from phasesweep.solvers.fem_axial_3d import (
    MIN_COLLAR_FACTOR,
    compute_edge_leakage,
    radial_edge_leakage_factor,
    solve_axial_sector,
)
from tests.conftest import requires_fem
from tests.crossval.axial_fem_contract import (
    CV1_ANALYTICAL_REL_TOL,
    CV4_FACTOR_ABS_TOL,
    load_parviainen_motor,
    slotless_reference_B1,
)

# Convergence gates, pinned from measured runs on the Parviainen fixture
# collar 4 -> 6 moved the deficit ratio by 1.3e-4 (0.7%
# of the deficit); maxh_fine edge/3 -> edge/6 moved it by 6.6e-4 with
# psi_geo(extruded) moving 6.8e-5 relative. Gates at ~3x the measured
# movement so a real regression trips them but discretization noise
# does not.
DOMAIN_EXTENT_RATIO_ABS_TOL = 1e-3
DOMAIN_EXTENT_DEFICIT_REL_TOL = 0.10
MESH_RATIO_ABS_TOL = 2e-3
MESH_PSI_REL_TOL = 5e-4


@pytest.fixture(scope="module")
def parviainen():
    return load_parviainen_motor()


@pytest.fixture(scope="module")
def base_pair(parviainen):
    """Finite + extruded solves at default settings (shared across tests)."""
    m = parviainen
    return compute_edge_leakage(
        m.geometry, m.n_p, m.B_rem, alpha_p=m.alpha_p, mu_r_pm=m.mu_r_pm,
    )


# ---------------------------------------------------------------------------
# Analytical factor (no FEM)
# ---------------------------------------------------------------------------

def test_radial_edge_factor_parviainen_value(parviainen):
    # Regression pin: Qu & Lipo magnet-to-rotor form, physical gap,
    # L_a = active_length (hand-checked: g=1.5 mm, l_pm=4 mm, L_a=65.5 mm).
    geo = parviainen.geometry
    k = radial_edge_leakage_factor(
        geo.airgap, geo.magnet_thickness, geo.active_length,
        parviainen.mu_r_pm,
    )
    assert k == pytest.approx(0.989264, abs=1e-5)


def test_radial_edge_factor_limits():
    # Vanishing gap -> no leakage path -> factor -> 1.
    assert radial_edge_leakage_factor(1e-9, 4e-3, 0.065, 1.05) == \
        pytest.approx(1.0, abs=1e-6)
    # Longer active length dilutes the edge effect -> factor closer to 1.
    short = radial_edge_leakage_factor(1.5e-3, 4e-3, 0.03, 1.05)
    long_ = radial_edge_leakage_factor(1.5e-3, 4e-3, 0.12, 1.05)
    assert short < long_ < 1.0
    with pytest.raises(ValueError):
        radial_edge_leakage_factor(0.0, 4e-3, 0.065, 1.05)


def test_solver_input_validation(parviainen):
    # These raise before any ngsolve import (runnable without FEM deps).
    m = parviainen
    with pytest.raises(ValueError, match="collar_factor"):
        solve_axial_sector(
            m.geometry, m.n_p, m.B_rem,
            collar_factor=MIN_COLLAR_FACTOR - 1.0,
        )
    with pytest.raises(ValueError, match="alpha_p"):
        solve_axial_sector(m.geometry, m.n_p, m.B_rem, alpha_p=1.5)
    coreless = dataclasses.replace(
        m.geometry, variant="coreless", stator_yoke=0.0, n_slots=0,
    )
    with pytest.raises(ValueError, match="coreless"):
        solve_axial_sector(coreless, m.n_p, m.B_rem)


# ---------------------------------------------------------------------------
# CV-1: extruded control vs the analytical layer series
# ---------------------------------------------------------------------------

@requires_fem
def test_extruded_matches_layer_series(parviainen, base_pair):
    m = parviainen
    geo = m.geometry
    ext = base_pair.extruded
    ref = slotless_reference_B1(geo, m.n_p, m.B_rem, m.mu_r_pm, m.alpha_p)
    i = int(np.argmin(np.abs(ext.r_list - geo.r_mean)))
    rel = abs(ext.B1_list[i] - ref) / ref
    assert rel < CV1_ANALYTICAL_REL_TOL, (
        f"extruded B1(r_mean)={ext.B1_list[i]:.5f} vs layer series "
        f"{ref:.5f} ({rel:.2%} > {CV1_ANALYTICAL_REL_TOL:.0%})"
    )
    # Translation invariance: away from the Neumann walls the profile is
    # flat — the mid-band spread stays within the same tolerance.
    band = ext.B1_list[len(ext.B1_list) // 3: 2 * len(ext.B1_list) // 3]
    assert np.ptp(band) / ref < CV1_ANALYTICAL_REL_TOL


# ---------------------------------------------------------------------------
# Convergence — both gates evaluated on the leakage deficit (audit F1)
# ---------------------------------------------------------------------------

@requires_fem
def test_domain_extent_convergence_on_deficit(parviainen, base_pair):
    m = parviainen
    wide_fin = solve_axial_sector(
        m.geometry, m.n_p, m.B_rem, alpha_p=m.alpha_p, mu_r_pm=m.mu_r_pm,
        finite=True, collar_factor=6.0,
    )
    ratio_base = base_pair.deficit_ratio
    ratio_wide = wide_fin.psi_geo / base_pair.extruded.psi_geo
    deficit = 1.0 - ratio_base
    assert deficit > 0.0
    assert abs(ratio_wide - ratio_base) < DOMAIN_EXTENT_RATIO_ABS_TOL
    assert abs(ratio_wide - ratio_base) < \
        DOMAIN_EXTENT_DEFICIT_REL_TOL * deficit, (
            f"far-field truncation not converged: deficit moved "
            f"{abs(ratio_wide - ratio_base):.2e} of {deficit:.2e} "
            f"going collar 4 -> 6"
        )


@requires_fem
def test_mesh_convergence_on_deficit(parviainen):
    m = parviainen
    geo = m.geometry
    edge = geo.airgap + geo.magnet_thickness
    res = {
        div: compute_edge_leakage(
            geo, m.n_p, m.B_rem, alpha_p=m.alpha_p, mu_r_pm=m.mu_r_pm,
            maxh_fine=edge / div,
        )
        for div in (3, 6)
    }
    d_ratio = abs(res[3].deficit_ratio - res[6].deficit_ratio)
    d_psi = abs(res[3].extruded.psi_geo / res[6].extruded.psi_geo - 1.0)
    assert d_ratio < MESH_RATIO_ABS_TOL, (
        f"deficit ratio moved {d_ratio:.2e} on maxh_fine halving"
    )
    assert d_psi < MESH_PSI_REL_TOL, (
        f"psi_geo(extruded) moved {d_psi:.2e} on maxh_fine halving"
    )


# ---------------------------------------------------------------------------
# CV-4 prep: deficit diagnostics (reported, not gated — arbitration in W3)
# ---------------------------------------------------------------------------

@requires_fem
def test_edge_leakage_deficit_diagnostics(parviainen, base_pair):
    m = parviainen
    geo = m.geometry
    r_mean, L_a = geo.r_mean, geo.active_length

    cases = {"base": base_pair}
    for s in (0.67, 1.5):
        g = dataclasses.replace(geo, airgap=geo.airgap * s)
        cases[f"airgap x{s}"] = compute_edge_leakage(
            g, m.n_p, m.B_rem, alpha_p=m.alpha_p, mu_r_pm=m.mu_r_pm,
        )
    for s in (0.67, 1.5):
        g = dataclasses.replace(
            geo, r_in=r_mean - s * L_a / 2, r_out=r_mean + s * L_a / 2,
        )
        cases[f"L_a x{s}"] = compute_edge_leakage(
            g, m.n_p, m.B_rem, alpha_p=m.alpha_p, mu_r_pm=m.mu_r_pm,
        )

    print("\nCV-4 prep — deficit ratio vs k_sigma_edge "
          f"(CV4_FACTOR_ABS_TOL={CV4_FACTOR_ABS_TOL}):")
    for label, r in cases.items():
        print(
            f"  {label:12s} deficit={r.deficit_ratio:.5f} "
            f"k_edge={r.k_sigma_edge:.5f} err={r.factor_error:+.5f} "
            f"{'within' if abs(r.factor_error) <= CV4_FACTOR_ABS_TOL else 'OUTSIDE'}"
        )

    for label, r in cases.items():
        # Sanity envelope only — the tight comparison is W3's call.
        assert 0.9 < r.deficit_ratio < 1.0, (label, r.deficit_ratio)
        assert 0.0 < r.k_sigma_edge < 1.0

    # Physical monotonicities of the FEM deficit itself.
    assert (cases["airgap x1.5"].deficit_ratio
            < cases["base"].deficit_ratio
            < cases["airgap x0.67"].deficit_ratio)
    assert (cases["L_a x0.67"].deficit_ratio
            < cases["base"].deficit_ratio
            < cases["L_a x1.5"].deficit_ratio)
