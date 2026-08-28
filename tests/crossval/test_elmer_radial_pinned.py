"""Radial Elmer pinned-result check — independent-code verification.

Validates the committed comparison JSON (produced once by
scripts/elmer_radial/run_elmer_radial.py against the native ElmerSolver
and the NGSolve fem model) against the shared contract. No Elmer and no
ngsolve at test time: the analytical references are pure numpy, the
cross-code comparands are both pinned, so this runs in the CI core
lane. Regenerating the JSON requires ElmerSolver + the fem extra — see
scripts/elmer_radial/README.md.
"""

import json

import pytest

from tests.conftest import ELMER_PINNED_SKIP, REPO_ROOT
from tests.crossval.radial_fem_contract import (
    ANALYTICAL_REL_TOL,
    B_REM,
    EXPECTED_CASES,
    MU_R_FE,
    MU_R_PM,
    N_P,
    NONTRIVIAL_PEAK_T,
    baseline_geometry,
    case_tolerance,
    slotless_reference_B1,
)

PINNED = REPO_ROOT / "scripts/elmer_radial/pinned/elmer_radial_cv.json"


@pytest.fixture(scope="module")
def pinned():
    if not PINNED.exists():
        pytest.skip(ELMER_PINNED_SKIP)
    return json.loads(PINNED.read_text())


def test_baseline_matches_contract(pinned):
    """The JSON was generated with the contract's baseline parameters —
    guards silent drift between a regeneration and this check."""
    base = pinned["baseline"]
    assert base["n_p"] == N_P
    assert base["B_rem_T"] == B_REM
    assert base["mu_r_pm"] == MU_R_PM
    assert base["mu_r_fe"] == MU_R_FE
    assert set(pinned["cases"]) == EXPECTED_CASES


def test_nontrivial_fields(pinned):
    """Silent-zero guard: every pinned Elmer field is a real solve
    (two known Elmer failure modes exit 0 with a trivially zero field)."""
    for name, case in pinned["cases"].items():
        assert case["elmer"]["peak_T"] >= NONTRIVIAL_PEAK_T, name


@pytest.mark.parametrize("name", sorted(EXPECTED_CASES))
def test_cross_code_agreement(pinned, name):
    """Elmer B1 vs the pinned NGSolve fem B1, per-case gate."""
    case = pinned["cases"][name]
    B1_e = case["elmer"]["B1_T"]
    B1_n = case["ngsolve"]["B1_T"]
    assert abs(B1_e - B1_n) / B1_n <= case_tolerance(case), name


@pytest.mark.parametrize("name", ["smooth_bore", "arcs_a080", "arcs_a095"])
def test_linear_slotless_vs_analytical(pinned, name):
    """Linear slotless cases vs the live Zhu/Howe/Chan series at r_ag.

    mu_r_fe = 5000 leaves a small physical yoke-MMF residual vs the
    ideal-iron series; the gate absorbs it (observed ~0.1%).
    """
    case = pinned["cases"][name]
    ref = slotless_reference_B1(baseline_geometry(), N_P, B_REM, MU_R_PM,
                                case["params"]["alpha_p"])
    B1_e = case["elmer"]["B1_T"]
    assert abs(B1_e - ref) / ref <= ANALYTICAL_REL_TOL, name


def test_current_sheet_effect_matches(pinned):
    """The armature sheet's fundamental rise agrees between codes —
    checks the j_s path specifically, not just each absolute level."""
    open_c = pinned["cases"]["slotted_open"]
    js_c = pinned["cases"]["slotted_js"]
    rise_e = js_c["elmer"]["B1_T"] / open_c["elmer"]["B1_T"]
    rise_n = js_c["ngsolve"]["B1_T"] / open_c["ngsolve"]["B1_T"]
    assert abs(rise_e - rise_n) / rise_n <= 0.005
