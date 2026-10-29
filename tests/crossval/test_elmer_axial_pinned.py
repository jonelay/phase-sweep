"""Track C pinned-result check — Elmer independent-code verification.

Validates the committed comparison JSON (produced once by
scripts/elmer_axial/run_elmer_axial.py, Elmer-in-Docker) against the
shared CV contract. No Docker and no ngsolve at test time: the contract
references are pure-numpy analytical values, so this runs in the CI
core lane. Regenerating the JSON requires Docker — see
scripts/elmer_axial/README.md.
"""

import json

import pytest

from tests.conftest import ELMER_PINNED_SKIP, REPO_ROOT, requires_fem
from tests.crossval.axial_fem_contract import (
    CV1_ANALYTICAL_REL_TOL,
    CV1_CROSS_CODE_REL_TOL,
    CV2_REL_TOL,
    NONTRIVIAL_FIELD_T,
    circuit_limit_Bz,
    load_parviainen_motor,
    slotless_reference_B1,
)

PINNED = REPO_ROOT / "scripts/elmer_axial/pinned/elmer_axial_cv.json"


@pytest.fixture(scope="module")
def pinned():
    if not PINNED.exists():
        pytest.skip(ELMER_PINNED_SKIP)
    return json.loads(PINNED.read_text())


def test_nontrivial_fields(pinned):
    """Silent-zero guard: every pinned Elmer field is a real solve
    (known Elmer failure modes exit 0 with a trivially zero field)."""
    for name, case in pinned["cases"].items():
        value = case.get("Bz_plateau_T", case.get("B1_T"))
        assert value is not None, f"{name}: no field magnitude in pinned JSON"
        assert value >= NONTRIVIAL_FIELD_T, name


def test_cv2_circuit_limit(pinned):
    """CV-2: Elmer plateau vs the 1D circuit limit (convention check)."""
    motor = load_parviainen_motor()
    geo = motor.geometry
    ref = circuit_limit_Bz(motor.B_rem, motor.mu_r_pm,
                           geo.magnet_thickness, geo.airgap)
    plateau = pinned["cases"]["cv2_circuit"]["Bz_plateau_T"]
    assert abs(plateau - ref) / ref <= CV2_REL_TOL


def test_cv1_slotless_vs_analytical(pinned):
    """CV-1: Elmer slotless B1 at r_mean vs the analytical layer series.

    The 1% cross-code gate is Elmer vs Track A (compared at merge);
    against the ideal-iron analytical series the mu_r_fe=1000 plane
    carries a physical ~0.9% iron-MMF offset, so only the 2% analytical
    gate applies here.
    """
    motor = load_parviainen_motor()
    ref = slotless_reference_B1(motor.geometry, motor.n_p, motor.B_rem,
                                motor.mu_r_pm, motor.alpha_p)
    B1 = pinned["cases"]["cv1_slotless"]["B1_T"]
    assert abs(B1 - ref) / ref <= CV1_ANALYTICAL_REL_TOL


def test_cv1_ideal_iron_diagnostic(pinned):
    """Ideal-iron plane must collapse onto the analytical series (0.5%),
    attributing the CV-1 residual to yoke MMF rather than a code error."""
    motor = load_parviainen_motor()
    ref = slotless_reference_B1(motor.geometry, motor.n_p, motor.B_rem,
                                motor.mu_r_pm, motor.alpha_p)
    B1 = pinned["cases"]["cv1_slotless_ideal_iron"]["B1_T"]
    assert abs(B1 - ref) / ref <= 0.005


@requires_fem
@pytest.mark.parametrize(("case", "mu_r_fe"), [
    ("cv1_slotless", None),                 # the motor's 1000: -0.39 %
    ("cv1_slotless_ideal_iron", 1.0e6),     # -0.001 %
])
def test_cv1_cross_code_ngsolve_vs_elmer(pinned, case, mu_r_fe):
    """The cross-code gate itself: fem_axial's slotless plane against the
    pinned Elmer B1 at the same iron permeability (it was only compared
    at merge, so the published delta could drift unseen)."""
    from dataclasses import replace

    from phasesweep.solvers.fem_axial import fundamental_amplitude, solve_plane

    motor = load_parviainen_motor()
    geo = replace(motor.geometry, n_slots=0, slot_depth=0.0)
    mu = motor.mu_r_fe if mu_r_fe is None else mu_r_fe
    assert pinned["cases"][case].get("params", {}).get("mu_r_fe", mu) == mu
    _, Bz = solve_plane(geo, geo.r_mean, motor.n_p, motor.B_rem,
                        motor.mu_r_pm, mu, motor.alpha_p)
    elmer = pinned["cases"][case]["B1_T"]
    assert abs(fundamental_amplitude(Bz) - elmer) / elmer <= CV1_CROSS_CODE_REL_TOL
