"""Pinned-result checks for the computed k_end records.

Validates the committed provenance records produced by
scripts/compute_k_end.py — no FEM at test time, so this runs in the CI
core lane (pattern: test_elmer_axial_pinned.py). Guards three things:

1. the pinned numeric values (regeneration drift is loud),
2. record ↔ motor-TOML consistency (a TOML k_end must equal the record
   it cites, and the record's geometry config_id must match the current
   TOML geometry — a geometry edit stales the record).
"""

import json

import pytest

from phasesweep.machines.configs import load_motor
from tests.conftest import REPO_ROOT

STEEL_RECORD = REPO_ROOT / "data/outrunner_14mm_steel/k_end.record.json"
STEEL_TOML = REPO_ROOT / "motors/outrunner_14mm_steel.toml"

# Same-machine repeatability is ~1e-15; the band is mesher/library
# headroom (workstation-pinned results, like the Elmer harnesses).
PIN_ABS = 1e-3


@pytest.fixture(scope="module")
def steel():
    if not STEEL_RECORD.exists():
        pytest.skip("steel k_end record not present")
    return json.loads(STEEL_RECORD.read_text())


class TestSteelRecord:

    def test_pinned_values(self, steel):
        r = steel["results"]
        assert r["k_end"] == pytest.approx(0.918486, abs=PIN_ABS)
        assert r["k_end_linear"] == pytest.approx(0.969365, abs=PIN_ABS)
        # Saturation costs ~5 points at this overhang (nonlinear < linear).
        assert r["k_end"] < r["k_end_linear"]
        # B_rem sensitivity: milder saturation at the calibrated-grade
        # B_rem — the headline is B_rem-dependent and says so.
        assert r["k_end_at_B_rem_sens"] == pytest.approx(0.943382,
                                                         abs=PIN_ABS)
        assert r["k_end"] < r["k_end_at_B_rem_sens"] < r["k_end_linear"]

    def test_toml_k_end_matches_record(self, steel):
        motor = load_motor(STEEL_TOML)
        assert motor.k_end == pytest.approx(steel["results"]["k_end"],
                                            abs=1e-6)

    def test_record_geometry_current(self, steel):
        # A geometry edit (radii, overhang) stales the record: recompute
        # with scripts/compute_k_end.py rather than loosening this.
        motor = load_motor(STEEL_TOML)
        assert steel["geometry_config_id"] == motor.geometry.config_id
        assert steel["magnet_overhang_m"] == pytest.approx(
            motor.geometry.magnet_overhang)
        assert steel["L_stk_m"] == pytest.approx(motor.L_stk)
        # n_p is not part of Geometry.config_id — check it explicitly
        assert steel["n_p"] == motor.n_p
