"""Dataset-integrity tests: JMT 1806 2400KV back-EMF sweep.

12N14P outrunner, previously reachable only via scripts/analyze_jmt_1806.py.
These tests pin the distilled quantities, the peak/phase-to-Y conventions,
and the catalog-KV cross-check so the dataset can't drift silently.

Model cross-validation is NOT possible yet: the teardown yielded the
turn count but no radial
geometry (gap, magnet thickness, rotor radii), so there is no motors/ TOML
to run the field models against. Blocked on caliper measurements.
"""

from __future__ import annotations

import json
import math
import tomllib

import pytest

from tests.conftest import REPO_ROOT

DATA_DIR = REPO_ROOT / "data" / "jmt_1806"


@pytest.fixture(scope="module")
def sweep():
    with open(DATA_DIR / "backemf_speed_sweep.json") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def reference():
    with open(DATA_DIR / "reference.toml", "rb") as f:
        return tomllib.load(f)


class TestJmt1806Quantities:

    def test_ke_psi_kt_self_consistent(self, sweep):
        """psi_f = Ke/n_p (peak, phase-to-Y) and Kt = 1.5*n_p*psi_f."""
        q = sweep["quantities"]
        n_p = q["n_p"]
        assert q["psi_f"] == pytest.approx(q["Ke_pk"] / n_p, rel=1e-3)
        assert q["Kt"] == pytest.approx(1.5 * n_p * q["psi_f"], rel=1e-3)

    def test_kv_vendor_convention(self, sweep):
        """kv = 60/(2*pi*Ke_LL_pk) with Ke_LL_pk = sqrt(3)*Ke_phase_pk."""
        q = sweep["quantities"]
        kv = 60.0 / (2 * math.pi * math.sqrt(3) * q["Ke_pk"])
        assert q["kv"] == pytest.approx(kv, rel=1e-3)

    def test_kv_matches_catalog(self, sweep, reference):
        """Measured KV within 1% of the catalog 2400 — independently
        confirms the phase-to-Y probing interpretation."""
        kv_catalog = reference["electrical"]["kv_catalog"]
        assert sweep["quantities"]["kv"] == pytest.approx(kv_catalog, rel=0.01)

    def test_reference_toml_matches_sweep_json(self, sweep, reference):
        """Distilled TOML electricals agree with the sweep JSON."""
        q = sweep["quantities"]
        e = reference["electrical"]
        assert e["psi_f_uWb"] * 1e-6 == pytest.approx(q["psi_f"], rel=1e-3)
        assert e["Ke_mV_per_rad_s"] * 1e-3 == pytest.approx(q["Ke_pk"], rel=1e-3)
        assert e["Kt_mNm_per_A"] * 1e-3 == pytest.approx(q["Kt"], rel=1e-3)
        assert e["kv"] == pytest.approx(q["kv"], rel=1e-3)
        assert reference["circuit"]["n_p"] == q["n_p"]

    def test_r_s_is_half_line_to_line(self, reference):
        """Wye: phase resistance = R_LL/2."""
        e = reference["electrical"]
        assert e["R_s_phase_mOhm"] == pytest.approx(e["R_LL_mOhm"] / 2, rel=0.02)


class TestJmt1806SpeedSweep:

    def test_fit_quality(self, sweep):
        assert sweep["fit"]["R_squared"] > 0.999

    def test_pole_pairs_from_electrical_frequency(self, sweep):
        """n_p falls out of f_elec/mech_rps on the non-slipping files."""
        n_p = sweep["quantities"]["n_p"]
        clean = [p for p in sweep["speed_sweep"] if abs(p["slip_pct"]) < 1.0]
        assert len(clean) >= 3
        for p in clean:
            assert round(p["f_elec_Hz"] / p["mech_rps"]) == n_p

    def test_per_point_ke_internal_consistency(self, sweep):
        """Each row's Ke equals V_pk / (2*pi*mech_rps)."""
        for p in sweep["speed_sweep"]:
            ke = p["V_pk_LN_V"] / (2 * math.pi * p["mech_rps"])
            assert p["Ke_mV_per_rads"] * 1e-3 == pytest.approx(ke, rel=2e-3), (
                f"{p['label_rps']} RPS row inconsistent"
            )

    def test_ke_constant_above_snr_floor(self, sweep):
        """Ke flat to ~0.5% for >=30 RPS (20 RPS reads low on SNR)."""
        ke_fit = sweep["quantities"]["Ke_pk"]
        for p in sweep["speed_sweep"]:
            if p["label_rps"] < 30:
                continue
            assert p["Ke_mV_per_rads"] * 1e-3 == pytest.approx(ke_fit, rel=5e-3)

    def test_slip_recovery_kept_heavy_files(self, sweep):
        """Slipping files (labels read high) are retained with recovered
        mech RPS, not dropped — the fit uses f_elec/n_p."""
        slipped = [p for p in sweep["speed_sweep"] if p["slip_pct"] < -2.0]
        assert slipped, "expected backdriver slip on the heavier files"
        for p in slipped:
            assert p["mech_rps"] < p["label_rps"]
