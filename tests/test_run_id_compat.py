"""Run-ID compatibility vectors — two-sided gate for the sweepstore migration.

Side A: compute_run_id(rc) == expected_run_id (exercises phase-sweep's codec).
Side B: md5(hash_string)[:12] == expected_run_id (exercises the hash primitive,
        decoupled from domain types).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from phasesweep import RunConfig, compute_run_id, load_motor

VECTORS = json.loads((Path(__file__).parent / "run_id_vectors.json").read_text())
MOTOR = load_motor(Path(__file__).parent.parent / VECTORS["_motor_toml"])


class TestRunIdCompatVectors:
    @pytest.mark.parametrize(
        "case", VECTORS["cases"],
        ids=[c["label"] for c in VECTORS["cases"]],
    )
    def test_hash_string_reproduces_run_id(self, case):
        got = hashlib.md5(case["hash_string"].encode()).hexdigest()[:12]
        assert got == case["expected_run_id"], (
            f"{case['label']}: md5({case['hash_string']!r})[:12] = {got}, "
            f"expected {case['expected_run_id']}"
        )

    @pytest.mark.parametrize(
        "case", VECTORS["cases"],
        ids=[c["label"] for c in VECTORS["cases"]],
    )
    def test_compute_run_id_matches_pinned(self, case):
        kw: dict = {"model": case["model"]}
        if case.get("dataset_id"):
            kw["dataset_id"] = case["dataset_id"]
        if "rotation" in case.get("hash_string", "") and "rotation=" in case["hash_string"]:
            kw["rotation"] = 0.1
        if "nonlinear_solver=picard" in case.get("hash_string", ""):
            kw["nonlinear"] = True
            kw["nonlinear_solver"] = "picard"
        if "cogging_points=24" in case.get("hash_string", ""):
            kw["cogging_points"] = 24
        if "n_planes=5" in case.get("hash_string", ""):
            kw["n_planes"] = 5
        if "slip=" in case.get("hash_string", ""):
            kw["slip"] = 0.03
        if "i_fault=250" in case.get("hash_string", ""):
            kw["i_fault"] = 250.0
            kw["nonlinear"] = True
        if "duty_profile=" in case.get("hash_string", ""):
            kw["duty_profile"] = ((100.0, 60.0), (50.0, 120.0))
        if "tau_ref=" in case.get("hash_string", ""):
            kw["w_mech"] = 502.6548245743669
            kw["tau_ref"] = 125.0
        rc = RunConfig(motor=MOTOR, **kw)
        got = compute_run_id(rc)
        assert got == case["expected_run_id"], (
            f"{case['label']}: compute_run_id gave {got}, "
            f"expected {case['expected_run_id']}"
        )


class TestRunIdVectorFileIntegrity:
    def test_minimum_case_count(self):
        assert len(VECTORS["cases"]) >= 11

    def test_all_cases_have_hash_string(self):
        for c in VECTORS["cases"]:
            assert "hash_string" in c, f"{c['label']}: missing hash_string"

    def test_motor_config_id_pinned(self):
        assert MOTOR.config_id == VECTORS["_motor_config_id"]

    def test_unique_labels(self):
        labels = [c["label"] for c in VECTORS["cases"]]
        assert len(labels) == len(set(labels))

    def test_consistent_ids(self):
        by_id: dict[str, str] = {}
        for c in VECTORS["cases"]:
            rid = c["expected_run_id"]
            hs = c["hash_string"]
            if rid in by_id:
                assert by_id[rid] == hs, (
                    f"same run_id {rid} but different hash strings"
                )
            by_id[rid] = hs
