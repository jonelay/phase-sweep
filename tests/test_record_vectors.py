"""Pin stored-record shape against phasesweep.record_schema.

Vectors are real store rows (scripts/gen_record_vectors.py); each must
round-trip through Motor/RunConfig and hash to the recorded run id, and
the TypedDict key sets must match what to_dict actually writes.
"""
from __future__ import annotations

import pytest

from phasesweep.machines.motor import _OPTIONAL_SCALARS, Motor
from phasesweep.record_schema import (
    MOTOR_COMPOUND_KEYS,
    MotorDict,
    RunConfigDict,
    RunResultDict,
    StoredRecord,
    load_record_vectors,
)
from phasesweep.sweep_types import RunConfig, RunResult, compute_run_id

CASES = load_record_vectors()["cases"]
CASE_IDS = [c["label"] for c in CASES]


@pytest.fixture(params=CASES, ids=CASE_IDS)
def case(request):
    return request.param


def test_motor_round_trip(case):
    stored = case["stored"]["config"]["motor"]
    assert Motor.from_dict(stored).to_dict() == case["expected"]["motor"]


def test_config_round_trip(case):
    assert RunConfig.from_dict(case["stored"]["config"]).to_dict() == case["expected"]["config"]


def test_run_id_matches(case):
    rc = RunConfig.from_dict(case["stored"]["config"])
    expected = case["expected"]["run_id"]
    assert compute_run_id(rc) == expected
    assert case["stored"].get("run_id", expected) == expected


def test_required_keys_present(case):
    assert MotorDict.__required_keys__ <= set(case["expected"]["motor"])
    assert RunConfigDict.__required_keys__ <= set(case["expected"]["config"])
    # Pre-stamp rows lack model_version on disk (the store treats them as
    # stale); the round trip through RunResult always writes it.
    assert RunResultDict.__required_keys__ - {"model_version"} <= set(case["stored"])
    assert RunResultDict.__required_keys__ <= set(RunResult.from_dict(case["stored"]).to_dict())


def test_no_undeclared_keys(case):
    assert set(case["expected"]["motor"]) <= set(MotorDict.__annotations__)
    assert set(case["expected"]["config"]) <= set(RunConfigDict.__annotations__)
    assert set(case["stored"]) <= set(StoredRecord.__annotations__)


def test_vectors_cover_sources_and_run_id():
    sources = {c["stored"]["source"] for c in CASES}
    assert {"computed", "measured", "published"} <= sources
    assert any(" " in s for s in sources), "need a provenance-qualified source"
    assert any("run_id" in c["stored"] for c in CASES)
    assert any("tolerances" in c["stored"] for c in CASES)
    assert any("flux_map" in c["stored"]["config"]["motor"] for c in CASES)


def test_to_dict_optional_tuple_matches_schema():
    assert set(_OPTIONAL_SCALARS) == MotorDict.__optional_keys__ - MOTOR_COMPOUND_KEYS
