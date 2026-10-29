"""Test staleness and attachment rules against portable test vectors.

Vectors ship with the vendored sweepstore
(`phasesweep._sweepstore.load_vectors()`); the contract is
docs/result-store-contract.md.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from phasesweep._sweepstore import attached, load_vectors
from phasesweep.result_store import version_current

VECTORS = load_vectors()


class _FakeModelInfo:
    def __init__(self, version: int):
        self.version = version


def _make_registry(model: str, live_version: int | None) -> dict:
    if live_version is None:
        return {}
    return {model: _FakeModelInfo(live_version)}


@pytest.mark.parametrize(
    "case",
    VECTORS["staleness"]["cases"],
    ids=[c["label"] for c in VECTORS["staleness"]["cases"]],
)
def test_staleness(case):
    source = case["source"] if case["source"] is not None else "computed"
    registry = _make_registry(case["model"], case["live_version"])

    with patch("phasesweep.registry.MODEL_REGISTRY", registry):
        is_current = version_current(case["model"], case["stamped_version"])

    if source != "computed":
        serves = True
    else:
        serves = is_current

    expected = case["expect"] == "serve"
    assert serves == expected, (
        f"{case['label']}: expected {'serve' if expected else 'stale'}, "
        f"got {'serve' if serves else 'stale'}"
    )


@pytest.mark.parametrize(
    "case",
    VECTORS["attachment"]["cases"],
    ids=[c["label"] for c in VECTORS["attachment"]["cases"]],
)
def test_attachment(case):
    result = attached(
        case["source"],
        case["record_config_id"],
        case["record_name"],
        case["target_config_id"],
        case["target_name"],
    )
    expected = case["expect"] == "attached"
    assert result == expected, (
        f"{case['label']}: expected {'attached' if expected else 'detached'}, "
        f"got {'attached' if result else 'detached'}"
    )
