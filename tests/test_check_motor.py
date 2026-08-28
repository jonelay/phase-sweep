"""Contract tests for scripts/check_motor.py status semantics.

These protect the evidence-level wording and objective-failure boundary:
solver agreement must never read as validation, and a psi_f/B_rem conflict
must warn without invalidating the config.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import check_motor as cm

MOTORS_DIR = Path(__file__).resolve().parent.parent / "motors"


def test_malformed_toml_is_objective_failure(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text("[circuit\nn_p = 4\n")
    report = cm.build_report(bad)
    assert report.config_status == "INVALID"
    assert report.exit_code == 1


def test_impossible_radii_is_objective_failure(tmp_path: Path) -> None:
    src = (MOTORS_DIR / "outrunner_14mm_steel.toml").read_text()
    bad = tmp_path / "radii.toml"
    bad.write_text(src.replace("r_outer = 0.009665", "r_outer = 0.001"))
    report = cm.build_report(bad)
    assert report.config_status == "INVALID"
    assert report.exit_code == 1


def test_circuit_only_motor_is_runnable_without_field_models() -> None:
    report = cm.build_report(MOTORS_DIR / "prius_2004.toml")
    assert report.exit_code == 0
    assert report.config_status == "valid"
    assert report.execution_status.startswith("runnable")
    assert report.crosscheck_status == "not run"
    text = "\n".join(report.lines)
    assert "skip analytical" in text


def test_conflict_warns_but_does_not_invalidate(tmp_path: Path) -> None:
    src = (MOTORS_DIR / "outrunner_14mm_steel.toml").read_text()
    conflicted = tmp_path / "conflict.toml"
    conflicted.write_text(
        src.replace(
            "# psi_f omitted — derive from B_rem via winding params",
            "psi_f = 0.02",
        )
    )
    report = cm.build_report(conflicted)
    assert report.exit_code == 0
    assert report.config_status == "valid"
    assert report.consistency_status.startswith("warning")
    text = "\n".join(report.lines)
    assert "Geometry-led:" in text
    assert "Measurement-led:" in text


@pytest.mark.parametrize(
    "toml", ["prius_2004.toml", "outrunner_14mm_steel.toml"]
)
def test_no_false_validation_wording(toml: str) -> None:
    report = cm.build_report(MOTORS_DIR / toml)
    text = "\n".join(report.lines)
    assert cm.OVERALL_LINE in text
    assert report.measurement_status == "not assessed"
    assert "valid motor" not in text.lower()
    assert "MEASUREMENT: not assessed" in text


def test_status_dimensions_are_stable() -> None:
    report = cm.build_report(MOTORS_DIR / "prius_2004.toml")
    text = "\n".join(report.lines)
    for dim in ("CONFIG:", "EXECUTION:", "CONSISTENCY:",
                "CROSS-CHECK:", "MEASUREMENT:"):
        assert dim in text
    assert cm.OVERALL_LINE == (
        "Overall: computationally checked; NOT measurement-validated."
    )


def test_runtime_input_model_is_applicable_but_not_executed() -> None:
    report = cm.build_report(MOTORS_DIR / "outrunner_14mm_steel.toml")
    text = "\n".join(report.lines)
    assert (
        "thermal_duty: applicable but needs run-time input(s): "
        "duty_profile; not executed by this check"
    ) in text
    assert report.execution_status == "runnable (4/4 fast models)"
