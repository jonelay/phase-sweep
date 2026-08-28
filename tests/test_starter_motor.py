"""Acceptance test for the hypothetical starter motor.

Protects the starter's regression behavior with deliberately broad ranges.
These are numerical regression bounds on assumed inputs, not physical
accuracy claims; the motor does not exist and must never become a
validation anchor.
"""

from __future__ import annotations

import sys
from pathlib import Path

from phasesweep import load_motor
from phasesweep.registry import MODEL_REGISTRY
from phasesweep.sweep_types import RunConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import check_motor as cm

STARTER = (
    Path(__file__).resolve().parent.parent
    / "motors" / "examples" / "starter_radial_spm.toml"
)


def test_starter_is_disclaimed_hypothetical() -> None:
    text = STARTER.read_text()
    assert "INTENTIONALLY HYPOTHETICAL" in text
    assert "not a measurement" in text


def test_starter_is_geometry_led() -> None:
    motor = load_motor(STARTER)
    assert motor.psi_f is None
    assert motor.B_rem is not None
    assert motor.geometry is not None


def test_starter_model_applicability() -> None:
    motor = load_motor(STARTER)
    for name in ("analytical", "fem", "rated_torque", "stall_torque",
                 "torque_speed"):
        MODEL_REGISTRY[name].validate(motor)  # raises if not applicable


def test_starter_fast_results_in_broad_ranges() -> None:
    motor = load_motor(STARTER)
    analytical = MODEL_REGISTRY["analytical"].fn(
        RunConfig(motor=motor, model="analytical")
    )
    # Broad numerical regression bounds, not physical claims.
    assert 0.9 < analytical["fundamental"] < 1.5
    assert 0.01 < analytical["flux_linkage_peak"] < 0.03
    # Design intent: comfortable back-EMF headroom on the 48 V bus.
    assert 10.0 < analytical["backemf_fundamental"] < 27.7

    rated = MODEL_REGISTRY["rated_torque"].fn(
        RunConfig(motor=motor, model="rated_torque")
    )
    assert 1.0 < rated["tau_mtpa"] < 3.0


def test_starter_checker_status() -> None:
    report = cm.build_report(STARTER)
    assert report.exit_code == 0
    assert report.config_status == "valid"
    assert report.consistency_status == "ok (single source)"
    assert report.execution_status.startswith("runnable")
    assert report.measurement_status == "not assessed"
