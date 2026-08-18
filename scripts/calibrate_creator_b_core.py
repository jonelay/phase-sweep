"""Re-derive CREATOR's B_core through the calibration framework.

The shipped motors/creator_case_pmsm.toml B_core was back-solved by hand so
p_fe matches the measured no-load loss at the rated point. This script does
the same inversion through calibrate(), starting from the UNFITTED estimate
(FEM per-pole flux), and writes the provenance record the hand calculation
never had.

    uv run python scripts/calibrate_creator_b_core.py

Writes data/creator_case_pmsm/b_core_calibration.record.json. No calibrated
TOML is written: the shipped motor already carries the fitted value, and
write_motor_toml refuses to overwrite it by design.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phasesweep.machines.motor import Motor

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data/creator_case_pmsm"
MOTOR_TOML = REPO_ROOT / "motors/creator_case_pmsm.toml"
DATASET = DATA / "iron_loss_noload.json"
RECORD = DATA / "b_core_calibration.record.json"

# Per-pole flux from a CREATOR fem solve — a SNAPSHOT, not recomputed here.
# A fem version bump that moves the airgap field must refresh this and the
# matching comment in motors/creator_case_pmsm.toml. This is the only code
# definition; tests/test_iron_loss.py imports the function below rather than
# repeating the arithmetic.
PHI_POLE = 188.1e-6


def unfitted_b_core(motor: Motor) -> float:
    """CREATOR's unfitted B_core: PHI_POLE spread over the yoke (2 x 11.6 mm
    x L_stk) and 1.5 teeth per pole (14.8 mm x L_stk), mass-and-alpha-weighted.
    The dimensions are CREATOR's, so this is not a general helper."""
    L, b_yoke_w, tooth_w = 0.0301, 0.0116, 0.0148
    m_yoke, m_teeth = 0.8453, 0.4063
    a = motor.alpha_fe
    B_y = PHI_POLE / (2 * b_yoke_w * L)
    B_t = PHI_POLE / (1.5 * tooth_w * L)
    return float(((m_yoke * B_y**a + m_teeth * B_t**a)
                  / (m_yoke + m_teeth)) ** (1 / a))


def main() -> None:
    from phasesweep.machines.configs import load_motor
    from phasesweep.validation.calibration import calibrate
    from phasesweep.validation.measured import (
        MeasuredResult,
        measured_run_result,
        validate_measured,
    )

    shipped = load_motor(MOTOR_TOML)
    source = replace(shipped, B_core=unfitted_b_core(shipped),
                     name=f"{shipped.name} [B_core unfitted]")

    data = MeasuredResult.from_dict(json.loads(DATASET.read_text()))
    validate_measured(data)
    measured = [measured_run_result(data, source, DATASET.stem)]

    # +82% from the unfitted estimate to the measured loss: the default
    # +/-30% delta bound cannot reach it.
    result = calibrate(
        source, measured, params=["B_core"], quantities=["p_fe"],
        bounds={"B_core": (-0.5, 2.0)},
    )
    record = result.record

    for p in record.params:
        print(f"  {p.param}: {p.initial:.6g} -> {p.final:.6g} ({p.delta:+.2%})")
    for before, after in zip(record.residuals_before, record.residuals_after):
        print(f"  {before['quantity']}: {before['rel_pct']:.2f}% -> "
              f"{after['rel_pct']:.4f}% (tol {after['tol_pct']:.0f}%)")
    for w in record.warnings:
        print(f"  WARNING: {w}")
    print(f"  closed: {record.closed}")
    print(f"  shipped B_core: {shipped.B_core}")

    record.save(RECORD)
    print(f"\nWrote: {RECORD.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
