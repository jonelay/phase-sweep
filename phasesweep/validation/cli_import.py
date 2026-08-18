"""CLI entry point for importing measured data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import measured test data into phase-sweep results",
    )
    parser.add_argument("json_file", type=Path, help="Path to measured JSON file")
    parser.add_argument("--motor-dir", type=Path, default=Path("motors"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    args = parser.parse_args()

    import json

    raw = json.loads(args.json_file.read_text())
    motor_name = raw.get("motor_name", "")
    if not motor_name:
        print("Error: JSON missing motor_name", file=sys.stderr)
        sys.exit(1)

    from phasesweep.machines.configs import load_motors
    motors = load_motors(args.motor_dir)
    motor = motors.get(motor_name)
    if motor is None:
        available = ", ".join(sorted(motors.keys())) or "(none)"
        print(
            f"Error: motor {motor_name!r} not found in {args.motor_dir}. "
            f"Available: {available}",
            file=sys.stderr,
        )
        sys.exit(1)

    from phasesweep.validation.measured import import_measured
    result = import_measured(args.json_file, motor, args.output_dir)
    print(f"Imported: model={result.model}, source={result.source}, "
          f"status={result.status}")
    if result.metrics:
        for k, v in result.metrics.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
