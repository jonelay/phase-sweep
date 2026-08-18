"""CLI entry point for per-motor parameter calibration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parse_bound(spec: str) -> tuple[str, tuple[float, float]]:
    try:
        param, rng = spec.split("=", 1)
        lo_s, hi_s = rng.split(":", 1)
        lo, hi = float(lo_s), float(hi_s)
    except ValueError:
        raise SystemExit(
            f"--bound expects PARAM=LO:HI (fractional deltas), got {spec!r}"
        ) from None
    if not lo < hi:
        raise SystemExit(f"--bound {spec!r}: lower bound must be < upper")
    return param, (lo, hi)


def _residual_pairs(record) -> list[tuple[dict, dict | None]]:
    """Pair before/after residual rows by key — residuals_after can be
    shorter than residuals_before, so a plain zip would misalign."""
    def key(r: dict) -> tuple:
        return (r["dataset"], r["quantity"],
                r["measured_model"], r["computed_model"])
    after = {key(r): r for r in record.residuals_after}
    return [(b, after.get(key(b))) for b in record.residuals_before]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit selected Motor parameters against measured/published "
                    "datasets; write a calibrated TOML + provenance record",
    )
    parser.add_argument("motor_toml", type=Path,
                        help="Source motor TOML (never overwritten)")
    parser.add_argument("--data", type=Path, action="append", required=True,
                        help="Measured dataset JSON (import format); repeatable")
    parser.add_argument("--params", action="append", required=True,
                        help="Parameter to fit (repeatable): e.g. B_rem, psi_f")
    parser.add_argument("--quantities", action="append", required=True,
                        help="Measured quantity to fit against (repeatable)")
    parser.add_argument("--models", action="append", default=None,
                        help="Computed model(s) to run in the loop "
                             "(default: auto-select fast models)")
    parser.add_argument("--bound", action="append", default=[],
                        metavar="PARAM=LO:HI",
                        help="Fractional-delta bounds, e.g. B_rem=-0.3:0.1 "
                             "(default ±0.3)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Calibrated TOML path (default: "
                             "<motor>_calibrated.toml next to the source)")
    args = parser.parse_args()

    from phasesweep.machines.configs import load_motor
    from phasesweep.validation.calibration import calibrate, write_motor_toml
    from phasesweep.validation.measured import (
        MeasuredResult,
        measured_run_result,
        validate_measured,
    )

    try:
        motor = load_motor(args.motor_toml)
    except (OSError, ValueError) as e:
        print(f"Error loading motor: {e}", file=sys.stderr)
        sys.exit(1)

    measured = []
    for path in args.data:
        try:
            data = MeasuredResult.from_dict(json.loads(path.read_text()))
            validate_measured(data)
        except (OSError, KeyError, ValueError) as e:
            print(f"Error loading {path}: {e}", file=sys.stderr)
            sys.exit(1)
        measured.append(measured_run_result(data, motor, path.stem))

    out = args.out or args.motor_toml.with_name(
        f"{args.motor_toml.stem}_calibrated.toml")
    if out.resolve() == args.motor_toml.resolve():
        print("Error: --out must not be the source TOML", file=sys.stderr)
        sys.exit(1)

    bounds = dict(_parse_bound(s) for s in args.bound)
    try:
        result = calibrate(
            motor, measured,
            params=args.params, quantities=args.quantities,
            models=args.models, bounds=bounds or None,
        )
    except ValueError as e:
        print(f"Calibration refused: {e}", file=sys.stderr)
        sys.exit(1)

    record = result.record
    print(f"=== Calibration: {motor.name} ===\n")
    for p in record.params:
        stderr = f" ± {p.stderr:.4g}" if p.stderr is not None else ""
        print(f"  {p.param}: {p.initial:.6g} -> {p.final:.6g}{stderr} "
              f"({p.delta:+.2%})")
    print("\nResiduals (before -> after):")
    for before, after in _residual_pairs(record):
        if after is None:
            print(f"  {before['quantity']:<25} {before['rel_pct']:6.2f}% -> "
                  f"(no comparison row after fit)")
            continue
        print(f"  {before['quantity']:<25} {before['rel_pct']:6.2f}% -> "
              f"{after['rel_pct']:6.2f}%  (tol {after['tol_pct']:.0f}%, "
              f"{'PASS' if after['passed'] else 'FAIL'})")
    for w in record.warnings:
        print(f"  WARNING: {w}")
    print(f"\nClosed within tolerance: {record.closed}")

    try:
        write_motor_toml(
            motor=result.motor, path=out,
            header=(
                f"Calibrated from {args.motor_toml.name} "
                f"(source config_id {record.source_config_id})\n"
                f"Fitted {', '.join(p.param for p in record.params)} against "
                f"{', '.join(record.dataset_ids)} — see {out.stem}.record.json"
            ),
        )
    except FileExistsError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    record_path = out.with_suffix(".record.json")
    record.save(record_path)
    print(f"\nWrote: {out}")
    print(f"Wrote: {record_path}")
    if not record.closed:
        sys.exit(2)


if __name__ == "__main__":
    main()
