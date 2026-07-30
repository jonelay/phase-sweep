"""CLI entry point for cross-validation comparison."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phasesweep.sweep_types import RunResult


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare computed and measured results for a motor",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--motor", type=str, default=None,
                        help="Filter by motor name (substring match)")
    parser.add_argument("--plot", action="store_true",
                        help="Generate overlay plots")
    parser.add_argument("--strict", action="store_true",
                        help="Exit 1 unless every comparison passes "
                             "(CI regression gate)")
    args = parser.parse_args()

    from phasesweep.crossval import (
        compare_all,
        diagnose,
        diagnose_detailed,
        format_table,
    )
    from phasesweep.result_store import ResultStore, version_current

    store = ResultStore(args.output_dir)
    # H1 staleness rule (same as load_slim / _dedupe_for_matrix): computed
    # records with a superseded model version must not reach PASS/FAIL or
    # the --strict exit code; measured/published describe hardware, not
    # physics code, and always serve.
    results = [r for r in store.load_results()
               if r.source != "computed"
               or version_current(r.model, r.model_version)]
    if not results:
        print(f"No results found in {args.output_dir}", file=sys.stderr)
        sys.exit(1)

    if args.motor:
        results = [r for r in results
                   if args.motor.lower() in r.config.motor.name.lower()]
        if not results:
            print(f"No results matching motor {args.motor!r}", file=sys.stderr)
            sys.exit(1)

    # Group by motor_config_id
    groups: dict[str, list[RunResult]] = {}
    for r in results:
        groups.setdefault(r.motor_config_id, []).append(r)

    all_ok = True
    for group in groups.values():
        motor_name = group[0].config.motor.name
        models = sorted({r.model for r in group})
        print(f"\n=== {motor_name} ({len(group)} results: {', '.join(models)}) ===\n")

        rows = compare_all(group)
        print(format_table(rows))

        diag = diagnose(group)
        print(f"\nDiagnosis: {diag}\n")
        if args.strict and not diagnose_detailed(group).all_pass:
            all_ok = False

    if args.plot:
        _plot_overlays(groups, args.output_dir)

    if args.strict and not all_ok:
        print("STRICT: at least one comparison failed", file=sys.stderr)
        sys.exit(1)


def _plot_overlays(
    groups: dict[str, list[RunResult]],
    output_dir: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for group in groups.values():
        # Overlay waveforms (theta_list + B_r_list) if available
        waveform_results = [
            (r, r.metrics) for r in group
            if r.metrics and "theta_list" in r.metrics and "B_r_list" in r.metrics
        ]
        if len(waveform_results) < 2:
            continue

        motor_name = group[0].config.motor.name
        fig, ax = plt.subplots(figsize=(10, 5))
        for r, metrics in waveform_results:
            label = f"{r.model} ({r.source})"
            ax.plot(metrics["theta_list"], metrics["B_r_list"], label=label)
        ax.set_xlabel("theta (rad)")
        ax.set_ylabel("B_r (T)")
        ax.set_title(f"Waveform Overlay — {motor_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        safe_name = motor_name.replace(" ", "_").replace(":", "")
        fig.savefig(output_dir / f"crossval_{safe_name}.png", dpi=150)
        plt.close(fig)
        print(f"Saved: crossval_{safe_name}.png")


if __name__ == "__main__":
    main()
