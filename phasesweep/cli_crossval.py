"""CLI entry point for cross-validation comparison."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare computed and measured results for a motor",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--motor", type=str, default=None,
                        help="Filter by motor name (substring match)")
    parser.add_argument("--plot", action="store_true",
                        help="Generate overlay plots")
    args = parser.parse_args()

    from phasesweep.crossval import compare_all, diagnose, format_table
    from phasesweep.result_store import ResultStore

    store = ResultStore(args.output_dir)
    results = store.load_results()
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
    groups: dict[str, list] = {}
    for r in results:
        groups.setdefault(r.motor_config_id, []).append(r)

    for mcid, group in groups.items():
        motor_name = group[0].config.motor.name
        models = sorted({r.model for r in group})
        print(f"\n=== {motor_name} ({len(group)} results: {', '.join(models)}) ===\n")

        rows = compare_all(group)
        print(format_table(rows))

        diag = diagnose(group)
        print(f"\nDiagnosis: {diag}\n")

    if args.plot:
        _plot_overlays(groups, args.output_dir)


def _plot_overlays(
    groups: dict[str, list],
    output_dir: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for mcid, group in groups.items():
        # Overlay waveforms (theta_list + B_r_list) if available
        waveform_results = [
            r for r in group
            if r.metrics and "theta_list" in r.metrics and "B_r_list" in r.metrics
        ]
        if len(waveform_results) < 2:
            continue

        motor_name = group[0].config.motor.name
        fig, ax = plt.subplots(figsize=(10, 5))
        for r in waveform_results:
            label = f"{r.model} ({r.source})"
            ax.plot(r.metrics["theta_list"], r.metrics["B_r_list"], label=label)
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
