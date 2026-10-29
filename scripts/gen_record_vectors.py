"""Generate phasesweep/_vectors/record_vectors.json from output/results.jsonl.

Picks one real store row per (motor name, model) label, trims metrics to
a handful of scalar keys, and records the expected Motor/RunConfig
round-trip dicts and run id. For labels already in the committed file the
stored row is kept verbatim (a pinned vector must not drift when the
store is repopulated or a motor re-keys) and only its expected round
trips and run id are recomputed; new labels are read from the store.
`--check` regenerates and diffs against the committed file.

Usage:
    uv run python scripts/gen_record_vectors.py [--check]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from phasesweep.machines.motor import Motor
from phasesweep.sweep_types import RunConfig, compute_run_id

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "output" / "results.jsonl"
TARGET = ROOT / "phasesweep" / "_vectors" / "record_vectors.json"

# (label, motor name, model). No drive_sim row: sim_plan needs the sim extra.
CASES: list[tuple[str, str, str]] = [
    ("creator-thermal-duty", "CREATOR Case PMSM", "thermal_duty"),
    ("lrk-outrunner-fem", "FEMM LRK 14p/12s Outrunner", "fem"),
    ("belkhadir-airgap-published", "Belkhadir 22p/24s ER-PMSM", "airgap_flux_test"),
    ("deylami-airgap-published-qualified", "Deylami 8p/12s Cooling Fan", "airgap_flux_test"),
    ("creator-backemf-measured", "CREATOR Case PMSM", "backemf_capture"),
    ("i3-stall-torque-stored-run-id", "BMW i3 2016 IPM", "stall_torque"),
    ("i3-torque-speed-flux-map", "BMW i3 2016 IPM", "torque_speed"),
    ("i3-operating-point-flux-map", "BMW i3 2016 IPM", "operating_point"),
    ("i3-motor-thermal-flux-map", "BMW i3 2016 IPM", "motor_thermal"),
]
MAX_METRIC_KEYS = 5
# Cases whose consumer (heat-sweep's passthrough) reads every scalar.
KEEP_ALL_SCALARS = {"i3-operating-point-flux-map", "i3-motor-thermal-flux-map"}


def _rows() -> list[dict[str, Any]]:
    return [json.loads(line) for line in SOURCE.read_text().splitlines() if line.strip()]


def _pick(rows: list[dict[str, Any]], name: str, model: str, want_flux_map: bool) -> dict[str, Any]:
    matches = [
        r for r in rows
        if r.get("model") == model
        and r["config"]["motor"]["name"] == name
        and ("flux_map" in r["config"]["motor"]) == want_flux_map
    ]
    if not matches:
        raise SystemExit(f"no row for {name!r} / {model!r} in {SOURCE}")
    return matches[-1]


def _trim_metrics(metrics: dict[str, Any] | None, keep_all: bool) -> dict[str, Any] | None:
    if metrics is None:
        return None
    scalars = {k: v for k, v in metrics.items() if not isinstance(v, (list, dict))}
    if keep_all:
        return scalars
    return dict(list(scalars.items())[:MAX_METRIC_KEYS])


def _committed() -> dict[str, dict[str, Any]]:
    if not TARGET.exists():
        return {}
    return {c["label"]: c for c in json.loads(TARGET.read_text())["cases"]}


def build() -> dict[str, Any]:
    rows = _rows()
    committed = _committed()
    cases = []
    for label, name, model in CASES:
        if label in committed:
            row = committed[label]["stored"]
        else:
            row = dict(_pick(rows, name, model, want_flux_map=label.endswith("flux-map")))
            row["metrics"] = _trim_metrics(row.get("metrics"), label in KEEP_ALL_SCALARS)
        rc = RunConfig.from_dict(row["config"])
        run_id = compute_run_id(rc)
        if row.get("run_id", run_id) != run_id:
            raise SystemExit(f"{label}: stored run_id {row['run_id']} != recomputed {run_id}")
        cases.append({
            "label": label,
            "stored": row,
            "expected": {
                "motor": Motor.from_dict(row["config"]["motor"]).to_dict(),
                "config": rc.to_dict(),
                "run_id": run_id,
            },
        })
    return {"description": "Real store rows pinned against phasesweep.record_schema", "cases": cases}


def main(argv: list[str]) -> int:
    text = json.dumps(build(), sort_keys=True, indent=1) + "\n"
    if "--check" in argv:
        current = TARGET.read_text() if TARGET.exists() else ""
        if current != text:
            print(f"{TARGET} is out of date; rerun without --check")
            return 1
        print(f"{TARGET} up to date")
        return 0
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(text)
    print(f"wrote {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
