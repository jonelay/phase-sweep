"""Populate the output store so the dashboard has results on cold start.

Runs fast models (analytical, rated_torque, stall_torque, thermal_duty,
torque_speed, iron_loss) + FEM for every motor in motors/. Skips models
the motor can't run (missing R_s, L_d, etc.) and configs already in the
store at the current model version. Purges version-stale computed
records first.

--verify recomputes the cached records and compares, catching the
staleness a version stamp cannot see: a record written by code that has
since changed without a bump, or by a bad batch. Read-only unless
--prune-mismatched, which drops the failures so a later run recomputes.

As a physics-change gate, run --verify *before* bumping any version:
a mismatch means the edit moved outputs (bump required), a clean pass
substantiates an output-identity claim. Pair it with --require-model so
a store that holds no record for the edited model fails loudly instead
of passing on an empty check.

Usage:
    uv run python scripts/populate_cache.py [--output-dir DIR] [--skip-fem]
                                            [--purge-only] [--workers N]
    uv run python scripts/populate_cache.py --verify [--verify-fem]
                                            [--require-model MODEL]
                                            [--prune-mismatched]
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

FAST_MODELS = ["analytical", "rated_torque", "stall_torque",
               "thermal_duty", "torque_speed", "iron_loss"]
ALL_MODELS = [*FAST_MODELS, "fem"]


def _purge_stale(output_dir: Path) -> int:
    """Remove computed records whose model version is superseded.

    Rewrites results.jsonl in place (atomic rename) and rebuilds the
    index. Measured/published records always survive.
    """
    from phasesweep.result_store import ResultStore, version_current

    store = ResultStore(output_dir)
    raw = store.load_all()
    if not raw:
        return 0

    kept = []
    purged = 0
    for d in raw:
        source = d.get("source", "computed")
        model = d.get("model", d.get("config", {}).get("model", ""))
        ver = d.get("model_version")
        if source == "computed" and not version_current(model, ver):
            purged += 1
        else:
            kept.append(d)

    if purged == 0:
        return 0

    _rewrite_store(output_dir, kept)
    return purged


def _rewrite_store(output_dir: Path, kept: list[dict]) -> None:
    """Replace results.jsonl with `kept` (atomic rename) and rebuild the
    index from the survivors."""
    from phasesweep.result_store import ResultStore, compute_run_id
    from phasesweep.sweep_types import RunConfig

    tmp = output_dir / "results.jsonl.tmp"
    with open(tmp, "w") as f:
        for d in kept:
            f.write(json.dumps(d, default=str) + "\n")
    tmp.replace(output_dir / "results.jsonl")

    index = {}
    for d in kept:
        try:
            rc = RunConfig.from_dict(d["config"])
            index[compute_run_id(rc)] = d.get("status", "ERROR")
        except (KeyError, ValueError, TypeError):
            pass
    store = ResultStore(output_dir)
    store._write_index(index)
    store._index_cache = None


def _flatten(v: Any) -> Any:
    if isinstance(v, list):
        for x in v:
            yield from _flatten(x)
    else:
        yield v


def _metric_diff(stored: Any, fresh: Any, rtol: float) -> str | None:
    """Describe how two metric values differ, or None if they agree.

    Numbers compare on relative deviation (the worst one in a list is the
    one reported); everything else compares by equality.
    """
    s, f = list(_flatten(stored)), list(_flatten(fresh))
    if len(s) != len(f):
        return f"length {len(s)} != {len(f)}"

    worst = 0.0
    worst_pair: tuple[float, float] | None = None
    for a, b in zip(s, f):
        numeric = (isinstance(a, (int, float)) and isinstance(b, (int, float))
                   and not isinstance(a, bool) and not isinstance(b, bool))
        if not numeric:
            if a != b:
                return f"{a!r} != {b!r}"
            continue
        if math.isnan(a) and math.isnan(b):
            continue
        scale = max(abs(a), abs(b))
        rel = 0.0 if scale == 0 else abs(a - b) / scale
        if rel > worst:
            worst, worst_pair = rel, (a, b)

    if worst <= rtol or worst_pair is None:
        return None
    a, b = worst_pair
    return f"{a:.6g} != {b:.6g} (rel {worst:.2e})"


def _verify(output_dir: Path, rtol: float, *,
            include_fem: bool) -> tuple[set[int], dict[str, int]]:
    """Recompute stored records and compare against what is on disk.

    Catches the staleness a version stamp cannot see: a record written by
    code that has since changed without a version bump, or by a bad batch.
    Only current-version computed records are checked — superseded ones
    are _purge_stale's job, and measured/published records have nothing to
    recompute from.

    Returns the positions (in load_all order) of the records that failed,
    and how many records were checked per model — a model with zero
    records was not tested at all, which must not read as a pass.
    """
    from phasesweep.registry import MODEL_REGISTRY
    from phasesweep.result_store import ResultStore, version_current
    from phasesweep.sweep_types import RunConfig

    log = logging.getLogger("populate")
    models = set(ALL_MODELS if include_fem else FAST_MODELS)

    checked = 0
    by_model: dict[str, int] = {}
    bad: set[int] = set()
    for i, d in enumerate(ResultStore(output_dir).load_all()):
        if d.get("source", "computed") != "computed" or d.get("status") != "OK":
            continue
        model = d.get("model", d.get("config", {}).get("model", ""))
        if model not in models or not version_current(model, d.get("model_version")):
            continue

        name = d.get("config", {}).get("motor", {}).get("name", "?")
        label = f"{name} / {model}"
        info = MODEL_REGISTRY[model]
        checked += 1
        by_model[model] = by_model.get(model, 0) + 1
        try:
            rc = RunConfig.from_dict(d["config"])
            if info.fn is None:
                raise ValueError(f"model {model!r} has no fn")
            fresh = info.fn(rc)
        except Exception as e:
            bad.add(i)
            log.warning("  RECOMPUTE-FAILED %-45s %s", label, e)
            continue

        stored = d.get("metrics") or {}
        diffs = []
        for key in sorted(set(stored) | set(fresh)):
            if key not in stored or key not in fresh:
                diffs.append(f"{key}: {'missing in store' if key not in stored else 'no longer produced'}")
                continue
            reason = _metric_diff(stored[key], fresh[key], rtol)
            if reason:
                diffs.append(f"{key}: {reason}")

        if diffs:
            bad.add(i)
            log.warning("  MISMATCH %-45s", label)
            for line in diffs:
                log.warning("      %s", line)
        else:
            log.info("  ok       %-45s", label)

    log.info("verified %d record(s): %d match, %d mismatched",
             checked, checked - len(bad), len(bad))
    for m in sorted(by_model):
        log.info("  %-18s %d record(s)", m, by_model[m])
    return bad, by_model


def _default_duty_profile(motor: Any) -> tuple[tuple[float, float], ...]:
    """S1 continuous duty at rated torque (60 s cycle).

    The rated_torque model must validate first, so I_rated and psi_f are
    guaranteed present.
    """
    from phasesweep.rated_torque import magnet_torque_constant
    from phasesweep.solver_params import prepare_rated_torque
    p = prepare_rated_torque(motor)
    k_t = magnet_torque_constant(motor.n_p, p.psi_f)
    tau_rated = k_t * motor.I_rated
    return ((tau_rated, 60.0),)


def _existing_ids(output_dir: Path) -> set[str]:
    from phasesweep.result_store import ResultStore
    return ResultStore(output_dir).get_known_ids()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    parser.add_argument("--motors-dir", type=Path, default=ROOT / "motors")
    parser.add_argument("--skip-fem", action="store_true",
                        help="only run fast models, skip FEM")
    parser.add_argument("--purge-only", action="store_true",
                        help="purge stale records and exit")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true",
                        help="show what would run without executing")
    parser.add_argument("--verify", action="store_true",
                        help="recompute cached records and compare; exit 1 on mismatch")
    parser.add_argument("--verify-fem", action="store_true",
                        help="include FEM in --verify (slow)")
    parser.add_argument("--rtol", type=float, default=1e-9,
                        help="relative tolerance for --verify (default 1e-9)")
    parser.add_argument("--prune-mismatched", action="store_true",
                        help="with --verify: drop the records that failed, so a "
                             "later populate run recomputes them")
    parser.add_argument("--require-model", action="append", default=[],
                        metavar="MODEL",
                        help="with --verify: fail unless at least one record was "
                             "checked for MODEL (repeatable, or comma-separated). "
                             "A model the store does not cover verifies vacuously")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("populate")

    # --- verify (read-only unless --prune-mismatched) ---
    if args.verify:
        required = [m.strip() for arg in args.require_model
                    for m in arg.split(",") if m.strip()]
        unknown = [m for m in required if m not in ALL_MODELS]
        if unknown:
            log.error("--require-model: %s not checked by --verify (coverable: %s)",
                      ", ".join(unknown), ", ".join(ALL_MODELS))
            sys.exit(1)
        include_fem = args.verify_fem or "fem" in required
        if include_fem and not args.verify_fem:
            log.info("--require-model fem implies --verify-fem")

        bad, by_model = _verify(args.output_dir, args.rtol, include_fem=include_fem)
        missing = [m for m in required if not by_model.get(m)]
        if missing:
            log.error("no records verified for %s — the gate proved nothing; "
                      "populate the store for that model first",
                      ", ".join(missing))

        if bad and args.prune_mismatched:
            from phasesweep.result_store import ResultStore
            raw = ResultStore(args.output_dir).load_all()
            _rewrite_store(args.output_dir,
                           [d for i, d in enumerate(raw) if i not in bad])
            log.info("pruned %d mismatched record(s) — re-run to recompute", len(bad))
            bad = set()
        if bad or missing:
            sys.exit(1)
        return

    # --- purge stale ---
    n_purged = _purge_stale(args.output_dir)
    if n_purged:
        log.info("purged %d stale record(s)", n_purged)
    else:
        log.info("no stale records to purge")

    if args.purge_only:
        return

    # --- discover motors ---
    from phasesweep.configs import load_motor
    from phasesweep.registry import MODEL_REGISTRY
    from phasesweep.result_store import compute_run_id
    from phasesweep.sweep_types import RunConfig

    motor_paths = sorted(args.motors_dir.glob("*.toml"))
    if not motor_paths:
        log.error("no .toml files in %s", args.motors_dir)
        sys.exit(1)

    models = FAST_MODELS if args.skip_fem else ALL_MODELS
    existing = _existing_ids(args.output_dir)

    # --- build job list ---
    jobs: list[dict] = []
    for mp in motor_paths:
        motor = load_motor(mp)
        for model_key in models:
            info = MODEL_REGISTRY[model_key]
            try:
                info.validate(motor)
            except (ValueError, TypeError):
                log.info("  skip %-35s %-18s (missing fields)", motor.name, model_key)
                continue

            config_kw: dict[str, Any] = {}
            if model_key == "thermal_duty":
                config_kw["duty_profile"] = _default_duty_profile(motor)

            rc = RunConfig(motor=motor, model=model_key, **config_kw)
            rid = compute_run_id(rc)
            if rid in existing:
                log.info("  skip %-35s %-18s (cached)", motor.name, model_key)
                continue

            jobs.append({
                "motor_dict": motor.to_dict(),
                "model_key": model_key,
                "config_kw": config_kw,
                "run_id": rid,
                "_label": f"{motor.name} / {model_key}",
            })

    if not jobs:
        log.info("nothing to run — cache is fully populated")
        return

    log.info("%d job(s) to run:", len(jobs))
    for j in jobs:
        log.info("  %s", j["_label"])

    if args.dry_run:
        return

    # --- execute ---
    from phasesweep.motor import Motor
    from phasesweep.parallel import execute_parallel
    from phasesweep.result_store import ResultStore

    store = ResultStore(args.output_dir)
    t0 = time.perf_counter()
    n_ok = 0

    def on_complete(result: dict, done: int, total: int) -> None:
        nonlocal n_ok
        label = result.get("_label", result.get("run_id", "?"))
        status = result.get("status", "ERROR")
        elapsed = result.get("elapsed_s", 0)
        if status == "OK":
            n_ok += 1
            log.info("  [%d/%d] OK  %-45s (%.1fs)", done, total, label, elapsed)
        else:
            log.warning("  [%d/%d] %s %-45s %s", done, total, status, label,
                        result.get("error_msg", ""))

    results = execute_parallel(
        jobs, workers=args.workers, on_complete=on_complete,
        timeout_s=600.0,
    )

    # --- persist ---
    # Results arrive in completion order (as_completed), not submission
    # order — match by run_id, which _run_worker passes through.
    from phasesweep.sweep_types import RunResult
    jobs_by_id = {j["run_id"]: j for j in jobs}
    for result in results:
        if result.get("status") != "OK":
            continue
        job = jobs_by_id[result["run_id"]]
        motor = Motor.from_dict(job["motor_dict"])
        rc = RunConfig(motor=motor, model=job["model_key"],
                       **job.get("config_kw", {}))
        rr = RunResult(
            config=rc,
            model=job["model_key"],
            status="OK",
            metrics=result["metrics"],
            elapsed_s=result.get("elapsed_s", 0),
        )
        store.save(rr)

    wall = time.perf_counter() - t0
    log.info("done: %d/%d OK in %.1fs", n_ok, len(jobs), wall)
    if n_ok < len(jobs):
        sys.exit(1)


if __name__ == "__main__":
    main()
