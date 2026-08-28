"""Staged motor-check: objective checks and evidence-level report for one TOML.

Runs staged load and structural checks, input-source and derived-value report,
model applicability,
fast computations, optional exploratory FEM cross-check, provenance presence,
and an evidence-level summary that never conflates solver agreement with
measurement validation.

Usage:
    uv run python scripts/check_motor.py motors/my_motor.toml
    uv run python scripts/check_motor.py motors/my_motor.toml --fem quick

Exit codes:
    0  all objective checks passed (engineering warnings may be present)
    1  objective failure: load/schema error, solver exception, non-finite
       output, or FEM error/timeout
"""

from __future__ import annotations

import argparse
import math
import sys
import tomllib
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from phasesweep import load_motor
from phasesweep.machines.motor import Motor
from phasesweep.registry import MODEL_REGISTRY
from phasesweep.solver_params import _derive_B_rem_from_psi_f
from phasesweep.sweep_types import RunConfig

# Fields whose scalar values are worth showing per fast model.
_HEADLINE_METRICS = {
    "analytical": ["fundamental", "peak_Br", "flux_linkage_peak",
                   "backemf_fundamental", "k_end", "k_end_source"],
    "analytical_axial": ["fundamental", "peak_Br", "flux_linkage_peak",
                         "backemf_fundamental"],
    "rated_torque": ["tau_mtpa", "k_T", "gamma_opt_deg"],
    "stall_torque": ["tau_stall", "I_stall", "saturation_warning"],
    "torque_speed": ["p_max_peak", "base_speed_peak", "max_speed_peak"],
    "iron_loss": [],
    "thermal_duty": [],
}

OVERALL_LINE = "Overall: computationally checked; NOT measurement-validated."

CONFLICT_REMEDIATION = (
    "  Geometry-led: omit psi_f and derive it from magnet, geometry, and\n"
    "  winding inputs. Measurement-led: retain measured psi_f, attach its\n"
    "  provenance, and treat B_rem-derived field results as a separate\n"
    "  model assumption."
)


@dataclass
class Report:
    """Structured result of a check run; main() renders and exits on it."""

    path: Path
    lines: list[str] = field(default_factory=list)
    config_status: str = "valid"
    execution_status: str = "not run"
    consistency_status: str = "not applicable"
    crosscheck_status: str = "not run"
    measurement_status: str = "not assessed"
    failures: list[str] = field(default_factory=list)
    warnings_seen: list[str] = field(default_factory=list)

    @property
    def exit_code(self) -> int:
        return 1 if self.failures else 0

    def say(self, text: str = "") -> None:
        self.lines.append(text)

    def fail(self, text: str) -> None:
        self.failures.append(text)
        self.lines.append(f"FAIL: {text}")

    def warn(self, text: str) -> None:
        self.warnings_seen.append(text)
        self.lines.append(f"warning: {text}")


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _scan_finite(name: str, value: Any, report: Report) -> None:
    """Fail on NaN/Inf anywhere in a metrics value (None is allowed)."""
    if isinstance(value, float) and not math.isfinite(value):
        report.fail(f"{name}: non-finite value {value!r}")
    elif isinstance(value, (list, tuple)):
        for item in value:
            _scan_finite(name, item, report)
    elif isinstance(value, dict):
        for key, item in value.items():
            _scan_finite(f"{name}.{key}", item, report)


def _stage_load(report: Report) -> Motor | None:
    report.say(f"== Load: {report.path}")
    try:
        motor = load_motor(report.path)
    except (ValueError, OSError, tomllib.TOMLDecodeError) as exc:
        report.config_status = "INVALID"
        report.fail(f"load: {exc}")
        return None
    kind = "induction" if motor.induction is not None else "synchronous PM"
    geo = "geometry-bearing" if motor.geometry is not None else "circuit-only"
    report.say(f"  name: {motor.name}")
    report.say(f"  topology: {kind}, {geo}, n_p={motor.n_p}")
    report.say("  CONFIG: valid (schema and objective invariants only;")
    report.say("  no claim about physical plausibility)")
    return motor


def _stage_inputs(report: Report, motor: Motor) -> None:
    report.say("== Inputs: psi_f / B_rem authority")
    psi_src = "explicit" if motor.psi_f is not None else "absent (derivable from B_rem + geometry + winding)"
    b_src = "explicit" if motor.B_rem is not None else "absent"
    report.say(f"  psi_f: {psi_src}" + (f" = {_fmt(motor.psi_f)} Wb" if motor.psi_f is not None else ""))
    report.say(f"  B_rem: {b_src}" + (f" = {_fmt(motor.B_rem)} T" if motor.B_rem is not None else ""))

    if motor.psi_f is None or motor.B_rem is None:
        report.consistency_status = "ok (single source)"
        return
    try:
        derived = _derive_B_rem_from_psi_f(motor)
    except ValueError as exc:
        report.consistency_status = f"not derivable ({exc})"
        return
    if derived is None:
        report.consistency_status = "not derivable (missing winding/geometry inputs)"
        return
    rel = abs(derived - motor.B_rem) / max(abs(motor.B_rem), 1e-12)
    report.say(f"  B_rem derived from psi_f: {_fmt(derived)} T "
               f"(disagrees with explicit B_rem by {rel * 100:.1f}%)")
    if rel > 0.50:
        report.consistency_status = f"warning ({rel * 100:.1f}% psi_f/B_rem disagreement)"
        report.warn("psi_f and B_rem disagree; choose one evidence path:")
        report.say(CONFLICT_REMEDIATION)
    else:
        report.consistency_status = f"ok ({rel * 100:.1f}% agreement)"


def _stage_applicability(report: Report, motor: Motor) -> list[str]:
    report.say("== Model applicability")
    runnable: list[str] = []
    for name, info in MODEL_REGISTRY.items():
        if info.source != "computed" or info.validate is None or info.fn is None:
            continue
        try:
            info.validate(motor)
        except (ValueError, ImportError) as exc:
            first = str(exc).splitlines()[0]
            report.say(f"  skip {name} ({info.cost}): {first}")
            continue
        runnable.append(name)
        report.say(f"  ok   {name} ({info.cost})")
    if not runnable:
        report.say("  no computed model is applicable")
    return runnable


def _run_model(report: Report, motor: Motor, name: str,
               **overrides: Any) -> dict[str, Any] | None:
    config = RunConfig(motor=motor, model=name, **overrides)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            metrics = MODEL_REGISTRY[name].fn(config)  # type: ignore[misc]
        except Exception as exc:
            report.fail(f"{name}: {exc}")
            return None
    for w in caught:
        body = str(w.message)
        if any(seen.endswith(body) for seen in report.warnings_seen):
            continue
        report.warn(f"{name}: {body}")
    for key, value in metrics.items():
        _scan_finite(f"{name}.{key}", value, report)
    return metrics


def _stage_fast(report: Report, motor: Motor, runnable: list[str]) -> dict[str, dict[str, Any]]:
    report.say("== Fast computations")
    results: dict[str, dict[str, Any]] = {}
    fast = []
    for n in runnable:
        if MODEL_REGISTRY[n].cost != "fast":
            continue
        needs_runtime = MODEL_REGISTRY[n].needs_runtime
        if needs_runtime:
            fields = ", ".join(sorted(needs_runtime))
            report.say(f"  {n}: applicable but needs run-time input(s): "
                       f"{fields}; not executed by this check")
            continue
        fast.append(n)
    if not fast:
        report.say("  no applicable fast model")
        return results
    for name in fast:
        metrics = _run_model(report, motor, name)
        if metrics is None:
            continue
        results[name] = metrics
        shown = _HEADLINE_METRICS.get(name, [])
        for key in shown:
            if key in metrics and metrics[key] is not None:
                report.say(f"  {name}: {key} = {_fmt(metrics[key])}")
        if not shown:
            report.say(f"  {name}: ran (finite outputs)")
    ran = len(results)
    report.execution_status = (
        f"runnable ({ran}/{len(fast)} fast models)" if ran else "failed"
    )
    if ran < len(fast):
        report.execution_status = f"failed ({ran}/{len(fast)} fast models)"
    return results


def _stage_fem_quick(report: Report, motor: Motor, runnable: list[str],
                     analytical: dict[str, Any] | None) -> None:
    from phasesweep.solvers.fem_runner import run_fem_safe

    report.say("== FEM cross-check (profile: quick - EXPLORATORY settings)")
    if "fem" not in runnable:
        report.say("  fem model not applicable; skipped")
        return
    settings = {"n_theta": 180, "maxh_fraction": 0.10}
    report.say(f"  resolved settings: n_theta={settings['n_theta']}, "
               f"maxh_fraction={settings['maxh_fraction']}, no-load (j_s=0),")
    report.say("  linear then nonlinear (default material B-H model)")
    fem_b1: dict[str, float] = {}
    for label, nonlinear in (("linear", False), ("nonlinear", True)):
        result = run_fem_safe(
            RunConfig(motor=motor, model="fem", nonlinear=nonlinear, **settings),
            timeout_s=300,
        )
        if result.status != "OK" or result.metrics is None:
            report.fail(f"fem ({label}): {result.status} {result.error_msg or ''}".strip())
            continue
        metrics = result.metrics
        _scan_finite(f"fem.{label}", {k: v for k, v in metrics.items()
                                      if k in ("fundamental", "peak_Br", "b_iron_max")},
                     report)
        fem_b1[label] = metrics["fundamental"]
        report.say(f"  {label}: fundamental B1 = {_fmt(metrics['fundamental'])} T"
                   + (f", peak stator |B| = {_fmt(metrics['b_iron_max'])} T"
                      if metrics.get("b_iron_max") is not None else ""))
        b_iron = metrics.get("b_iron_max")
        if b_iron is not None and b_iron > 1.8:
            report.warn(f"fem ({label}): peak stator |B| {_fmt(b_iron)} T exceeds "
                        "typical electrical-steel knee (~1.5-1.8 T); saturation risk."
                        " Engineering concern, not an objective failure.")
    ref = analytical.get("fundamental") if analytical else None
    if ref and fem_b1:
        deltas = []
        for label, b1 in fem_b1.items():
            delta = (b1 - ref) / ref * 100
            deltas.append(abs(delta))
            report.say(f"  analytical vs FEM ({label}): {delta:+.2f}%")
        report.crosscheck_status = (
            f"ran ({max(deltas):.1f}% max delta, exploratory mesh - numerical"
            " cross-check, NOT physical validation)"
        )
    elif fem_b1:
        report.crosscheck_status = "FEM ran; no analytical reference to compare"


def _stage_provenance(report: Report) -> None:
    report.say("== Evidence and provenance")
    try:
        raw = tomllib.loads(report.path.read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return
    for section, meaning, why in (
        ("provenance", "input provenance", "non-obvious inputs are undocumented"),
        ("validation", "measured values", "no bench measurements are recorded"),
    ):
        present = "present" if section in raw else "absent"
        report.say(f"  [{section}] ({meaning}): {present}")
        if section not in raw:
            report.warn(f"no [{section}] section; {why}")
    report.say("  measurement comparison: not performed by this checker;")
    report.say("  use phasesweep-crossval with a registered dataset")


def build_report(path: Path, fem: str = "off") -> Report:
    report = Report(path=path)
    motor = _stage_load(report)
    if motor is None:
        return report
    _stage_inputs(report, motor)
    runnable = _stage_applicability(report, motor)
    results = _stage_fast(report, motor, runnable)
    if fem == "quick":
        _stage_fem_quick(report, motor, runnable, results.get("analytical"))
    _stage_provenance(report)

    report.say("== Status")
    report.say(f"  CONFIG:      {report.config_status}")
    report.say(f"  EXECUTION:   {report.execution_status}")
    report.say(f"  CONSISTENCY: {report.consistency_status}")
    report.say(f"  CROSS-CHECK: {report.crosscheck_status}")
    report.say(f"  MEASUREMENT: {report.measurement_status}")
    report.say("")
    report.say(OVERALL_LINE)
    report.say("Passing this check means the file is internally consistent and the")
    report.say("software can solve it. It does not establish that the motor is")
    report.say("manufacturable or that the models predict measured performance.")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("toml", type=Path, help="motor TOML file to check")
    parser.add_argument(
        "--fem", choices=["off", "quick"], default="off",
        help="run an exploratory coarse FEM cross-check (default: off)",
    )
    args = parser.parse_args()
    report = build_report(args.toml, fem=args.fem)
    print("\n".join(report.lines))
    sys.exit(report.exit_code)


if __name__ == "__main__":
    main()
