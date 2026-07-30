"""Timeout-safe motulator simulation runner via subprocess isolation.

Entry point: python -m phasesweep.sim_runner '<config_json>'

Subprocess isolation guarantees that hung or crashing simulations are killed
cleanly rather than wedging the parent process. Default timeout: 60s.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from typing import Any

from phasesweep.sweep_types import RunConfig, RunResult

_DEFAULT_TIMEOUT = 60


def _run_sim_impl(config: RunConfig) -> dict[str, Any]:
    """Run motulator simulation from RunConfig. Executes inside subprocess."""
    from phasesweep.sim import build_sim, extract_metrics, extract_waveforms
    from phasesweep.solver_params import prepare_drive_sim

    params = prepare_drive_sim(config.motor)
    plan = config.sim_plan
    if plan is None:
        raise ValueError("RunConfig.sim_plan is required for drive_sim")
    sim = build_sim(params, plan)
    res = sim.simulate(t_stop=plan.t_stop)
    assert params.drive.W_REF is not None
    metrics: dict[str, Any] = extract_metrics(res, plan=plan, w_ref=params.drive.W_REF)
    metrics.update(extract_waveforms(res))
    return metrics


def run_sim_safe(config: RunConfig, timeout_s: int = _DEFAULT_TIMEOUT) -> RunResult:
    """Run simulation with true timeout via subprocess."""
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "phasesweep.sim_runner",
             json.dumps(config.to_dict())],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if proc.returncode != 0:
            return RunResult(
                config=config, model="drive_sim", status="ERROR", metrics=None,
                elapsed_s=time.time() - t0,
                error_msg=proc.stderr[:500] if proc.stderr else f"Exit {proc.returncode}",
            )
        return RunResult(
            config=config, model="drive_sim", status="OK",
            metrics=json.loads(proc.stdout),
            elapsed_s=time.time() - t0,
        )
    except subprocess.TimeoutExpired:
        return RunResult(
            config=config, model="drive_sim", status="TIMEOUT", metrics=None,
            elapsed_s=float(timeout_s),
            error_msg=f"Exceeded {timeout_s}s timeout",
        )
    except Exception as e:
        return RunResult(
            config=config, model="drive_sim", status="ERROR", metrics=None,
            elapsed_s=time.time() - t0,
            error_msg=str(e),
        )


def _cli_main() -> None:
    def _json_safe(obj: Any) -> float:
        if hasattr(obj, "item"):  # numpy scalar
            return obj.item()
        raise TypeError(f"{type(obj)} is not JSON serializable")

    if len(sys.argv) != 2:
        print("Usage: python -m phasesweep.sim_runner '<config_json>'", file=sys.stderr)
        sys.exit(1)
    try:
        config = RunConfig.from_dict(json.loads(sys.argv[1]))
        metrics = _run_sim_impl(config)
        print(json.dumps(metrics, default=_json_safe))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    _cli_main()
