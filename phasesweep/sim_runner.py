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

from phasesweep.sweep_types import MotorSweepConfig, SweepResult

_DEFAULT_TIMEOUT = 60


def _run_sim_impl(config: MotorSweepConfig) -> dict[str, float]:
    """Run motulator simulation. Executes inside subprocess."""
    from phasesweep.sim import build_sim, extract_metrics

    sim = build_sim("sweep", config.to_motor_config(),
                     load_torque=config.load_torque, load_time=config.load_time)
    res = sim.simulate(t_stop=config.t_stop)
    return extract_metrics(res, config.n_p)


def run_sim_safe(config: MotorSweepConfig, timeout_s: int = _DEFAULT_TIMEOUT) -> SweepResult:
    """Run simulation with true timeout via subprocess.

    subprocess.run(timeout=) kills the process on expiry, unlike
    ProcessPoolExecutor.future.result(timeout=) which only times out the wait.
    """
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "phasesweep.sim_runner", json.dumps(config.to_dict())],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if proc.returncode != 0:
            return SweepResult(
                config=config, run_type="sim", status="ERROR", metrics=None,
                elapsed_s=time.time() - t0,
                error_msg=proc.stderr[:500] if proc.stderr else f"Exit {proc.returncode}",
            )
        return SweepResult(
            config=config, run_type="sim", status="OK",
            metrics=json.loads(proc.stdout),
            elapsed_s=time.time() - t0,
        )
    except subprocess.TimeoutExpired:
        return SweepResult(
            config=config, run_type="sim", status="TIMEOUT", metrics=None,
            elapsed_s=float(timeout_s),
            error_msg=f"Exceeded {timeout_s}s timeout",
        )
    except Exception as e:
        return SweepResult(
            config=config, run_type="sim", status="ERROR", metrics=None,
            elapsed_s=time.time() - t0,
            error_msg=str(e),
        )


def _cli_main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python -m phasesweep.sim_runner '<config_json>'", file=sys.stderr)
        sys.exit(1)
    try:
        config = MotorSweepConfig.from_dict(json.loads(sys.argv[1]))
        metrics = _run_sim_impl(config)

        def _json_safe(obj: Any) -> float:
            if hasattr(obj, "item"):  # numpy scalar
                return obj.item()
            raise TypeError(f"{type(obj)} is not JSON serializable")

        print(json.dumps(metrics, default=_json_safe))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    _cli_main()
