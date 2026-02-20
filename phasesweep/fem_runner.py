"""Timeout-safe NGSolve FEM runner via subprocess isolation.

Entry point: python -m phasesweep.fem_runner '<config_json>'

FEM solves can occasionally hang (mesh generation, sparse solver). Subprocess
isolation guarantees they are killed cleanly on timeout. Default timeout: 300s.

Metrics returned:
  peak_Br      — max |B_r| at air-gap centre (proxy units)
  fundamental  — amplitude at n_p-th harmonic
  thd_pct      — total harmonic distortion (% of fundamental)
  sh_pct       — slot harmonic (Q - n_p order) amplitude (% of fundamental)
  theta_list   — angle samples (list[float], for reconstruction)
  B_r_list     — radial flux density samples (list[float], for reconstruction)
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from typing import Any

import numpy as np

from phasesweep.sweep_types import MotorSweepConfig, SweepResult

_DEFAULT_TIMEOUT = 300


def _run_fem_impl(config: MotorSweepConfig) -> dict[str, Any]:
    """Run NGSolve FEM solve. Executes inside subprocess."""
    from phasesweep.fem_field import solve_field_fem, harmonics_1sided

    theta, B_r = solve_field_fem(
        n_p=config.n_p,
        psi_f=config.psi_f,
        L_d=config.L_d,
        L_q=config.L_q,
        n_theta=config.n_theta,
        maxh=config.maxh,
        n_slots=config.n_slots,
        j_s=config.j_s,
        nonlinear=config.nonlinear,
        N=config.N,
        k_w=config.k_w,
        L_stk=config.L_stk,
    )

    amps = harmonics_1sided(B_r)

    fund_idx = min(config.n_p, len(amps) - 1)
    peak_B_r = float(np.nanmax(np.abs(B_r)))
    fundamental = float(amps[fund_idx])
    fund_or_1 = fundamental or 1.0

    thd_pct = float(
        np.sqrt(max(np.sum(amps[1:] ** 2) - amps[fund_idx] ** 2, 0.0)) / fund_or_1 * 100
    )

    sh_pct = 0.0
    if config.n_slots > 0:
        sh_idx = config.n_slots - config.n_p
        if 0 < sh_idx < len(amps):
            sh_pct = float(amps[sh_idx] / fund_or_1 * 100)

    return {
        "peak_Br": peak_B_r,
        "fundamental": fundamental,
        "thd_pct": thd_pct,
        "sh_pct": sh_pct,
        "theta_list": theta.tolist(),
        "B_r_list": B_r.tolist(),
    }


def run_fem_safe(config: MotorSweepConfig, timeout_s: int = _DEFAULT_TIMEOUT) -> SweepResult:
    """Run FEM solve with true timeout via subprocess."""
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "phasesweep.fem_runner", json.dumps(config.to_dict())],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if proc.returncode != 0:
            return SweepResult(
                config=config, run_type="fem", status="ERROR", metrics=None,
                elapsed_s=time.time() - t0,
                error_msg=proc.stderr[:500] if proc.stderr else f"Exit {proc.returncode}",
            )
        return SweepResult(
            config=config, run_type="fem", status="OK",
            metrics=json.loads(proc.stdout),
            elapsed_s=time.time() - t0,
        )
    except subprocess.TimeoutExpired:
        return SweepResult(
            config=config, run_type="fem", status="TIMEOUT", metrics=None,
            elapsed_s=float(timeout_s),
            error_msg=f"Exceeded {timeout_s}s timeout",
        )
    except Exception as e:
        return SweepResult(
            config=config, run_type="fem", status="ERROR", metrics=None,
            elapsed_s=time.time() - t0,
            error_msg=str(e),
        )


def _cli_main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python -m phasesweep.fem_runner '<config_json>'", file=sys.stderr)
        sys.exit(1)
    try:
        config = MotorSweepConfig.from_dict(json.loads(sys.argv[1]))
        metrics = _run_fem_impl(config)
        print(json.dumps(metrics))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    _cli_main()
