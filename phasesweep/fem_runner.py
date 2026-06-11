"""Timeout-safe NGSolve FEM runner via subprocess isolation.

Entry point: python -m phasesweep.fem_runner '<config_json>'

FEM solves can occasionally hang (mesh generation, sparse solver). Subprocess
isolation guarantees they are killed cleanly on timeout. Default timeout: 300s.

Metrics returned:
  peak_Br      — max |B_r| at air-gap centre (T)
  fundamental  — amplitude at n_p-th harmonic
  thd_pct      — total harmonic distortion (% of fundamental)
  sh_pct       — slot harmonic (Q - n_p order) amplitude (% of fundamental)
  theta_list   — angle samples (list[float], for reconstruction)
  B_r_list     — radial flux density samples (list[float], for reconstruction)
  b_iron_max   — peak |B| in iron regions (T)
  picard_*     — nonlinear solves only: iterations, relax_final, residual
                 (relaxed step), residual_raw (unrelaxed step)
"""

from __future__ import annotations

import json
import subprocess
import sys
import time

import numpy as np

from phasesweep.sweep_types import RunConfig, RunResult

_DEFAULT_TIMEOUT = 300


def _run_fem_impl(config: RunConfig) -> dict[str, float | list[float]]:
    """Run NGSolve FEM solve from RunConfig. Executes inside subprocess."""
    from phasesweep.fem_field import compute_thd, harmonics_1sided, solve_field_fem
    from phasesweep.solver_params import prepare_fem

    params = prepare_fem(config.motor)
    geo = params.geometry

    solve_info: dict[str, float] = {}
    theta, B_r = solve_field_fem(
        geo=geo, n_p=params.n_p, B_rem=params.B_rem,
        mu_r_pm=params.mu_r_pm, mu_r_fe=params.mu_r_fe,
        maxh_fraction=config.maxh_fraction,
        n_theta=config.n_theta, nonlinear=config.nonlinear,
        j_s=config.j_s,
        k_w=config.motor.k_w if config.motor.k_w is not None else 0.966,
        alpha_p=params.alpha_p,
        info=solve_info,
    )

    amps = harmonics_1sided(B_r)
    n_p = params.n_p
    fund_idx = min(n_p, len(amps) - 1)
    peak_B_r = float(np.nanmax(np.abs(B_r)))
    fundamental = float(amps[fund_idx])
    fund_or_1 = fundamental or 1.0

    thd_pct = compute_thd(amps, fund_idx)

    sh_pct = 0.0
    n_slots = geo.n_slots
    if n_slots > 0:
        sh_idx = n_slots - n_p
        if 0 < sh_idx < len(amps):
            sh_pct = float(amps[sh_idx] / fund_or_1 * 100)

    return {
        "peak_Br": peak_B_r,
        "fundamental": fundamental,
        "thd_pct": thd_pct,
        "sh_pct": sh_pct,
        "theta_list": theta.tolist(),
        "B_r_list": B_r.tolist(),
        **solve_info,
    }


def run_fem_safe(config: RunConfig, timeout_s: int = _DEFAULT_TIMEOUT) -> RunResult:
    """Run FEM solve with true timeout via subprocess."""
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "phasesweep.fem_runner",
             json.dumps(config.to_dict())],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if proc.returncode != 0:
            return RunResult(
                config=config, model="fem", status="ERROR", metrics=None,
                elapsed_s=time.time() - t0,
                error_msg=proc.stderr[:500] if proc.stderr else f"Exit {proc.returncode}",
            )
        return RunResult(
            config=config, model="fem", status="OK",
            metrics=json.loads(proc.stdout),
            elapsed_s=time.time() - t0,
        )
    except subprocess.TimeoutExpired:
        return RunResult(
            config=config, model="fem", status="TIMEOUT", metrics=None,
            elapsed_s=float(timeout_s),
            error_msg=f"Exceeded {timeout_s}s timeout",
        )
    except Exception as e:
        return RunResult(
            config=config, model="fem", status="ERROR", metrics=None,
            elapsed_s=time.time() - t0,
            error_msg=str(e),
        )


def _cli_main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python -m phasesweep.fem_runner '<config_json>'", file=sys.stderr)
        sys.exit(1)
    try:
        config = RunConfig.from_dict(json.loads(sys.argv[1]))
        metrics = _run_fem_impl(config)
        print(json.dumps(metrics))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    _cli_main()
