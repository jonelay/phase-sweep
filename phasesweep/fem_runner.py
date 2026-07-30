"""Timeout-safe NGSolve FEM runner via subprocess isolation.

Entry point: python -m phasesweep.fem_runner '<config_json>'

FEM solves can occasionally hang (mesh generation, sparse solver). Subprocess
isolation guarantees they are killed cleanly on timeout. Default timeout: 300s.

Metrics returned:
  peak_Br      — max |B_r| at air-gap centre (T)
  fundamental  — amplitude at n_p-th harmonic
  thd_pct      — total harmonic distortion (% of fundamental)
  sh_pct       — lower slot-harmonic sideband (|Q - n_p| order) amplitude
                 (% of fundamental)
  sh_upper_pct — upper slot-harmonic sideband (Q + n_p order) amplitude
                 (% of fundamental); slot harmonics come in pairs Q ± n_p
  theta_list   — angle samples (list[float], for reconstruction)
  B_r_list     — radial flux density samples (list[float], for reconstruction)
  B_mag_grid   — |B| on a Cartesian grid over the cross-section (rows = y,
                 cols = x; null outside the motor domain) for the dashboard
                 heatmap
  grid_coords_list — the shared x/y axis of B_mag_grid (m)
  b_iron_max   — peak |B| in iron regions (T)
  tau_maxwell_per_m — armature runs (j_s != 0) only: Maxwell-stress rotor
                 torque per metre of stack (N·m/m, mid-gap contour)
  tau_maxwell  — tau_maxwell_per_m · L_stk (N·m), when Motor.L_stk is set
  picard_*     — nonlinear solves only: iterations, relax_final, residual
                 (raw/unrelaxed step, the stop criterion)
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from typing import Any

import numpy as np

from phasesweep.sweep_types import RunConfig, RunResult

_DEFAULT_TIMEOUT = 300
_RASTER_N_GRID = 120


def _run_fem_impl(config: RunConfig) -> dict[str, Any]:
    """Run NGSolve FEM solve from RunConfig. Executes inside subprocess."""
    from phasesweep.fem_field import solve_field_fem
    from phasesweep.harmonics import compute_thd, harmonics_1sided
    from phasesweep.solver_params import prepare_fem

    params = prepare_fem(config.motor)
    geo = params.geometry

    if config.j_s != 0.0 and config.motor.k_w is None:
        raise ValueError(
            f"Motor {config.motor.name!r}: j_s != 0 requires an explicit "
            "k_w (armature current sheet scales with the winding factor)"
        )
    if config.n_theta <= 2 * params.n_p:
        # before the solve — a bad config must not cost a FEM run
        raise ValueError(
            f"n_theta={config.n_theta} cannot resolve the fundamental at "
            f"n_p={params.n_p}; need n_theta > 2*n_p"
        )

    solve_info: dict[str, float] = {}
    theta, B_r, mesh_obj, gfu = solve_field_fem(
        geo=geo, n_p=params.n_p, B_rem=params.B_rem,
        mu_r_pm=params.mu_r_pm, mu_r_fe=params.mu_r_fe,
        maxh_fraction=config.maxh_fraction,
        n_theta=config.n_theta, nonlinear=config.nonlinear,
        j_s=config.j_s,
        k_w=config.motor.k_w if config.motor.k_w is not None else 0.966,
        alpha_p=params.alpha_p,
        info=solve_info,
        return_full=True,
    )
    torque_info: dict[str, float] = {}
    if config.j_s != 0.0:
        from phasesweep.fem_field import maxwell_stress_torque
        # Torque on the region enclosed by the mid-gap circle = the rotor
        # for an inrunner; sign-flip for an outrunner (rotor outside).
        rotor_sign = 1.0 if geo.topology == "inrunner" else -1.0
        tau_per_m = rotor_sign * maxwell_stress_torque(mesh_obj, gfu, geo.r_ag)
        torque_info["tau_maxwell_per_m"] = tau_per_m
        if config.motor.L_stk is not None:
            torque_info["tau_maxwell"] = tau_per_m * config.motor.L_stk

    from phasesweep.fem_field import rasterise_cross_section
    xi, _yi, _Az, Bmag = rasterise_cross_section(
        mesh_obj, gfu, n_grid=_RASTER_N_GRID, r_bound=geo.r_outer)
    B_mag_grid = [
        [round(float(v), 5) if np.isfinite(v) else None for v in row]
        for row in Bmag
    ]
    grid_coords = np.round(xi[0], 6).tolist()

    amps = harmonics_1sided(B_r)
    n_p = params.n_p
    if np.isnan(B_r).any():
        raise RuntimeError(
            f"FEM B_r sampling produced {int(np.isnan(B_r).sum())} NaN "
            f"point(s) of {B_r.size} — mesh evaluation failed at the "
            "sampling radius"
        )
    peak_B_r = float(np.max(np.abs(B_r)))
    fundamental = float(amps[n_p])
    fund_or_1 = fundamental or 1.0

    thd_pct = compute_thd(amps, n_p)

    sh_pct = 0.0
    sh_upper_pct = 0.0
    n_slots = geo.n_slots
    if n_slots > 0:
        sh_idx = abs(n_slots - n_p)
        if 0 < sh_idx < len(amps):
            sh_pct = float(amps[sh_idx] / fund_or_1 * 100)
        sh_upper_idx = n_slots + n_p
        if sh_upper_idx < len(amps):
            sh_upper_pct = float(amps[sh_upper_idx] / fund_or_1 * 100)

    return {
        "peak_Br": peak_B_r,
        "fundamental": fundamental,
        "thd_pct": thd_pct,
        "sh_pct": sh_pct,
        "sh_upper_pct": sh_upper_pct,
        "theta_list": theta.tolist(),
        "B_r_list": B_r.tolist(),
        "B_mag_grid": B_mag_grid,
        "grid_coords_list": grid_coords,
        **torque_info,
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
