"""Cogging torque model — rotor sweep at zero armature current.

Entry point: python -m phasesweep.solvers.cogging '<config_json>'

Sweeps solve_field_fem over one cogging period with j_s=0, computing
Arkkio torque at each angle.  The full sweep runs in one subprocess call
(run_cogging_safe) with timeout scaled by the angle count.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from typing import Any

import numpy as np

from phasesweep.sweep_types import RunConfig, RunResult

_PER_ANGLE_TIMEOUT = 300


def _run_cogging_impl(config: RunConfig) -> dict[str, Any]:
    """Run cogging torque sweep from RunConfig.  Executes inside subprocess."""
    from phasesweep.solver_params import prepare_fem
    from phasesweep.solvers.fem_field import arkkio_torque, solve_field_fem
    from phasesweep.solvers.harmonics import cogging_angles, harmonics_1sided

    if config.j_s != 0.0:
        raise ValueError(
            "cogging_torque v1 requires j_s=0 (locked-rotor ripple "
            "under load is out of scope)"
        )

    params = prepare_fem(config.motor)
    geo = params.geometry

    if geo.n_slots == 0:
        raise ValueError(
            "cogging_torque requires a slotted geometry (n_slots > 0)"
        )

    if config.cogging_points < 4:
        raise ValueError(
            f"cogging_points={config.cogging_points} too low; "
            "need >= 4 (12 recommended for adequate p-p resolution)"
        )

    angles = cogging_angles(
        geo.n_slots, params.n_p,
        points_per_period=config.cogging_points,
    )

    g = abs(geo.r_stator - geo.r_magnet)
    margin = 0.05 * g
    if geo.topology == "inrunner":
        r_ark_inner = geo.r_magnet + margin
        r_ark_outer = geo.r_stator - margin
    else:
        r_ark_inner = geo.r_stator + margin
        r_ark_outer = geo.r_magnet - margin

    rotor_sign = 1.0 if geo.topology == "inrunner" else -1.0

    tau_list = []
    for phi in angles:
        _, _, mesh_obj, gfu = solve_field_fem(
            geo=geo, n_p=params.n_p, B_rem=params.B_rem,
            mu_r_pm=params.mu_r_pm, mu_r_fe=params.mu_r_fe,
            maxh_fraction=config.maxh_fraction,
            n_theta=config.n_theta, nonlinear=config.nonlinear,
            j_s=0.0, alpha_p=params.alpha_p,
            rotation=float(phi),
            return_full=True,
        )
        tau_per_m = rotor_sign * arkkio_torque(
            mesh_obj, gfu, r_ark_inner, r_ark_outer,
            n_theta=config.n_theta,
        )
        tau_list.append(tau_per_m)

    tau_arr = np.array(tau_list)

    from math import gcd
    lcm_val = geo.n_slots * 2 * params.n_p // gcd(geo.n_slots, 2 * params.n_p)
    n_cogging_periods = lcm_val

    amps = harmonics_1sided(tau_arr)
    dominant_bin = int(np.argmax(amps[1:])) + 1
    dominant_order = dominant_bin * n_cogging_periods

    tau_pp = float(tau_arr.max() - tau_arr.min())

    result: dict[str, Any] = {
        "rotation_list": angles.tolist(),
        "tau_cogging_list": tau_arr.tolist(),
        "tau_cogging_pp": tau_pp,
        "dominant_order": dominant_order,
        "n_cogging_periods": n_cogging_periods,
    }

    if config.motor.L_stk is not None:
        result["tau_cogging_pp_Nm"] = tau_pp * config.motor.L_stk

    return result


def run_cogging_safe(
    config: RunConfig, timeout_s: int | None = None,
) -> RunResult:
    """Run cogging torque sweep with timeout via subprocess."""
    if timeout_s is None:
        timeout_s = _PER_ANGLE_TIMEOUT * max(config.cogging_points, 1)
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "phasesweep.solvers.cogging",
             json.dumps(config.to_dict())],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if proc.returncode != 0:
            return RunResult(
                config=config, model="cogging_torque", status="ERROR",
                metrics=None, elapsed_s=time.time() - t0,
                error_msg=proc.stderr[:500] if proc.stderr else f"Exit {proc.returncode}",
            )
        return RunResult(
            config=config, model="cogging_torque", status="OK",
            metrics=json.loads(proc.stdout),
            elapsed_s=time.time() - t0,
        )
    except subprocess.TimeoutExpired:
        return RunResult(
            config=config, model="cogging_torque", status="TIMEOUT",
            metrics=None, elapsed_s=float(timeout_s),
            error_msg=f"Exceeded {timeout_s}s timeout",
        )
    except Exception as e:
        return RunResult(
            config=config, model="cogging_torque", status="ERROR",
            metrics=None, elapsed_s=time.time() - t0,
            error_msg=str(e),
        )


def _cli_main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python -m phasesweep.solvers.cogging '<config_json>'",
              file=sys.stderr)
        sys.exit(1)
    try:
        config = RunConfig.from_dict(json.loads(sys.argv[1]))
        metrics = _run_cogging_impl(config)
        print(json.dumps(metrics))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    _cli_main()
