#!/usr/bin/env python3
"""Reference torque-mode run for cross-checking an external PMSM plant model.

Scenario: torque step at id*=0, steady state. Demonstrates the analytical
anchor `|i_s| = tau_ref / (1.5 * n_p * psi_f)` and emits the agreed output
set (i_d, i_q, |i_s|, tau_M, u_d, u_q, w_M) as steady-state means over the
final ss_window of the run.

Default params are synthetic, non-proprietary placeholder values (the
GBM2804H label is nominal only — psi_f/U_DC are orders of magnitude
above any real gimbal motor). They were chosen so the operating point
lies well inside the linear regime; the cross-check exercises the
plant equations, not a physical motor.

Writes JSON to output/torque_crosscheck_reference/result.json and a
console summary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from motulator.drive.utils import Step

from phasesweep.motor import DriveParams
from phasesweep.sim import build_sim, plan_torque_sim
from phasesweep.solver_params import DriveSimParams

# Synthetic placeholder params (nominal "GBM2804H" label; see module docstring)
PARAMS = DriveSimParams(
    n_p=2,
    R_s=5.2,
    L_d=0.063,
    L_q=0.133,
    psi_f=1.819,
    J=0.0011,
    drive=DriveParams(U_DC=400.0, MAX_I_S=5.0),
)
TAU_REF = 3.0  # Nm

OUT = ROOT / "output" / "torque_crosscheck_reference"


def run() -> dict[str, float]:
    plan = plan_torque_sim(PARAMS, tau_ref=TAU_REF)
    torque_ref = Step(step_time=plan.load_time, step_value=TAU_REF, initial_value=0)
    sim = build_sim(PARAMS, plan, torque_ref=torque_ref)
    res = sim.simulate(t_stop=plan.t_stop)

    t = res.mdl.t
    ss = t >= (plan.t_stop - plan.ss_window)

    i_s_ab = res.mdl.machine.i_s_ab
    i_s_dq = res.mdl.machine.i_s_dq
    tau_M = res.mdl.machine.tau_M
    u_s_dq = res.mdl.machine.u_s_ab / res.mdl.machine.exp_j_theta_m
    w_M = res.mdl.mechanics.w_M

    return {
        "i_d": float(np.mean(i_s_dq.real[ss])),
        "i_q": float(np.mean(i_s_dq.imag[ss])),
        "i_s_mag": float(np.mean(np.abs(i_s_ab)[ss])),
        "tau_M": float(np.mean(tau_M[ss])),
        "u_d": float(np.mean(u_s_dq.real[ss])),
        "u_q": float(np.mean(u_s_dq.imag[ss])),
        "w_M": float(np.mean(w_M[ss])),
        "ss_window_s": plan.ss_window,
        "t_stop_s": plan.t_stop,
        "tau_ref_Nm": TAU_REF,
    }


def main() -> int:
    result = run()
    analytical_i_s = TAU_REF / (1.5 * PARAMS.n_p * PARAMS.psi_f)
    rel_err = (result["i_s_mag"] - analytical_i_s) / analytical_i_s

    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "params": {
            "n_p": PARAMS.n_p,
            "R_s": PARAMS.R_s,
            "L_d": PARAMS.L_d,
            "L_q": PARAMS.L_q,
            "psi_f": PARAMS.psi_f,
            "J": PARAMS.J,
            "U_DC": PARAMS.drive.U_DC,
            "MAX_I_S": PARAMS.drive.MAX_I_S,
        },
        "scenario": "torque_step_id_zero",
        "tau_ref_Nm": TAU_REF,
        "analytical_i_s_A": analytical_i_s,
        "steady_state": result,
        "anchor_rel_err": rel_err,
    }
    (OUT / "result.json").write_text(json.dumps(payload, indent=2))

    print(f"torque cross-check reference — tau_ref = {TAU_REF} Nm")
    print(f"  analytical |i_s| = {analytical_i_s:.4f} A")
    print(f"  simulated  |i_s| = {result['i_s_mag']:.4f} A  ({rel_err * 100:+.2f}%)")
    print(f"  i_d / i_q        = {result['i_d']:+.4f} / {result['i_q']:+.4f} A")
    print(f"  u_d / u_q        = {result['u_d']:+.2f} / {result['u_q']:+.2f} V")
    print(f"  tau_M            = {result['tau_M']:.4f} Nm")
    print(f"  w_M              = {result['w_M']:+.4f} rad/s")
    print(f"  written: {OUT / 'result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
