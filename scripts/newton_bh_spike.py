"""Newton-vs-Picard comparison for the nonlinear BH solve (Track C).

Drives the shipped Newton solver (``solve_field_fem(nonlinear_solver=
"newton")``, same discrete system as Picard — element-wise L2^0 nu from
the shared 19-point BH table) against the Picard default on three
production nonlinear cases (deep-dive nonlinear_bh pinned case, CREATOR
inset+slots, 14 mm steel actuator), twice each: an operational pass
(shipping picard_tol vs Newton's rtol 1e-4 — the fair iteration and
wall-time comparison at matched B1 accuracy) and a fixed-point pass
(picard_tol 1e-4 vs Newton driven to rtol 1e-8) that confirms both
iterations share one converged solution.

Known limitation of the shipped Newton:
the Jacobian evaluates grad(A) pointwise while the residual's nu uses
the element-averaged |B|, so the asymptotic rate is linear, not
quadratic — irrelevant at the 1e-4 operating point (2-19 iterations),
visible only when driving to 1e-8 in the fixed-point pass.

Usage:
    uv run python scripts/newton_bh_spike.py [--quick]

Writes output/newton_spike/newton_vs_picard.json and prints a summary.
--quick runs only the deep-dive case.
"""
from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

import phasesweep.solvers.fem_field as ff
from phasesweep.machines.configs import load_motor
from phasesweep.solvers.fem_field import solve_field_fem
from phasesweep.solvers.fem_runner import prepare_fem
from phasesweep.solvers.harmonics import harmonics_1sided

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "output" / "newton_spike"


@contextmanager
def newton_rtol(rtol: float):
    orig = ff._NEWTON_RTOL
    ff._NEWTON_RTOL = rtol
    try:
        yield
    finally:
        ff._NEWTON_RTOL = orig


def _creator_slot_width_ratio(motor) -> float:
    import tomllib
    with open(ROOT / "motors" / "creator_case_pmsm.toml", "rb") as f:
        raw = tomllib.load(f)
    geom = raw["geometry"]
    pitch = geom["stator_id"] * np.pi / motor.geometry.n_slots
    return geom["slot_opening_width"] / pitch


def _cases(quick: bool) -> list[dict]:
    from phasesweep.machines.geometry import default_inrunner

    cases = [
        # The deep-dive §5.3 / elmer_radial nonlinear_bh pinned case:
        # Picard took 26 iterations here, Elmer's Newton took 3.
        dict(name="zhu_a080_nl", geo=default_inrunner(), n_p=4, B_rem=1.2,
             mu_r_pm=1.05, mu_r_fe=5000.0, alpha_p=0.8,
             kw=dict(n_slots=0), maxh=0.05),
    ]
    if quick:
        return cases

    creator = load_motor(ROOT / "motors" / "creator_case_pmsm.toml")
    pc = prepare_fem(creator)
    cases.append(
        dict(name="creator_inset", geo=pc.geometry, n_p=pc.n_p,
             B_rem=pc.B_rem, mu_r_pm=pc.mu_r_pm, mu_r_fe=pc.mu_r_fe,
             alpha_p=pc.alpha_p,
             kw=dict(n_slots=pc.geometry.n_slots,
                     slot_width_ratio=_creator_slot_width_ratio(creator)),
             maxh=0.03 / pc.geometry.r_outer))

    steel = load_motor(ROOT / "motors" / "outrunner_14mm_steel.toml")
    ps = prepare_fem(steel)
    cases.append(
        dict(name="steel_actuator", geo=ps.geometry, n_p=ps.n_p,
             B_rem=ps.B_rem, mu_r_pm=ps.mu_r_pm, mu_r_fe=ps.mu_r_fe,
             alpha_p=ps.alpha_p, kw={}, maxh=0.05))
    return cases


def _run(case: dict, solver: str, picard_tol: float | None) -> dict:
    info: dict[str, float] = {}
    kw = dict(geo=case["geo"], n_p=case["n_p"], B_rem=case["B_rem"],
              mu_r_pm=case["mu_r_pm"], mu_r_fe=case["mu_r_fe"],
              maxh_fraction=case["maxh"], n_theta=360,
              alpha_p=case["alpha_p"], nonlinear=True,
              nonlinear_solver=solver, info=info, **case["kw"])
    if picard_tol is not None:
        kw["picard_tol"] = picard_tol
    t0 = time.perf_counter()
    _, B_r = solve_field_fem(**kw)
    dt = time.perf_counter() - t0
    amps = harmonics_1sided(B_r)
    return dict(B1=float(amps[case["n_p"]]), Br=B_r,
                wall_s=round(dt, 2), info=info)


def _compare(case: dict, picard_tol: float | None, rtol: float,
             label: str) -> dict:
    pic = _run(case, "picard", picard_tol)
    with newton_rtol(rtol):
        newt = _run(case, "newton", picard_tol)

    d_B1 = (newt["B1"] - pic["B1"]) / pic["B1"]
    d_wave = float(np.max(np.abs(newt["Br"] - pic["Br"])))
    fell_back = "newton_fell_back" in newt["info"]
    row = dict(
        case=case["name"], pass_label=label,
        picard_tol=picard_tol, newton_rtol=rtol,
        picard=dict(B1=pic["B1"], wall_s=pic["wall_s"],
                    iterations=pic["info"].get("picard_iterations"),
                    relax_final=pic["info"].get("picard_relax_final")),
        newton=dict(B1=newt["B1"], wall_s=newt["wall_s"],
                    iterations=newt["info"].get("newton_iterations"),
                    backtracks=newt["info"].get("newton_backtracks"),
                    fell_back=fell_back),
        delta_B1_rel=d_B1, delta_waveform_max_T=d_wave)
    newton_desc = ("FELL BACK to Picard" if fell_back else
                   f"{newt['info'].get('newton_iterations'):.0f} it"
                   f" ({newt['info'].get('newton_backtracks'):.0f} backtrack)")
    print(f"  {case['name']:16s} [{label}]  Picard "
          f"{pic['info'].get('picard_iterations'):.0f} it {pic['wall_s']:6.1f}s"
          f" | Newton {newton_desc} {newt['wall_s']:6.1f}s"
          f" | dB1 {d_B1:+.2e} | max|dBr| {d_wave:.2e} T")
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    cases = _cases(args.quick)
    rows = []
    print("Newton-vs-Picard comparison (shared discrete system; "
          "operational then fixed-point pass)")
    for case in cases:
        # Warm the mesh cache so wall times compare solvers, not meshing
        # (inset geometries reject linear solves, hence nonlinear=True).
        solve_field_fem(geo=case["geo"], n_p=case["n_p"], B_rem=case["B_rem"],
                        mu_r_pm=case["mu_r_pm"], mu_r_fe=case["mu_r_fe"],
                        maxh_fraction=case["maxh"], n_theta=360,
                        alpha_p=case["alpha_p"], nonlinear=True, **case["kw"])
        # Operational: Picard at shipping tol vs Newton at its shipped
        # rtol — the fair iteration/wall-time comparison.
        rows.append(_compare(case, None, ff._NEWTON_RTOL, "operational"))
        # Fixed-point: both driven hard — confirms the two iterations
        # share one converged solution (dB1 expected ~1e-6 or below).
        rows.append(_compare(case, 1e-4, 1e-8, "fixed-point"))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "newton_vs_picard.json"
    out.write_text(json.dumps(dict(results=rows), indent=1) + "\n")
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
