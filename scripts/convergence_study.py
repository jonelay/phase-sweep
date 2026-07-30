"""Mesh and sampling convergence audit for the FEM reference cases.

Sweeps maxh_fraction in sqrt(2) steps around the production value for the
three reference configurations (Zhu default inrunner, Belkhadir smooth-bore
outrunner, Deylami slotted outrunner), Richardson-extrapolates the
fundamental B1 from the three finest levels, and reports the observed
convergence order plus the discretization error at production settings.
A second pass sweeps n_theta at the production mesh to bound sampling error.

Usage:
    uv run python scripts/convergence_study.py [--quick]

Writes output/convergence/mesh_convergence.json and prints a summary table.
--quick drops the finest mesh level (fast smoke run).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from phasesweep.configs import load_motor
from phasesweep.fem_field import solve_field_fem
from phasesweep.geometry import default_inrunner
from phasesweep.harmonics import compute_thd, harmonics_1sided

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "output" / "convergence"

REFINE = math.sqrt(2.0)
N_THETA_SWEEP = (180, 360, 720, 1440, 2880)

# Real-size arc motors clamp the global maxh to max(interpole_gap/2, 0.5mm)
# (fem_field._get_or_build_mesh), so maxh_fraction only binds once the
# requested maxh drops below that clamp — levels are chosen per case to
# actually change the mesh. The airgap face maxh (gap/3) stays fixed
# throughout; it bounds the achievable B1 accuracy independently of these
# levels (see the companion analytical_limit_study.py for absolute error).


def _levels(clamp_frac: float, n: int) -> list[float]:
    return [clamp_frac / REFINE**k for k in range(n)]


def _cases() -> list[dict]:
    belkhadir = load_motor(ROOT / "motors" / "belkhadir_outrunner.toml")
    deylami = load_motor(ROOT / "motors" / "deylami_fan.toml")
    return [
        dict(name="zhu_inrunner", geo=default_inrunner(), n_p=4, B_rem=1.2,
             mu_r_pm=1.05, mu_r_fe=5000.0, alpha_p=1.0, kw={},
             maxh_prod=0.05, levels=[0.05 * REFINE**k for k in range(3, -4, -1)]),
        dict(name="belkhadir_outrunner", geo=belkhadir.geometry,
             n_p=belkhadir.n_p, B_rem=belkhadir.B_rem,
             mu_r_pm=belkhadir.mu_r_pm, mu_r_fe=belkhadir.mu_r_fe,
             alpha_p=belkhadir.alpha_p, kw=dict(n_slots=0), maxh_prod=0.05,
             levels=_levels(0.0193, 5)),
        dict(name="deylami_slotted", geo=deylami.geometry, n_p=deylami.n_p,
             B_rem=deylami.B_rem, mu_r_pm=deylami.mu_r_pm,
             mu_r_fe=deylami.mu_r_fe, alpha_p=deylami.alpha_p, kw={},
             maxh_prod=0.03, levels=_levels(0.01814, 5)),
    ]


def _solve(case: dict, maxh: float, n_theta: int) -> dict:
    t0 = time.perf_counter()
    theta, B_r, mesh, gfu = solve_field_fem(
        geo=case["geo"], n_p=case["n_p"], B_rem=case["B_rem"],
        mu_r_pm=case["mu_r_pm"], mu_r_fe=case["mu_r_fe"],
        maxh_fraction=maxh, n_theta=n_theta, alpha_p=case["alpha_p"],
        return_full=True, **case["kw"],
    )
    dt = time.perf_counter() - t0
    amps = harmonics_1sided(B_r)
    return dict(
        maxh_fraction=maxh, n_theta=n_theta,
        B1=float(amps[case["n_p"]]),
        thd_pct=compute_thd(amps, case["n_p"]),
        B_peak=float(np.max(np.abs(B_r))),
        n_elements=int(mesh.ne), ndof=int(gfu.space.ndof),
        solve_s=round(dt, 2),
    )


def _richardson(rows: list[dict], key: str) -> dict:
    """Observed order + extrapolated value from the three finest levels."""
    f1, f2, f3 = (r[key] for r in rows[-3:])  # coarse -> fine
    if f2 == f3 or f1 == f2:
        return dict(order=None, extrapolated=f3, note="converged to sampling floor")
    ratio = (f1 - f2) / (f2 - f3)
    if ratio <= 1.0:
        return dict(order=None, extrapolated=f3, note="non-monotone tail")
    p = math.log(ratio) / math.log(REFINE)
    f_ext = f3 + (f3 - f2) / (REFINE**p - 1.0)
    return dict(order=round(p, 2), extrapolated=f_ext, note=None)


def run_case(case: dict, quick: bool) -> dict:
    prod = case["maxh_prod"]
    levels = case["levels"][:-1] if quick else case["levels"]
    print(f"\n=== {case['name']} (production maxh_fraction={prod}) ===")
    mesh_rows = []
    for maxh in levels:
        row = _solve(case, maxh, n_theta=720)
        mesh_rows.append(row)
        print(f"  maxh={maxh:.5f}  ne={row['n_elements']:>7}  "
              f"B1={row['B1']:.6f}  THD={row['thd_pct']:.3f}%  "
              f"({row['solve_s']}s)")

    rich = _richardson(mesh_rows, "B1")
    # For the arc motors the production request lands on the clamp, which is
    # exactly the coarsest level swept — that row is the production mesh.
    prod_row = next(
        (r for r in mesh_rows if math.isclose(r["maxh_fraction"], prod)),
        mesh_rows[0],
    )
    err_prod = abs(prod_row["B1"] - rich["extrapolated"]) / abs(rich["extrapolated"])
    err_fine = abs(mesh_rows[-1]["B1"] - rich["extrapolated"]) / abs(rich["extrapolated"])

    ntheta_rows = [_solve(case, prod, n_theta=n) for n in N_THETA_SWEEP]
    b1s = [r["B1"] for r in ntheta_rows]
    ntheta_spread = (max(b1s) - min(b1s)) / abs(b1s[-1])

    print(f"  observed order p={rich['order']}  "
          f"B1_ext={rich['extrapolated']:.6f}  "
          f"B1 err @production={err_prod:.3e}  @finest={err_fine:.3e}  "
          f"n_theta spread={ntheta_spread:.2e}"
          + (f"  [{rich['note']}]" if rich["note"] else ""))
    return dict(
        name=case["name"], maxh_production=prod, mesh_sweep=mesh_rows,
        richardson_B1=rich, B1_rel_err_at_production=err_prod,
        B1_rel_err_at_finest=err_fine, ntheta_sweep=ntheta_rows,
        ntheta_B1_rel_spread=ntheta_spread,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    results = [run_case(c, args.quick) for c in _cases()]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "mesh_convergence.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
