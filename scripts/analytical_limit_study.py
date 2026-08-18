"""Analytical-limit validity check: slotless FEM vs the Zhu closed form.

The Zhu, Howe & Chan (2002) model is the exact solution of the slotless
concentric-annulus problem with infinitely permeable iron boundaries. This
study drives the FEM solver into that regime and measures convergence of
the FEM fundamental B1 toward the analytical value under three controls:

  1. mu_r_fe sweep at fixed fine mesh — the FEM/analytical gap should
     shrink as iron permeability approaches the model's mu -> inf boundary.
  2. mesh refinement at high mu_r_fe — the remaining gap is the true
     discretization error (Richardson-free absolute error meter).
  3. alpha_p sweep — pole-arc handling agreement away from full arcs,
     kept above the known interpole-gap collapse threshold.

Usage:
    uv run python scripts/analytical_limit_study.py

Writes output/convergence/analytical_limits.json and prints a summary.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from phasesweep.machines.geometry import default_inrunner
from phasesweep.solvers.analytical import zhu_howe_Br_series
from phasesweep.solvers.fem_field import solve_field_fem
from phasesweep.solvers.harmonics import harmonics_1sided

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "output" / "convergence"

N_P = 4
B_REM = 1.2
MU_R_PM = 1.05
N_THETA = 720

MU_FE_SWEEP = (500.0, 2000.0, 5000.0, 2e4, 1e5)
MAXH_SWEEP = (0.1, 0.0707, 0.05, 0.0354, 0.025, 0.0177)
ALPHA_P_SWEEP = (1.0, 0.95, 0.9, 0.8, 0.7)


def _fem_B1(geo, mu_r_fe: float, maxh: float, alpha_p: float) -> float:
    _, B_r = solve_field_fem(
        geo=geo, n_p=N_P, B_rem=B_REM, mu_r_pm=MU_R_PM, mu_r_fe=mu_r_fe,
        maxh_fraction=maxh, n_theta=N_THETA, alpha_p=alpha_p,
    )
    return float(harmonics_1sided(B_r)[N_P])


def _ana_B1(geo, alpha_p: float) -> float:
    theta = np.linspace(0.0, 2.0 * np.pi, N_THETA, endpoint=False)
    B_r = zhu_howe_Br_series(
        theta, N_P, B_REM,
        r_stator=geo.r_stator, r_magnet=geo.r_magnet, r_rotor=geo.r_rotor,
        mu_r_pm=MU_R_PM, alpha_p=alpha_p,
    )
    return float(harmonics_1sided(B_r)[N_P])


def main() -> None:
    geo = default_inrunner()
    results: dict = {}

    ana = _ana_B1(geo, 1.0)
    print(f"analytical B1 (alpha_p=1) = {ana:.6f} T")

    print("\n--- mu_r_fe sweep (maxh=0.025, alpha_p=1) ---")
    rows = []
    for mu in MU_FE_SWEEP:
        b1 = _fem_B1(geo, mu, 0.025, 1.0)
        rows.append(dict(mu_r_fe=mu, B1_fem=b1, rel_err=(b1 - ana) / ana))
        print(f"  mu_r_fe={mu:>8g}  B1={b1:.6f}  rel err={(b1 - ana) / ana:+.3e}")
    results["mu_fe_sweep"] = dict(B1_analytical=ana, rows=rows)

    print("\n--- mesh sweep (mu_r_fe=1e5, alpha_p=1) ---")
    rows = []
    for maxh in MAXH_SWEEP:
        b1 = _fem_B1(geo, 1e5, maxh, 1.0)
        rows.append(dict(maxh_fraction=maxh, B1_fem=b1, rel_err=(b1 - ana) / ana))
        print(f"  maxh={maxh:.4f}  B1={b1:.6f}  rel err={(b1 - ana) / ana:+.3e}")
    results["mesh_sweep"] = dict(B1_analytical=ana, rows=rows)

    print("\n--- alpha_p sweep (mu_r_fe=1e5, maxh=0.025) ---")
    rows = []
    for ap in ALPHA_P_SWEEP:
        ana_ap = _ana_B1(geo, ap)
        b1 = _fem_B1(geo, 1e5, 0.025, ap)
        rows.append(dict(alpha_p=ap, B1_analytical=ana_ap, B1_fem=b1,
                         rel_err=(b1 - ana_ap) / ana_ap))
        print(f"  alpha_p={ap:.2f}  ana={ana_ap:.6f}  fem={b1:.6f}  "
              f"rel err={(b1 - ana_ap) / ana_ap:+.3e}")
    results["alpha_p_sweep"] = rows

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "analytical_limits.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
