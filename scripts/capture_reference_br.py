"""Capture reference FEM B_r arrays to npz files.

Saves (theta, B_r) for 3 motor configurations. Originally a
pre-refactor baseline for the geometry builder (that migration is
long complete); kept because the cached npz feeds README figure 3
(scripts/generate_readme_figures.py). --verify re-runs and diffs
against the stored arrays.

Usage:
    uv run python scripts/capture_reference_br.py [--verify]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from phasesweep.machines.configs import load_motor
from phasesweep.solvers.fem_field import solve_field_fem

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "output" / "reference_br"


def capture_zhu_inrunner():
    """Zhu & Howe 8-pole inrunner with default geometry."""
    from phasesweep.machines.geometry import default_inrunner
    geo = default_inrunner()
    theta, B_r = solve_field_fem(
        geo=geo, n_p=4, B_rem=1.2, mu_r_pm=1.05, mu_r_fe=5000,
        maxh_fraction=0.05, n_theta=360,
    )
    return theta, B_r


def capture_zhu_outrunner():
    """Zhu & Howe 8-pole outrunner (Belkhadir geometry, smooth-bore)."""
    motor = load_motor(ROOT / "motors" / "belkhadir_outrunner.toml")
    geo = motor.geometry
    theta, B_r = solve_field_fem(
        geo=geo, n_p=motor.n_p, B_rem=motor.B_rem,
        mu_r_pm=motor.mu_r_pm, mu_r_fe=motor.mu_r_fe,
        maxh_fraction=0.05, n_theta=360, n_slots=0,
        alpha_p=motor.alpha_p,
    )
    return theta, B_r


def capture_deylami_slotted():
    """Deylami 8p/12s outrunner with slots."""
    motor = load_motor(ROOT / "motors" / "deylami_fan.toml")
    geo = motor.geometry
    theta, B_r = solve_field_fem(
        geo=geo, n_p=motor.n_p, B_rem=motor.B_rem,
        mu_r_pm=motor.mu_r_pm, mu_r_fe=motor.mu_r_fe,
        maxh_fraction=0.03, n_theta=720,
        alpha_p=motor.alpha_p,
    )
    return theta, B_r


def capture_all():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    print("Capturing Zhu inrunner...")
    theta, B_r = capture_zhu_inrunner()
    np.savez(OUT_DIR / "zhu_inrunner.npz", theta=theta, B_r=B_r)
    results["zhu_inrunner"] = (theta, B_r)
    print(f"  peak={np.max(np.abs(B_r)):.4f} T, {len(theta)} pts")

    print("Capturing Zhu outrunner (Belkhadir smooth-bore)...")
    theta, B_r = capture_zhu_outrunner()
    np.savez(OUT_DIR / "zhu_outrunner.npz", theta=theta, B_r=B_r)
    results["zhu_outrunner"] = (theta, B_r)
    print(f"  peak={np.max(np.abs(B_r)):.4f} T, {len(theta)} pts")

    print("Capturing Deylami slotted...")
    theta, B_r = capture_deylami_slotted()
    np.savez(OUT_DIR / "deylami_slotted.npz", theta=theta, B_r=B_r)
    results["deylami_slotted"] = (theta, B_r)
    print(f"  peak={np.max(np.abs(B_r)):.4f} T, {len(theta)} pts")

    print(f"\nSaved to {OUT_DIR}/")
    return results


def verify():
    print("Verifying against reference arrays...")
    all_ok = True
    for name in ["zhu_inrunner", "zhu_outrunner", "deylami_slotted"]:
        ref = np.load(OUT_DIR / f"{name}.npz")
        if name == "zhu_inrunner":
            theta, B_r = capture_zhu_inrunner()
        elif name == "zhu_outrunner":
            theta, B_r = capture_zhu_outrunner()
        else:
            theta, B_r = capture_deylami_slotted()

        max_diff = np.max(np.abs(B_r - ref["B_r"]))
        ok = max_diff < 1e-12
        status = "PASS" if ok else "FAIL"
        print(f"  {name}: max_diff={max_diff:.2e} [{status}]")
        if not ok:
            all_ok = False

    return all_ok


if __name__ == "__main__":
    if "--verify" in sys.argv:
        ok = verify()
        sys.exit(0 if ok else 1)
    else:
        capture_all()
