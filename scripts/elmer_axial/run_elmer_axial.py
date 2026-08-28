"""Track C runner — Elmer-in-Docker independent verification of the
axial-flux r_mean plane. Not a shipped dependency.

Builds the mesh (mesh_writer), fills the SIF template, runs ElmerSolver
in the pinned Docker image, parses the SaveLine mid-airgap output, and
compares against the shared CV contract:

  CV-2 first: alpha_p = 1, n_p = 1 at fixed r_mean (tau_p >> gap stack),
    near-ideal iron -> plateau must hit the 1D circuit limit. This is
    the magnetization-convention check; nothing else is trusted before
    it passes.
  CV-1: Parviainen slotless plane at r_mean vs the analytical layer
    series fundamental, reported against both the 2% analytical gate
    and the 1% cross-code gate (A-vs-C uses the same tolerance).

Usage (from the repo root, repo venv):
  .venv/bin/python scripts/elmer_axial/run_elmer_axial.py \
      [--image elmer-axial:local]
  .venv/bin/python scripts/elmer_axial/run_elmer_axial.py --check
      (re-validate the pinned JSON against the contract, no Docker)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import date
from math import pi
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from mesh_writer import write_pole_mesh
from parse_saveline import (
    fundamental_B1,
    plateau_Bz,
    read_gapline,
)

from tests.crossval.axial_fem_contract import (
    CV1_ANALYTICAL_REL_TOL,
    CV1_CROSS_CODE_REL_TOL,
    CV2_REL_TOL,
    circuit_limit_Bz,
    load_parviainen_motor,
    slotless_reference_B1,
)

MU0 = 1.25663706143592e-6   # H/m — matches the SIF Constants block
DEFAULT_IMAGE = "elmer-axial:local"
RESULTS_JSON = SCRIPT_DIR / "pinned" / "elmer_axial_cv.json"
TEMPLATE = SCRIPT_DIR / "axial_pole.sif.template"
N_LINE = 400                # SaveLine polyline divisions
N_WAVEFORM_STORE = 200      # decimated waveform points kept in the JSON


def run_case(
    case_dir: Path,
    image: str,
    *,
    tau_p: float,
    alpha_p: float,
    B_rem: float,
    mu_r_pm: float,
    mu_r_fe: float,
    t_rotor_yoke: float,
    t_pm: float,
    t_gap: float,
    t_stator_yoke: float,
    nx: int,
    ny_yoke: int = 8,
    ny_pm: int = 8,
    ny_gap: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Mesh + SIF + docker ElmerSolver; returns (x [m], B_z [T])."""
    case_dir.mkdir(parents=True, exist_ok=True)
    info = write_pole_mesh(
        case_dir / "mesh",
        tau_p=tau_p, alpha_p=alpha_p,
        t_rotor_yoke=t_rotor_yoke, t_pm=t_pm, t_gap=t_gap,
        t_stator_yoke=t_stator_yoke,
        nx=nx, ny_yoke=ny_yoke, ny_pm=ny_pm, ny_gap=ny_gap,
    )
    M_y = B_rem / (MU0 * mu_r_pm)  # A/m — B = mu0*mu_r*(H + M) convention
    sif = TEMPLATE.read_text().format(
        y_mid_gap=info["y_mid_gap"], tau_p=tau_p, n_line=N_LINE,
        mu_r_fe=mu_r_fe, mu_r_pm=mu_r_pm, M_y=M_y, M_y_neg=-M_y,
    )
    (case_dir / "case.sif").write_text(sif)
    cmd = [
        "docker", "run", "--rm",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-v", f"{case_dir}:/case", "-w", "/case",
        image, "ElmerSolver", "case.sif",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    (case_dir / "solver.log").write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ElmerSolver failed (exit {proc.returncode}) in {case_dir}; "
            f"see solver.log. Tail:\n{proc.stdout[-2000:]}"
        )
    return read_gapline(case_dir / "gapline.dat")


def _decimate(x: np.ndarray, Bz: np.ndarray, n: int) -> dict:
    idx = np.linspace(0, len(x) - 1, min(n, len(x))).round().astype(int)
    return {"x_m": x[idx].tolist(), "Bz_T": Bz[idx].tolist()}


def _image_id(image: str) -> str:
    out = subprocess.run(
        ["docker", "inspect", image, "--format", "{{.Id}}"],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def build_results(image: str, cases_root: Path) -> dict:
    motor = load_parviainen_motor()
    geo = motor.geometry
    n_p = motor.n_p
    B_rem = motor.B_rem
    mu_r_pm = motor.mu_r_pm
    alpha_p = motor.alpha_p
    mu_r_fe = motor.mu_r_fe

    results: dict = {
        "generated_by": "scripts/elmer_axial/run_elmer_axial.py",
        "date": date.today().isoformat(),
        "elmer_image": {"tag": image, "id": _image_id(image)},
        "cases": {},
    }

    # --- CV-2: magnetization-convention / 1D circuit limit ------------
    # alpha_p = 1, n_p = 1 at fixed r_mean -> tau_p/gap-stack ~ 75.
    # Near-ideal iron (mu_r 1e6) and thick yokes so the reference
    # formula's infinite-permeability assumption holds to << 1%.
    tau_p_cv2 = pi * geo.r_mean / 1.0
    x2, Bz2 = run_case(
        cases_root / "cv2_circuit", image,
        tau_p=tau_p_cv2, alpha_p=1.0, B_rem=B_rem, mu_r_pm=mu_r_pm,
        mu_r_fe=1e6,
        t_rotor_yoke=0.05, t_pm=geo.magnet_thickness, t_gap=geo.airgap,
        t_stator_yoke=0.05, nx=512, ny_yoke=6,
    )
    plateau = plateau_Bz(x2, Bz2, tau_p_cv2)
    ref2 = circuit_limit_Bz(B_rem, mu_r_pm, geo.magnet_thickness, geo.airgap)
    err2 = abs(plateau - ref2) / ref2
    results["cases"]["cv2_circuit"] = {
        "description": "1D circuit limit: alpha_p=1, n_p=1, mu_r_fe=1e6",
        "params": {
            "tau_p_m": tau_p_cv2, "alpha_p": 1.0, "B_rem_T": B_rem,
            "mu_r_pm": mu_r_pm, "mu_r_fe": 1e6,
            "l_pm_m": geo.magnet_thickness, "g_m": geo.airgap, "nx": 512,
        },
        "Bz_plateau_T": plateau,
        "reference_T": ref2,
        "rel_err": err2,
        "tol": CV2_REL_TOL,
        "pass": bool(err2 <= CV2_REL_TOL),
        "waveform": _decimate(x2, Bz2, N_WAVEFORM_STORE),
    }
    print(f"CV-2: plateau {plateau:.6f} T vs circuit {ref2:.6f} T "
          f"-> rel err {err2:.2%} (gate {CV2_REL_TOL:.0%}, "
          f"{'PASS' if err2 <= CV2_REL_TOL else 'FAIL'})")

    # --- CV-1: slotless Parviainen plane at r_mean --------------------
    def cv1(nx: int, ny_scale: int = 1, mu_fe: float = mu_r_fe,
            tag: str = "") -> tuple[float, dict]:
        x1, Bz1 = run_case(
            cases_root / f"cv1_slotless_nx{nx}{tag}", image,
            tau_p=pi * geo.r_mean / n_p, alpha_p=alpha_p, B_rem=B_rem,
            mu_r_pm=mu_r_pm, mu_r_fe=mu_fe,
            t_rotor_yoke=geo.rotor_yoke, t_pm=geo.magnet_thickness,
            t_gap=geo.airgap, t_stator_yoke=geo.stator_yoke,
            nx=nx, ny_yoke=8 * ny_scale, ny_pm=8 * ny_scale,
            ny_gap=6 * ny_scale,
        )
        return fundamental_B1(x1, Bz1, pi * geo.r_mean / n_p), \
            _decimate(x1, Bz1, N_WAVEFORM_STORE)

    ref1 = slotless_reference_B1(geo, n_p, B_rem, mu_r_pm, alpha_p)
    B1_coarse, wf_coarse = cv1(nx=192)
    B1_fine, wf_fine = cv1(nx=384, ny_scale=2)
    err1 = abs(B1_fine - ref1) / ref1
    mesh_delta = abs(B1_fine - B1_coarse) / B1_fine
    results["cases"]["cv1_slotless"] = {
        "description": "Parviainen slotless plane at r_mean, g'=g",
        "params": {
            "tau_p_m": pi * geo.r_mean / n_p, "alpha_p": alpha_p,
            "B_rem_T": B_rem, "mu_r_pm": mu_r_pm, "mu_r_fe": mu_r_fe,
            "l_pm_m": geo.magnet_thickness, "g_m": geo.airgap,
            "nx_coarse": 192, "nx_fine": 384,
        },
        "B1_T": B1_fine,
        "B1_coarse_T": B1_coarse,
        "mesh_refinement_delta": mesh_delta,
        "reference_B1_T": ref1,
        "rel_err": err1,
        "analytical_tol": CV1_ANALYTICAL_REL_TOL,
        "pass_analytical": bool(err1 <= CV1_ANALYTICAL_REL_TOL),
        "cross_code_tol": CV1_CROSS_CODE_REL_TOL,
        "within_cross_code_tol_of_analytical":
            bool(err1 <= CV1_CROSS_CODE_REL_TOL),
        "note": ("cross-code gate is Elmer vs Track A B1; compared at "
                 "merge — the value above is vs the analytical series"),
        "waveform": wf_fine,
    }
    print(f"CV-1: B1 {B1_fine:.6f} T (coarse {B1_coarse:.6f}, mesh delta "
          f"{mesh_delta:.2%}) vs analytical {ref1:.6f} T -> rel err "
          f"{err1:.2%} (2% gate "
          f"{'PASS' if err1 <= CV1_ANALYTICAL_REL_TOL else 'FAIL'}; "
          f"1% cross-code band "
          f"{'within' if err1 <= CV1_CROSS_CODE_REL_TOL else 'outside'})")

    # Diagnostic: same plane with near-ideal iron. The analytical layer
    # series assumes infinitely permeable yokes; the gap between the
    # mu_r_fe=1000 result and the series should collapse here, which
    # attributes the CV-1 residual to yoke MMF rather than a code error.
    B1_ideal, _ = cv1(nx=384, ny_scale=2, mu_fe=1e6, tag="_idealiron")
    err_ideal = abs(B1_ideal - ref1) / ref1
    results["cases"]["cv1_slotless_ideal_iron"] = {
        "description": ("diagnostic: CV-1 plane with mu_r_fe=1e6 — "
                        "isolates the finite-iron MMF share of the "
                        "CV-1 residual (analytical series assumes "
                        "ideal iron)"),
        "B1_T": B1_ideal,
        "reference_B1_T": ref1,
        "rel_err": err_ideal,
    }
    print(f"CV-1 ideal-iron diagnostic: B1 {B1_ideal:.6f} T -> rel err "
          f"{err_ideal:.2%} vs analytical (residual attributed to iron: "
          f"{err1 - err_ideal:+.2%})")
    return results


def check_pinned(json_path: Path) -> int:
    """Re-validate a pinned results JSON against the contract (no Docker)."""
    data = json.loads(json_path.read_text())
    motor = load_parviainen_motor()
    geo = motor.geometry
    ok = True
    cv2 = data["cases"]["cv2_circuit"]
    ref2 = circuit_limit_Bz(motor.B_rem, motor.mu_r_pm,
                            geo.magnet_thickness, geo.airgap)
    err2 = abs(cv2["Bz_plateau_T"] - ref2) / ref2
    ok &= err2 <= CV2_REL_TOL
    print(f"CV-2 pinned: rel err {err2:.3%} vs gate {CV2_REL_TOL:.0%} "
          f"-> {'PASS' if err2 <= CV2_REL_TOL else 'FAIL'}")
    cv1 = data["cases"]["cv1_slotless"]
    ref1 = slotless_reference_B1(geo, motor.n_p, motor.B_rem,
                                 motor.mu_r_pm, motor.alpha_p)
    err1 = abs(cv1["B1_T"] - ref1) / ref1
    ok &= err1 <= CV1_ANALYTICAL_REL_TOL
    print(f"CV-1 pinned: rel err {err1:.3%} vs gate "
          f"{CV1_ANALYTICAL_REL_TOL:.0%} "
          f"-> {'PASS' if err1 <= CV1_ANALYTICAL_REL_TOL else 'FAIL'}; "
          f"1% cross-code band "
          f"{'within' if err1 <= CV1_CROSS_CODE_REL_TOL else 'outside'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--check", action="store_true",
                    help="re-validate the pinned JSON, no Docker run")
    ap.add_argument("--out", type=Path, default=RESULTS_JSON)
    args = ap.parse_args()
    if args.check:
        return check_pinned(args.out)
    cases_root = SCRIPT_DIR / "cases"
    results = build_results(args.image, cases_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
