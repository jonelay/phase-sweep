"""Elmer independent-code verification of the radial FEM solver.

Runs a native ElmerSolver install (see README)
on a structured polar mesh of the normalized Zhu baseline, runs the
NGSolve `fem` model on the same cases, and pins the comparison in
pinned/elmer_radial_cv.json. The committed JSON is re-checked in the CI
core lane by tests/crossval/test_elmer_radial_pinned.py with no Elmer
and no NGSolve. Shared parameters, tolerances and analytical references:
tests/crossval/radial_fem_contract.py.

Cases:
smooth-bore linear, discrete arcs alpha_p 0.8/0.95, nonlinear BH on the
shared 19-point table, slotted open-circuit, slotted + j_s current
sheet. Rotation/cogging is deliberately excluded: at these mesh levels
the sub-0.1% ripple is discretization noise in both codes and only the
mean agrees — a cross-check needs a convergence study first (§5.5).

Usage (from the repo root, repo venv):
  .venv/bin/python scripts/elmer_radial/run_elmer_radial.py
  .venv/bin/python scripts/elmer_radial/run_elmer_radial.py --check
      (re-validate the pinned JSON against the contract; no Elmer,
       no NGSolve)
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from mesh_writer import write_radial_mesh
from parse_saveline import fundamental_B1, read_gapline

from tests.crossval.radial_fem_contract import (
    ANALYTICAL_REL_TOL,
    B_REM,
    EXPECTED_CASES,
    J_S,
    K_W,
    MU_R_FE,
    MU_R_PM,
    N_P,
    NONTRIVIAL_PEAK_T,
    baseline_geometry,
    case_tolerance,
    slotless_reference_B1,
    slotted_geometry,
)

MU0 = 1.25663706143592e-6   # H/m — matches the SIF Constants block
RESULTS_JSON = SCRIPT_DIR / "pinned" / "elmer_radial_cv.json"
TEMPLATE = SCRIPT_DIR / "radial.sif.template"
N_WAVEFORM_STORE = 200


def _iron_block(nonlinear: bool) -> str:
    if not nonlinear:
        return f"  Relative Permeability = {MU_R_FE:.6e}\n"
    # The fem solver's generic 19-point soft-iron table, written H-first
    # per Elmer's H-B Curve row order (phasesweep stores it as H(B)).
    # Block form is mandatory: the sized-array form H-B Curve(19,2)
    # fails to parse (README trap 3).
    from phasesweep.solvers.fem_materials import _BH_B, _BH_H
    rows = "\n      ".join(f"{h:.6e} {b:.6e}" for h, b in zip(_BH_H, _BH_B))
    return ('  H-B Curve = Variable "dummy"\n'
            '    Real Monotone Cubic\n'
            f'      {rows}\n'
            '    End\n')


def _nl_block(nonlinear: bool) -> str:
    # Nonlinear System Max Iterations defaults to 1 — omitting it runs a
    # nonlinear case as one linear iteration, silently (README trap 4).
    if not nonlinear:
        return "  Nonlinear System Max Iterations = 1\n"
    return ("  Nonlinear System Max Iterations = 100\n"
            "  Nonlinear System Convergence Tolerance = 1e-7\n"
            "  Nonlinear System Newton After Tolerance = 1e-3\n"
            "  Nonlinear System Relaxation Factor = 1.0\n")


def _bodyforce(j_s: float, n_p: int) -> tuple[str, str]:
    if j_s == 0.0:
        return "", ""
    section = (
        "Body Force 1\n"
        "  Current Density = Variable Coordinate 1, Coordinate 2\n"
        f'    Real MATC "{-j_s * K_W:.10e}*cos({n_p}*atan2(tx(1),tx(0)))"\n'
        "End\n")
    return "  Body Force = 1\n", section


def run_elmer_case(
    case_dir: Path, *, geo, alpha_p: float, n_theta_mesh: int,
    nonlinear: bool = False, j_s: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Mesh + SIF + native ElmerSolver; returns (theta, B_r, info)."""
    case_dir.mkdir(parents=True, exist_ok=True)
    info = write_radial_mesh(
        case_dir / "mesh", r_outer=geo.r_outer, r_stator=geo.r_stator,
        r_ag=geo.r_ag, r_magnet=geo.r_magnet, r_rotor=geo.r_rotor,
        n_p=N_P, alpha_p=alpha_p, n_theta=n_theta_mesh,
        n_slots=geo.n_slots, slot_depth=geo.slot_depth,
        slot_width_ratio=geo.slot_width_ratio)
    M = B_REM / (MU0 * MU_R_PM)
    body6_extra, bf_section = _bodyforce(j_s, N_P)
    (case_dir / "case.sif").write_text(TEMPLATE.read_text().format(
        iron_block=_iron_block(nonlinear), nl_block=_nl_block(nonlinear),
        body6_extra=body6_extra, bodyforce_section=bf_section,
        mu_r_pm=MU_R_PM, M=M, n_p=N_P))
    proc = subprocess.run(["ElmerSolver", "case.sif"], cwd=case_dir,
                          capture_output=True, text=True)
    log = proc.stdout + proc.stderr
    (case_dir / "solver.log").write_text(log)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ElmerSolver failed (exit {proc.returncode}) in {case_dir}; "
            f"see solver.log. Tail:\n{log[-3000:]}")
    th, Br = read_gapline(case_dir / "gapline.dat")
    peak = float(np.max(np.abs(Br)))
    if peak < NONTRIVIAL_PEAK_T:
        # Known silent-zero class: exit 0, no warning, trivially zero
        # field (magnetization in a Body Force; bad Mortar BC index).
        raise RuntimeError(
            f"Elmer returned a trivial field in {case_dir}: peak |B_r| = "
            f"{peak:.3e} T < {NONTRIVIAL_PEAK_T} T non-triviality floor")
    it = re.findall(r"NS \(ITER=\s*(\d+)\).*:: mgdyn2d", log)
    info["nonlinear_iterations"] = int(it[-1]) if (nonlinear and it) else None
    # The MAIN: Version: line carries the build identity (rev + compile
    # date) and is stable across runs; the ELMER SOLVER banner only has
    # a run timestamp.
    info["version_banner"] = next(
        (ln.split("MAIN:", 1)[1].strip() for ln in log.splitlines()
         if "MAIN:" in ln and "Version:" in ln),
        next((ln.strip() for ln in log.splitlines()
              if "ELMER SOLVER" in ln), ""))
    return th, Br, info


def run_ngsolve_case(*, geo, alpha_p: float, n_theta: int,
                     nonlinear: bool = False, j_s: float = 0.0) -> dict:
    from phasesweep.solvers.fem_field import solve_field_fem
    from phasesweep.solvers.harmonics import harmonics_1sided
    kwargs = dict(geo=geo, n_p=N_P, B_rem=B_REM, mu_r_pm=MU_R_PM,
                  mu_r_fe=MU_R_FE, maxh_fraction=0.05, n_theta=n_theta,
                  alpha_p=alpha_p)
    if nonlinear:
        kwargs["nonlinear"] = True
    if j_s:
        kwargs.update(j_s=j_s, k_w=K_W)
    _, Br = solve_field_fem(**kwargs)
    return {"B1_T": float(harmonics_1sided(Br)[N_P]),
            "peak_T": float(np.max(np.abs(Br))),
            "maxh_fraction": 0.05, "n_theta": n_theta}


def _decimate(th: np.ndarray, Br: np.ndarray, n: int) -> dict:
    idx = np.linspace(0, len(th) - 1, min(n, len(th))).round().astype(int)
    return {"theta_rad": th[idx].tolist(), "Br_T": Br[idx].tolist()}


CASES = [
    dict(name="smooth_bore",
         description="full PM ring, linear iron, alpha_p=1",
         alpha_p=1.0, slotted=False, nonlinear=False, j_s=0.0,
         ntm=1440, ntm_coarse=360, ngsolve_n_theta=360),
    dict(name="arcs_a080",
         description="discrete magnet arcs, alpha_p=0.8, linear iron",
         alpha_p=0.80, slotted=False, nonlinear=False, j_s=0.0,
         ntm=720, ngsolve_n_theta=360),
    dict(name="arcs_a095",
         description=("discrete magnet arcs, alpha_p=0.95, linear iron "
                      "(angular quantization worst case: mesh solves "
                      "alpha_p ~ 0.956 at n_theta=720)"),
         alpha_p=0.95, slotted=False, nonlinear=False, j_s=0.0,
         ntm=720, ngsolve_n_theta=360),
    dict(name="nonlinear_bh",
         description=("nonlinear BH on the shared 19-point table, "
                      "alpha_p=0.8; Elmer Newton vs NGSolve Picard"),
         alpha_p=0.80, slotted=False, nonlinear=True, j_s=0.0,
         ntm=720, ngsolve_n_theta=360),
    dict(name="slotted_open",
         description="24 slots, open circuit (j_s=0), alpha_p=1",
         alpha_p=1.0, slotted=True, nonlinear=False, j_s=0.0,
         ntm=1440, ngsolve_n_theta=720),
    dict(name="slotted_js",
         description="24 slots + armature current sheet j_s=5e6 A/m^2",
         alpha_p=1.0, slotted=True, nonlinear=False, j_s=J_S,
         ntm=1440, ngsolve_n_theta=720),
]


def build_results(cases_root: Path) -> dict:
    geo_smooth = baseline_geometry()
    geo_slotted = slotted_geometry()
    results: dict = {
        "generated_by": "scripts/elmer_radial/run_elmer_radial.py",
        "date": date.today().isoformat(),
        "elmer": {},
        "baseline": {
            "geometry": "default_inrunner (normalized Zhu)",
            "n_p": N_P, "B_rem_T": B_REM, "mu_r_pm": MU_R_PM,
            "mu_r_fe": MU_R_FE, "k_w": K_W,
        },
        "cases": {},
    }
    all_pass = True
    for c in CASES:
        geo = geo_slotted if c["slotted"] else geo_smooth
        th, Br, info = run_elmer_case(
            cases_root / c["name"], geo=geo, alpha_p=c["alpha_p"],
            n_theta_mesh=c["ntm"], nonlinear=c["nonlinear"], j_s=c["j_s"])
        if not results["elmer"]:
            results["elmer"] = {"version_banner": info["version_banner"],
                                "install": "native (Ubuntu PPA "
                                           "elmer-csc-ubuntu/elmer-csc-ppa)"}
        B1_e = fundamental_B1(th, Br, N_P)
        rec: dict = {
            "description": c["description"],
            "params": {
                "alpha_p": c["alpha_p"], "n_slots": geo.n_slots,
                "slot_depth": geo.slot_depth,
                "slot_width_ratio": geo.slot_width_ratio,
                "nonlinear": c["nonlinear"], "j_s_A_per_m2": c["j_s"],
                "n_theta_mesh": c["ntm"],
            },
            "elmer": {"B1_T": B1_e, "peak_T": float(np.max(np.abs(Br))),
                      "n_mesh_nodes": info["n_nodes"],
                      "nonlinear_iterations": info["nonlinear_iterations"]},
        }
        if "ntm_coarse" in c:
            th_c, Br_c, _ = run_elmer_case(
                cases_root / f"{c['name']}_coarse", geo=geo,
                alpha_p=c["alpha_p"], n_theta_mesh=c["ntm_coarse"],
                nonlinear=c["nonlinear"], j_s=c["j_s"])
            B1_coarse = fundamental_B1(th_c, Br_c, N_P)
            rec["elmer"]["B1_coarse_T"] = B1_coarse
            rec["elmer"]["mesh_refinement_delta"] = \
                abs(B1_e - B1_coarse) / B1_e
        ng = run_ngsolve_case(geo=geo, alpha_p=c["alpha_p"],
                              n_theta=c["ngsolve_n_theta"],
                              nonlinear=c["nonlinear"], j_s=c["j_s"])
        rec["ngsolve"] = ng
        sdiff = (B1_e - ng["B1_T"]) / ng["B1_T"]
        diff = abs(sdiff)
        tol = case_tolerance(rec)
        rec["rel_diff_cross_code"] = diff
        rec["rel_diff_cross_code_signed"] = sdiff
        rec["cross_code_tol"] = tol
        rec["pass_cross_code"] = bool(diff <= tol)
        line = (f"{c['name']:14s} Elmer B1 {B1_e:.6f} T vs NGSolve "
                f"{ng['B1_T']:.6f} T -> {sdiff:+.3%} (gate {tol:.1%}, "
                f"{'PASS' if diff <= tol else 'FAIL'})")
        if not c["slotted"] and not c["nonlinear"]:
            ref = slotless_reference_B1(geo, N_P, B_REM, MU_R_PM,
                                        c["alpha_p"])
            err = abs(B1_e - ref) / ref
            rec["analytical_B1_T"] = ref
            rec["rel_err_vs_analytical"] = err
            rec["analytical_tol"] = ANALYTICAL_REL_TOL
            rec["pass_analytical"] = bool(err <= ANALYTICAL_REL_TOL)
            line += (f"; vs analytical {ref:.6f} T {err:.3%} "
                     f"({'PASS' if err <= ANALYTICAL_REL_TOL else 'FAIL'})")
        rec["waveform"] = _decimate(th, Br, N_WAVEFORM_STORE)
        results["cases"][c["name"]] = rec
        all_pass &= rec["pass_cross_code"] and rec.get("pass_analytical", True)
        print(line)
    results["all_pass"] = all_pass
    return results


def check_pinned(json_path: Path) -> int:
    """Re-validate a pinned results JSON against the contract — the same
    gates as tests/crossval/test_elmer_radial_pinned.py (no Elmer, no
    NGSolve)."""
    data = json.loads(json_path.read_text())
    base = data["baseline"]
    geo_smooth = baseline_geometry()
    ok = (base["n_p"] == N_P and base["B_rem_T"] == B_REM
          and base["mu_r_pm"] == MU_R_PM and base["mu_r_fe"] == MU_R_FE)
    if not ok:
        print("FAIL: baseline parameters differ from the contract")
    if set(data["cases"]) != EXPECTED_CASES:
        ok = False
        print(f"FAIL: case set {sorted(data['cases'])} != expected "
              f"{sorted(EXPECTED_CASES)}")
    for name, case in data["cases"].items():
        B1_e = case["elmer"]["B1_T"]
        nontrivial = case["elmer"]["peak_T"] >= NONTRIVIAL_PEAK_T
        ok &= nontrivial
        diff = abs(B1_e - case["ngsolve"]["B1_T"]) / case["ngsolve"]["B1_T"]
        tol = case_tolerance(case)
        ok &= diff <= tol
        line = (f"{name:14s} cross-code {diff:.3%} (gate {tol:.1%}, "
                f"{'PASS' if diff <= tol else 'FAIL'})")
        if not nontrivial:
            line += "; TRIVIAL FIELD"
        if "analytical_B1_T" in case:
            ref = slotless_reference_B1(
                geo_smooth, base["n_p"], base["B_rem_T"], base["mu_r_pm"],
                case["params"]["alpha_p"])
            err = abs(B1_e - ref) / ref
            ok &= err <= ANALYTICAL_REL_TOL
            line += (f"; vs analytical {err:.3%} "
                     f"({'PASS' if err <= ANALYTICAL_REL_TOL else 'FAIL'})")
        print(line)
    if {"slotted_open", "slotted_js"} <= set(data["cases"]):
        rise_e = (data["cases"]["slotted_js"]["elmer"]["B1_T"]
                  / data["cases"]["slotted_open"]["elmer"]["B1_T"])
        rise_n = (data["cases"]["slotted_js"]["ngsolve"]["B1_T"]
                  / data["cases"]["slotted_open"]["ngsolve"]["B1_T"])
        rise_ok = abs(rise_e - rise_n) / rise_n <= 0.005
        ok &= rise_ok
        print(f"current-sheet B1 rise: Elmer {rise_e:.4f} vs NGSolve "
              f"{rise_n:.4f} ({'PASS' if rise_ok else 'FAIL'})")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="re-validate the pinned JSON; no Elmer/NGSolve run")
    ap.add_argument("--out", type=Path, default=RESULTS_JSON)
    args = ap.parse_args()
    if args.check:
        return check_pinned(args.out)
    results = build_results(SCRIPT_DIR / "cases")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1) + "\n")
    print(f"wrote {args.out}")
    return 0 if results["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
