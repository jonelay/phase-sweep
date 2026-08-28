"""Per-motor offline 3D wedge k_end solve with a committed provenance record.

Runs fem_wedge.solve_wedge_endeffect at production settings for one motor
TOML and writes a JSON record (linear + nonlinear k_end, settings, wall
times). The nonlinear k_end is the headline value intended for the motor's
`[geometry] k_end` field; the record is the provenance behind it.

The magnet overhang comes from the TOML's `[geometry] magnet_overhang`
(meters per side, beyond the active stack) or the --overhang override.
Idealizations inherited from the wedge solver: smooth bore, full-pitch
(alpha_p = 1) magnet ring, distributed-winding fundamental, flux linkage
truncated at the active stack.

B_rem cancels in the LINEAR k_end ratio only. The nonlinear headline
depends on the assumed B_rem through iron saturation, so it is computed
at the nominal datasheet B_rem (recorded) and the record also carries a
sensitivity solve at a second B_rem — if the two differ materially, the
headline should be revisited at the motor's calibrated effective B_rem.
The nonlinear value also depends on the shared generic BH table
(fem_materials._BH_B/_BH_H); a table change stales every record.

Results are workstation-pinned like the Elmer harnesses: --verify
recomputes and compares against the stored record.

Usage:
    uv run python scripts/compute_k_end.py motors/outrunner_14mm_steel.toml
    uv run python scripts/compute_k_end.py <toml> --overhang 0.0 --out <json>
    uv run python scripts/compute_k_end.py <toml> --verify
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Production solve settings — passed to solve_wedge_endeffect EXPLICITLY
# (not left to its defaults, so a default drift cannot silently diverge
# from what the committed records claim). VERIFY_RTOL covers same-machine
# repeatability (~1e-15) plus mesher/library-version headroom.
SETTINGS = {
    "z_box_factor": 6.0,
    "order": 2,
    "n_z": 97,
    "n_theta": 31,
    "B_rem": 1.45,
    "mu_r_pm": 1.05,
    "mu_r_fe_linear": 1.0e4,
}
# Second nonlinear solve at this B_rem quantifies the saturation
# sensitivity of the headline (B_rem does NOT cancel in the nonlinear
# ratio); chosen near the fleet's calibrated effective values.
B_REM_SENSITIVITY = 1.25
VERIFY_RTOL = 1e-3


def compute_record(toml_path: Path, overhang: float | None,
                   yoke_overhang: float | None,
                   note: str | None = None) -> dict:
    from phasesweep.machines.configs import load_motor
    from phasesweep.machines.geometry import Geometry
    from phasesweep.solvers.fem_wedge import solve_wedge_endeffect

    motor = load_motor(toml_path)
    geo = motor.geometry
    if geo is None or not isinstance(geo, Geometry):
        raise SystemExit(f"{toml_path}: k_end needs a radial Geometry")
    if motor.L_stk is None:
        raise SystemExit(f"{toml_path}: k_end needs L_stk")
    if overhang is None:
        overhang = geo.magnet_overhang
    if overhang is None:
        raise SystemExit(
            f"{toml_path}: no magnet overhang — set [geometry] "
            f"magnet_overhang (0.0 = measured co-extensive) or pass "
            f"--overhang"
        )
    if yoke_overhang is None:
        yoke_overhang = overhang  # rotor cup follows the magnet

    # Mesh sizes made explicit (they are also the fem_wedge defaults);
    # recorded numerically below so the record is self-describing.
    gap = abs(geo.r_stator - geo.r_magnet)
    maxh = gap / 3.0
    maxh_bulk = geo.r_outer / 12.0
    kw = dict(
        n_p=motor.n_p, L_stk=motor.L_stk,
        mag_overhang=overhang, yoke_overhang=yoke_overhang,
        mu_r_pm=SETTINGS["mu_r_pm"], mu_r_fe=SETTINGS["mu_r_fe_linear"],
        z_box_factor=SETTINGS["z_box_factor"], order=SETTINGS["order"],
        n_z=SETTINGS["n_z"], n_theta=SETTINGS["n_theta"],
        maxh=maxh, maxh_bulk=maxh_bulk,
    )

    print(f"{motor.name}: linear solve ...", flush=True)
    t0 = time.perf_counter()
    lin = solve_wedge_endeffect(geo, B_rem=SETTINGS["B_rem"], **kw)
    t_lin = time.perf_counter() - t0
    print(f"  k_end_linear = {lin.k_end:.6f}  ({t_lin:.0f} s)", flush=True)

    print(f"{motor.name}: nonlinear solve (newton, picard fallback) ...",
          flush=True)
    t0 = time.perf_counter()
    nl = solve_wedge_endeffect(geo, B_rem=SETTINGS["B_rem"], nonlinear=True,
                               nonlinear_solver="newton", **kw)
    t_nl = time.perf_counter() - t0
    print(f"  k_end (nonlinear) = {nl.k_end:.6f}  ({t_nl:.0f} s)", flush=True)

    print(f"{motor.name}: nonlinear sensitivity at B_rem = "
          f"{B_REM_SENSITIVITY} ...", flush=True)
    t0 = time.perf_counter()
    sens = solve_wedge_endeffect(geo, B_rem=B_REM_SENSITIVITY,
                                 nonlinear=True,
                                 nonlinear_solver="newton", **kw)
    t_sens = time.perf_counter() - t0
    print(f"  k_end @ {B_REM_SENSITIVITY} T = {sens.k_end:.6f}  "
          f"({t_sens:.0f} s)", flush=True)

    record = {
        "motor_name": motor.name,
        "motor_toml": str(toml_path.resolve().relative_to(ROOT)),
        "geometry_config_id": geo.config_id,
        "n_p": motor.n_p,
        "L_stk_m": motor.L_stk,
        "magnet_overhang_m": overhang,
        "yoke_overhang_m": yoke_overhang,
        "settings": {
            **SETTINGS,
            "maxh_m": maxh,
            "maxh_bulk_m": maxh_bulk,
            "bh_table": "fem_materials generic 19-point (_BH_B/_BH_H) — a "
                        "table change stales the nonlinear values",
        },
        "results": {
            # headline: nonlinear (BH iron), the value for [geometry] k_end
            "k_end": round(nl.k_end, 6),
            "k_end_linear": round(lin.k_end, 6),
            "k_end_bore": round(nl.k_end_bore, 6),
            "flux_droop_ratio": round(nl.flux_droop_ratio, 6),
            # B_rem does not cancel in the nonlinear ratio: saturation
            # sensitivity of the headline value
            "k_end_at_B_rem_sens": round(sens.k_end, 6),
            "B_rem_sens": B_REM_SENSITIVITY,
        },
        "wall_s": {"linear": round(t_lin, 1), "nonlinear": round(t_nl, 1),
                   "sensitivity": round(t_sens, 1)},
        "idealizations": (
            "smooth bore; full-pitch (alpha_p=1) magnet; "
            "distributed-winding fundamental; flux linkage truncated at "
            "the active stack; verified against the 2D analytic limit + "
            "convergence, NOT an independent 3D benchmark"
        ),
        "solver": "fem_wedge.solve_wedge_endeffect (NGSolve scalar-phi)",
        "date": datetime.date.today().isoformat(),
    }
    if note:
        record["note"] = note
    return record


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("toml", type=Path, help="motor TOML path")
    ap.add_argument("--overhang", type=float, default=None,
                    help="magnet overhang override (m per side)")
    ap.add_argument("--yoke-overhang", type=float, default=None,
                    help="yoke overhang (m per side; default = magnet)")
    ap.add_argument("--out", type=Path, default=None,
                    help="record path (default data/<stem>/k_end.record.json)")
    ap.add_argument("--note", type=str, default=None,
                    help="provenance note stored in the record (e.g. where "
                         "the overhang number comes from)")
    ap.add_argument("--verify", action="store_true",
                    help="recompute and compare against the stored record")
    args = ap.parse_args()

    out = args.out
    if out is None:
        out = ROOT / "data" / args.toml.stem / "k_end.record.json"

    if args.verify:
        if not out.exists():
            print(f"no record at {out}", file=sys.stderr)
            return 1
        stored = json.loads(out.read_text())
        fresh = compute_record(
            args.toml,
            stored["magnet_overhang_m"] if args.overhang is None
            else args.overhang,
            stored["yoke_overhang_m"] if args.yoke_overhang is None
            else args.yoke_overhang,
        )
        ok = True
        for key in ("k_end", "k_end_linear", "k_end_bore",
                    "flux_droop_ratio", "k_end_at_B_rem_sens"):
            a, b = stored["results"][key], fresh["results"][key]
            rel = abs(a - b) / abs(a)
            status = "ok" if rel <= VERIFY_RTOL else "MISMATCH"
            if rel > VERIFY_RTOL:
                ok = False
            print(f"  {status:8s} {key}: stored {a} vs fresh {b} "
                  f"(rel {rel:.2e})")
        print("PASS" if ok else "FAIL")
        return 0 if ok else 1

    record = compute_record(args.toml, args.overhang, args.yoke_overhang,
                            note=args.note)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
