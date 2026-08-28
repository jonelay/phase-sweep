#!/usr/bin/env python3
"""Download the CREATOR induction-motor dataset and extract what phase-sweep needs.

The CREATOR open-benchmark induction machine dataset (DOI
10.3217/kh3v7-dhn98) is published under CC BY-NC 4.0 (note: NC, not
NC-ND — the PMSM sibling dataset carries the stricter ND clause). We do
not redistribute its contents; this script fetches from the canonical
source and extracts the files the test suite expects:

  - the M800-50A steel loss tables (9 frequencies, 50-1000 Hz) and BH curve
  - the measured equivalent-circuit parameter CSVs (deep-bar R_r(f),
    leakage L_sigma(f), magnetizing saturation L_m(I_m))
  - the no-load / locked-rotor iron and friction+windage loss CSVs
  - flat CSV conversions of the raw locked-rotor and no-load test .mat
    structs (one row per measured point)

Usage:
    python scripts/fetch_creator_im_dataset.py           # default: data/creator_case_im/
    python scripts/fetch_creator_im_dataset.py /tmp/out  # custom output directory
"""
from __future__ import annotations

import csv
import io
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

DATASET_DOI = "10.3217/kh3v7-dhn98"
ZIP_URL = "https://repository.tugraz.at/records/kh3v7-dhn98/files/Induction_motor.zip?download=1"
ZIP_NAME = "Induction_motor.zip"

# Members extracted verbatim, preserving the zip's tree layout (tests read
# them at these paths, mirroring the PMSM dataset's extracted-tree pattern).
TREE_MEMBERS = [
    "Induction_motor/IM_general_specification.csv",
    "Induction_motor/Design_parameters/Electrical_parameters/Electrical_properties_of_IM.csv",
    "Induction_motor/Design_parameters/Motor_geometry/Geometry_parameters.csv",
    "Induction_motor/Design_parameters/Winding_scheme/Winding_properties_of_IM.csv",
    "Induction_motor/Design_parameters/Material_properties/Stator_and_rotor_iron/B_H.csv",
    "Induction_motor/Design_parameters/Material_properties/Stator_and_rotor_iron/Stator_and_rotor_iron_properties.csv",
    "Induction_motor/Measurement_results/Equivalent_circuit_parameters/Equivalent_circuit_parameters.csv",
    "Induction_motor/Measurement_results/Equivalent_circuit_parameters/Rr.csv",
    "Induction_motor/Measurement_results/Equivalent_circuit_parameters/L_sigmas.csv",
    "Induction_motor/Measurement_results/Equivalent_circuit_parameters/Lm.csv",
    "Induction_motor/Measurement_results/Locked_rotor_and_no_load_tests/Pfe_flux_sq.csv",
    "Induction_motor/Measurement_results/Locked_rotor_and_no_load_tests/Pfw_flux_sq.csv",
]

STEEL_DATA_PREFIX = (
    "Induction_motor/Design_parameters/Material_properties/"
    "Stator_and_rotor_iron/Stator_and_rotor_iron_data/"
)

MAT_MEMBERS = {
    "Induction_motor/Measurement_results/Locked_rotor_and_no_load_tests/"
    "Locked_rotor_data.mat": ("LRs", "locked_rotor_tests.csv"),
    "Induction_motor/Measurement_results/Locked_rotor_and_no_load_tests/"
    "No_load_data.mat": ("NLs", "no_load_tests.csv"),
}

REPO_ROOT = Path(__file__).resolve().parent.parent


def _extract_member(zf: zipfile.ZipFile, member: str, out_dir: Path) -> None:
    dest = out_dir / member
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(zf.read(member))


def _convert_mat(zf: zipfile.ZipFile, member: str, struct_name: str,
                 dest: Path) -> int:
    """Flatten a locked-rotor / no-load .mat struct to one row per point.

    The struct is (1, n_freq); each frequency row carries parallel arrays
    Vs / Is / P / f / Tshaft (per-point) and a scalar Rs (the DC stator
    resistance measured alongside that run). Vs is PER-PHASE rms — verified
    by power balance: at the 10 Hz locked-rotor points
    P = 3·Vs·Is·pf closes with the circuit-model power factor to ~1.3%.
    """
    import numpy as np
    from scipy.io import loadmat

    m = loadmat(io.BytesIO(zf.read(member)))
    s = m[struct_name]
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow([
        "f_supply_Hz", "V_phase_rms_V", "I_phase_rms_A", "P_in_W",
        "T_shaft_Nm", "R_s_dc_ohm",
    ])
    n = 0
    for i in range(s.shape[1]):
        row = s[0, i]
        Vs = np.atleast_1d(row["Vs"]).ravel()
        Is = np.atleast_1d(row["Is"]).ravel()
        P = np.atleast_1d(row["P"]).ravel()
        f = np.atleast_1d(row["f"]).ravel()
        T = np.atleast_1d(row["Tshaft"]).ravel()
        Rs = float(np.atleast_1d(row["Rs"]).ravel()[0])
        for v, i_a, p, fi, t in zip(Vs, Is, P, f, T):
            writer.writerow([
                f"{fi:.6g}", f"{v:.6g}", f"{i_a:.6g}", f"{p:.6g}",
                f"{t:.6g}", f"{Rs:.6g}",
            ])
            n += 1
    dest.write_text(out.getvalue())
    return n


def main(out_dir: Path | None = None) -> None:
    if out_dir is None:
        out_dir = REPO_ROOT / "data" / "creator_case_im"
    out_dir.mkdir(parents=True, exist_ok=True)

    zip_path = out_dir / ZIP_NAME
    if not zip_path.exists():
        print(f"Downloading CREATOR IM dataset from {ZIP_URL} ...")
        urlretrieve(ZIP_URL, zip_path)
        print(f"  saved to {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"Using cached {zip_path}")

    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        steel = sorted(n for n in names
                       if n.startswith(STEEL_DATA_PREFIX) and n.endswith(".csv"))
        for member in TREE_MEMBERS + steel:
            _extract_member(zf, member, out_dir)
        print(f"  extracted {len(TREE_MEMBERS)} parameter/measurement CSVs "
              f"+ {len(steel)} steel loss tables")

        for member, (struct, dest_name) in MAT_MEMBERS.items():
            n = _convert_mat(zf, member, struct, out_dir / dest_name)
            print(f"  {dest_name}: {n} points from {Path(member).name}")

    print("\nDone.")


if __name__ == "__main__":
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    main(dest)
