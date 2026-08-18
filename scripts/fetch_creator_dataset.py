#!/usr/bin/env python3
"""Download the CREATOR dataset and extract the files phase-sweep needs.

The CREATOR open-benchmark PMSM dataset (DOI 10.3217/sns1d-77m43) is
published under CC BY-NC-ND 4.0. We do not redistribute its contents;
this script fetches from the canonical source and writes the derived
CSVs and JSON import-format files that the test suite expects.

Usage:
    python scripts/fetch_creator_dataset.py          # default: data/creator_case_pmsm/
    python scripts/fetch_creator_dataset.py /tmp/out  # custom output directory
"""
from __future__ import annotations

import csv
import io
import json
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

DATASET_DOI = "10.3217/sns1d-77m43"
ZIP_URL = "https://repository.tugraz.at/records/sns1d-77m43/files/PM_synchronous_motor.zip?download=1"
ZIP_NAME = "PM_synchronous_motor.zip"

EXTRACTIONS = {
    "ferrite_bh.csv": {
        "zip_path": "PM_synchronous_motor/Design_parameters/Material_properties/Ferrite/Ferrite_BH.csv",
        "header": "B_T,H_A_m",
    },
    "iron_losses_measured.csv": {
        "zip_path": "PM_synchronous_motor/Measurement_results/No_load_tests/No_load_iron_losses.csv",
        "header": "frequency_Hz,iron_loss_W",
    },
    "m250_35a_bh_50hz.csv": {
        "zip_path": "PM_synchronous_motor/Design_parameters/Material_properties/Electrical_steel/Data/50_Hz.csv",
        "header": "B_T,H_A_m,loss_W_kg,mu_r",
    },
    "no_load_torque_measured.csv": {
        "zip_path": "PM_synchronous_motor/Measurement_results/No_load_tests/M20231018.csv",
        "header": "speed_rpm,torque_Nm",
    },
}

BACK_EMF_ZIP_PATH = (
    "PM_synchronous_motor/Measurement_results/No_load_tests/Back_emf.csv"
)
N_P = 2  # pole pairs for the CREATOR 4-pole motor

REPO_ROOT = Path(__file__).resolve().parent.parent


def _reheader(raw_bytes: bytes, new_header: str) -> str:
    text = raw_bytes.decode("utf-8-sig").strip()
    lines = text.splitlines()
    rows = list(csv.reader(lines))
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(new_header.split(","))
    for row in rows[1:]:
        writer.writerow([float(v) for v in row])
    return out.getvalue()


def _generate_backemf_json(zf: zipfile.ZipFile, out_dir: Path) -> None:
    """Generate backemf_measured.json from Back_emf.csv via FFT."""
    raw = zf.read(BACK_EMF_ZIP_PATH)
    text = raw.decode("utf-8-sig").strip()
    lines = text.splitlines()
    reader = csv.reader(lines[1:])
    u_phase = np.array([float(row[1]) for row in reader])

    fft_mag = 2.0 * np.abs(np.fft.rfft(u_phase)) / len(u_phase)
    backemf_fundamental = round(float(fft_mag[N_P]), 2)

    data = {
        "motor_name": "CREATOR Case PMSM",
        "test_type": "backemf_capture",
        "conditions": {
            "speed_rpm": 2000,
            "temperature_C": 22.0,
            "load_torque_Nm": 0.0,
            "date": "2024-01-15",
            "instrument": "oscilloscope",
            "notes": "no-load back-EMF measurement",
        },
        "quantities": {"backemf_fundamental": backemf_fundamental},
        "waveforms": {},
        "uncertainty": {"backemf_fundamental": 0.5},
        "source_file": (
            f"CREATOR dataset (DOI {DATASET_DOI}), {BACK_EMF_ZIP_PATH}"
        ),
        "source": "measured",
        "tolerances": {"backemf_fundamental": 5.0},
        "provenance_note": (
            "Genuine lab measurement from the CREATOR dataset "
            f"(TU Graz EALS, DOI {DATASET_DOI}), NOT a passthrough of "
            "the paper's Table 10 circuit value. "
            f"{backemf_fundamental} V is the electrical-fundamental peak "
            "from an FFT of Back_emf.csv phase U (mechanical harmonic "
            f"{N_P} = electrical fundamental for n_p={N_P}). "
            "Independent of the stored psi_f=0.1144 Wb."
        ),
    }
    dest = out_dir / "backemf_measured.json"
    dest.write_text(json.dumps(data, indent=2) + "\n")
    print(f"  backemf_measured.json: fundamental={backemf_fundamental} V")


def _generate_iron_loss_json(out_dir: Path) -> None:
    """Generate iron_loss_noload.json from the already-extracted CSV."""
    csv_path = out_dir / "iron_losses_measured.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} must be extracted first (run CSV extractions)"
        )
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        freqs, losses = [], []
        for row in reader:
            freqs.append(float(row["frequency_Hz"]))
            losses.append(float(row["iron_loss_W"]))

    rated_freq = 66.667
    rated_idx = min(range(len(freqs)), key=lambda i: abs(freqs[i] - rated_freq))
    p_fe_rated = losses[rated_idx]

    data = {
        "motor_name": "CREATOR Case PMSM",
        "test_type": "iron_loss_test",
        "conditions": {
            "speed_rpm": 2000.0,
            "temperature_C": 20.0,
            "load_torque_Nm": 0.0,
            "date": "2025-01-27",
            "instrument": (
                "CREATOR dataset no-load loss measurement "
                f"(dataset DOI {DATASET_DOI})"
            ),
            "notes": (
                f"No-load iron loss at the rated point, 2000 rpm mechanical "
                f"= {rated_freq} Hz electrical for n_p = {N_P}. The single "
                f"scalar is the {rated_freq} Hz row of "
                "iron_losses_measured.csv verbatim; the full "
                f"{len(freqs)}-point {freqs[0]}-{freqs[-1]} Hz sweep is "
                "carried in waveforms for reference."
            ),
        },
        "quantities": {"p_fe": p_fe_rated},
        "waveforms": {
            "frequency_Hz": freqs,
            "p_fe_W": losses,
        },
        "uncertainty": {},
        "tolerances": {"p_fe": 5.0},
        "source_file": (
            f"data/creator_case_pmsm/iron_losses_measured.csv "
            f"(extracted from CREATOR dataset DOI {DATASET_DOI})"
        ),
        "source": "published",
        "provenance_note": (
            "DELIBERATELY UNTAGGED, and the reason is the opposite of the "
            "usual one. The shipped motors/creator_case_pmsm.toml B_core WAS "
            "derived from this dataset -- it is the calibration output, "
            "fitted here -- so by the letter of the derived_params rule this "
            "dataset would be tagged ['B_core'] and every fit against it "
            "refused. That would leave the calibration with no reproducible "
            "record at all. The tag stays off so the fit is runnable from the "
            "UNFITTED starting value (B_core = 0.273 T, the FEM per-pole-flux "
            "estimate), which is what scripts/calibrate_creator_b_core.py "
            "does; see b_core_calibration.record.json. The hazard the tag "
            "would have covered is instead named outright: fitting B_core "
            "against this dataset starting from the SHIPPED motor returns a "
            "zero delta and a closed record, and that is an echo, not "
            "agreement. CREATOR does not validate iron_loss."
        ),
    }
    dest = out_dir / "iron_loss_noload.json"
    dest.write_text(json.dumps(data, indent=2) + "\n")
    print(f"  iron_loss_noload.json: p_fe={p_fe_rated} W at {rated_freq} Hz")


def main(out_dir: Path | None = None) -> None:
    if out_dir is None:
        out_dir = REPO_ROOT / "data" / "creator_case_pmsm"
    out_dir.mkdir(parents=True, exist_ok=True)

    zip_path = out_dir / ZIP_NAME
    if not zip_path.exists():
        print(f"Downloading CREATOR dataset from {ZIP_URL} ...")
        urlretrieve(ZIP_URL, zip_path)
        print(f"  saved to {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"Using cached {zip_path}")

    with zipfile.ZipFile(zip_path) as zf:
        for dest_name, spec in EXTRACTIONS.items():
            raw = zf.read(spec["zip_path"])
            content = _reheader(raw, spec["header"])
            dest = out_dir / dest_name
            dest.write_text(content)
            n_rows = content.count("\n") - 1
            print(f"  {dest_name}: {n_rows} rows from {spec['zip_path']}")

        _generate_backemf_json(zf, out_dir)

    _generate_iron_loss_json(out_dir)

    full_dir = out_dir / "PM_synchronous_motor"
    if not full_dir.exists():
        print("\nFull dataset not extracted. For Tier 2 tests, run:")
        print(f"  cd {out_dir} && unzip {ZIP_NAME}")

    print("\nDone.")


if __name__ == "__main__":
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    main(dest)
