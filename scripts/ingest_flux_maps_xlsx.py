"""Extract angle-averaged flux linkage maps from Gierczynski FEM xlsx.

Reads FEMResults_1000rpm.xlsx (Gierczynski & Grzesiak, Energies 2021),
averages ψ_d and ψ_q waveforms over one electrical cycle at each
(i_d, i_q) operating point, and writes a JSON file with the 7×7 maps.

Cross-checks the extracted torque against 1.5·n_p·(ψ_d·i_q − ψ_q·i_d).

Usage:
    uv run --with openpyxl python scripts/ingest_flux_maps_xlsx.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "bmw_i3"
XLSX = DATA_DIR / "FEMResults_1000rpm.xlsx"
OUT = DATA_DIR / "flux_maps_7x7.json"

N_P = 6
I_Q_VALUES = [0, 100, 200, 300, 400, 500, 600]
I_D_VALUES = [-600, -500, -400, -300, -200, -100, 0]
N_IQ = len(I_Q_VALUES)
N_ID = len(I_D_VALUES)

DATA_ROW_START = 6
DATA_ROW_END = 205
N_ANGLES = DATA_ROW_END - DATA_ROW_START + 1

COL_OFFSET = 5


def _read_grid(ws) -> list[list[float]]:
    """Read waveforms and return angle-averaged 7×7 grid [i_q][i_d]."""
    grid = []
    for iq_idx in range(N_IQ):
        row_means = []
        for id_idx in range(N_ID):
            col = COL_OFFSET + iq_idx * N_ID + id_idx
            values = []
            for r in range(DATA_ROW_START, DATA_ROW_END + 1):
                v = ws.cell(r, col).value
                if v is None:
                    raise ValueError(f"Missing value at row {r}, col {col}")
                values.append(float(v))
            row_means.append(sum(values) / len(values))
        grid.append(row_means)
    return grid


def main() -> None:
    try:
        import openpyxl
    except ImportError:
        print("openpyxl required: uv run --with openpyxl python scripts/ingest_flux_maps_xlsx.py",
              file=sys.stderr)
        sys.exit(1)

    wb = openpyxl.load_workbook(str(XLSX), data_only=True)

    psi_d = _read_grid(wb["PsiD"])
    psi_q = _read_grid(wb["PsiQ"])
    torque = _read_grid(wb["Torque"])

    print(f"Extracted {N_IQ}×{N_ID} maps from {XLSX.name}")
    print(f"  ψ_d(0,0) = {psi_d[0][-1]:.2f} mWb  (expect ~43.6)")
    print(f"  ψ_q(0,0) = {psi_q[0][-1]:.2f} mWb  (expect ~0)")

    max_torque_err = 0.0
    max_torque_err_loaded = 0.0
    for iq_idx, i_q in enumerate(I_Q_VALUES):
        for id_idx, i_d in enumerate(I_D_VALUES):
            pd = psi_d[iq_idx][id_idx] * 1e-3
            pq = psi_q[iq_idx][id_idx] * 1e-3
            tau_formula = 1.5 * N_P * (pd * i_q - pq * i_d)
            tau_sheet = torque[iq_idx][id_idx]
            if abs(tau_sheet) > 1.0:
                err = abs(tau_formula - tau_sheet) / abs(tau_sheet)
                max_torque_err = max(max_torque_err, err)
                if abs(tau_sheet) > 10.0:
                    max_torque_err_loaded = max(max_torque_err_loaded, err)

    print(f"  Max torque cross-check error: {max_torque_err:.4%}"
          f" (loaded points >10 Nm: {max_torque_err_loaded:.4%})")
    if max_torque_err_loaded > 0.05:
        print("  WARNING: loaded-point torque cross-check error > 5%",
              file=sys.stderr)

    psi_d_wb = [[round(v, 4) for v in row] for row in psi_d]
    psi_q_wb = [[round(v, 4) for v in row] for row in psi_q]
    torque_wb = [[round(v, 4) for v in row] for row in torque]

    result = {
        "description": (
            "Angle-averaged d-q flux linkage maps from Gierczynski & Grzesiak, "
            "Energies 2021, 14(9), 2343, DOI 10.3390/en14092343. "
            "FEA-extracted (Ansys Maxwell 2D), not measured."
        ),
        "source": "data/bmw_i3/FEMResults_1000rpm.xlsx",
        "license": "CC BY 4.0",
        "motor": "BMW i3 2016 IPM",
        "n_p": N_P,
        "speed_rpm": 1000,
        "convention": {
            "current": "peak",
            "flux_linkage": "peak",
            "frame": "phase (not line-line)",
        },
        "axes": {
            "i_d_A": I_D_VALUES,
            "i_q_A": I_Q_VALUES,
            "note": "i_d <= 0 (demagnetizing), i_q >= 0 (motoring)",
        },
        "layout": "[i_q_idx][i_d_idx]",
        "psi_d_mWb": psi_d_wb,
        "psi_q_mWb": psi_q_wb,
        "torque_Nm": torque_wb,
        "cross_check": {
            "method": "1.5 * n_p * (psi_d * i_q - psi_q * i_d) vs Torque sheet",
            "max_relative_error": round(max_torque_err, 6),
            "max_relative_error_loaded": round(max_torque_err_loaded, 6),
            "note": (
                "Error decreases with load: ~2.3% at i_q=100 A, ~0.3% at "
                "i_q=600 A. Likely a FEM torque-computation-method "
                "discrepancy (dq cross-product vs virtual work); cause not "
                "confirmed. Near-zero torque points show larger relative "
                "error from numerical noise."
            ),
        },
    }

    with open(str(OUT), "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
