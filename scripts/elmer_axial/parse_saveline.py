"""Parse Elmer SaveLine ASCII output into {x, B_z, B1}.

SaveLine writes `<name>.dat` (whitespace columns, one row per sample
point) and `<name>.dat.names` (one column description per line, e.g.
"   7: magnetic flux density 2"). We locate columns by name instead of
position so solver/version reordering cannot silently break the parse.

Coordinate mapping (shared physics contract): plane x = circumferential
[m], plane y = axial; the machine's axial field B_z is the plane's
B_y = -dA_z/dx, i.e. Elmer's "magnetic flux density 2" [T].
"""

from __future__ import annotations

from math import pi
from pathlib import Path

import numpy as np


def _column_index(names_file: Path, wanted: str) -> int:
    """0-based column index whose description ends with `wanted`."""
    for line in names_file.read_text().splitlines():
        if ":" not in line:
            continue
        head, _, desc = line.partition(":")
        head = head.strip()
        if head.isdigit() and desc.strip().lower() == wanted.lower():
            return int(head) - 1
    raise KeyError(f"column '{wanted}' not found in {names_file}")


def read_gapline(dat_file: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read (x [m], B_z [T]) along the mid-airgap polyline, sorted by x."""
    dat_file = Path(dat_file)
    data = np.loadtxt(dat_file, ndmin=2)
    names = dat_file.with_suffix(dat_file.suffix + ".names")
    ix = _column_index(names, "coordinate 1")
    ib = _column_index(names, "magnetic flux density 2")
    order = np.argsort(data[:, ix])
    return data[order, ix], data[order, ib]


def fundamental_B1(x: np.ndarray, Bz: np.ndarray, tau_p: float) -> float:
    """Fundamental amplitude [T] of the mirror-extended waveform.

    The solved domain is half an electrical period (one pole pitch);
    mirror-extend with B(x + tau_p) = -B(x) to the full 2*tau_p period,
    resample uniformly (SaveLine points need not be uniform), and
    project onto the fundamental — same convention as the contract's
    slotless_reference_B1.
    """
    n = 4096
    xu = np.linspace(0.0, tau_p, n, endpoint=False)
    Bu = np.interp(xu, x, Bz)
    x_full = np.concatenate([xu, xu + tau_p])
    B_full = np.concatenate([Bu, -Bu])
    phase = np.exp(-1j * pi * x_full / tau_p)  # k=1 over period 2*tau_p
    return 2.0 * float(np.abs(np.mean(B_full * phase)))


def plateau_Bz(x: np.ndarray, Bz: np.ndarray, tau_p: float,
               frac: float = 0.05) -> float:
    """Mean |B_z| [T] over the first `frac` of the pole pitch.

    For the CV-2 case (alpha_p = 1, tau_p >> gap stack) the field is
    flattest at the magnet-center mirror plane x = 0; averaging the
    first few percent rejects single-point interpolation noise.
    """
    sel = x <= frac * tau_p
    if not np.any(sel):
        raise ValueError("no samples in plateau window")
    return float(np.mean(np.abs(Bz[sel])))
