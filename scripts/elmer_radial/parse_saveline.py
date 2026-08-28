"""SaveLine output parsing for the radial Elmer harness.

Columns are located by name from the `.dat.names` companion file, never
by position (the axial parse_saveline.py rule). read_gapline returns the
mid-airgap radial flux density (theta, B_r) sorted by theta.
"""
from __future__ import annotations

from math import pi
from pathlib import Path

import numpy as np


def col(names_file: Path, wanted: str) -> int:
    for line in names_file.read_text().splitlines():
        head, _, desc = line.partition(":")
        head = head.strip()
        if head.isdigit() and desc.strip().lower() == wanted.lower():
            return int(head) - 1
    raise KeyError(f"{wanted!r} not in {names_file}")


def read_gapline(dat: Path) -> tuple[np.ndarray, np.ndarray]:
    """(theta [rad, sorted], B_r [T]) from a SaveLine gapline.dat."""
    data = np.loadtxt(dat, ndmin=2)
    names = dat.with_suffix(dat.suffix + ".names")
    x = data[:, col(names, "coordinate 1")]
    y = data[:, col(names, "coordinate 2")]
    th = np.mod(np.arctan2(y, x), 2 * pi)
    Br = (data[:, col(names, "magnetic flux density 1")] * np.cos(th)
          + data[:, col(names, "magnetic flux density 2")] * np.sin(th))
    o = np.argsort(th)
    return th[o], Br[o]


def fundamental_B1(th: np.ndarray, Br: np.ndarray, n_p: int,
                   n: int = 2048) -> float:
    """n_p-th harmonic amplitude by projection on a uniform resample,
    independent of the (mesh-tied) SaveLine node spacing."""
    thu = np.linspace(0, 2 * pi, n, endpoint=False)
    Bu = np.interp(thu, th, Br, period=2 * pi)
    return 2.0 * float(np.abs(np.mean(Bu * np.exp(-1j * n_p * thu))))
