"""One-sided FFT harmonic decomposition for air-gap flux waveforms.

CuPy (GPU) accelerates the batch path when available; single-waveform
helpers are NumPy-only.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

try:
    import cupy as cp
    _CUPY = True
except ImportError:
    cp = None
    _CUPY = False


def _double_onesided(amps: NDArray[np.floating], n_pts: int) -> NDArray[np.floating]:
    """Double all bins except DC and (for even n) the Nyquist bin."""
    out = amps.copy()
    end = -1 if n_pts % 2 == 0 else None
    out[..., 1:end] *= 2
    return out


def harmonics_1sided(B_r: NDArray[np.floating]) -> NDArray[np.floating]:
    """One-sided FFT harmonic amplitudes for a single B_r waveform (NumPy)."""
    n_pts = len(B_r)
    amps = np.abs(np.fft.rfft(B_r)) / n_pts
    return _double_onesided(amps, n_pts)


def compute_thd(amps: NDArray[np.floating], fund_idx: int) -> float:
    """THD as percentage of fundamental from one-sided harmonic amplitudes."""
    fundamental = float(amps[fund_idx])
    if fundamental == 0.0:
        return float("nan")
    return float(
        np.sqrt(max(np.sum(amps[1:] ** 2) - amps[fund_idx] ** 2, 0.0))
        / fundamental * 100
    )


def cogging_angles(
    n_slots: int, n_p: int,
    points_per_period: int = 12, n_periods: int = 1,
) -> NDArray[np.floating]:
    """Uniform angle grid over ``n_periods`` cogging periods.

    One cogging period = 2*pi / lcm(n_slots, 2*n_p).  The grid is
    endpoint=False so the last point is one step before the period
    boundary (suitable for FFT periodicity).
    """
    from math import gcd, pi
    lcm_val = n_slots * 2 * n_p // gcd(n_slots, 2 * n_p)
    period = 2 * pi / lcm_val
    n_points = points_per_period * n_periods
    return np.linspace(0, period * n_periods, n_points, endpoint=False)


def batch_harmonics(B_r_dict: dict[str, NDArray[np.floating]]) -> dict[str, NDArray[np.floating]]:
    """One-sided FFT harmonic amplitudes. CuPy (GPU) if available, else NumPy."""
    names = list(B_r_dict.keys())
    arr = np.stack([B_r_dict[name] for name in names])
    n_pts = arr.shape[-1]

    if _CUPY:
        assert cp is not None
        arr_gpu = cp.asarray(arr)
        amps = cp.asnumpy(cp.abs(cp.fft.rfft(arr_gpu, axis=-1))) / n_pts
    else:
        amps = np.abs(np.fft.rfft(arr, axis=-1)) / n_pts

    amps = _double_onesided(amps, n_pts)
    return {name: amps[i] for i, name in enumerate(names)}
