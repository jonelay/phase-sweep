"""One-sided FFT harmonic decomposition for air-gap flux waveforms.

CuPy (GPU) accelerates the batch path when available; single-waveform
helpers are NumPy-only.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "assert_waveform_symmetry",
    "batch_harmonics",
    "cogging_angles",
    "compute_thd",
    "harmonics_1sided",
]

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
    if not (1 <= fund_idx < len(amps)):
        raise ValueError(
            f"fund_idx={fund_idx} out of range [1, {len(amps)})"
        )
    if not np.all(np.isfinite(amps)):
        raise ValueError("non-finite amplitude(s) in input")
    fundamental = float(amps[fund_idx])
    if fundamental == 0.0:
        return float("nan")
    return float(
        np.sqrt(max(np.sum(amps[1:] ** 2) - amps[fund_idx] ** 2, 0.0))
        / fundamental * 100
    )


def assert_waveform_symmetry(
    amps: NDArray[np.floating],
    n_p: int,
    *,
    label: str = "B_r",
) -> None:
    """Assert odd-harmonic cosine-series invariants on one-sided FFT amplitudes.

    True by construction for a sum of cos(k*n_p*theta), k odd:
    - DC bin is zero
    - Only odd multiples of n_p carry energy
    - (Half-wave antisymmetry and even symmetry about magnet centre follow
      from the cosine-only, odd-k structure.)

    Tolerance: 1e-9 x fundamental amplitude.  Raises RuntimeError on
    violation -- a non-zero forbidden bin means a code bug, not a physics
    issue.
    """
    fundamental = float(amps[n_p]) if n_p < len(amps) else 0.0
    tol = 1e-9 * max(fundamental, 1e-30)

    if abs(float(amps[0])) > tol:
        raise RuntimeError(
            f"{label} waveform symmetry: DC bin = {float(amps[0]):.3e}, "
            f"expected 0 (tol {tol:.1e})"
        )

    indices = np.arange(1, len(amps))
    allowed = np.zeros(len(amps), dtype=bool)
    allowed[n_p::2 * n_p] = True
    forbidden = ~allowed[1:]
    magnitudes = np.abs(amps[1:])
    violators = forbidden & (magnitudes > tol)
    if np.any(violators):
        idx = int(indices[violators][0])
        val = float(amps[idx])
        if idx % n_p != 0:
            raise RuntimeError(
                f"{label} waveform symmetry: bin {idx} = {val:.3e}, "
                f"expected 0 (not a multiple of n_p={n_p}, tol {tol:.1e})"
            )
        k = idx // n_p
        raise RuntimeError(
            f"{label} waveform symmetry: bin {idx} (k={k}*n_p) = "
            f"{val:.3e}, expected 0 (even k, tol {tol:.1e})"
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
