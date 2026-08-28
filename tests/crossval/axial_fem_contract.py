"""Shared cross-validation contract for the Phase 3 axial FEM tracks.

Single source of truth for the CV gate tolerances and reference values
used by Track A (fem_axial, ships), Track B (fem_axial_3d study), and
Track C (Elmer study), so the three implementations converge on one
comparison surface. Not a test module — imported by track tests.

Conventions all tracks must share:
- One gap circuit (per-winding-circuit; GAP_LINKAGE applied downstream).
- Mirror domain: one pole pitch, Dirichlet potential = 0 on the two
  magnet-center planes; tooth centered on the d-axis.
- B_z sampled on the mid-airgap line; waveform mirror-extended to the
  2*tau_p electrical period before FFT.
"""

from __future__ import annotations

from math import pi

import numpy as np

from phasesweep.machines.configs import load_motor
from phasesweep.solvers.analytical_axial import axial_layer_Bz_series
from tests.conftest import REPO_ROOT

PARVIAINEN_TOML = REPO_ROOT / "data/parviainen_afpm/parviainen_5kw_afpm.toml"

# Measured anchor (thesis Tables 3.1/3.3) — CV-5 ship gate.
E_PM_MEASURED_RMS = 211.0   # V rms at 300 rpm, operating temperature
SPEED_RPM = 300.0
CV5_REL_TOL = 0.05          # back-EMF validation band

# CV-0: mirror domain vs true periodic-identification pole-pair solve.
# Same code, same mesh density — only the BC treatment differs, so the
# agreement bound is set by discretization noise, not physics.
CV0_REL_TOL = 1e-3

# CV-1: slotless plane at r_mean vs the analytical layer series (the
# series is a true-periodic Fourier solution — independent of the
# mirror-BC assumption). A vs C compares the same physics in two codes.
CV1_ANALYTICAL_REL_TOL = 0.02
CV1_CROSS_CODE_REL_TOL = 0.01

# CV-2: 1D magnetic-circuit limit (alpha_p = 1, tau_p >> gap stack).
CV2_REL_TOL = 0.01

# CV-3 is a reported diagnostic, not a gate (audit F4: both sides share
# the slot_width_ratio Carter fallback; the comparison tests Carter's
# error, not the FEM's).

# CV-4: (finite annulus - extruded control) flux deficit vs k_sigma_edge,
# matched slotless/linear configs on both sides.
CV4_FACTOR_ABS_TOL = 0.005

# Silent-zero guard (mirrors radial_fem_contract.NONTRIVIAL_PEAK_T):
# known Elmer failure modes exit 0 with a trivially zero field, so every
# pinned field magnitude must clear this floor to count as a real solve.
NONTRIVIAL_FIELD_T = 0.1


def load_parviainen_motor():
    """The Parviainen 5 kW 1R2S anchor motor, loaded from its TOML."""
    return load_motor(PARVIAINEN_TOML)


def slotless_reference_B1(
    geo, n_p: int, B_rem: float, mu_r_pm: float, alpha_p: float,
    n_theta: int = 3600,
) -> float:
    """CV-1 reference: fundamental of the slotless (g' = g) layer series
    at r_mean.

    Fundamental amplitude extracted by projection so the number is
    independent of each track's FFT plumbing.
    """
    theta = np.linspace(0.0, 2.0 * pi, n_theta, endpoint=False)
    Bz = axial_layer_Bz_series(
        theta, n_p, B_rem,
        tau_p=pi * geo.r_mean / n_p,
        g_prime=geo.airgap,
        l_pm=geo.magnet_thickness,
        mu_r_pm=mu_r_pm,
        alpha_p=alpha_p,
        max_order=n_theta // 2,
    )
    return 2.0 * float(np.abs(np.mean(Bz * np.exp(-1j * n_p * theta))))


def circuit_limit_Bz(B_rem: float, mu_r_pm: float, l_pm: float,
                     g: float) -> float:
    """CV-2 reference: 1D iron-backed magnet + gap circuit plateau."""
    return B_rem * l_pm / (l_pm + mu_r_pm * g)
