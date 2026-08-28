"""Shared cross-validation contract for the radial Elmer harness.

Single source of truth for the case parameters, tolerances and
analytical references shared by scripts/elmer_radial/ (generation,
needs Elmer + NGSolve) and test_elmer_radial_pinned.py (core-lane
re-check, pure numpy). Mirrors axial_fem_contract.py. Not a test
module — imported by both sides.

Baseline: the normalized Zhu geometry (default_inrunner), the same
problem the fem-vs-analytical crossval gate runs on. B1 is the n_p-th
harmonic of B_r on the r_ag circle, extracted by projection.
"""

from __future__ import annotations

from math import pi

import numpy as np

from phasesweep.machines.geometry import Geometry, default_inrunner, inrunner
from phasesweep.solvers.analytical import zhu_howe_Br_series

# Baseline parameters, all cases
N_P = 4
B_REM = 1.2          # T
MU_R_PM = 1.05
MU_R_FE = 5000.0     # linear-iron cases
J_S = 5.0e6          # A/m^2, armature current-sheet case
K_W = 0.966

# Slotted variant of the baseline geometry
SLOT_N = 24
SLOT_DEPTH = 0.15
SLOT_WIDTH_RATIO = 0.6

# Cross-code gates: Elmer (structured polar mesh, nodal Lagrange) vs
# NGSolve fem (unstructured OCC mesh, H1 order 3). Spike band was
# 0.05–0.12% on slotless/nonlinear cases; slotted cases widen to ~0.2%
# because re-entrant slot corners are resolved differently by the two
# discretizations.
CROSS_CODE_REL_TOL = 0.005
CROSS_CODE_REL_TOL_SLOTTED = 0.01

# Linear slotless cases vs the Zhu/Howe/Chan series at r_ag. The series
# assumes ideal iron; mu_r_fe = 5000 leaves a small physical yoke-MMF
# residual inside this gate (observed ~0.1%).
ANALYTICAL_REL_TOL = 0.005

# Complete case set a pinned JSON must carry
EXPECTED_CASES = frozenset({"smooth_bore", "arcs_a080", "arcs_a095",
                            "nonlinear_bh", "slotted_open", "slotted_js"})

# Silent-zero guard: two known Elmer failure modes (magnetization in a
# Body Force; Mortar BC pointing at a target tag) exit 0 with a
# trivially zero field. Baseline peak |B_r| is ~0.7 T.
NONTRIVIAL_PEAK_T = 0.1


def baseline_geometry() -> Geometry:
    return default_inrunner()


def slotted_geometry() -> Geometry:
    return inrunner(
        r_outer=1.0, r_stator=0.70, r_magnet=0.64, r_rotor=0.30,
        n_slots=SLOT_N, slot_depth=SLOT_DEPTH,
        slot_width_ratio=SLOT_WIDTH_RATIO,
    )


def slotless_reference_B1(geo: Geometry, n_p: int, B_rem: float,
                          mu_r_pm: float, alpha_p: float,
                          n_theta: int = 4096) -> float:
    """Analytical reference: fundamental of the Zhu/Howe/Chan odd-harmonic
    series at r_ag (ideal iron, smooth bore)."""
    theta = np.linspace(0.0, 2.0 * pi, n_theta, endpoint=False)
    Br = zhu_howe_Br_series(
        theta, n_p, B_rem, geo.r_ag,
        r_stator=geo.r_stator, r_magnet=geo.r_magnet, r_rotor=geo.r_rotor,
        mu_r_pm=mu_r_pm, alpha_p=alpha_p, max_order=n_theta // 2,
    )
    return 2.0 * float(np.abs(np.mean(Br * np.exp(-1j * n_p * theta))))


def case_tolerance(case: dict) -> float:
    """Cross-code gate for one pinned case record."""
    if case["params"].get("n_slots", 0) > 0:
        return CROSS_CODE_REL_TOL_SLOTTED
    return CROSS_CODE_REL_TOL
