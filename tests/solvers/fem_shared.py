"""Shared geometry, constants, and the legacy-style solve helper for the
per-module FEM test files (split from the former tests/solvers/test_fem.py).

Not a test module — imported by test_fem_field/_mesh/_materials/_sampling/
_runner after their module-level ngsolve guards, so the fem_field import
here never runs without ngsolve present.
"""

from phasesweep.machines.geometry import default_inrunner, inrunner, outrunner
from phasesweep.solvers.analytical import _derive_B_rem
from phasesweep.solvers.fem_field import solve_field_fem

FAST = dict(n_theta=60, maxh_fraction=0.08)
_GEO = default_inrunner()
# zhu_howe_Br radii are required keywords
_RADII = dict(r_stator=_GEO.r_stator, r_magnet=_GEO.r_magnet,
              r_rotor=_GEO.r_rotor)


def _solve(n_p=2, psi_f=0.1, N=50, k_w=0.966, L_stk=0.10,
           n_slots=0, j_s=0.0, nonlinear=False, geo=None, **kw):
    """Helper: legacy-style call via new Geometry-based solve_field_fem."""
    _geo = geo or _GEO
    B_rem = _derive_B_rem(psi_f, n_p, N, k_w, L_stk,
                          r_stator=_geo.r_stator, r_magnet=_geo.r_magnet,
                          r_rotor=_geo.r_rotor)
    merged = {**FAST, **kw}
    return solve_field_fem(
        geo=_geo, n_p=n_p, B_rem=B_rem,
        mu_r_pm=1.05, mu_r_fe=1000.0,
        n_slots=n_slots, j_s=j_s, nonlinear=nonlinear,
        **merged,
    )


_SLOTTED_GEO_KW = dict(r_outer=1.0, r_stator=0.70, r_magnet=0.64, r_rotor=0.30,
                       r_inner=0.0, n_slots=12, slot_depth=0.05)
_SLOTTED_OUT_GEO_KW = dict(r_outer=0.10, r_rotor=0.08, r_magnet=0.06,
                           r_stator=0.04, r_inner=0.02, n_slots=6,
                           slot_depth=0.005)


_GEO_SMALL = inrunner(r_outer=0.05, r_stator=0.035, r_magnet=0.032,
                      r_rotor=0.015, r_inner=0.0)


_GEO_OUT = outrunner(r_outer=0.10, r_rotor=0.08, r_magnet=0.06,
                     r_stator=0.04, r_inner=0.02)


_TORQUE_FAST = dict(n_p=2, B_rem=1.0, mu_r_pm=1.05, mu_r_fe=1000.0,
                    k_w=0.966, n_theta=60, maxh_fraction=0.08,
                    return_full=True)


def _torque_geo():
    return inrunner(**_SLOTTED_GEO_KW)


def _gap_radii(geo, fractions=(0.25, 0.5, 0.75)):
    g = geo.r_stator - geo.r_magnet
    return [geo.r_magnet + f * g for f in fractions]


_HALBACH_GEO = inrunner(
    r_outer=0.060, r_stator=0.048, r_magnet=0.040, r_rotor=0.030,
)
_H_FAST = dict(n_theta=120, maxh_fraction=0.06)
