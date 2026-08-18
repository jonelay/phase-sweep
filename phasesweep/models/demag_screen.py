"""FEM demagnetization screen.

Solves the magnet field with a pure demagnetizing d-axis current sheet
(sheet_phase = +π/2) at a screening current i_fault, samples the magnet
operating point B_m = sign(M_r)·B_r inside the arcs, and reports the
worst-point margin against the knee B_knee(T).

A screen, not a recoil model: it predicts knee-crossing onset (where and
how much magnet area), not the post-fault k_T loss — no hysteresis. The
solve is a locked-rotor aligned snapshot (rotation = 0), which is the
classic worst case for a pure d-axis MMF. i_fault is the current
amplitude the caller chooses to screen at (drive limit, steady
short-circuit, asymmetric transient peak — the asymmetric peak can
exceed steady I_sc by ~2×); there is deliberately no drive-limit
fallback. The linear solve over-predicts the armature field (saturated
teeth add reluctance to the armature path), so linear is the
conservative screen and ``nonlinear=True`` the refinement. At α_p = 1
the square-wave pole-transition corner pins B_m_min and the screen is
over-conservative (see fem_field.sample_magnet_Bm).
"""

from __future__ import annotations

from math import pi
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from phasesweep.sweep_types import RunConfig


def run_demag_screen(config: RunConfig) -> dict[str, Any]:
    """Demag screen at config.i_fault (peak phase current, A)."""
    from phasesweep.solver_params import (
        j_s_from_phase_current,
        prepare_demag_screen,
    )
    from phasesweep.solvers.fem_field import demag_margin, solve_field_fem

    params = prepare_demag_screen(config.motor)
    if config.i_fault is None:
        raise ValueError(
            f"Motor '{config.motor.name}': demag_screen needs "
            f"RunConfig.i_fault (peak phase current to screen at) — no "
            f"drive-limit fallback; the screening current is a deliberate "
            f"choice (MAX_I_S, steady short-circuit, transient peak)"
        )
    j_s_fault = j_s_from_phase_current(config.motor, config.i_fault)

    f = params.fem
    solve_info: dict[str, float] = {}
    _, _, mesh, gfu = solve_field_fem(
        geo=f.geometry, n_p=f.n_p, B_rem=f.B_rem,
        mu_r_pm=f.mu_r_pm, mu_r_fe=f.mu_r_fe,
        maxh_fraction=config.maxh_fraction, n_theta=config.n_theta,
        nonlinear=config.nonlinear,
        j_s=j_s_fault, k_w=params.k_w, sheet_phase=pi / 2,
        alpha_p=f.alpha_p, info=solve_info, return_full=True,
    )
    out: dict[str, Any] = demag_margin(
        mesh, gfu, f.geometry, f.n_p, B_knee=params.B_knee,
        alpha_p=f.alpha_p,
    )
    out["B_knee"] = params.B_knee
    out["i_fault"] = config.i_fault
    out["j_s_fault"] = j_s_fault
    if "b_iron_max" in solve_info:
        out["b_iron_max"] = solve_info["b_iron_max"]
    return out
