"""FEM demagnetization screen.

Solves the magnet field with a pure demagnetizing d-axis current sheet
(sheet_phase = +π/2) at a screening current i_fault, samples the magnet
operating point B_m inside the arcs, and reports the worst-point margin
against the knee B_knee(T). For radial magnetization B_m = sign(M_r)*B_r;
for Halbach, B_m projects onto each segment's magnetization axis.

A screen, not a recoil model: it predicts knee-crossing onset (where and
how much magnet area), not the post-fault k_T loss — no hysteresis. The
solve is a locked-rotor aligned snapshot (rotation = 0), which is the
classic worst case for a pure d-axis MMF. i_fault is the current
amplitude the caller chooses to screen at (drive limit, steady
short-circuit, asymmetric transient peak — the asymmetric peak can
exceed steady I_sc by ~2×); there is deliberately no drive-limit
fallback. The linear solve over-predicts the armature field (saturated
teeth add reluctance to the armature path), so linear is the
conservative screen and ``nonlinear=True`` the refinement. At
magnetization discontinuities (pole transitions for radial,
segment boundaries for Halbach) the sharp M-direction jump can pin
B_m_min, making the screen over-conservative (see the caveat in
fem_sampling.sample_magnet_Bm).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, pi
from typing import TYPE_CHECKING, Any

from phasesweep.solver_params import (
    MAGNET_REF_TEMP_C,
    n_eff,
    reject_induction,
    require_geometry,
    resolve_b_rem,
)
from phasesweep.solvers.fem_runner import FemParams

if TYPE_CHECKING:
    from phasesweep.machines.motor import Motor
    from phasesweep.sweep_types import RunConfig

__all__ = [
    "DemagScreenParams",
    "prepare_demag_screen",
    "run_demag_screen",
]


def run_demag_screen(config: RunConfig) -> dict[str, Any]:
    """Demag screen at config.i_fault (peak phase current, A)."""
    from phasesweep.solver_params import j_s_from_phase_current
    from phasesweep.solvers.fem_field import solve_field_fem
    from phasesweep.solvers.fem_sampling import demag_margin

    params = prepare_demag_screen(config.motor)
    if config.i_fault is None:
        raise ValueError(
            f"Motor '{config.motor.name}': demag_screen needs "
            f"RunConfig.i_fault (peak phase current to screen at) — no "
            f"drive-limit fallback; the screening current is a deliberate "
            f"choice (MAX_I_S, steady short-circuit, transient peak)"
        )
    if config.i_fault <= 0 or not isfinite(config.i_fault):
        raise ValueError(
            f"RunConfig.i_fault must be > 0 and finite (peak fault current), "
            f"got {config.i_fault}"
        )
    j_s_fault = j_s_from_phase_current(config.motor, config.i_fault)

    f = params.fem
    solve_info: dict[str, float] = {}
    _, _, mesh, gfu = solve_field_fem(
        geo=f.geometry, n_p=f.n_p, B_rem=f.B_rem,
        mu_r_pm=f.mu_r_pm, mu_r_fe=f.mu_r_fe,
        maxh_fraction=config.maxh_fraction, n_theta=config.n_theta,
        nonlinear=config.nonlinear,
        nonlinear_solver=config.nonlinear_solver,
        j_s=j_s_fault, k_w=params.k_w, sheet_phase=pi / 2,
        alpha_p=f.alpha_p, info=solve_info, return_full=True,
        magnetization=f.magnetization,
        halbach_segments=f.halbach_segments,
        halbach_angles_rad=f.halbach_angles_rad,
    )
    out: dict[str, Any] = demag_margin(
        mesh, gfu, f.geometry, f.n_p, B_knee=params.B_knee,
        alpha_p=f.alpha_p,
        magnetization=f.magnetization,
        halbach_segments=f.halbach_segments,
        halbach_angles_rad=f.halbach_angles_rad,
    )
    out["B_knee"] = params.B_knee
    out["i_fault"] = config.i_fault
    out["j_s_fault"] = j_s_fault
    if "b_iron_max" in solve_info:
        out["b_iron_max"] = solve_info["b_iron_max"]
    return out


@dataclass(frozen=True)
class DemagScreenParams:
    """Validated inputs for the FEM demag screen.

    fem — field-solver params with B_rem derated to magnet_temp when the
    derating fields are set; B_knee — effective knee (T) at that same
    temperature; k_w — winding factor scaling the current sheet.
    """

    fem: FemParams
    B_knee: float
    k_w: float


def prepare_demag_screen(motor: Motor) -> DemagScreenParams:
    """Validate and extract demag-screen params from a Motor.

    Needs the FEM inputs plus [materials] B_knee, a slotted geometry (the
    d-axis current sheet integrates over slot faces) and the winding
    fields behind the j_s ↔ phase-current mapping (N, coils_series, k_w).
    When magnet_temp is set, BOTH alpha_Br and alpha_B_knee must be set —
    remanence and knee are evaluated at one consistent temperature:
    B_rem·[1 + alpha_Br·(T − 20)] and B_knee + alpha_B_knee·(T − 20)
    (the knee slope is absolute T/K — the knee can cross zero).
    """
    from phasesweep.solver_params import derate_motor

    reject_induction(motor, "demag_screen")
    geo = require_geometry(motor, "demag_screen")
    if geo.n_slots == 0 or geo.slot_depth == 0:
        raise ValueError(
            f"Motor '{motor.name}': demag_screen needs a slotted geometry "
            f"(n_slots > 0, slot_depth > 0) — the d-axis current sheet "
            f"integrates over slot faces"
        )
    b_rem = resolve_b_rem(motor)
    if motor.B_knee is None:
        raise ValueError(
            f"Motor '{motor.name}': demag_screen needs B_knee "
            f"([materials] TOML section)"
        )
    if motor.k_w is None:
        raise ValueError(
            f"Motor '{motor.name}': demag_screen needs an explicit k_w "
            f"(the current sheet scales with the winding factor)"
        )
    n_eff(motor)  # loud on missing N / coils_series (the j_s mapping inputs)

    b_knee = motor.B_knee
    if motor.magnet_temp is not None:
        alpha_Br, alpha_B_knee = motor.alpha_Br, motor.alpha_B_knee
        if alpha_Br is None or alpha_B_knee is None:
            fields = (("alpha_Br", alpha_Br), ("alpha_B_knee", alpha_B_knee))
            missing = [f for f, v in fields if v is None]
            raise ValueError(
                f"Motor '{motor.name}': magnet_temp is set but "
                f"{', '.join(missing)} is not — the demag screen evaluates "
                f"B_rem and B_knee at the same temperature (assuming the "
                f"20 °C knee at operating temperature is optimistic for "
                f"NdFeB)"
            )
        b_rem = derate_motor(motor, None, b_rem)[1]  # shared range guard
        b_knee = b_knee + alpha_B_knee * (motor.magnet_temp - MAGNET_REF_TEMP_C)

    from math import radians

    angles_rad = None
    if motor.halbach_angles_deg is not None:
        angles_rad = tuple(radians(a) for a in motor.halbach_angles_deg)

    return DemagScreenParams(
        fem=FemParams(
            geometry=geo,
            n_p=motor.n_p,
            B_rem=b_rem,
            mu_r_pm=motor.mu_r_pm,
            mu_r_fe=motor.mu_r_fe,
            alpha_p=motor.alpha_p,
            magnetization=motor.magnetization,
            halbach_segments=motor.halbach_segments,
            halbach_angles_rad=angles_rad,
        ),
        B_knee=b_knee,
        k_w=motor.k_w,
    )
