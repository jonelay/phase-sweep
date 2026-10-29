"""Shared solver-parameter utilities and Motor-completeness validation.

The per-model Params dataclasses and prepare_* factories live next to
their run functions (solvers/, models/, simulation/params.py); this
module keeps what they share: winding arithmetic (n_eff,
winding_transfer, the j_s ↔ phase-current mapping), the Carter-consistent
B_rem ↔ psi_f derivation and resolution (resolve_b_rem, resolve_psi_f,
psi_f_carter), and the common validation gates (reject_induction,
require_geometry, require_axial_geometry, require_drive_fields,
salient_pair).
"""

from __future__ import annotations

import warnings
from math import pi
from typing import TYPE_CHECKING

from phasesweep.defaults import COPPER_TEMP_COEFF
from phasesweep.machines.geometry import AxialFluxGeometry, Geometry
from phasesweep.solvers.field_solution import (
    carter_adjusted_radii,
    derive_B_rem,
    zhu_howe_Br,
)

if TYPE_CHECKING:
    from phasesweep.machines.motor import Motor

__all__ = [
    "MAGNET_REF_TEMP_C",
    "R_S_REF_TEMP_C",
    "backemf_from_flux",
    "backemf_source",
    "check_nyquist",
    "derate",
    "derate_motor",
    "derive_B_rem_from_psi_f",
    "derive_psi_f_smooth",
    "j_s_from_phase_current",
    "n_eff",
    "phase_current_from_j_s",
    "psi_f_at_magnet_temp",
    "psi_f_carter",
    "r_s_at_operating_temp",
    "reject_induction",
    "require_axial_geometry",
    "require_drive_fields",
    "require_geometry",
    "resolve_b_rem",
    "resolve_psi_f",
    "salient_pair",
    "winding_transfer",
]


# ---------------------------------------------------------------------------
# Temperature derating (shared by every circuit model)
# ---------------------------------------------------------------------------

# Reference temperature of a datasheet R_s. 20 °C matches the back-EMF
# derating reference; a 25 °C datasheet differs by ~2%,
# negligible beside the ~50% rise to a class-F winding limit.
R_S_REF_TEMP_C: float = 20.0

# Reference temperature of a datasheet B_rem / psi_f (magnet derating).
# Same 20 °C convention as R_S_REF_TEMP_C; unlike copper's
# universal alpha, the magnet coefficient is grade-specific and lives on
# Motor.alpha_Br.
MAGNET_REF_TEMP_C: float = 20.0


def psi_f_at_magnet_temp(
    psi_f_ref: float, magnet_temp: float | None, alpha_Br: float | None,
) -> float:
    """psi_f (Wb, peak) derated to the magnet operating temperature.

    psi_f(T) = psi_f_ref · [1 + alpha_Br·(T_magnet − T_ref)] — B_rem and
    psi_f share the linear coefficient since psi_f is linear in B_rem.
    With no magnet_temp there is nothing to derate to (cold reference
    returned); alpha_Br alone is inert material data. The magnet runs
    cooler than the winding hot-spot, so magnet_temp is its own field,
    not winding_temp_limit. The (-0.01, 0) range check on alpha_Br lives
    in Motor.__post_init__ and in `derate`.
    """
    if magnet_temp is None or alpha_Br is None:
        return psi_f_ref
    return psi_f_ref * (1.0 + alpha_Br * (magnet_temp - MAGNET_REF_TEMP_C))


def r_s_at_operating_temp(
    R_s_ref: float, winding_temp: float | None,
) -> float:
    """R_s (Ω) at the winding temperature winding_temp (°C).

    Every circuit model passes Motor.winding_temp_limit (worst-case S1);
    motor_thermal passes its solved winding node.

    R_s(T) = R_s_ref · [1 + α_cu·(T − T_ref)]. At the S1 thermal
    steady state the winding sits at winding_temp_limit, so passing the
    limit gives the conservative operating resistance with no fixed-point
    solve. With no winding_temp there is nothing to derate to, so R_s_ref
    is returned unchanged (cold). winding_temp_limit is an insulation hot-spot, marginally
    above the mean winding temperature that sets bulk R_s, so this slightly
    over-estimates R_s (conservative). Where derating affects the verdict vs.
    merely rescaling watts is documented at thermal_duty._resolve_thermal_budget.
    """
    if winding_temp is None:
        return R_s_ref
    return R_s_ref * (1.0 + COPPER_TEMP_COEFF * (winding_temp - R_S_REF_TEMP_C))


def derate(R_s: float | None, psi_f: float, winding_temp: float | None,
           magnet_temp: float | None, alpha_Br: float | None) -> tuple[float | None, float]:
    """(R_s_op, psi_f_op) with the inputs and results range-checked.

    The one derating rule every circuit model applies, so a motor with
    magnet_temp / winding_temp_limit set gives the same psi_f and R_s to
    rated_torque, stall_torque, torque_speed, operating_point, thermal_duty,
    demag_screen and drive_sim. R_s may be None for models that do not
    read it (rated_torque).

    NaN passes every inequality downstream and would persist as a result;
    a derated psi_f <= 0 is outside the linear alpha_Br model and flips
    the sign of the no-load voltage.
    """
    from math import isfinite

    for name, v in (("winding_temp", winding_temp),
                    ("magnet_temp", magnet_temp), ("alpha_Br", alpha_Br)):
        if v is not None and not isfinite(v):
            raise ValueError(f"{name} must be finite, got {v}")
    if magnet_temp is not None and alpha_Br is None:
        raise ValueError(
            "magnet_temp is set but alpha_Br is not — the B_rem temperature "
            "coefficient is grade-specific (per K, e.g. NdFeB N42 -0.0012, "
            "SmCo -0.0003); set alpha_Br or drop magnet_temp"
        )
    if alpha_Br is not None and not (-0.01 < alpha_Br < 0):
        raise ValueError(f"alpha_Br must be in (-0.01, 0) per K, got {alpha_Br}")
    psi_f_op = psi_f_at_magnet_temp(psi_f, magnet_temp, alpha_Br)
    if psi_f > 0.0 and psi_f_op <= 0.0:
        raise ValueError(
            f"psi_f derated to magnet_temp {magnet_temp} degC is {psi_f_op:.4g} "
            f"Wb, not > 0: outside the linear alpha_Br range")
    if R_s is None:
        return None, psi_f_op
    R_s_op = r_s_at_operating_temp(R_s, winding_temp)
    if R_s_op < 0.0:
        raise ValueError(
            f"R_s derated to winding_temp {winding_temp} degC is "
            f"{R_s_op:.4g} ohm, below zero")
    return R_s_op, psi_f_op


def derate_motor(motor: Motor, R_s: float | None, psi_f: float) -> tuple[float | None, float]:
    """`derate` against the motor's own thermal fields, error-prefixed."""
    try:
        return derate(R_s, psi_f, motor.winding_temp_limit, motor.magnet_temp,
                      motor.alpha_Br)
    except ValueError as e:
        raise ValueError(f"Motor '{motor.name}': {e}") from None


# ---------------------------------------------------------------------------
# B_rem <-> psi_f derivation (shared by analytical/FEM factories)
# ---------------------------------------------------------------------------

def _coils_per_phase(motor: Motor) -> int:
    """Coils connected in series per phase.

    Requires motor.coils_series to be set explicitly. Auto-derivation
    was removed because n_slots//3 computes coils_per_phase, not
    coils_series — the two differ for motors with parallel branches.
    Silent wrong N_eff is worse than a loud error.

    Set coils_series in [winding] TOML section:
        coils_series = coils_per_phase / parallel_branches
    For smooth-bore motors where N is already total series turns per
    phase, set coils_series = 1.
    """
    if motor.coils_series is not None:
        return motor.coils_series
    raise ValueError(
        f"Motor '{motor.name}': coils_series is required when N and k_w are "
        f"set (needed for N_eff = N × coils_series). "
        f"Add coils_series to [winding] in the TOML "
        f"(= coils_per_phase / parallel_branches)."
    )


def n_eff(motor: Motor) -> int:
    """Total series turns per phase: N (turns per coil) × coils_series.

    Raises ValueError when coils_series is unset — never silently
    derives from n_slots//3, which is wrong for parallel branches.
    """
    if motor.N is None:
        raise ValueError(f"Motor '{motor.name}': N is required for N_eff")
    return motor.N * _coils_per_phase(motor)


def winding_transfer(motor: Motor) -> float:
    """psi_f per unit peak fundamental air-gap B (winding formula at the
    original bore): 2 * N_eff * k_w * r_stator * L_stk / n_p.
    """
    if motor.geometry is None or motor.k_w is None or motor.L_stk is None:
        raise ValueError(
            f"Motor '{motor.name}': winding_transfer needs geometry, k_w, L_stk"
        )
    if isinstance(motor.geometry, AxialFluxGeometry):
        raise ValueError(
            f"Motor '{motor.name}': winding_transfer is radial-only — "
            f"axial-flux winding transfer is not yet implemented"
        )
    return (
        2 * n_eff(motor) * motor.k_w * motor.geometry.r_stator
        * motor.L_stk / motor.n_p
    )


def j_s_from_phase_current(
    motor: Motor, i_peak: float, *, slot_width_ratio: float | None = None,
) -> float:
    """FEM slot current-density amplitude j_s (A/m²) for a peak phase
    current, matching the winding's fundamental MMF.

    The FEM armature source J_z = -j_s·k_w·cos(n_p·θ) over the slot faces
    has ampere-turns-per-angle fundamental j_s·k_w·S (S = the slot faces'
    radial moment, ``fem_mesh.slot_source_moment``); a balanced 3-phase
    winding with N_eff series turns per phase at peak current Î has
    3·k_w·N_eff·Î/π. Equating gives

        j_s = 3 · N_eff · Î / (π · S)

    k_w cancels — the solver applies it to the sheet itself, so pass the
    raw phase current here. The mapping is phase-invariant: the
    solver's ``sheet_phase`` rotates the sheet's fundamental without
    changing its amplitude, so Î is the current magnitude at any current
    angle (sheet_phase = 0: pure i_q; ±π/2: pure i_d). At sheet_phase = 0
    the Maxwell-stress torque closes to the circuit tier's
    1.5·n_p·ψf·i_q analytically.

    Pass ``slot_width_ratio`` only when overriding it in solve_field_fem
    too — the mapping must integrate the same slot faces the mesh builds.
    """

    from phasesweep.solvers.fem_mesh import slot_source_moment

    geo = require_geometry(motor, "j_s_from_phase_current")
    S = slot_source_moment(geo, slot_width_ratio=slot_width_ratio,
                           n_p=motor.n_p)
    return 3 * n_eff(motor) * i_peak / (pi * S)


def phase_current_from_j_s(
    motor: Motor, j_s: float, *, slot_width_ratio: float | None = None,
) -> float:
    """Peak phase current (A) equivalent to a FEM sheet amplitude j_s —
    exact inverse of ``j_s_from_phase_current``."""

    from phasesweep.solvers.fem_mesh import slot_source_moment

    geo = require_geometry(motor, "phase_current_from_j_s")
    S = slot_source_moment(geo, slot_width_ratio=slot_width_ratio,
                           n_p=motor.n_p)
    return j_s * pi * S / (3 * n_eff(motor))


def _derivation_gaps(motor: Motor, source: str) -> list[str]:
    """Inputs blocking the psi_f <-> B_rem derivation, for error messages.

    Mirrors the None returns of derive_B_rem_from_psi_f / _derive_psi_f,
    including the geometry ones, so the error names the real blocker.
    """
    gaps = [source] if getattr(motor, source) is None else []
    gaps += [f for f in ("N", "k_w", "L_stk") if getattr(motor, f) is None]
    if motor.geometry is None:
        gaps.append("[geometry] (none on this motor)")
    elif isinstance(motor.geometry, AxialFluxGeometry):
        gaps.append("radial geometry (the transfer is radial-only; set "
                    "the value explicitly for an axial-flux motor)")
    return gaps


def derive_B_rem_from_psi_f(motor: Motor) -> float | None:
    """Derive B_rem from psi_f via Zhu & Howe transfer ratio.

    Carter-consistent: inverts the same model psi_f_carter evaluates —
    Carter-adjusted radii for the field, original bore for the winding —
    so psi_f → B_rem → flux_linkage_peak round-trips exactly.
    Returns derived B_rem, or None if derivation inputs are missing.
    N_eff = N (turns per coil) × coils_series = total series turns per phase.
    """
    if motor.psi_f is None:
        return None
    if motor.N is None or motor.k_w is None or motor.L_stk is None:
        return None
    if motor.geometry is None or isinstance(motor.geometry, AxialFluxGeometry):
        return None
    r_s_c, r_m_c, _k_c = carter_adjusted_radii(motor.geometry, motor.mu_r_pm)
    return derive_B_rem(
        motor.psi_f, motor.n_p, n_eff(motor), motor.k_w, motor.L_stk,
        r_stator=motor.geometry.r_stator,
        r_magnet=motor.geometry.r_magnet,
        r_rotor=motor.geometry.r_rotor,
        mu_r_pm=motor.mu_r_pm,
        alpha_p=motor.alpha_p,
        r_stator_c=r_s_c,
        r_magnet_c=r_m_c,
    )


def derive_psi_f_smooth(motor: Motor) -> float | None:
    """Derive psi_f from B_rem via Zhu & Howe — smooth-bore (no Carter).

    Explicit baseline helper: slotting is deliberately ignored. Production
    psi_f resolution uses the Carter-consistent _derive_psi_f below.
    Returns derived psi_f, or None if derivation inputs are missing.
    N_eff = N (turns per coil) × coils_series = total series turns per phase.
    """
    import numpy as np

    if motor.B_rem is None:
        return None
    if motor.N is None or motor.k_w is None or motor.L_stk is None:
        return None
    if motor.geometry is None or isinstance(motor.geometry, AxialFluxGeometry):
        return None
    geo = motor.geometry
    B_peak = zhu_howe_Br(
        np.array([0.0]), motor.n_p, motor.B_rem, r_eval=geo.r_stator,
        r_stator=geo.r_stator, r_magnet=geo.r_magnet, r_rotor=geo.r_rotor,
        mu_r_pm=motor.mu_r_pm, alpha_p=motor.alpha_p,
    )[0]
    return B_peak * winding_transfer(motor)


def _derive_psi_f(motor: Motor) -> float | None:
    """Derive psi_f from B_rem, Carter-consistently.

    Exact inverse of derive_B_rem_from_psi_f and identical to the registry's
    flux_linkage_peak: Carter-adjusted radii for the field, original bore
    for the evaluation point and winding formula. B_rem → psi_f → B_rem
    round-trips exactly.

    Motor.k_end is deliberately NOT applied here: the analytical runner
    corrects its reported flux_linkage_peak, but the circuit-tier psi_f
    resolution stays raw-2D so the psi_f ↔ B_rem round-trip and every
    torque-tier expectation remain unchanged. Extending the correction to
    the circuit tier is a separate, larger change (documented limitation).
    Returns derived psi_f, or None if derivation inputs are missing.
    """
    if motor.B_rem is None:
        return None
    if motor.N is None or motor.k_w is None or motor.L_stk is None:
        return None
    if motor.geometry is None or isinstance(motor.geometry, AxialFluxGeometry):
        return None
    r_s_c, r_m_c, _k_c = carter_adjusted_radii(motor.geometry, motor.mu_r_pm)
    return psi_f_carter(motor, r_s_c, r_m_c, motor.B_rem)


def psi_f_carter(
    motor: Motor, r_stator_c: float, r_magnet_c: float, B_rem: float,
) -> float | None:
    """psi_f (Wb, peak) with Carter-adjusted radii for the field solution,
    original bore for both the evaluation point and the winding formula.

    B_rem is passed explicitly so callers can supply the resolved value
    (which may be derived from psi_f) rather than motor.B_rem.
    Returns None if winding inputs (N, k_w, L_stk) are missing.
    """
    import numpy as np

    if motor.N is None or motor.k_w is None or motor.L_stk is None:
        return None
    if motor.geometry is None or isinstance(motor.geometry, AxialFluxGeometry):
        return None
    geo = motor.geometry
    B_peak = zhu_howe_Br(
        np.array([0.0]), motor.n_p, B_rem, r_eval=geo.r_stator,
        r_stator=r_stator_c, r_magnet=r_magnet_c, r_rotor=geo.r_rotor,
        mu_r_pm=motor.mu_r_pm, alpha_p=motor.alpha_p,
    )[0]
    return B_peak * winding_transfer(motor)


def resolve_b_rem(motor: Motor) -> float:
    """Get or derive B_rem from motor. Raises ValueError if impossible."""
    if motor.B_rem is not None:
        if motor.magnetization == "radial":
            _check_consistency(motor)
        return motor.B_rem

    if motor.is_reluctance:
        raise ValueError(
            f"Motor '{motor.name}': psi_f = 0 (reluctance machine, no PM "
            f"rotor) — field-solver models need a PM field; there is no "
            f"B_rem to derive"
        )

    if motor.magnetization == "halbach":
        raise ValueError(
            f"Motor '{motor.name}': Halbach motors require an explicit B_rem — "
            f"the psi_f→B_rem derivation uses the radial Zhu & Howe transfer "
            f"which is invalid for Halbach magnetization"
        )
    if motor.geometry is not None and getattr(motor.geometry, "magnet_mount", None) == "inset":
        raise ValueError(
            f"Motor '{motor.name}': inset motors require an explicit B_rem — "
            f"the psi_f→B_rem derivation uses the radial Zhu & Howe transfer "
            f"which assumes air between magnet arcs"
        )
    if motor.geometry is not None and getattr(motor.geometry, "bridge_thickness", None) is not None:
        raise ValueError(
            f"Motor '{motor.name}': bridged motors require an explicit B_rem — "
            f"the psi_f→B_rem derivation uses the radial Zhu & Howe transfer "
            f"which does not model the bridge ring flux shunt"
        )

    derived = derive_B_rem_from_psi_f(motor)
    if derived is not None:
        return derived

    raise ValueError(
        f"Motor '{motor.name}': cannot determine B_rem — set it, or "
        f"derive it from psi_f; missing for derivation: "
        f"{', '.join(_derivation_gaps(motor, 'psi_f'))}"
    )


def resolve_psi_f(motor: Motor) -> float:
    """Get or derive psi_f from motor. Raises ValueError if impossible."""
    if motor.psi_f is not None:
        if motor.magnetization == "radial":
            _check_consistency(motor)
        return motor.psi_f

    if motor.magnetization == "halbach":
        raise ValueError(
            f"Motor '{motor.name}': Halbach motors require an explicit psi_f — "
            f"the B_rem→psi_f derivation uses the radial Zhu & Howe transfer "
            f"which is invalid for Halbach magnetization"
        )
    if motor.geometry is not None and getattr(motor.geometry, "magnet_mount", None) == "inset":
        raise ValueError(
            f"Motor '{motor.name}': inset motors require an explicit psi_f — "
            f"the B_rem→psi_f derivation uses the radial Zhu & Howe transfer "
            f"which assumes air between magnet arcs"
        )
    if motor.geometry is not None and getattr(motor.geometry, "bridge_thickness", None) is not None:
        raise ValueError(
            f"Motor '{motor.name}': bridged motors require an explicit psi_f — "
            f"the B_rem→psi_f derivation uses the radial Zhu & Howe transfer "
            f"which does not model the bridge ring flux shunt"
        )

    derived = _derive_psi_f(motor)
    if derived is not None:
        return derived

    raise ValueError(
        f"Motor '{motor.name}': cannot determine psi_f — set it, or "
        f"derive it from B_rem; missing for derivation: "
        f"{', '.join(_derivation_gaps(motor, 'B_rem'))}"
    )


def _check_consistency(motor: Motor) -> None:
    """Warn if both B_rem and psi_f are present and disagree significantly.

    The B_rem <-> psi_f derivation uses a simplified winding formula
    (N_eff * k_w * Phi_1) which is ~20% accurate for FSCW motors.
    Threshold is 50% to avoid false warnings from model limitations.
    """
    if motor.B_rem is None or motor.psi_f is None:
        return
    if motor.is_reluctance:
        return
    if motor.geometry is not None and (
        getattr(motor.geometry, "magnet_mount", None) == "inset"
        or getattr(motor.geometry, "bridge_thickness", None) is not None
    ):
        return
    if motor.N is None or motor.k_w is None or motor.L_stk is None:
        return
    derived = derive_B_rem_from_psi_f(motor)
    if derived is None:
        return
    rel_err = abs(derived - motor.B_rem) / max(abs(motor.B_rem), 1e-12)
    if rel_err > 0.50:
        warnings.warn(
            f"B_rem from psi_f={derived:.4f} disagrees with "
            f"B_rem={motor.B_rem:.4f} by "
            f"{rel_err * 100:.1f}%",
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# Shared validation gates (used by the per-model prepare_* factories)
# ---------------------------------------------------------------------------

def reject_induction(motor: Motor, model: str) -> None:
    """Loud gate: synchronous-machine models refuse induction motors
    (topology rule)."""
    if motor.induction is not None:
        raise ValueError(
            f"Motor '{motor.name}': {model} models a synchronous machine "
            f"(d-q frame with a constant rotor flux); this motor has an "
            f"[induction] section (squirrel-cage rotor, slip degree of "
            f"freedom)."
        )


def require_geometry(motor: Motor, model: str) -> Geometry:
    if motor.geometry is None:
        raise ValueError(f"Motor '{motor.name}': {model} needs geometry")
    if isinstance(motor.geometry, AxialFluxGeometry):
        raise ValueError(
            f"Motor '{motor.name}': {model} requires radial-flux Geometry, "
            f"got AxialFluxGeometry (axial-flux solvers are not yet available)"
        )
    return motor.geometry


def salient_pair(motor: Motor) -> tuple[float | None, float | None]:
    """(L_d, L_q) for the salient torque form, or (None, None) when neither is set.

    One-sided saliency is an input error, not a non-salient machine.
    """
    if motor.L_d is None and motor.L_q is None:
        return None, None
    if motor.L_d is None or motor.L_q is None:
        raise ValueError(
            f"Motor '{motor.name}': L_d and L_q must be set together "
            f"(got L_d={motor.L_d}, L_q={motor.L_q})"
        )
    return motor.L_d, motor.L_q


def require_axial_geometry(motor: Motor, model: str) -> AxialFluxGeometry:
    if motor.geometry is None:
        raise ValueError(f"Motor '{motor.name}': {model} needs geometry")
    if not isinstance(motor.geometry, AxialFluxGeometry):
        raise ValueError(
            f"Motor '{motor.name}': {model} requires AxialFluxGeometry, "
            f"got radial-flux Geometry (use the 'analytical' model)"
        )
    return motor.geometry


def require_drive_fields(
    motor: Motor, model: str, fields: tuple[str, ...],
) -> None:
    """Raise ValueError if any named DriveParams fields are None."""
    missing = [f for f in fields if getattr(motor.drive, f) is None]
    if missing:
        raise ValueError(
            f"Motor '{motor.name}': {model} needs [drive] "
            f"{', '.join(missing)}"
        )


# ---------------------------------------------------------------------------
# Shared runner helpers
# ---------------------------------------------------------------------------

def check_nyquist(n_theta: int, n_p: int, *, series: bool = False) -> None:
    """Reject n_theta too small to resolve the fundamental.

    *series* = True uses the analytical-series bound (n_theta >= 2·n_p + 2,
    because max_order = n_theta // 2 would equal n_p and silently drop the
    k=1 term). *series* = False uses the FFT Nyquist bound (n_theta > 2·n_p).
    """
    if series:
        if n_theta <= 2 * n_p + 1:
            raise ValueError(
                f"n_theta={n_theta} cannot resolve the fundamental at "
                f"n_p={n_p}; need n_theta >= 2*n_p + 2"
            )
    else:
        if n_theta <= 2 * n_p:
            raise ValueError(
                f"n_theta={n_theta} cannot resolve the fundamental at "
                f"n_p={n_p}; need n_theta > 2*n_p"
            )


def backemf_from_flux(motor: Motor, flux_linkage_peak: float | None) -> float | None:
    """Derive back-EMF fundamental from motor speed and flux linkage."""
    if motor.drive.W_REF is None:
        return None
    w_e = motor.drive.W_REF * motor.n_p
    if motor.psi_f is not None:
        return float(w_e * motor.psi_f)
    if flux_linkage_peak is not None:
        return float(w_e * flux_linkage_peak)
    return None


def backemf_source(motor: Motor, backemf_fundamental: float | None) -> str | None:
    """Which input backemf_from_flux used: "psi_f" when the motor's stated
    flux linkage overrides the model's, "computed" when the
    model's own flux_linkage_peak was used, None when there is no back-EMF."""
    if backemf_fundamental is None:
        return None
    return "psi_f" if motor.psi_f is not None else "computed"


