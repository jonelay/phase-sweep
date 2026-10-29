"""Motor perturbation functions for sensitivity analysis.

Motor.k_end is deliberately NOT invalidated by the geometry-affecting
perturbations (OD, gap, L_stk), even though the computed value is tied
to the source geometry: a near-constant multiplicative factor cancels in
relative sensitivity, while dropping it would put an on/off k_end jump
into every perturbed-vs-base delta. Its true geometry dependence is
second-order at sweep deltas (±5%).
"""

from __future__ import annotations

import dataclasses as dc
import logging
from typing import Any, Literal

from phasesweep.machines.geometry import AxialFluxGeometry
from phasesweep.machines.motor import Motor

logger = logging.getLogger(__name__)

__all__ = [
    "PerturbParam",
    "perturb_motor",
    "scale_inductances",
]

PerturbParam = Literal[
    "OD", "gap", "L_stk", "B_rem", "k_w", "psi_f", "R_s", "L_d", "L_q",
    "B_core",
    "R_2", "L_m", "L_sigma_s", "L_sigma_r",
]

# Circuit- and iron-tier scalars: perturbed by direct scaling, no geometry
# needed. B_core is a lumped effective flux density, not a field the
# geometry produces, so nothing downstream is invalidated by scaling it.
_SCALAR_PARAMS: frozenset[str] = frozenset(
    {"psi_f", "R_s", "L_d", "L_q", "B_core"})

# Induction-circuit scalars, nested on Motor.induction. Each has a
# matching deep-bar/saturation table; the table's y-values are scaled by
# the same factor so the scalar reference and the table stay consistent
# (perturbing only the f=0 anchor of a deep-bar cage would be a lie).
_INDUCTION_PARAMS: dict[str, str] = {
    "R_2": "R_2_vs_f",
    "L_m": "L_m_vs_i",
    "L_sigma_s": "L_sigma_s_vs_f",
    "L_sigma_r": "L_sigma_r_vs_f",
}


_MAP_REFUSAL = "flux map is extracted, not derivable from geometry"


def _rederive_refusal(motor: Motor) -> str | None:
    """Why a geometry perturbation cannot re-derive psi_f, or None.

    OD/gap clear psi_f so the field models re-derive it from the new
    geometry. The radial Zhu & Howe derivation does not cover Halbach,
    inset or bridged rotors, and an extracted flux map cannot follow a
    geometry change — perturbing the scalars under a fixed map is a
    mixed state. Those motors read "n/a" rather than a wrong number.
    """
    if motor.flux_map is not None:
        return _MAP_REFUSAL
    if motor.magnetization == "halbach":
        return "Halbach psi_f is not derivable"
    geo = motor.geometry
    if getattr(geo, "magnet_mount", "surface") == "inset":
        return "inset psi_f is not derivable"
    if getattr(geo, "bridge_thickness", None) is not None:
        return "bridged psi_f is not derivable"
    return None


def scale_inductances(motor: Motor, factor: float) -> dict[str, Any]:
    """Scale L_d, L_q by factor. Returns kwargs for dc.replace()."""
    kw: dict[str, Any] = {}
    if motor.L_d is not None:
        kw["L_d"] = motor.L_d * factor
    if motor.L_q is not None:
        kw["L_q"] = motor.L_q * factor
    return kw


def perturb_motor(motor: Motor, param: PerturbParam, delta: float) -> Motor | None:
    if param in _INDUCTION_PARAMS:
        if motor.induction is None:
            return None
        val = getattr(motor.induction, param)
        if val is None:
            return None
        factor = 1.0 + delta
        new_val = val * factor
        if new_val <= 0:
            return None
        kw: dict[str, Any] = {param: new_val}
        table_name = _INDUCTION_PARAMS[param]
        table = getattr(motor.induction, table_name)
        if table is not None:
            kw[table_name] = tuple((x, y * factor) for x, y in table)
        return dc.replace(motor, induction=dc.replace(motor.induction, **kw),
                          name=f"{motor.name} [{param} {delta:+.1%}]")

    if param in _SCALAR_PARAMS:
        val = getattr(motor, param)
        if val is None:
            return None
        new_val = val * (1.0 + delta)
        if new_val <= 0:
            return None
        return dc.replace(motor, **{param: new_val},
                          name=f"{motor.name} [{param} {delta:+.1%}]")

    if motor.geometry is None or isinstance(motor.geometry, AxialFluxGeometry):
        return None
    if motor.induction is not None:
        # The geometry/magnet transforms are PM field-model levers; the
        # circuit tier never reads geometry, so a cage geometry would only
        # yield a flat 0% curve. Skip, as for a geometry-less motor.
        return None
    geo = motor.geometry
    topo = geo.topology

    # Every geometry/magnet lever clears psi_f. A flux map cannot follow
    # any of them; the non-derivable rotors only matter for OD/gap, where
    # the field models would otherwise re-derive from a changed geometry
    # (on L_stk/B_rem/k_w they raise explicitly in resolve_psi_f).
    if param in ("OD", "gap"):
        reason = _rederive_refusal(motor)
    elif motor.flux_map is not None:
        reason = _MAP_REFUSAL
    else:
        reason = None
    if reason is not None:
        logger.info("perturb_motor: %s %s skipped — %s", motor.name, param, reason)
        return None

    if param == "OD":
        # OEM frame-size model: stator and rotor grow together while air gap,
        # magnet thickness, and shaft (r_inner) stay fixed.  Bigger bore →
        # more flux linkage.  L_m ∝ r_bore/gap.
        r_outer_new = geo.r_outer * (1.0 + delta)
        shift = r_outer_new - geo.r_outer
        r_stator_new = geo.r_stator + shift
        # Smallest shifted radius must clear the shaft.
        r_min_shifted = r_stator_new if topo == "outrunner" else geo.r_rotor + shift
        if r_min_shifted <= geo.r_inner:
            return None
        try:
            # dc.replace carries every non-radial field (mount, overhang,
            # back iron, skew). Radial build is NOT scaled: slot_depth and
            # the [r_stator, r_outer] annulus are preserved by the shift —
            # a diameter change, not a uniform lamination scale. The
            # opening tracks the bore circumference (slot pitch grows with r).
            new_geo = dc.replace(
                geo, r_outer=r_outer_new, r_stator=r_stator_new,
                r_magnet=geo.r_magnet + shift, r_rotor=geo.r_rotor + shift,
                r_ag=geo.r_ag + shift,
                slot_opening_width=geo.slot_opening_width * (r_stator_new / geo.r_stator),
            )
        except ValueError:
            return None
        # L_m ∝ r_bore/gap (energy in gap ∝ r×g, B ∝ 1/g → W ∝ r/g).
        # Gap is fixed, so L scales linearly with bore radius.
        L_factor = r_stator_new / geo.r_stator
        return dc.replace(motor, geometry=new_geo, psi_f=None,
                          **scale_inductances(motor, L_factor),
                          name=f"{motor.name} [OD {delta:+.1%}]")

    elif param == "gap":
        # Move the stator surface, not the magnet face: the magnet ring
        # [r_magnet, r_rotor] is a fixed component in both topologies.
        gap = abs(geo.r_stator - geo.r_magnet)
        new_gap = gap * (1.0 + delta)
        sign = 1.0 if topo == "inrunner" else -1.0
        new_r_stator = geo.r_magnet + sign * new_gap
        if topo == "inrunner" and new_r_stator >= geo.r_outer:
            return None
        if topo == "outrunner" and new_r_stator <= geo.r_inner:
            return None
        # r_ag keeps its fractional position in the gap (mid-gap stays
        # mid-gap; a custom sampling radius moves with the stator face).
        frac = (geo.r_ag - geo.r_magnet) / (geo.r_stator - geo.r_magnet)
        try:
            new_geo = dc.replace(
                geo, r_stator=new_r_stator,
                r_ag=geo.r_magnet + frac * (new_r_stator - geo.r_magnet),
                slot_opening_width=geo.slot_opening_width * (new_r_stator / geo.r_stator),
            )
        except ValueError:
            return None
        # Magnetizing L scales ~1/g_eff with g_eff = g + l_m/mu_r_pm — the
        # magnet recoil path dominates for surface PM, so a pure-1/g scaling
        # overstates the sensitivity by ~g_eff/g. Leakage (k_w-independent,
        # often dominant for FSCW) is still not decomposed, so even this
        # overstates; same limitation class as the k_w branch below.
        l_m = abs(geo.r_magnet - geo.r_rotor)
        g_eff = gap + l_m / motor.mu_r_pm
        L_factor = g_eff / (new_gap + l_m / motor.mu_r_pm)
        return dc.replace(motor, geometry=new_geo, psi_f=None,
                          **scale_inductances(motor, L_factor),
                          name=f"{motor.name} [gap {delta:+.1%}]")

    elif param == "L_stk":
        if motor.L_stk is None:
            return None
        new_L = motor.L_stk * (1.0 + delta)
        if new_L <= 0:
            return None
        # R_s scales with slot length (first-order; end-turn fraction ignored)
        r_s_kw: dict[str, Any] = {}
        if motor.R_s is not None:
            r_s_kw["R_s"] = motor.R_s * (1.0 + delta)
        return dc.replace(motor, L_stk=new_L, psi_f=None,
                          **scale_inductances(motor, 1.0 + delta),
                          **r_s_kw,
                          name=f"{motor.name} [L_stk {delta:+.1%}]")

    elif param == "B_rem":
        if motor.B_rem is None:
            return None
        new_B = motor.B_rem * (1.0 + delta)
        if new_B <= 0:
            return None
        return dc.replace(motor, B_rem=new_B, psi_f=None,
                          name=f"{motor.name} [B_rem {delta:+.1%}]")

    elif param == "k_w":
        if motor.k_w is None:
            return None
        new_k_w = motor.k_w * (1.0 + delta)
        # k_w is physically bounded to [0.75, 1.0]. Below 0.75 is not a
        # realistic winding; above 1.0 is nonphysical. Return None so the
        # caller skips this point explicitly.
        if not (0.75 <= new_k_w <= 1.0):
            return None
        # NOTE: L_d and L_q are NOT scaled here. Magnetizing inductance scales
        # as k_w² but total L includes leakage (typically 15–25% of total) which
        # is k_w-independent. Without leakage decomposition the correct scaling
        # factor is unknown. A documented model limitation.
        # Impact: k_w sensitivity on torque/back-EMF is correct (via psi_f
        # re-derivation); sensitivity on current-loop dynamics (L_d/L_q) is
        # underestimated.
        return dc.replace(motor, k_w=new_k_w, psi_f=None,
                          name=f"{motor.name} [k_w {delta:+.1%}]")

    return None
