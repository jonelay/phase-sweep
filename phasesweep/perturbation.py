"""Motor perturbation functions for sensitivity analysis."""

from __future__ import annotations

import dataclasses as dc
from typing import Literal

from phasesweep.geometry import inrunner, outrunner
from phasesweep.motor import Motor

PerturbParam = Literal["OD", "gap", "L_stk", "B_rem", "k_w"]


def scale_inductances(motor: Motor, factor: float) -> dict[str, float]:
    """Scale L_d, L_q by factor. Returns kwargs for dc.replace()."""
    kw: dict[str, float] = {}
    if motor.L_d is not None:
        kw["L_d"] = motor.L_d * factor
    if motor.L_q is not None:
        kw["L_q"] = motor.L_q * factor
    return kw


def perturb_motor(motor: Motor, param: PerturbParam, delta: float) -> Motor | None:
    geo = motor.geometry
    topo = geo.topology

    if param == "OD":
        # OEM frame-size model: stator and rotor grow together while air gap,
        # magnet thickness, and shaft (r_inner) stay fixed.  Bigger bore →
        # more flux linkage.  L_m ∝ r_bore/gap.
        r_outer_new = geo.r_outer * (1.0 + delta)
        shift = r_outer_new - geo.r_outer
        r_stator_new = geo.r_stator + shift
        r_magnet_new = geo.r_magnet + shift
        r_rotor_new = geo.r_rotor + shift
        # Smallest shifted radius must clear the shaft.
        r_min_shifted = r_stator_new if topo == "outrunner" else r_rotor_new
        if r_min_shifted <= geo.r_inner:
            return None
        builder = outrunner if topo == "outrunner" else inrunner
        try:
            new_geo = builder(
                r_outer=r_outer_new, r_stator=r_stator_new,
                r_magnet=r_magnet_new, r_rotor=r_rotor_new,
                r_inner=geo.r_inner,
                n_slots=geo.n_slots, slot_depth=geo.slot_depth,
                slot_width_ratio=geo.slot_width_ratio,
                # opening scales with bore: ratio stays constant under
                # frame-size scaling (whole lamination scales)
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
        # Change gap thickness while keeping magnet and yoke unchanged
        if topo == "inrunner":
            gap = geo.r_stator - geo.r_magnet
            new_gap = gap * (1.0 + delta)
            new_r_stator = geo.r_magnet + new_gap
            if new_r_stator >= geo.r_outer:
                return None
            try:
                new_geo = inrunner(
                    r_outer=geo.r_outer, r_stator=new_r_stator,
                    r_magnet=geo.r_magnet, r_rotor=geo.r_rotor,
                    r_inner=geo.r_inner,
                    n_slots=geo.n_slots, slot_depth=geo.slot_depth,
                    slot_width_ratio=geo.slot_width_ratio,
                    slot_opening_width=geo.slot_opening_width * (new_r_stator / geo.r_stator),
                )
            except ValueError:
                return None
        else:
            gap = geo.r_magnet - geo.r_stator
            new_gap = gap * (1.0 + delta)
            new_r_magnet = geo.r_stator + new_gap
            if new_r_magnet >= geo.r_rotor:
                return None
            try:
                new_geo = outrunner(
                    r_outer=geo.r_outer, r_rotor=geo.r_rotor,
                    r_magnet=new_r_magnet, r_stator=geo.r_stator,
                    r_inner=geo.r_inner,
                    n_slots=geo.n_slots, slot_depth=geo.slot_depth,
                    slot_width_ratio=geo.slot_width_ratio,
                    slot_opening_width=geo.slot_opening_width,
                )
            except ValueError:
                return None
        # L scales ~1/gap (magnetizing component dominates; leakage not decomposed)
        return dc.replace(motor, geometry=new_geo, psi_f=None,
                          **scale_inductances(motor, 1.0 / (1.0 + delta)),
                          name=f"{motor.name} [gap {delta:+.1%}]")

    elif param == "L_stk":
        if motor.L_stk is None:
            return None
        new_L = motor.L_stk * (1.0 + delta)
        if new_L <= 0:
            return None
        # R_s scales with slot length (first-order; end-turn fraction ignored)
        r_s_kw: dict[str, float] = {}
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
        # factor is unknown. This is a documented limitation.
        # Impact: k_w sensitivity on torque/back-EMF is correct (via psi_f
        # re-derivation); sensitivity on current-loop dynamics (L_d/L_q) is
        # underestimated.
        return dc.replace(motor, k_w=new_k_w, psi_f=None,
                          name=f"{motor.name} [k_w {delta:+.1%}]")

    return None
