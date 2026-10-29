"""Validated drive-simulation parameter types and factory.

Deliberately separate from sim.py: that module raises ImportError
without the [sim] extra (motulator), while these params must stay
importable everywhere — RunConfig carries TwoMassLoad and the registry
validates drive_sim configs on hosts without the simulation stack.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from phasesweep.solver_params import (
    derate_motor,
    reject_induction,
    require_drive_fields,
    resolve_psi_f,
)

if TYPE_CHECKING:
    from phasesweep.machines.geometry import AxialFluxGeometry, Geometry
    from phasesweep.machines.motor import DriveParams, FluxMap, Motor

__all__ = [
    "DriveSimParams",
    "SimPlan",
    "TwoMassLoad",
    "prepare_drive_sim",
    "uses_flux_map",
]


@dataclass(frozen=True)
class SimPlan:
    """Self-contained execution plan for a drive simulation.

    Carries load parameters AND metric extraction timing windows.
    Built by plan_sim(); consumed by build_sim() and extract_metrics().
    """

    # Execution
    load_torque: float      # Nm
    load_time: float        # s — when load step is applied
    t_stop: float           # s — simulation end
    speed_step_time: float  # s — when speed ref steps from 0

    # Metric extraction windows
    settle_threshold: float            # fraction of w_ref (e.g. 0.05)
    ss_window: float                   # s — averaging window before t_stop
    droop_window: float                # s — post-load window for speed dip
    accel_window: tuple[float, float]  # (start, end) for peak torque

    # Controller tuning
    alpha_s: float          # speed-controller bandwidth (rad/s)
    alpha_c: float          # current-controller bandwidth (rad/s)
    T_s: float              # control sampling period (s)

    # Diagnostic
    tau_m: float            # mechanical time constant (s)

    def to_dict(self) -> dict[str, Any]:
        return {
            "load_torque": self.load_torque,
            "load_time": self.load_time,
            "t_stop": self.t_stop,
            "speed_step_time": self.speed_step_time,
            "settle_threshold": self.settle_threshold,
            "ss_window": self.ss_window,
            "droop_window": self.droop_window,
            "accel_window": list(self.accel_window),
            "alpha_s": self.alpha_s,
            "alpha_c": self.alpha_c,
            "T_s": self.T_s,
            "tau_m": self.tau_m,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SimPlan:
        aw = d["accel_window"]
        return cls(
            load_torque=d["load_torque"],
            load_time=d["load_time"],
            t_stop=d["t_stop"],
            speed_step_time=d["speed_step_time"],
            settle_threshold=d["settle_threshold"],
            ss_window=d["ss_window"],
            droop_window=d["droop_window"],
            accel_window=(aw[0], aw[1]),
            alpha_s=d.get("alpha_s", 2 * math.pi * 4),
            alpha_c=d.get("alpha_c", 2 * math.pi * 200),
            T_s=d.get("T_s", 125e-6),
            tau_m=d["tau_m"],
        )


@dataclass(frozen=True)
class TwoMassLoad:
    """Load-side parameters for a two-mass mechanical system.

    J_L — load inertia (kg·m²); K_S — shaft torsional stiffness (N·m/rad);
    C_S — shaft torsional damping (N·m·s/rad); B_L — viscous load friction
    (N·m·s/rad).
    """

    J_L: float
    K_S: float
    C_S: float
    B_L: float = 0.0

    def __post_init__(self) -> None:
        if self.J_L <= 0:
            raise ValueError(f"J_L must be > 0, got {self.J_L}")
        if self.K_S <= 0:
            raise ValueError(f"K_S must be > 0, got {self.K_S}")
        if self.C_S < 0:
            raise ValueError(f"C_S must be >= 0, got {self.C_S}")
        if self.B_L < 0:
            raise ValueError(f"B_L must be >= 0, got {self.B_L}")

    def to_dict(self) -> dict[str, float]:
        return {"J_L": self.J_L, "K_S": self.K_S, "C_S": self.C_S, "B_L": self.B_L}

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> TwoMassLoad:
        return cls(J_L=d["J_L"], K_S=d["K_S"], C_S=d["C_S"], B_L=d.get("B_L", 0.0))


@dataclass(frozen=True)
class DriveSimParams:
    """Validated inputs for motulator PMSM drive simulation.

    R_s — stator resistance (Ω); L_d / L_q — inductances (H);
    psi_f — PM flux linkage (Wb, peak); J — rotor inertia (kg·m²);
    drive — DriveParams operating point (V / A peak / mechanical rad/s).

    `geometry` is optional — `build_sim()` does not read it, it exists for
    phase-sweep's hashing/traceability pipeline. Dynamics-only consumers
    (e.g. cross-checks against an external plant model) may omit it.

    `flux_map` — optional tabulated ψ_d/ψ_q surface; when it covers the
    drive current limit (see `uses_flux_map`) the machine model and the
    controller run on it instead of the constant L_d/L_q/psi_f triple,
    which then only feeds the timing plan and the saliency guards.
    """

    n_p: int
    R_s: float
    L_d: float
    L_q: float
    psi_f: float
    J: float
    drive: DriveParams
    geometry: Geometry | AxialFluxGeometry | None = None
    flux_map: FluxMap | None = None

    @property
    def is_reluctance(self) -> bool:
        return self.psi_f == 0.0

    def __post_init__(self) -> None:
        if self.n_p < 2:
            raise ValueError(f"n_p must be >= 2, got {self.n_p}")
        if self.R_s < 0:
            raise ValueError(f"R_s must be >= 0, got {self.R_s}")
        if self.L_d <= 0:
            raise ValueError(f"L_d must be > 0, got {self.L_d}")
        if self.L_q <= 0:
            raise ValueError(f"L_q must be > 0, got {self.L_q}")
        if self.psi_f < 0:
            raise ValueError(f"psi_f must be >= 0, got {self.psi_f}")
        if self.J <= 0:
            raise ValueError(f"J must be > 0, got {self.J}")


def uses_flux_map(params: DriveSimParams) -> bool:
    """True when the drive sim runs on the flux-linkage surface: a map is
    present, the origin is a grid node (`has_origin` — conservative; not
    required by any code path, since motulator derives psi_f from the
    inverse), and the tabulated grid covers MAX_I_S on both axes
    (torque_speed gates on I_LIMIT-or-MAX_I_S instead; MAX_I_S is the
    right limit here because it is the controller's `i_s_max`).
    Otherwise the constant-L model is used."""
    fm = params.flux_map
    if fm is None or not fm.has_origin:
        return False
    i_max = params.drive.MAX_I_S
    if i_max is None:
        return False
    return i_max <= min(-fm.i_d_A[0], fm.i_q_A[-1])


def prepare_drive_sim(motor: Motor) -> DriveSimParams:
    """Derive, validate, and extract drive sim params from a Motor.

    Unlike rated_torque/torque_speed, does not fall back to flux_map_psi_f
    when psi_f/B_rem are absent — L_d/L_q/psi_f are still required for
    the timing plan and saliency guards even when a map supplies the
    machine model. R_s, psi_f and the map are at the motor's
    winding_temp_limit / magnet_temp (solver_params.derate)."""
    from phasesweep.models.flux_map import derated_flux_map

    reject_induction(motor, "drive_sim")
    psi_f = resolve_psi_f(motor)

    if motor.R_s is None or motor.L_d is None or motor.L_q is None or motor.J is None:
        missing = [f for f in ("R_s", "L_d", "L_q", "J")
                   if getattr(motor, f) is None]
        raise ValueError(
            f"Motor '{motor.name}': drive_sim needs {', '.join(missing)}"
        )
    require_drive_fields(motor, "drive_sim", ("U_DC", "MAX_I_S", "W_REF"))
    R_s_op, psi_f_op = derate_motor(motor, motor.R_s, psi_f)
    assert R_s_op is not None

    return DriveSimParams(
        geometry=motor.geometry,
        n_p=motor.n_p,
        R_s=R_s_op,
        L_d=motor.L_d,
        L_q=motor.L_q,
        psi_f=psi_f_op,
        J=motor.J,
        drive=motor.drive,
        flux_map=derated_flux_map(motor),
    )
