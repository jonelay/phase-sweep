"""Validated drive-simulation parameter types and factory.

Deliberately separate from sim.py: that module raises ImportError
without the [sim] extra (motulator), while these params must stay
importable everywhere — RunConfig carries TwoMassLoad and the registry
validates drive_sim configs on hosts without the simulation stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from phasesweep.solver_params import (
    _reject_induction,
    _require_drive_fields,
    _resolve_psi_f,
)

if TYPE_CHECKING:
    from phasesweep.machines.geometry import AxialFluxGeometry, Geometry
    from phasesweep.machines.motor import DriveParams, Motor


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
    """

    n_p: int
    R_s: float
    L_d: float
    L_q: float
    psi_f: float
    J: float
    drive: DriveParams
    geometry: Geometry | AxialFluxGeometry | None = None

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


def prepare_drive_sim(motor: Motor) -> DriveSimParams:
    """Derive, validate, and extract drive sim params from a Motor."""
    _reject_induction(motor, "drive_sim")
    psi_f = _resolve_psi_f(motor)

    if motor.R_s is None or motor.L_d is None or motor.L_q is None or motor.J is None:
        missing = [f for f in ("R_s", "L_d", "L_q", "J")
                   if getattr(motor, f) is None]
        raise ValueError(
            f"Motor '{motor.name}': drive_sim needs {', '.join(missing)}"
        )
    _require_drive_fields(motor, "drive_sim", ("U_DC", "MAX_I_S", "W_REF"))

    return DriveSimParams(
        geometry=motor.geometry,
        n_p=motor.n_p,
        R_s=motor.R_s,
        L_d=motor.L_d,
        L_q=motor.L_q,
        psi_f=psi_f,
        J=motor.J,
        drive=motor.drive,
    )
