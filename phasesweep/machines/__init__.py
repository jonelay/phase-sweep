"""Machine definition and loading."""

from phasesweep.machines.configs import load_motor, load_motors, motor_from_dict
from phasesweep.machines.geometry import (
    AxialFluxGeometry,
    Geometry,
    axial_flux,
    geometry_from_toml,
    inrunner,
    outrunner,
)
from phasesweep.machines.motor import DriveParams, InductionParams, Motor
from phasesweep.machines.perturbation import perturb_motor

__all__ = [
    "AxialFluxGeometry",
    "DriveParams",
    "Geometry",
    "InductionParams",
    "Motor",
    "axial_flux",
    "geometry_from_toml",
    "inrunner",
    "load_motor",
    "load_motors",
    "motor_from_dict",
    "outrunner",
    "perturb_motor",
]
