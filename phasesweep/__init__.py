"""phasesweep — PMSM air-gap flux density and drive simulation.

Public API: motor/geometry types, TOML loaders, validated solver-param
factories, the model registry, and run/result types. Internal modules
(fem_field, sim, plots, runners) are importable directly but their
interfaces may change between minor versions.
"""

from phasesweep.configs import load_motor, load_motors
from phasesweep.geometry import Geometry, inrunner, outrunner
from phasesweep.motor import DriveParams, Motor
from phasesweep.registry import MODEL_REGISTRY, ModelInfo
from phasesweep.result_store import ResultStore
from phasesweep.solver_params import (
    AnalyticalParams,
    DriveSimParams,
    FemParams,
    RatedTorqueParams,
    StallTorqueParams,
    ThermalDutyParams,
    prepare_analytical,
    prepare_drive_sim,
    prepare_fem,
    prepare_rated_torque,
    prepare_stall_torque,
    prepare_thermal_duty,
)
from phasesweep.sweep_types import RunConfig, RunResult, compute_run_id

__version__ = "0.3.0"

__all__ = [
    "MODEL_REGISTRY",
    "AnalyticalParams",
    "DriveParams",
    "DriveSimParams",
    "FemParams",
    "Geometry",
    "ModelInfo",
    "Motor",
    "RatedTorqueParams",
    "ResultStore",
    "RunConfig",
    "RunResult",
    "StallTorqueParams",
    "ThermalDutyParams",
    "__version__",
    "compute_run_id",
    "inrunner",
    "load_motor",
    "load_motors",
    "outrunner",
    "prepare_analytical",
    "prepare_drive_sim",
    "prepare_fem",
    "prepare_rated_torque",
    "prepare_stall_torque",
    "prepare_thermal_duty",
]
