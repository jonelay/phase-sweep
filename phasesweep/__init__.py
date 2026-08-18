"""phasesweep — PMSM air-gap flux density and drive simulation.

Public API: motor/geometry types, TOML loaders, validated solver-param
factories, the model registry, and run/result types. Internal modules
(fem_field, sim, plots, runners) are importable directly but their
interfaces may change between minor versions.
"""

from phasesweep.machines.configs import load_motor, load_motors
from phasesweep.machines.geometry import Geometry, inrunner, outrunner
from phasesweep.machines.motor import DriveParams, Motor
from phasesweep.registry import MODEL_REGISTRY, ModelInfo
from phasesweep.result_store import ResultStore
from phasesweep.solver_params import (
    AnalyticalParams,
    DriveSimParams,
    FemParams,
    RatedTorqueParams,
    StallTorqueParams,
    ThermalDutyParams,
    TwoMassLoad,
    prepare_analytical,
    prepare_drive_sim,
    prepare_fem,
    prepare_rated_torque,
    prepare_stall_torque,
    prepare_thermal_duty,
)
from phasesweep.sweep_types import RunConfig, RunResult, compute_run_id

__version__ = "0.4.0"

_DEBUG_DEPS = ("numpy", "scipy", "ngsolve", "motulator", "matplotlib", "fastapi", "cupy")


def _dep_version(module: str) -> str:
    import importlib

    try:
        mod = importlib.import_module(module)
    except ModuleNotFoundError:
        return "not installed"
    except Exception as e:
        return f"broken ({type(e).__name__}: {e})"
    return getattr(mod, "__version__", "installed")


def debug_info() -> None:
    """Print environment info for bug reports."""
    import platform

    print(f"phasesweep : {__version__}")
    print(f"python     : {platform.python_version()} ({platform.python_implementation()})")
    print(f"platform   : {platform.platform()}")
    for dep in _DEBUG_DEPS:
        print(f"{dep:<11}: {_dep_version(dep)}")


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
    "TwoMassLoad",
    "__version__",
    "compute_run_id",
    "debug_info",
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
