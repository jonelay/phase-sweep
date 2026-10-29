"""phasesweep — PMSM air-gap flux density and drive simulation.

Public API: motor/geometry types, TOML loaders, validated solver-param
factories, the model registry, and run/result types. The dict shape of
a stored result row is declared in ``phasesweep.record_schema``, with
``docs/result-store-contract.md`` as the prose contract; together they
are the consumer contract (downstream adapters check against them; no
top-level re-export). Internal modules (fem_field, sim, plots, runners)
are importable directly but their interfaces may change between minor
versions.
"""

from phasesweep.machines.configs import load_motor, load_motors
from phasesweep.machines.geometry import AxialFluxGeometry, Geometry, axial_flux, inrunner, outrunner
from phasesweep.machines.motor import DriveParams, InductionParams, Motor
from phasesweep.models.im_circuit import ImCircuitParams, prepare_im_circuit
from phasesweep.models.rated_torque import (
    RatedTorqueParams,
    StallTorqueParams,
    prepare_rated_torque,
    prepare_stall_torque,
)
from phasesweep.models.thermal_duty import ThermalDutyParams, prepare_thermal_duty
from phasesweep.registry import MODEL_REGISTRY, ModelInfo
from phasesweep.result_store import ResultStore
from phasesweep.simulation.params import (
    DriveSimParams,
    TwoMassLoad,
    prepare_drive_sim,
)
from phasesweep.solvers.analytical import AnalyticalParams, prepare_analytical
from phasesweep.solvers.fem_runner import FemParams, prepare_fem
from phasesweep.sweep_types import RunConfig, RunResult, compute_run_id

__version__ = "0.6.0"

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
    "AxialFluxGeometry",
    "DriveParams",
    "DriveSimParams",
    "FemParams",
    "Geometry",
    "ImCircuitParams",
    "InductionParams",
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
    "axial_flux",
    "compute_run_id",
    "debug_info",
    "inrunner",
    "load_motor",
    "load_motors",
    "outrunner",
    "prepare_analytical",
    "prepare_drive_sim",
    "prepare_fem",
    "prepare_im_circuit",
    "prepare_rated_torque",
    "prepare_stall_torque",
    "prepare_thermal_duty",
]
