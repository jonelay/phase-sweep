"""Type-safe, validated configuration and result objects for motor runs."""

from __future__ import annotations

import hashlib
import json
import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, get_args

if TYPE_CHECKING:
    from phasesweep.machines.motor import Motor

from phasesweep.record_schema import RunConfigDict, RunResultDict
from phasesweep.simulation.params import SimPlan, TwoMassLoad

__all__ = [
    "BoundaryCondition",
    "Magnetization",
    "NonlinearSolver",
    "RunConfig",
    "RunResult",
    "Source",
    "Status",
    "compute_run_id",
    "resolve_model",
]

Status = Literal["OK", "TIMEOUT", "ERROR"]

NonlinearSolver = Literal["picard", "newton"]
Magnetization = Literal["radial", "halbach"]
BoundaryCondition = Literal["neumann", "dirichlet"]

_DEFAULTS: dict[str, Any] = {
    "maxh_fraction": 0.05,
    "n_theta": 360,
    "nonlinear": False,
    "nonlinear_solver": "newton",
    "j_s": 0.0,
    "rotation": 0.0,
    "cogging_points": 12,
    "n_planes": 3,
}


def _is_finite_real(v: object) -> bool:
    return (not isinstance(v, bool) and isinstance(v, numbers.Real)
            and math.isfinite(v))


@dataclass(frozen=True)
class RunConfig:
    """Run configuration composing a Motor with solver/sim parameters.

    model — registry key ("analytical", "fem", "drive_sim", ...);
    maxh_fraction — max FEM mesh element size as fraction of r_outer;
    n_theta — airgap angular samples per revolution;
    nonlinear — enable B-H curve iteration (FEM);
    nonlinear_solver — BH iteration scheme for nonlinear FEM solves:
    "newton" (default, line-search damped with Picard fallback) or
    "picard"; validated by the solver;
    sim_plan — drive-sim timing/load plan (SimPlan);
    j_s — armature slot current density amplitude (A/m², 0 = open circuit);
    i_fault — peak phase current (A) the demag_screen model screens at;
    required by that model (no drive-limit fallback — the screen current
    is a deliberate choice: drive limit, short-circuit peak, ...);
    duty_profile — thermal-duty torque-time segments ((torque_Nm,
    duration_s), ...) for the thermal_duty model;
    dataset_id — identity of an imported measured dataset (None for
    computed models); distinguishes repeat captures of the same test;
    load_mech — two-mass load parameters (TwoMassLoad) for the
    drive_sim_two_mass model; None for single-mass drive_sim;
    rotation — mechanical rotor angle (rad) for rotated FEM solves;
    0.0 = aligned (identity-preserving default, no version bump);
    cogging_points — angular samples per cogging period for the
    cogging_torque model (default 12);
    n_planes — quasi-3D computation planes for the fem_axial model
    (default 3; odd values put the middle plane at r_mean);
    slip — rotor slip ((0, 1], the motoring range) for induction-machine
    models; no default (the operating slip is a deliberate choice:
    rated slip, standstill, ...). Slip is an operating point, so it
    lives here and in hash_fields, not on the Motor.
    w_mech / tau_ref — operating-point demand (mechanical rad/s, N·m) for
    the operating_point model; no defaults, both finite and >= 0.
    T_coolant — coolant (sink) temperature (°C) for the motor_thermal
    model; an operating condition like w_mech, so it lives here, not on
    the Motor; no default, finite.
    """

    motor: Motor
    model: str

    maxh_fraction: float = _DEFAULTS["maxh_fraction"]
    n_theta: int = _DEFAULTS["n_theta"]
    nonlinear: bool = _DEFAULTS["nonlinear"]
    nonlinear_solver: NonlinearSolver = _DEFAULTS["nonlinear_solver"]

    sim_plan: SimPlan | None = None

    j_s: float = _DEFAULTS["j_s"]

    i_fault: float | None = None

    duty_profile: tuple[tuple[float, float], ...] | None = None

    dataset_id: str | None = None

    load_mech: TwoMassLoad | None = None

    rotation: float = _DEFAULTS["rotation"]

    cogging_points: int = _DEFAULTS["cogging_points"]

    n_planes: int = _DEFAULTS["n_planes"]

    slip: float | None = None

    w_mech: float | None = None

    tau_ref: float | None = None

    T_coolant: float | None = None

    def __post_init__(self) -> None:
        from phasesweep.machines.geometry import int_normalize

        # Every field below reaches compute_run_id: a mistyped value
        # ("false", "180", NaN) would hash as its own configuration.
        for name in ("n_theta", "cogging_points", "n_planes"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (numbers.Integral, float)):
                raise ValueError(f"{name} must be an integer, got {v!r}")
        int_normalize(self, ("n_theta", "cogging_points", "n_planes"), "RunConfig")
        for name in ("n_theta", "cogging_points", "n_planes"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        if not isinstance(self.nonlinear, bool):
            raise ValueError(f"nonlinear must be a bool, got {self.nonlinear!r}")
        if self.nonlinear_solver not in get_args(NonlinearSolver):
            raise ValueError(
                f"nonlinear_solver must be one of {get_args(NonlinearSolver)}, "
                f"got {self.nonlinear_solver!r}")
        for name in ("maxh_fraction", "j_s", "rotation", "i_fault", "slip"):
            v = getattr(self, name)
            if v is None and name in ("i_fault", "slip"):
                continue
            if not _is_finite_real(v):
                raise ValueError(f"{name} must be a finite number, got {v!r}")
        if self.maxh_fraction <= 0:
            raise ValueError(f"maxh_fraction must be > 0, got {self.maxh_fraction}")
        if self.duty_profile is not None:
            for seg in self.duty_profile:
                if (len(seg) != 2 or not all(_is_finite_real(x) for x in seg)
                        or seg[1] <= 0):
                    raise ValueError(
                        f"duty_profile segments must be finite (torque_Nm, "
                        f"duration_s > 0), got {seg!r}")
        # NaN passes every inequality, so isfinite comes first.
        for name in ("w_mech", "tau_ref"):
            v = getattr(self, name)
            if v is None:
                continue
            if not math.isfinite(v) or v < 0:
                raise ValueError(f"{name} must be finite and >= 0, got {v}")
        if self.T_coolant is not None and not math.isfinite(self.T_coolant):
            raise ValueError(f"T_coolant must be finite, got {self.T_coolant}")

    def to_dict(self) -> RunConfigDict:
        d: RunConfigDict = {
            "motor": self.motor.to_dict(),
            "model": self.model,
            "maxh_fraction": self.maxh_fraction,
            "n_theta": self.n_theta,
            "nonlinear": self.nonlinear,
            "j_s": self.j_s,
        }
        if self.i_fault is not None:
            d["i_fault"] = self.i_fault
        if self.sim_plan is not None:
            d["sim_plan"] = self.sim_plan.to_dict()
        if self.duty_profile is not None:
            d["duty_profile"] = [list(seg) for seg in self.duty_profile]
        if self.dataset_id is not None:
            d["dataset_id"] = self.dataset_id
        if self.load_mech is not None:
            d["load_mech"] = self.load_mech.to_dict()
        if self.nonlinear_solver != _DEFAULTS["nonlinear_solver"]:
            d["nonlinear_solver"] = self.nonlinear_solver
        if self.rotation != _DEFAULTS["rotation"]:
            d["rotation"] = self.rotation
        if self.cogging_points != _DEFAULTS["cogging_points"]:
            d["cogging_points"] = self.cogging_points
        if self.n_planes != _DEFAULTS["n_planes"]:
            d["n_planes"] = self.n_planes
        if self.slip is not None:
            d["slip"] = self.slip
        if self.w_mech is not None:
            d["w_mech"] = self.w_mech
        if self.tau_ref is not None:
            d["tau_ref"] = self.tau_ref
        if self.T_coolant is not None:
            d["T_coolant"] = self.T_coolant
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> RunConfig:
        from phasesweep.machines.motor import Motor as MotorCls

        motor = MotorCls.from_dict(d["motor"])
        sim_plan = None
        if "sim_plan" in d and d["sim_plan"] is not None:
            sim_plan = SimPlan.from_dict(d["sim_plan"])
        duty_profile = None
        if d.get("duty_profile") is not None:
            duty_profile = tuple((float(t), float(dt)) for t, dt in d["duty_profile"])
        load_mech = None
        if d.get("load_mech") is not None:
            load_mech = TwoMassLoad.from_dict(d["load_mech"])
        return cls(
            motor=motor,
            model=d["model"],
            maxh_fraction=d.get("maxh_fraction", _DEFAULTS["maxh_fraction"]),
            n_theta=d.get("n_theta", _DEFAULTS["n_theta"]),
            nonlinear=d.get("nonlinear", _DEFAULTS["nonlinear"]),
            nonlinear_solver=d.get("nonlinear_solver", _DEFAULTS["nonlinear_solver"]),
            sim_plan=sim_plan,
            j_s=d.get("j_s", _DEFAULTS["j_s"]),
            i_fault=d.get("i_fault"),
            duty_profile=duty_profile,
            dataset_id=d.get("dataset_id"),
            load_mech=load_mech,
            rotation=d.get("rotation", _DEFAULTS["rotation"]),
            cogging_points=d.get("cogging_points", _DEFAULTS["cogging_points"]),
            n_planes=d.get("n_planes", _DEFAULTS["n_planes"]),
            slip=d.get("slip"),
            w_mech=d.get("w_mech"),
            tau_ref=d.get("tau_ref"),
            T_coolant=d.get("T_coolant"),
        )


Source = Literal["computed", "measured", "published"]

_SOURCES: tuple[str, ...] = ("computed", "measured", "published")
_STATUSES: tuple[str, ...] = ("OK", "TIMEOUT", "ERROR")


def normalize_source(raw: Any) -> Source:
    """Map a stored source to the Source Literal; raise if it is none.

    Legacy rows carry a descriptive suffix ("published (ANSYS FEM)");
    the class is the part before " (".
    """
    if not isinstance(raw, str):
        raise ValueError(f"source must be a string, got {raw!r}")
    s = raw.split(" (")[0]
    if s not in _SOURCES:
        raise ValueError(f"source must be one of {_SOURCES}, got {raw!r}")
    return s  # type: ignore[return-value]


def check_status(raw: Any) -> Status:
    if raw not in _STATUSES:
        raise ValueError(f"status must be one of {_STATUSES}, got {raw!r}")
    return raw

# Legacy compat: old persisted JSONL may have "sim" instead of "drive_sim".
_LEGACY_MODEL_MAP: dict[str, str] = {
    "sim": "drive_sim",
}


def resolve_model(d: Mapping[str, Any]) -> str:
    """Resolve model key from v2.0 'model' or v1.0 'run_type' field."""
    raw = d.get("model") or d.get("run_type")
    if raw is None:
        raise KeyError("missing 'model' (or legacy 'run_type')")
    return _LEGACY_MODEL_MAP.get(raw, raw)


@dataclass
class RunResult:
    """Result from a single run (FEM, sim, or analytical).

    metrics keys/units are model-specific (see MODEL_REGISTRY produces
    and docs/glossary.md); elapsed_s — wall-clock solve time (s).
    """

    config: RunConfig
    model: str
    status: Status
    metrics: dict[str, Any] | None
    elapsed_s: float
    error_msg: str | None = None
    source: Source = "computed"
    tolerances: dict[str, float] | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    # Registry model-code version at save time (stamped by ResultStore.save,
    # including v1 — an unstamped record must not be confusable with a v1
    # one). None on records loaded from pre-v2.1 stores: unknown vintage,
    # treated as stale by the read-side checks.
    model_version: int | None = None
    schema_version: str = "v2.1"

    @property
    def motor_config_id(self) -> str:
        return self.config.motor.config_id

    def to_dict(self) -> RunResultDict:
        d: RunResultDict = {
            "config": self.config.to_dict(),
            "model": self.model,
            "source": self.source,
            "motor_config_id": self.motor_config_id,
            "status": self.status,
            "metrics": self.metrics,
            "elapsed_s": self.elapsed_s,
            "error_msg": self.error_msg,
            "timestamp": self.timestamp,
            "model_version": self.model_version,
            "schema_version": self.schema_version,
        }
        if self.tolerances:
            d["tolerances"] = self.tolerances
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> RunResult:
        return cls(
            config=RunConfig.from_dict(d["config"]),
            model=resolve_model(d),
            status=check_status(d["status"]),
            metrics=d.get("metrics"),
            elapsed_s=d["elapsed_s"],
            error_msg=d.get("error_msg"),
            source=normalize_source(d.get("source", "computed")),
            tolerances=d.get("tolerances"),
            timestamp=d.get("timestamp", ""),
            model_version=d.get("model_version"),
            schema_version=d.get("schema_version", "v2.0"),
        )


def _run_id_key(
    rc: RunConfig, needs: frozenset[str] | None = None,
) -> str:
    """Build the canonical pipe-delimited hash string for a RunConfig.

    Canonicalization is consumer-owned (sweepstore never reformats the
    input). This function is the single source of truth for the hash
    string shape; compute_run_id hashes its output.
    """
    from phasesweep.registry import MODEL_REGISTRY
    info = MODEL_REGISTRY.get(rc.model)
    if needs is None and info is not None:
        needs = info.hash_fields

    parts = [rc.motor.config_id, rc.model]
    if info is not None and info.version > 1:
        parts.append(f"model_v={info.version}")
    if rc.dataset_id is not None:
        parts.append(f"dataset={rc.dataset_id}")

    drive = rc.motor.drive
    solver_params: dict[str, str] = {
        "maxh_fraction": f"{rc.maxh_fraction:.4f}",
        "n_theta": str(rc.n_theta),
        "nonlinear": str(rc.nonlinear),
        "j_s": f"{rc.j_s:.6f}",
        "i_fault": "none" if rc.i_fault is None else f"{rc.i_fault:.6g}",
        "U_DC": "none" if drive.U_DC is None else f"{drive.U_DC:.6g}",
        "MAX_I_S": "none" if drive.MAX_I_S is None else f"{drive.MAX_I_S:.6g}",
        "W_REF": "none" if drive.W_REF is None else f"{drive.W_REF:.6g}",
        "I_LIMIT": "none" if drive.I_LIMIT is None else f"{drive.I_LIMIT:.6g}",
    }
    if rc.sim_plan is not None:
        solver_params["sim_plan"] = json.dumps(
            rc.sim_plan.to_dict(), sort_keys=True
        )
    if rc.duty_profile is not None:
        solver_params["duty_profile"] = json.dumps(
            [list(seg) for seg in rc.duty_profile], sort_keys=True
        )
    if rc.load_mech is not None:
        solver_params["load_mech"] = json.dumps(
            rc.load_mech.to_dict(), sort_keys=True
        )
    if rc.nonlinear_solver != _DEFAULTS["nonlinear_solver"]:
        solver_params["nonlinear_solver"] = rc.nonlinear_solver
    if rc.rotation != _DEFAULTS["rotation"]:
        solver_params["rotation"] = f"{rc.rotation:.10f}"
    if rc.cogging_points != _DEFAULTS["cogging_points"]:
        solver_params["cogging_points"] = str(rc.cogging_points)
    if rc.n_planes != _DEFAULTS["n_planes"]:
        solver_params["n_planes"] = str(rc.n_planes)
    if rc.slip is not None:
        solver_params["slip"] = f"{rc.slip:.10g}"
    # Emitted only when set: an unconditional "none" would re-key every
    # unregistered-model ID (those hash all solver params).
    if rc.w_mech is not None:
        solver_params["w_mech"] = f"{rc.w_mech:.10g}"
    if rc.tau_ref is not None:
        solver_params["tau_ref"] = f"{rc.tau_ref:.10g}"
    if rc.T_coolant is not None:
        solver_params["T_coolant"] = f"{rc.T_coolant:.10g}"

    if needs is not None:
        solver_params = {k: v for k, v in solver_params.items() if k in needs}

    for k in sorted(solver_params):
        parts.append(f"{k}={solver_params[k]}")

    return "|".join(parts)


def compute_run_id(
    rc: RunConfig, needs: frozenset[str] | None = None,
) -> str:
    """Model-aware run ID. Hashes motor.config_id + model + relevant solver params.

    md5 of the key, first 12 hex characters: the same primitive as
    ``sweepstore.RunID.from_string``, inlined so the core needs no store.
    tests/run_id_vectors.json pins both sides.
    """
    return hashlib.md5(_run_id_key(rc, needs).encode()).hexdigest()[:12]
