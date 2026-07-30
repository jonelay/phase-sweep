"""Type-safe, validated configuration and result objects for motor runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from phasesweep.motor import Motor
    from phasesweep.sim import SimPlan

Status = Literal["OK", "TIMEOUT", "ERROR"]


@dataclass(frozen=True)
class RunConfig:
    """Run configuration composing a Motor with solver/sim parameters.

    model — registry key ("analytical", "fem", "drive_sim", ...);
    maxh_fraction — max FEM mesh element size as fraction of r_outer;
    n_theta — airgap angular samples per revolution;
    nonlinear — enable B-H curve Picard iteration (FEM);
    sim_plan — drive-sim timing/load plan (SimPlan);
    j_s — armature slot current density amplitude (A/m², 0 = open circuit);
    i_fault — peak phase current (A) the demag_screen model screens at;
    required by that model (no drive-limit fallback — the screen current
    is a deliberate choice: drive limit, short-circuit peak, ...);
    duty_profile — thermal-duty torque-time segments ((torque_Nm,
    duration_s), ...) for the thermal_duty model;
    dataset_id — identity of an imported measured dataset (None for
    computed models); distinguishes repeat captures of the same test.
    """

    motor: Motor
    model: str

    maxh_fraction: float = 0.05
    n_theta: int = 360
    nonlinear: bool = False

    sim_plan: SimPlan | None = None

    j_s: float = 0.0

    i_fault: float | None = None

    duty_profile: tuple[tuple[float, float], ...] | None = None

    dataset_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
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
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunConfig:
        from phasesweep.motor import Motor as MotorCls
        from phasesweep.sim import SimPlan

        motor = MotorCls.from_dict(d["motor"])
        sim_plan = None
        if "sim_plan" in d and d["sim_plan"] is not None:
            sim_plan = SimPlan.from_dict(d["sim_plan"])
        duty_profile = None
        if d.get("duty_profile") is not None:
            duty_profile = tuple((float(t), float(dt)) for t, dt in d["duty_profile"])
        return cls(
            motor=motor,
            model=d["model"],
            maxh_fraction=d.get("maxh_fraction", 0.05),
            n_theta=d.get("n_theta", 360),
            nonlinear=d.get("nonlinear", False),
            sim_plan=sim_plan,
            j_s=d.get("j_s", 0.0),
            i_fault=d.get("i_fault"),
            duty_profile=duty_profile,
            dataset_id=d.get("dataset_id"),
        )


Source = Literal["computed", "measured", "published"]

# Legacy compat: old persisted JSONL may have "sim" instead of "drive_sim".
_LEGACY_MODEL_MAP: dict[str, str] = {
    "sim": "drive_sim",
}


def _resolve_model(d: dict[str, Any]) -> str:
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

    def to_dict(self) -> dict[str, Any]:
        d = {
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
    def from_dict(cls, d: dict[str, Any]) -> RunResult:
        return cls(
            config=RunConfig.from_dict(d["config"]),
            model=_resolve_model(d),
            status=d["status"],
            metrics=d.get("metrics"),
            elapsed_s=d["elapsed_s"],
            error_msg=d.get("error_msg"),
            source=d.get("source", "computed"),
            tolerances=d.get("tolerances"),
            timestamp=d.get("timestamp", ""),
            model_version=d.get("model_version"),
            schema_version=d.get("schema_version", "v2.0"),
        )


def compute_run_id(
    rc: RunConfig, needs: frozenset[str] | None = None,
) -> str:
    """Model-aware run ID. Hashes motor.config_id + model + relevant solver params.

    When needs is None, looks up hash_fields from MODEL_REGISTRY automatically.
    Falls back to hashing all solver params for unknown models.
    Includes the registry model-code version (when > 1) so results from a
    superseded physics convention cannot serve from the store.
    """
    from phasesweep.registry import MODEL_REGISTRY
    info = MODEL_REGISTRY.get(rc.model)
    if needs is None and info is not None:
        needs = info.hash_fields

    parts = [rc.motor.config_id, rc.model]
    if info is not None and info.version > 1:
        parts.append(f"model_v={info.version}")
    # Dataset identity is unconditional (not a solver param): repeat
    # captures of the same motor/test must not collide to one run ID
    if rc.dataset_id is not None:
        parts.append(f"dataset={rc.dataset_id}")

    drive = rc.motor.drive
    solver_params: dict[str, str] = {
        "maxh_fraction": f"{rc.maxh_fraction:.4f}",
        "n_theta": str(rc.n_theta),
        "nonlinear": str(rc.nonlinear),
        "j_s": f"{rc.j_s:.6f}",
        "i_fault": "none" if rc.i_fault is None else f"{rc.i_fault:.6g}",
        # Drive params live on the motor but are excluded from
        # Motor.config_id (keeps archived motor_config_id groupings valid)
        "U_DC": "none" if drive.U_DC is None else f"{drive.U_DC:.6g}",
        "MAX_I_S": "none" if drive.MAX_I_S is None else f"{drive.MAX_I_S:.6g}",
        "W_REF": "none" if drive.W_REF is None else f"{drive.W_REF:.6g}",
        "I_LIMIT": "none" if drive.I_LIMIT is None else f"{drive.I_LIMIT:.6g}",
    }
    if rc.sim_plan is not None:
        # Hash the full plan — controller tuning and extraction windows
        # change metrics, so they must change the run ID too
        solver_params["sim_plan"] = json.dumps(
            rc.sim_plan.to_dict(), sort_keys=True
        )
    if rc.duty_profile is not None:
        # The torque-time profile drives every thermal-duty metric
        solver_params["duty_profile"] = json.dumps(
            [list(seg) for seg in rc.duty_profile], sort_keys=True
        )

    if needs is not None:
        solver_params = {k: v for k, v in solver_params.items() if k in needs}

    for k in sorted(solver_params):
        parts.append(f"{k}={solver_params[k]}")

    key = "|".join(parts)
    return hashlib.md5(key.encode()).hexdigest()[:12]
