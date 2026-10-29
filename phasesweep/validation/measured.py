"""Measured data schema, validation, and import logic.

Defines MeasurementConditions and MeasuredResult frozen dataclasses for
importing real test data alongside computed results.
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, get_args

from phasesweep.machines.motor import InductionParams, Motor
from phasesweep.sweep_types import RunConfig, RunResult, Source

__all__ = [
    "BoundRef",
    "CurveRef",
    "KeyMapping",
    "MeasuredFormatError",
    "MeasuredResult",
    "MeasurementConditions",
    "import_measured",
    "measured_run_result",
    "validate_measured",
]

# Measured data files carry real-world data; "computed" is excluded.
_VALID_SOURCES = tuple(s for s in get_args(Source) if s != "computed")

Extract = Literal["interp", "max", "min", "rms"]
Relation = Literal["gte", "lte", "gt", "lt"]


class MeasuredFormatError(ValueError):
    """A measured-data payload is malformed (missing key, wrong type or
    value). Raised before anything is persisted."""


def _finite_real(where: str, v: Any) -> None:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise MeasuredFormatError(f"{where} must be a finite number, got {v!r}")


@dataclass(frozen=True)
class MeasurementConditions:
    speed_rpm: float
    temperature_C: float
    load_torque_Nm: float
    date: str
    instrument: str
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "speed_rpm": self.speed_rpm,
            "temperature_C": self.temperature_C,
            "load_torque_Nm": self.load_torque_Nm,
            "date": self.date,
            "instrument": self.instrument,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MeasurementConditions:
        return cls(
            speed_rpm=d["speed_rpm"],
            temperature_C=d["temperature_C"],
            load_torque_Nm=d["load_torque_Nm"],
            date=d["date"],
            instrument=d["instrument"],
            notes=d.get("notes", ""),
        )


@dataclass(frozen=True)
class CurveRef:
    curve_x: str
    curve_y: str
    at_x: float | None = None
    extract: Extract = "interp"

    def __post_init__(self) -> None:
        if self.extract not in get_args(Extract):
            raise MeasuredFormatError(
                f"curve_compare extract must be one of {get_args(Extract)}, "
                f"got {self.extract!r}")
        if self.extract == "interp" and self.at_x is None:
            raise MeasuredFormatError("curve_compare extract 'interp' needs at_x")
        if self.at_x is not None:
            _finite_real("curve_compare at_x", self.at_x)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"curve_x": self.curve_x, "curve_y": self.curve_y, "extract": self.extract}
        if self.at_x is not None:
            d["at_x"] = self.at_x
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CurveRef:
        return cls(curve_x=d["curve_x"], curve_y=d["curve_y"],
                   at_x=d.get("at_x"), extract=d.get("extract", "interp"))


@dataclass(frozen=True)
class BoundRef:
    computed_key: str
    relation: Relation

    def __post_init__(self) -> None:
        if self.relation not in get_args(Relation):
            raise MeasuredFormatError(
                f"bound_compare relation must be one of {get_args(Relation)}, "
                f"got {self.relation!r}")

    def to_dict(self) -> dict[str, Any]:
        return {"computed_key": self.computed_key, "relation": self.relation}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BoundRef:
        return cls(computed_key=d["computed_key"], relation=d["relation"])


@dataclass(frozen=True)
class KeyMapping:
    computed_key: str
    semantic_note: str

    def to_dict(self) -> dict[str, Any]:
        return {"computed_key": self.computed_key, "semantic_note": self.semantic_note}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> KeyMapping:
        return cls(computed_key=d["computed_key"], semantic_note=d["semantic_note"])


@dataclass(frozen=True)
class MeasuredResult:
    motor_name: str
    test_type: str
    conditions: MeasurementConditions
    quantities: dict[str, float]
    waveforms: dict[str, list[float]]
    uncertainty: dict[str, float]
    source_file: str
    tolerances: dict[str, float] = field(default_factory=dict)
    source: Source = "measured"
    curve_compare: dict[str, CurveRef] = field(default_factory=dict)
    bound_compare: dict[str, BoundRef] = field(default_factory=dict)
    key_mapping: dict[str, KeyMapping] = field(default_factory=dict)
    # Motor params whose TOML values were derived from THIS dataset
    # (e.g. psi_f from a datasheet Ke). The calibration circularity guard
    # refuses to fit these params against this dataset.
    derived_params: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "motor_name": self.motor_name,
            "test_type": self.test_type,
            "conditions": self.conditions.to_dict(),
            "quantities": self.quantities,
            "waveforms": self.waveforms,
            "uncertainty": self.uncertainty,
            "source_file": self.source_file,
            "source": self.source,
        }
        if self.tolerances:
            d["tolerances"] = self.tolerances
        if self.curve_compare:
            d["curve_compare"] = {k: v.to_dict() for k, v in self.curve_compare.items()}
        if self.bound_compare:
            d["bound_compare"] = {k: v.to_dict() for k, v in self.bound_compare.items()}
        if self.key_mapping:
            d["key_mapping"] = {k: v.to_dict() for k, v in self.key_mapping.items()}
        if self.derived_params:
            d["derived_params"] = list(self.derived_params)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MeasuredResult:
        try:
            return cls._from_dict(d)
        except MeasuredFormatError:
            raise
        except (KeyError, TypeError, AttributeError, ValueError) as e:
            detail = f"missing key {e}" if isinstance(e, KeyError) else str(e)
            raise MeasuredFormatError(detail) from e

    @classmethod
    def _from_dict(cls, d: dict[str, Any]) -> MeasuredResult:
        return cls(
            motor_name=d["motor_name"],
            test_type=d["test_type"],
            conditions=MeasurementConditions.from_dict(d["conditions"]),
            quantities=d["quantities"],
            waveforms=d.get("waveforms", {}),
            uncertainty=d.get("uncertainty", {}),
            source_file=d.get("source_file", ""),
            tolerances=d.get("tolerances", {}),
            source=d.get("source", "measured"),
            curve_compare={k: CurveRef.from_dict(v) for k, v in d.get("curve_compare", {}).items()},
            bound_compare={k: BoundRef.from_dict(v) for k, v in d.get("bound_compare", {}).items()},
            key_mapping={k: KeyMapping.from_dict(v) for k, v in d.get("key_mapping", {}).items()},
            derived_params=tuple(d.get("derived_params", ())),
        )


def validate_measured(data: MeasuredResult) -> None:
    from phasesweep.registry import MODEL_REGISTRY

    if not data.motor_name:
        raise ValueError("motor_name must be non-empty")

    # Values reach crossval and the API as numbers; a string would drop
    # out of the comparison silently or 500 the validation endpoint.
    for section in ("quantities", "uncertainty", "tolerances"):
        table = getattr(data, section)
        if not isinstance(table, dict):
            raise MeasuredFormatError(f"{section} must be a table, got {table!r}")
        for k, v in table.items():
            _finite_real(f"{section}.{k}", v)
    for k in ("speed_rpm", "temperature_C", "load_torque_Nm"):
        _finite_real(f"conditions.{k}", getattr(data.conditions, k))

    if data.source not in _VALID_SOURCES:
        raise ValueError(
            f"source must be one of {_VALID_SOURCES}, got {data.source!r} "
            f"— descriptive detail belongs in conditions.notes or source_file"
        )

    info = MODEL_REGISTRY.get(data.test_type)
    if info is None or info.source != "measured":
        raise ValueError(
            f"unknown measured test_type: {data.test_type!r}"
        )

    resolved = (
        info.produces
        | set(data.curve_compare.keys())
        | set(data.bound_compare.keys())
        | set(data.key_mapping.keys())
    )
    unresolved = set(data.quantities.keys()) - resolved
    if unresolved:
        raise ValueError(
            f"quantities keys {unresolved} have no resolution path "
            f"(not in produces, curve_compare, bound_compare, or key_mapping) "
            f"for {data.test_type}"
        )

    motor_fields = {f.name for f in dataclasses.fields(Motor)}
    motor_fields |= {f.name for f in dataclasses.fields(InductionParams)}
    unknown = set(data.derived_params) - motor_fields
    if unknown:
        raise ValueError(
            f"derived_params {sorted(unknown)} are not Motor or "
            f"InductionParams fields"
        )


def measured_run_result(
    data: MeasuredResult, motor: Motor, dataset_id: str,
) -> RunResult:
    """Build a RunResult from validated measured data (no store side effects)."""
    config = RunConfig(
        motor=motor, model=data.test_type, dataset_id=dataset_id,
    )
    metrics: dict[str, Any] = dict(data.quantities)
    metrics["_conditions"] = data.conditions.to_dict()
    if data.waveforms:
        metrics["_waveforms"] = data.waveforms
    if data.uncertainty:
        metrics["_uncertainty"] = data.uncertainty
    if data.curve_compare:
        metrics["_curve_compare"] = {k: v.to_dict() for k, v in data.curve_compare.items()}
    if data.bound_compare:
        metrics["_bound_compare"] = {k: v.to_dict() for k, v in data.bound_compare.items()}
    if data.key_mapping:
        metrics["_key_mapping"] = {k: v.to_dict() for k, v in data.key_mapping.items()}
    if data.derived_params:
        metrics["_derived_params"] = list(data.derived_params)

    return RunResult(
        config=config,
        model=data.test_type,
        status="OK",
        metrics=metrics,
        elapsed_s=0.0,
        source=data.source,
        tolerances=data.tolerances or None,
    )


def import_measured(
    path: Path, motor: Motor, output_dir: Path,
) -> RunResult:
    raw = json.loads(path.read_text())
    data = MeasuredResult.from_dict(raw)
    validate_measured(data)
    result = measured_run_result(data, motor, path.stem)

    from phasesweep.result_store import ResultStore

    store = ResultStore(output_dir)
    store.save(result)
    return result
