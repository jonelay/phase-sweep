"""Domain result store wrapping the vendored sweepstore.Store.

Layers phase-sweep's Motor/RunConfig types, MODEL_REGISTRY lookups,
and the load_results/load_slim filtering on top of sweepstore's
schema-agnostic JSONL + index mechanics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, NamedTuple, cast

from phasesweep._sweepstore import Store as _Store
from phasesweep._sweepstore import attached
from phasesweep._sweepstore import record_is_current as _ss_record_is_current
from phasesweep._sweepstore import version_current as _ss_version_current
from phasesweep.machines.motor import Motor
from phasesweep.sweep_types import (
    RunResult,
    Source,
    Status,
    check_status,
    compute_run_id,
    normalize_source,
    resolve_model,
)

__all__ = [
    "ResultStore",
    "SlimResult",
    "record_is_current",
    "version_current",
]


def version_current(model: str, stamped: Any) -> bool:
    """Staleness check: a stored result serves only when
    its stamped model version matches the live registry."""
    from phasesweep.registry import MODEL_REGISTRY
    info = MODEL_REGISTRY.get(model)
    live = info.version if info is not None else None
    return _ss_version_current(stamped, live)


def record_is_current(source: str, model: str, model_version: int | None) -> bool:
    """Staleness rule: computed records must carry the current registry
    version; measured/published always serve."""
    from phasesweep.registry import MODEL_REGISTRY
    info = MODEL_REGISTRY.get(model)
    live = info.version if info is not None else None
    return _ss_record_is_current(source, model_version, live)


def _row_run_id(d: dict[str, Any]) -> str | None:
    """Run id of a stored row: the stamped `run_id`, or recomputed from
    the config for legacy rows without one. None if the config cannot
    be deserialized (including sim rows without phasesweep[sim])."""
    from phasesweep.sweep_types import RunConfig
    rid = d.get("run_id")
    if rid is not None:
        return str(rid)
    try:
        return compute_run_id(RunConfig.from_dict(d["config"]))
    except (KeyError, ValueError, TypeError, ImportError):
        return None


def _json_default(o: Any) -> Any:
    """Serialize what json can't, preferring the value over its repr.

    numpy scalars have to unwrap to their Python equivalent: str() on a
    np.bool_ stores "False", and bool("False") reads back True. Duck-typed
    so this module stays numpy-free.
    """
    if getattr(o, "ndim", None) == 0 and hasattr(o, "item"):
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


class SlimResult(NamedTuple):
    config_id: str
    status: Status
    metrics: dict[str, Any] | None
    motor_config_id: str = ""
    model: str = ""
    source: Source = "computed"
    motor_name: str = ""
    timestamp: str = ""


class ResultStore:
    """Single-writer incremental result storage wrapping sweepstore.Store."""

    output_dir: Path

    def __init__(self, output_dir: Path | str) -> None:
        self.output_dir = Path(output_dir)
        self._store = _Store(self.output_dir)
        self.results_file = self.output_dir / "results.jsonl"
        self.index_file = self.output_dir / "index.json"

    def save(self, result: RunResult) -> None:
        """Append single result immediately (crash-safe).

        Stamps the registry model version so the read side can tell
        superseded-physics records from current ones.
        """
        d = cast(dict[str, Any], result.to_dict())
        mv = d.pop("model_version", None)
        if mv is None:
            from phasesweep.registry import MODEL_REGISTRY
            info = MODEL_REGISTRY.get(result.model)
            if info is not None:
                mv = info.version
        run_id = compute_run_id(result.config)
        self._store.save(run_id, d, model_version=mv, json_default=_json_default)

    def get_known_ids(self) -> set[str]:
        return self._store.get_known_ids()

    def load_all(self) -> list[dict[str, Any]]:
        return self._store.load_all()

    def load_results(
        self,
        *,
        motor: Motor | None = None,
        motor_config_id: str | None = None,
        model: str | None = None,
        source: str | None = None,
        current_only: bool = False,
    ) -> list[RunResult]:
        """Load results as RunResult objects, with optional filtering.

        `motor` applies the attach rule (see `attached`);
        `motor_config_id` is a plain exact-config filter. `current_only`
        applies the `record_is_current` staleness rule that `load_slim`
        always applies.
        """
        target = (motor.config_id, motor.name) if motor is not None else None
        results: list[RunResult] = []
        for d in self.load_all():
            try:
                rr = RunResult.from_dict(d)
            except (KeyError, ValueError, TypeError, ImportError):
                logging.warning("Skipping undeserializable result in %s", self.results_file)
                continue
            if target is not None and not attached(
                    rr.source, rr.motor_config_id, rr.config.motor.name, *target):
                continue
            if motor_config_id is not None and rr.motor_config_id != motor_config_id:
                continue
            if model is not None and rr.model != model:
                continue
            if source is not None and rr.source != source:
                continue
            if current_only and not record_is_current(
                    rr.source, rr.model, rr.model_version):
                continue
            results.append(rr)
        return results

    def load_slim(
        self,
        *,
        motor: Motor | None = None,
        motor_config_id: str | None = None,
        model: str | None = None,
        source: str | None = None,
    ) -> dict[str, SlimResult]:
        """Load results as lightweight SlimResult objects keyed by config_id.

        Always applies record_is_current staleness filtering, before any
        deserialization: stale rows exit on dict reads alone. Current rows
        key by the stored `run_id`; only legacy rows without one (or
        without `motor_config_id`) rebuild the RunConfig to recompute it.
        """
        from phasesweep.sweep_types import RunConfig
        target = (motor.config_id, motor.name) if motor is not None else None
        out: dict[str, SlimResult] = {}
        for d in self.load_all():
            try:
                d_model = resolve_model(d)
            except KeyError:
                logging.warning("Skipping result without model in %s", self.results_file)
                continue
            try:
                d_source = normalize_source(d.get("source", "computed"))
                d_status = check_status(d.get("status", "ERROR"))
            except ValueError as e:
                logging.warning("Skipping result in %s: %s", self.results_file, e)
                continue
            if not record_is_current(d_source, d_model, d.get("model_version")):
                continue
            cid = d.get("run_id")
            d_motor_cid = d.get("motor_config_id")
            try:
                name = d["config"]["motor"]["name"]
            except (KeyError, TypeError):
                logging.warning("Skipping result without motor name in %s", self.results_file)
                continue
            if cid is None or d_motor_cid is None:
                try:
                    rc = RunConfig.from_dict(d["config"])
                except ImportError as e:
                    logging.warning("Skipping result needing an optional extra in %s: %s",
                                    self.results_file, e)
                    continue
                except (KeyError, ValueError, TypeError):
                    logging.warning("Skipping undeserializable result in %s", self.results_file)
                    continue
                cid = compute_run_id(rc)
                d_motor_cid = rc.motor.config_id
            if target is not None and not attached(
                    d_source, d_motor_cid, name, *target):
                continue
            if motor_config_id is not None and d_motor_cid != motor_config_id:
                continue
            if model is not None and d_model != model:
                continue
            if source is not None and d_source != source:
                continue
            out[cid] = SlimResult(
                config_id=cid,
                status=d_status,
                metrics=d.get("metrics"),
                motor_config_id=d_motor_cid,
                model=d_model,
                source=d_source,
                motor_name=name,
                timestamp=d.get("timestamp", ""),
            )
        return out
