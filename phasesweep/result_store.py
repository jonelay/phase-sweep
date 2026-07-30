"""Crash-safe incremental result storage using JSONL.

Design:
- JSONL (not JSON array): append-only, no parse-whole-file on each save
- Separate index file: O(1) lookup for completed IDs vs O(n) scan
- Single writer: main process saves; parallel workers return via futures
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, NamedTuple

from phasesweep.motor import Motor
from phasesweep.sweep_types import RunResult, Source, Status, _resolve_model, compute_run_id


def version_current(model: str, stamped: Any) -> bool:
    """Staleness check: a stored result serves only when
    its stamped model version matches the live registry. Missing stamp =
    unknown vintage = stale, exactly like a mismatch — pre-stamp stores
    re-run rather than serve superseded physics."""
    from phasesweep.registry import MODEL_REGISTRY
    info = MODEL_REGISTRY.get(model)
    return info is not None and stamped == info.version


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


def _attached(source: str, motor_config_id: str, motor_name: str,
              target_config_id: str, target_name: str) -> bool:
    """Spec 05 §10.1 attach rule: computed results belong to an exact motor
    config; measured/published describe the hardware, so they follow the
    Motor name through parameter edits."""
    return (motor_config_id == target_config_id
            or (source != "computed" and motor_name == target_name))


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
    """Single-writer incremental result storage using JSONL."""

    output_dir: Path
    results_file: Path
    index_file: Path

    def __init__(self, output_dir: Path | str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_file = self.output_dir / "results.jsonl"
        self.index_file = self.output_dir / "index.json"
        self._index_cache: dict[str, str] | None = None

    def save(self, result: RunResult) -> None:
        """Append single result immediately (crash-safe).

        Stamps the registry model version (v1 included) so the read side
        can tell superseded-physics records from current ones. The stamp
        goes into the serialized record only — the caller's RunResult is
        never mutated. Caveat: the stamp applies to any
        UNstamped result, so a migration that load→saves pre-stamp legacy
        records marks them current — migrations must set model_version
        explicitly instead of relying on this default.
        """
        d = result.to_dict()
        if d.get("model_version") is None:
            from phasesweep.registry import MODEL_REGISTRY
            info = MODEL_REGISTRY.get(result.model)
            if info is not None:
                d["model_version"] = info.version
        with open(self.results_file, "a") as f:
            f.write(json.dumps(d, default=_json_default) + "\n")
        config_id = compute_run_id(result.config)
        self._update_index(config_id, result.status)

    def _write_index(self, index: dict[str, str]) -> None:
        fd, tmp = tempfile.mkstemp(dir=self.output_dir, suffix=".tmp")
        try:
            os.write(fd, json.dumps(index).encode())
            os.close(fd)
            fd = -1
            os.replace(tmp, self.index_file)
        except BaseException:
            if fd >= 0:
                os.close(fd)
            os.unlink(tmp)
            raise

    def _load_index(self, *, refresh: bool = False) -> dict[str, str]:
        if self._index_cache is None or refresh:
            self._index_cache = {}
            if self.index_file.exists():
                with contextlib.suppress(json.JSONDecodeError):
                    self._index_cache = json.loads(self.index_file.read_text())
        return self._index_cache

    def _update_index(self, config_id: str, status: Status) -> None:
        # refresh before mutate-and-write-back: the whole dict is written
        # out, so a stale cache would clobber entries another store
        # instance added since our last read
        index = self._load_index(refresh=True)
        index[config_id] = status
        self._write_index(index)

    def get_known_ids(self) -> set[str]:
        return set(self._load_index(refresh=True).keys())

    def load_all(self) -> list[dict[str, Any]]:
        if not self.results_file.exists():
            return []
        results = []
        for line in self.results_file.read_text().splitlines():
            if line.strip():
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    logging.warning("Skipping malformed JSON line in %s", self.results_file)
        return results

    def load_results(
        self,
        *,
        motor: Motor | None = None,
        motor_config_id: str | None = None,
        model: str | None = None,
        source: str | None = None,
    ) -> list[RunResult]:
        """Load results as RunResult objects, with optional filtering.

        `motor` applies the attach rule (see `_attached`);
        `motor_config_id` is a plain exact-config filter.
        """
        target = (motor.config_id, motor.name) if motor is not None else None
        results: list[RunResult] = []
        for d in self.load_all():
            try:
                rr = RunResult.from_dict(d)
            except (KeyError, ValueError, TypeError):
                logging.warning("Skipping undeserializable result in %s", self.results_file)
                continue
            if target is not None and not _attached(
                    rr.source, rr.motor_config_id, rr.config.motor.name, *target):
                continue
            if motor_config_id is not None and rr.motor_config_id != motor_config_id:
                continue
            if model is not None and rr.model != model:
                continue
            if source is not None and rr.source != source:
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

        Deserializes RunConfig to compute run_id. Last-write-wins for
        duplicate config_ids (correct after mark_pending + re-run).
        `motor` applies the attach rule (see `_attached`);
        the other filters are plain field matches.
        """
        from phasesweep.sweep_types import RunConfig
        target = (motor.config_id, motor.name) if motor is not None else None
        out: dict[str, SlimResult] = {}
        for d in self.load_all():
            try:
                rc = RunConfig.from_dict(d["config"])
                cid = compute_run_id(rc)
            except (KeyError, ValueError, TypeError):
                logging.warning("Skipping undeserializable result in %s", self.results_file)
                continue
            try:
                d_model = _resolve_model(d)
            except KeyError:
                d_model = rc.model
            # H1: run IDs are recomputed at the *current* registry version,
            # so a superseded-physics record would re-key to a fresh id and
            # serve as a cache hit — drop computed records not stamped
            # current. Measured/published describe hardware, not physics
            # code, and can't be regenerated by a re-run (fn=None) — they
            # always serve (same rule as _dedupe_for_matrix).
            d_source = d.get("source", "computed")
            if d_source == "computed" and not version_current(
                    d_model, d.get("model_version")):
                continue
            d_motor_cid = d.get("motor_config_id", rc.motor.config_id)
            if target is not None and not _attached(
                    d_source, d_motor_cid, rc.motor.name, *target):
                continue
            if motor_config_id is not None and d_motor_cid != motor_config_id:
                continue
            if model is not None and d_model != model:
                continue
            if source is not None and d_source != source:
                continue
            out[cid] = SlimResult(
                config_id=cid,
                status=d.get("status", "ERROR"),
                metrics=d.get("metrics"),
                motor_config_id=d_motor_cid,
                model=d_model,
                source=d_source,
                motor_name=rc.motor.name,
                timestamp=d.get("timestamp", ""),
            )
        return out

    def get_stats(self) -> dict[str, int]:
        results = self.load_all()
        if not results:
            return {"total": 0, "ok": 0, "timeout": 0, "error": 0}
        return {
            "total": len(results),
            "ok": sum(1 for r in results if r["status"] == "OK"),
            "timeout": sum(1 for r in results if r["status"] == "TIMEOUT"),
            "error": sum(1 for r in results if r["status"] == "ERROR"),
        }

    def mark_pending(self, config_ids: set[str]) -> None:
        """Remove config IDs from index to force re-run."""
        index = self._load_index(refresh=True)
        if not index:
            return
        for config_id in config_ids:
            index.pop(config_id, None)
        self._write_index(index)
