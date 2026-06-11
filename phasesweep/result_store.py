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

from phasesweep.sweep_types import RunResult, Source, Status, _resolve_model, compute_run_id


class SlimResult(NamedTuple):
    config_id: str
    status: Status
    metrics: dict[str, Any] | None
    motor_config_id: str = ""
    model: str = ""
    source: Source = "computed"


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

    def save(self, result: RunResult) -> None:
        """Append single result immediately (crash-safe)."""
        with open(self.results_file, "a") as f:
            f.write(json.dumps(result.to_dict(), default=str) + "\n")
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

    def _update_index(self, config_id: str, status: Status) -> None:
        index: dict[str, str] = {}
        if self.index_file.exists():
            with contextlib.suppress(json.JSONDecodeError):
                index = json.loads(self.index_file.read_text())
        index[config_id] = status
        self._write_index(index)

    def get_known_ids(self) -> set[str]:
        if not self.index_file.exists():
            return set()
        try:
            return set(json.loads(self.index_file.read_text()).keys())
        except json.JSONDecodeError:
            return set()

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
        motor_config_id: str | None = None,
        model: str | None = None,
        source: str | None = None,
    ) -> list[RunResult]:
        """Load results as RunResult objects, with optional filtering."""
        results: list[RunResult] = []
        for d in self.load_all():
            try:
                rr = RunResult.from_dict(d)
            except (KeyError, ValueError, TypeError):
                logging.warning("Skipping undeserializable result in %s", self.results_file)
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
        motor_config_id: str | None = None,
        model: str | None = None,
        source: str | None = None,
    ) -> dict[str, SlimResult]:
        """Load results as lightweight SlimResult objects keyed by config_id.

        Deserializes RunConfig to compute run_id. Last-write-wins for
        duplicate config_ids (correct after mark_pending + re-run).
        Optional filters narrow results by motor_config_id, model, or source.
        """
        from phasesweep.sweep_types import RunConfig
        out: dict[str, SlimResult] = {}
        for d in self.load_all():
            try:
                rc = RunConfig.from_dict(d["config"])
                cid = compute_run_id(rc)
            except (KeyError, ValueError, TypeError):
                continue
            try:
                d_model = _resolve_model(d)
            except KeyError:
                d_model = rc.model
            d_source = d.get("source", "computed")
            d_motor_cid = d.get("motor_config_id", rc.motor.config_id)
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
        if not self.index_file.exists():
            return
        try:
            index = json.loads(self.index_file.read_text())
        except json.JSONDecodeError:
            return
        for config_id in config_ids:
            index.pop(config_id, None)
        self._write_index(index)
