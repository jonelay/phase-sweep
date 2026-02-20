"""Crash-safe incremental result storage using JSONL.

Design:
- JSONL (not JSON array): append-only, no parse-whole-file on each save
- Separate index file: O(1) lookup for completed IDs vs O(n) scan
- Single writer: main process saves; parallel workers return via futures
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, NamedTuple

from phasesweep.sweep_types import Status, SweepResult


class SlimResult(NamedTuple):
    config_id: str
    status: str
    metrics: dict[str, Any] | None


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

    def save(self, result: SweepResult) -> None:
        """Append single result immediately (crash-safe)."""
        with open(self.results_file, "a") as f:
            f.write(json.dumps(result.to_dict(), default=str) + "\n")
        self._update_index(result.config.config_id, result.status)

    def _update_index(self, config_id: str, status: Status) -> None:
        index: dict[str, str] = {}
        if self.index_file.exists():
            try:
                index = json.loads(self.index_file.read_text())
            except json.JSONDecodeError:
                pass
        index[config_id] = status
        self.index_file.write_text(json.dumps(index))

    def get_known_ids(self) -> set[str]:
        """IDs of already-completed configs for resume support."""
        if not self.index_file.exists():
            return set()
        try:
            return set(json.loads(self.index_file.read_text()).keys())
        except json.JSONDecodeError:
            return set()

    def load_all(self) -> list[dict[str, Any]]:
        """Load all results as raw dicts from JSONL."""
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

    def load_results(self) -> list[SweepResult]:
        """Load all results as SweepResult objects."""
        results: list[SweepResult] = []
        for d in self.load_all():
            try:
                results.append(SweepResult.from_dict(d))
            except (KeyError, ValueError, TypeError):
                logging.warning("Skipping undeserializable result in %s", self.results_file)
        return results

    def load_slim(self) -> dict[str, SlimResult]:
        """Load results as lightweight SlimResult objects keyed by config_id.

        Skips full MotorSweepConfig deserialization. Last-write-wins for
        duplicate config_ids (correct after mark_pending + re-run).
        """
        out: dict[str, SlimResult] = {}
        for d in self.load_all():
            cfg = d.get("config", {})
            cid = cfg.get("config_id")
            if cid is None:
                continue
            out[cid] = SlimResult(
                config_id=cid,
                status=d.get("status", "ERROR"),
                metrics=d.get("metrics"),
            )
        return out

    def get_stats(self) -> dict[str, int]:
        """Summary counts by status."""
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
        """Remove config IDs from index to force re-run.

        Old results remain in JSONL; new results append.
        """
        if not self.index_file.exists():
            return
        try:
            index = json.loads(self.index_file.read_text())
        except json.JSONDecodeError:
            return
        for config_id in config_ids:
            index.pop(config_id, None)
        self.index_file.write_text(json.dumps(index))
