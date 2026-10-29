"""Crash-safe incremental state-point storage using JSONL.

Single-writer, append-only JSONL with an atomic index file for O(1) lookup.
Schema-agnostic — stores any JSON-serializable dict inside a versioned envelope.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any


class Store:
    """Single-writer incremental JSONL store with atomic index."""

    def __init__(self, path: Path | str) -> None:
        """Constructing a Store does not touch the filesystem; the directory
        is created on first write."""
        self._dir = Path(path)
        self._results_file = self._dir / "results.jsonl"
        self._index_file = self._dir / "index.json"

    @property
    def path(self) -> Path:
        return self._dir

    def save(
        self,
        run_id: str,
        record: dict[str, Any],
        *,
        model_version: int | None = None,
        json_default: Any = None,
    ) -> None:
        """Append *record* inside a versioned envelope and update the index."""
        row = dict(record)
        row["run_id"] = run_id
        if model_version is not None:
            row["model_version"] = model_version
        row.setdefault("model_version", None)
        row.setdefault("timestamp", datetime.now().isoformat())
        self._dir.mkdir(parents=True, exist_ok=True)
        with open(self._results_file, "a") as f:
            f.write(json.dumps(row, default=json_default) + "\n")
        self._update_index(run_id, record.get("status", "OK"))

    def load_all(self) -> list[dict[str, Any]]:
        """All JSONL rows, including superseded ones from prior invalidation cycles."""
        if not self._results_file.exists():
            return []
        results: list[dict[str, Any]] = []
        for line in self._results_file.read_text().splitlines():
            if line.strip():
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    logging.warning("Skipping malformed JSON line in %s", self._results_file)
        return results

    def load_current(self) -> dict[str, dict[str, Any]]:
        """Index-filtered rows keyed by run_id."""
        index = self._load_index()
        out: dict[str, dict[str, Any]] = {}
        for row in self.load_all():
            rid = row.get("run_id")
            if rid is not None and rid in index:
                out[rid] = row
        return out

    def get_known_ids(self) -> set[str]:
        """Run IDs present in the index."""
        return set(self._load_index().keys())

    def invalidate(self, run_ids: Iterable[str]) -> None:
        """Remove *run_ids* from the index; JSONL rows are preserved."""
        index = self._load_index()
        for rid in run_ids:
            index.pop(rid, None)
        self._write_index(index)

    # -- index internals --

    def _load_index(self) -> dict[str, str]:
        index: dict[str, str] = {}
        if self._index_file.exists():
            with contextlib.suppress(json.JSONDecodeError):
                index = json.loads(self._index_file.read_text())
        return index

    def _update_index(self, run_id: str, status: str) -> None:
        index = self._load_index()
        index[run_id] = status
        self._write_index(index)

    def _write_index(self, index: dict[str, str]) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self._dir, suffix=".tmp")
        try:
            os.write(fd, json.dumps(index).encode())
            os.close(fd)
            fd = -1
            os.replace(tmp, self._index_file)
        except BaseException:
            if fd >= 0:
                os.close(fd)
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
