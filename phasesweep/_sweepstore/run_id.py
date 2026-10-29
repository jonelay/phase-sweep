"""Deterministic run-ID generation from consumer-built hash strings."""

from __future__ import annotations

import hashlib


class RunID:
    @staticmethod
    def from_string(s: str) -> str:
        """Hash *s* with MD5 and return the first 12 hex characters."""
        return hashlib.md5(s.encode()).hexdigest()[:12]
