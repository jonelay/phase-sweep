"""Pure staleness and attachment predicates.

These functions encode the contract from result-store-contract.md.
The consumer supplies the live version — sweepstore has no registry.
"""

from __future__ import annotations

from typing import Any


def version_current(stamped: Any, live: Any) -> bool:
    """True only when *stamped* equals *live* and both are non-None."""
    return live is not None and stamped == live


def record_is_current(source: str | None, stamped_version: Any, live_version: Any) -> bool:
    """Staleness rule: computed records must carry the current version;
    measured/published always serve."""
    if source is None:
        source = "computed"
    return source != "computed" or version_current(stamped_version, live_version)


def attached(
    source: str | None,
    record_config_id: str,
    record_name: str,
    target_config_id: str,
    target_name: str,
) -> bool:
    """Attachment predicate: computed by config_id, measured/published by name."""
    if source is None:
        source = "computed"
    return record_config_id == target_config_id or (source != "computed" and record_name == target_name)
