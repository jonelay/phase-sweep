"""Vendored copy of sweepstore 0.1.1 (Apache-2.0, same author).

sweepstore is not yet on a public package index, so the store ships
inside phasesweep until it is. Keep the modules identical to the
upstream tag apart from the package name in imports.
"""

from phasesweep._sweepstore.run_id import RunID
from phasesweep._sweepstore.staleness import attached, record_is_current, version_current
from phasesweep._sweepstore.store import Store
from phasesweep._sweepstore.vectors import load_vectors

__version__ = "0.1.1"

__all__ = [
    "RunID",
    "Store",
    "attached",
    "load_vectors",
    "record_is_current",
    "version_current",
]
