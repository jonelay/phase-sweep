"""Install hints for optional extras.

The PyPI name ``phasesweep`` belongs to an unrelated project, so a hint of
the form ``phasesweep[fem]`` against the package index would install the wrong package.
Hints name the extra against a source checkout instead.
"""

from __future__ import annotations

from importlib.util import find_spec

SERVER_MODULES = ("fastapi", "uvicorn", "structlog", "tomli_w", "mcp")


class MissingExtra(ImportError):
    """An optional extra this code path needs is not installed.

    Distinct from ImportError so callers (the server) can report a missing
    extra without masking an import bug inside phasesweep itself.
    """

    def __init__(self, what: str, extra: str) -> None:
        super().__init__(f"{what}; {install_hint(extra)}")
        self.extra = extra


def install_hint(extra: str) -> str:
    """Instruction for installing *extra* from a phase-sweep checkout."""
    return f'install the {extra} extra from a phase-sweep checkout: pip install -e ".[{extra}]"'


def missing_modules(modules: tuple[str, ...]) -> list[str]:
    return [m for m in modules if find_spec(m) is None]


def require_extra(extra: str, modules: tuple[str, ...], what: str) -> None:
    """Raise MissingExtra naming *extra* if any of *modules* is missing."""
    missing = missing_modules(modules)
    if missing:
        raise MissingExtra(f"{what} needs {', '.join(missing)}", extra)
