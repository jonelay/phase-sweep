"""Install hints for optional extras.

The PyPI name ``phasesweep`` belongs to an unrelated project, so a hint of
the form ``phasesweep[fem]`` against the package index would install the wrong package.
Hints name the extra against a source checkout instead.
"""

from __future__ import annotations

from importlib.util import find_spec

SERVER_MODULES = ("fastapi", "uvicorn", "structlog", "tomli_w")


def install_hint(extra: str) -> str:
    """Instruction for installing *extra* from a phase-sweep checkout."""
    return f'install the {extra} extra from a phase-sweep checkout: pip install -e ".[{extra}]"'


def missing_modules(modules: tuple[str, ...]) -> list[str]:
    return [m for m in modules if find_spec(m) is None]


def require_extra(extra: str, modules: tuple[str, ...], what: str) -> None:
    """Raise ImportError naming *extra* if any of *modules* is missing."""
    missing = missing_modules(modules)
    if missing:
        raise ImportError(f"{what} needs {', '.join(missing)}; {install_hint(extra)}")
