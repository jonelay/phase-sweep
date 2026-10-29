"""Missing optional extras fail with an install hint, not a raw error.

Each check runs in a subprocess with one dependency hidden through
`sys.modules[name] = None`, which both `import` and `find_spec` treat as
not installed, so the installed copy cannot mask the failure path.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent


def _run(hidden: str, body: str) -> subprocess.CompletedProcess[str]:
    code = f"import sys\nsys.modules[{hidden!r}] = None\n" + textwrap.dedent(body)
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=120, cwd=ROOT,
    )


def test_hint_never_names_the_pypi_distribution():
    # `phasesweep` on PyPI is an unrelated project.
    hits = subprocess.run(
        ["git", "grep", "-n", "pip install phasesweep", "--", "phasesweep"],
        capture_output=True, text=True, cwd=ROOT,
    ).stdout
    assert hits == ""


@pytest.mark.parametrize(("hidden", "module", "extra"), [
    ("fastapi", "phasesweep.server.app", "server"),
    ("structlog", "phasesweep.server.jobs", "server"),
    ("tomli_w", "phasesweep.server.routes", "server"),
    ("matplotlib", "phasesweep.vis.plots", "viz"),
    ("motulator", "phasesweep.simulation.sim", "sim"),
])
def test_module_import_names_the_extra(hidden, module, extra):
    r = _run(hidden, f"""
        try:
            import {module}
        except ImportError as e:
            print(e)
        else:
            raise SystemExit("imported without {hidden}")
    """)
    assert r.returncode == 0, r.stderr
    assert hidden in r.stdout
    assert f'pip install -e ".[{extra}]"' in r.stdout


@pytest.mark.parametrize(("hidden", "model", "extra"), [
    ("ngsolve", "fem", "fem"),
    ("motulator", "drive_sim", "sim"),
])
def test_model_run_names_the_extra(hidden, model, extra):
    r = _run(hidden, f"""
        from phasesweep import MODEL_REGISTRY, RunConfig, load_motor
        cfg = RunConfig(motor=load_motor("motors/belkhadir_outrunner.toml"), model={model!r})
        try:
            MODEL_REGISTRY[{model!r}].fn(cfg)
        except ImportError as e:
            print(e)
        else:
            raise SystemExit("ran without {hidden}")
    """)
    assert r.returncode == 0, r.stderr
    assert f'pip install -e ".[{extra}]"' in r.stdout


def test_server_cli_exits_with_one_line_hint():
    r = _run("uvicorn", """
        sys.argv = ["phasesweep-server"]
        from phasesweep.server.__main__ import main
        main()
    """)
    assert r.returncode == 1
    assert "Traceback" not in r.stderr
    assert r.stderr.strip().count("\n") == 0
    assert 'pip install -e ".[server]"' in r.stderr
