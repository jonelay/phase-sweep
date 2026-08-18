"""Tests for phasesweep.debug_info()."""

import platform

import numpy

import phasesweep
from phasesweep import _dep_version, debug_info


def test_debug_info_runs(capsys):
    result = debug_info()
    assert result is None
    out = capsys.readouterr().out
    assert phasesweep.__version__ in out
    assert platform.python_version() in out
    for dep in phasesweep._DEBUG_DEPS:
        assert dep in out


def test_dep_version_installed():
    assert _dep_version("numpy") == numpy.__version__


def test_dep_version_missing():
    assert _dep_version("nonexistent_module_xyz") == "not installed"


def test_dep_version_broken(monkeypatch):
    def _raise_import_error(name):
        raise ImportError("libfoo.so: cannot open shared object file")

    monkeypatch.setattr("importlib.import_module", _raise_import_error)
    result = _dep_version("numpy")
    assert result.startswith("broken (")
    assert "ImportError" in result


def test_dep_version_no_version_attr(monkeypatch):
    import types

    fake = types.ModuleType("fake_module")
    monkeypatch.setattr("importlib.import_module", lambda name: fake)
    assert _dep_version("fake_module") == "installed"
