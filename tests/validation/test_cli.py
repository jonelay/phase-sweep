"""Smoke tests for the console-script entry points (cli_import, cli_crossval)."""

import json
import sys

import pytest

from phasesweep.validation import cli_crossval, cli_import
from tests.conftest import REPO_ROOT

MEASURED_JSON = REPO_ROOT / "data" / "belkhadir_outrunner" / "torque_rated.json"


def _run_main(monkeypatch, module, argv):
    monkeypatch.setattr(sys, "argv", argv)
    module.main()


class TestCliImport:

    def test_import_happy_path(self, monkeypatch, capsys, tmp_path):
        _run_main(monkeypatch, cli_import, [
            "phase-sweep-import", str(MEASURED_JSON),
            "--motor-dir", str(REPO_ROOT / "motors"),
            "--output-dir", str(tmp_path),
        ])
        out = capsys.readouterr().out
        assert "Imported: model=" in out
        assert "status=OK" in out
        assert (tmp_path / "index.json").exists()

    def test_import_missing_motor_name_exits(self, monkeypatch, capsys, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"test_type": "torque_test"}))
        with pytest.raises(SystemExit) as exc:
            _run_main(monkeypatch, cli_import, [
                "phase-sweep-import", str(bad),
                "--output-dir", str(tmp_path),
            ])
        assert exc.value.code == 1
        assert "missing motor_name" in capsys.readouterr().err

    def test_import_unknown_motor_exits(self, monkeypatch, capsys, tmp_path):
        bad = tmp_path / "orphan.json"
        bad.write_text(json.dumps(
            {"motor_name": "No Such Motor", "test_type": "torque_test"}
        ))
        with pytest.raises(SystemExit) as exc:
            _run_main(monkeypatch, cli_import, [
                "phase-sweep-import", str(bad),
                "--motor-dir", str(REPO_ROOT / "motors"),
                "--output-dir", str(tmp_path),
            ])
        assert exc.value.code == 1
        assert "not found" in capsys.readouterr().err


class TestCliCrossval:

    def test_crossval_happy_path(self, monkeypatch, capsys, tmp_path):
        from phasesweep.machines.configs import load_motor
        from phasesweep.validation.measured import import_measured

        motor = load_motor(REPO_ROOT / "motors" / "belkhadir_outrunner.toml")
        import_measured(MEASURED_JSON, motor, tmp_path)

        _run_main(monkeypatch, cli_crossval, [
            "phase-sweep-crossval", "--output-dir", str(tmp_path),
        ])
        out = capsys.readouterr().out
        assert "===" in out
        assert "Diagnosis:" in out

    def test_crossval_empty_store_exits(self, monkeypatch, capsys, tmp_path):
        with pytest.raises(SystemExit) as exc:
            _run_main(monkeypatch, cli_crossval, [
                "phase-sweep-crossval", "--output-dir", str(tmp_path),
            ])
        assert exc.value.code == 1
        assert "No results found" in capsys.readouterr().err

    def test_crossval_motor_filter_no_match_exits(self, monkeypatch, capsys, tmp_path):
        from phasesweep.machines.configs import load_motor
        from phasesweep.validation.measured import import_measured

        motor = load_motor(REPO_ROOT / "motors" / "belkhadir_outrunner.toml")
        import_measured(MEASURED_JSON, motor, tmp_path)

        with pytest.raises(SystemExit) as exc:
            _run_main(monkeypatch, cli_crossval, [
                "phase-sweep-crossval", "--output-dir", str(tmp_path),
                "--motor", "nonexistent",
            ])
        assert exc.value.code == 1
        assert "No results matching" in capsys.readouterr().err

    def test_crossval_strict_passes_when_all_pass(self, monkeypatch, capsys, tmp_path):
        from phasesweep.machines.configs import load_motor
        from phasesweep.validation.measured import import_measured

        motor = load_motor(REPO_ROOT / "motors" / "belkhadir_outrunner.toml")
        import_measured(MEASURED_JSON, motor, tmp_path)

        _run_main(monkeypatch, cli_crossval, [
            "phase-sweep-crossval", "--output-dir", str(tmp_path), "--strict",
        ])
        assert "Diagnosis:" in capsys.readouterr().out

    def test_crossval_excludes_stale_computed(self, monkeypatch, capsys, tmp_path):
        """A superseded-physics computed record must
        not reach PASS/FAIL or the --strict gate; the measured record
        keeps serving. The stale record carries a poison metric that
        would fail any tolerance if it leaked through."""
        from phasesweep.machines.configs import load_motor
        from phasesweep.result_store import ResultStore
        from phasesweep.sweep_types import RunConfig, RunResult
        from phasesweep.validation.measured import import_measured

        motor = load_motor(REPO_ROOT / "motors" / "belkhadir_outrunner.toml")
        import_measured(MEASURED_JSON, motor, tmp_path)
        store = ResultStore(tmp_path)
        rc = RunConfig(motor=motor, model="rated_torque")
        store.save(RunResult(config=rc, model="rated_torque", status="OK",
                             metrics={"tau_rated": 999.0}, elapsed_s=0.1,
                             model_version=1))  # superseded (registry v4)

        _run_main(monkeypatch, cli_crossval, [
            "phase-sweep-crossval", "--output-dir", str(tmp_path), "--strict",
        ])  # no SystemExit: the stale FAIL-bait record was filtered out
        out = capsys.readouterr().out
        assert "rated_torque" not in out
        assert "Diagnosis:" in out

    def test_crossval_strict_exits_on_failure(self, monkeypatch, capsys, tmp_path):
        from phasesweep.machines.configs import load_motor
        from phasesweep.validation import crossval
        from phasesweep.validation.measured import import_measured

        motor = load_motor(REPO_ROOT / "motors" / "belkhadir_outrunner.toml")
        import_measured(MEASURED_JSON, motor, tmp_path)

        class _FailDiag:
            all_pass = False

        monkeypatch.setattr(
            crossval, "diagnose_detailed", lambda group: _FailDiag())
        with pytest.raises(SystemExit) as exc:
            _run_main(monkeypatch, cli_crossval, [
                "phase-sweep-crossval", "--output-dir", str(tmp_path), "--strict",
            ])
        assert exc.value.code == 1
        assert "STRICT" in capsys.readouterr().err
