"""Tests for motor config loading and TOML motor definitions."""

from pathlib import Path

import pytest

from phasesweep.configs import CONFIGS, FullMotorConfig, load_motor, load_motors

MOTORS_DIR = Path(__file__).parent.parent / "motors"
CREATOR_TOML = MOTORS_DIR / "creator_case_pmsm.toml"


def test_load_motor():
    motor = load_motor(CREATOR_TOML)
    cfg = motor.config
    assert cfg["n_p"] == 2
    assert cfg["R_s"] == pytest.approx(8.9462)
    assert cfg["L_d"] == pytest.approx(0.2055)
    assert cfg["L_q"] == pytest.approx(0.3320)
    assert cfg["psi_f"] == pytest.approx(0.1144)
    assert cfg["J"] == pytest.approx(0.00011348)
    assert cfg["n_slots"] == 6
    assert cfg["N"] == 328
    assert cfg["k_w"] == pytest.approx(0.866)
    assert cfg["L_stk"] == pytest.approx(0.0301)


def test_load_motor_drive_params():
    motor = load_motor(CREATOR_TOML)
    assert motor.drive["U_DC"] == pytest.approx(326.0)
    assert motor.drive["MAX_I_S"] == pytest.approx(0.42)
    assert motor.drive["W_REF"] == pytest.approx(209.4)


def test_load_motor_name():
    motor = load_motor(CREATOR_TOML)
    assert motor.name == "CREATOR Case PMSM"


def test_load_motor_metadata():
    motor = load_motor(CREATOR_TOML)
    assert "geometry" in motor.metadata
    assert "materials" in motor.metadata
    assert "provenance" in motor.metadata
    assert motor.metadata["provenance"]["doi"] == "10.3217/sns1d-77m43"


def test_load_motor_missing_file():
    with pytest.raises(FileNotFoundError):
        load_motor("nonexistent.toml")


def test_load_motors_directory():
    motors = load_motors(MOTORS_DIR)
    assert "CREATOR Case PMSM" in motors
    assert motors["CREATOR Case PMSM"].config["n_p"] == 2


def test_configs_have_winding_fields():
    for name, cfg in CONFIGS.items():
        assert "N" in cfg, f"{name} missing N"
        assert "k_w" in cfg, f"{name} missing k_w"
        assert "L_stk" in cfg, f"{name} missing L_stk"
