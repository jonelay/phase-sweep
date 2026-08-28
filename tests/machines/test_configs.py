"""Tests for motor config loading and TOML motor definitions."""

import logging
import math
from pathlib import Path

import pytest

from phasesweep.machines.configs import load_motor, load_motors
from phasesweep.machines.geometry import geometry_from_toml

MOTORS_DIR = Path(__file__).parent.parent.parent / "motors"
CREATOR_TOML = MOTORS_DIR / "creator_case_pmsm.toml"


def test_load_motor():
    motor = load_motor(CREATOR_TOML)
    assert motor.n_p == 2
    assert motor.R_s == pytest.approx(8.9462)
    assert motor.L_d == pytest.approx(0.2055)
    assert motor.L_q == pytest.approx(0.3320)
    assert motor.psi_f == pytest.approx(0.1144)
    assert motor.J == pytest.approx(0.00011348)
    assert motor.geometry.n_slots == 6
    assert motor.N == 328
    assert motor.k_w == pytest.approx(0.866)
    assert motor.L_stk == pytest.approx(0.0301)


def test_load_motor_drive_params():
    motor = load_motor(CREATOR_TOML)
    assert motor.drive.U_DC == pytest.approx(326.0)
    assert motor.drive.MAX_I_S == pytest.approx(0.42)
    assert motor.drive.W_REF == pytest.approx(209.4)


def test_load_motor_name():
    motor = load_motor(CREATOR_TOML)
    assert motor.name == "CREATOR Case PMSM"


def test_load_motor_geometry():
    motor = load_motor(CREATOR_TOML)
    assert motor.geometry.topology == "inrunner"
    assert motor.geometry.r_outer == pytest.approx(0.113 / 2)
    assert motor.geometry.r_stator == pytest.approx(0.0478 / 2)
    assert motor.geometry.r_magnet == pytest.approx(0.047 / 2)


def test_load_motor_k_end_and_overhang(tmp_path):
    toml = tmp_path / "kend.toml"
    toml.write_text("""
[motor]
topology = "outrunner"
[circuit]
n_p = 6
[geometry]
r_outer = 0.009665
r_rotor = 0.008765
r_magnet = 0.007478
r_stator = 0.006950
r_inner = 0.003
L_stk = 0.007
magnet_overhang = 0.00025
k_end = 0.9252
""")
    motor = load_motor(toml)
    assert motor.k_end == pytest.approx(0.9252)
    assert motor.geometry.magnet_overhang == pytest.approx(0.00025)


def test_load_motor_derives_slot_opening_ratio():
    # slot_opening_width 3.23mm over the 6-slot pitch at the 23.9mm bore;
    # n_slots comes from [winding], so the rebuild path must re-derive
    motor = load_motor(CREATOR_TOML)
    assert motor.geometry.slot_opening_ratio == pytest.approx(0.129, abs=0.001)


def test_explicit_slot_opening_width_wins():
    geo = geometry_from_toml({
        "r_outer": 1.0, "r_stator": 0.7, "r_magnet": 0.64, "r_rotor": 0.3,
        "n_slots": 6, "slot_opening_ratio": 0.2, "slot_opening_width": 0.1,
    })
    assert geo.slot_opening_width == 0.1
    assert geo.slot_opening_ratio == pytest.approx(0.1 / (2 * math.pi * 0.7 / 6))


def test_n_slots_mismatch_raises(tmp_path):
    toml = tmp_path / "bad_motor.toml"
    toml.write_text(
        "[circuit]\nn_p = 4\n"
        "[winding]\nn_slots = 9\n"
        "[geometry]\nn_slots = 12\n"
        "r_outer = 1.0\nr_stator = 0.7\nr_magnet = 0.64\nr_rotor = 0.3\n"
    )
    with pytest.raises(ValueError, match="n_slots mismatch"):
        load_motor(toml)


def test_toml_slot_opening_ratio_converts_to_width():
    geo = geometry_from_toml({
        "r_outer": 1.0, "r_stator": 0.7, "r_magnet": 0.64, "r_rotor": 0.3,
        "n_slots": 6, "slot_opening_ratio": 0.2,
    })
    assert geo.slot_opening_width == pytest.approx(0.2 * 2 * math.pi * 0.7 / 6)
    assert geo.slot_opening_ratio == pytest.approx(0.2, rel=1e-12)


def test_load_motor_materials():
    motor = load_motor(CREATOR_TOML)
    assert motor.B_rem == pytest.approx(0.415)
    assert motor.mu_r_pm == pytest.approx(1.1)


def test_load_motor_missing_file():
    with pytest.raises(FileNotFoundError):
        load_motor("nonexistent.toml")


def test_load_motor_missing_drive_gives_none(tmp_path, caplog):
    p = tmp_path / "minimal.toml"
    p.write_text(
        "[circuit]\nn_p = 4\n"
        "[geometry]\n"
        "r_outer = 1.0\nr_stator = 0.7\nr_magnet = 0.64\nr_rotor = 0.3\n"
    )
    with caplog.at_level(logging.WARNING):
        motor = load_motor(p)
    assert motor.drive.U_DC is None
    assert motor.drive.MAX_I_S is None
    assert motor.drive.W_REF is None
    assert motor.mu_r_fe == pytest.approx(1000.0)
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "mu_r_fe" in messages


def test_load_motor_missing_circuit_section(tmp_path):
    p = tmp_path / "bad.toml"
    p.write_text('[motor]\nname = "bad"\n')
    with pytest.raises(ValueError, match=r"missing required section \[circuit\]"):
        load_motor(p)


def test_load_motor_missing_n_p(tmp_path):
    p = tmp_path / "bad.toml"
    p.write_text('[circuit]\nR_s = 0.2\n[geometry]\nr_outer = 1.0\nr_stator = 0.7\nr_magnet = 0.6\nr_rotor = 0.3\n')
    with pytest.raises(ValueError, match=r"\[circuit\] missing required field 'n_p'"):
        load_motor(p)


def test_load_motor_missing_geometry_section(tmp_path):
    # [geometry] is optional: a circuit-only (datasheet) motor loads with
    # geometry=None and runs the circuit tier.
    p = tmp_path / "circuit_only.toml"
    p.write_text('[circuit]\nn_p = 2\nR_s = 0.5\npsi_f = 0.1\n')
    m = load_motor(p)
    assert m.geometry is None
    assert m.R_s == 0.5
    assert m.psi_f == 0.1


def test_load_motor_bad_geometry_radii(tmp_path):
    p = tmp_path / "bad.toml"
    p.write_text('[circuit]\nn_p = 2\n[geometry]\nr_outer = 0.5\n')
    with pytest.raises(ValueError, match=str(p)):
        load_motor(p)


def test_load_motor_invalid_n_p(tmp_path):
    p = tmp_path / "bad.toml"
    p.write_text('[circuit]\nn_p = 1\n[geometry]\nr_outer = 1.0\nr_stator = 0.7\nr_magnet = 0.6\nr_rotor = 0.3\n')
    with pytest.raises(ValueError, match=str(p)):
        load_motor(p)


def test_load_motor_error_includes_path(tmp_path):
    """All validation errors include the file path for diagnosis."""
    p = tmp_path / "my_motor.toml"
    p.write_text('[motor]\nname = "x"\n')
    with pytest.raises(ValueError, match=r"my_motor\.toml"):
        load_motor(p)


def test_load_motors_directory():
    motors = load_motors(MOTORS_DIR)
    assert "CREATOR Case PMSM" in motors
    assert motors["CREATOR Case PMSM"].n_p == 2


def test_load_motors_strict_raises_on_bad_file(tmp_path):
    (tmp_path / "good.toml").write_text("[circuit]\nn_p = 4\n")
    (tmp_path / "bad.toml").write_text('[motor]\nname = "x"\n')
    with pytest.raises(ValueError, match=r"bad\.toml"):
        load_motors(tmp_path, strict=True)


def test_load_motors_default_skips_bad_file(tmp_path, caplog):
    (tmp_path / "good.toml").write_text("[circuit]\nn_p = 4\n")
    (tmp_path / "bad.toml").write_text('[motor]\nname = "x"\n')
    with caplog.at_level(logging.WARNING):
        motors = load_motors(tmp_path)
    assert len(motors) == 1
    assert any("bad.toml" in r.getMessage() for r in caplog.records)


class TestHalbachToml:

    def test_halbach_loads(self, tmp_path):
        p = tmp_path / "halbach.toml"
        p.write_text(
            "[circuit]\nn_p = 10\n"
            "[materials]\nB_rem = 1.35\nmu_r_pm = 1.09\n"
            "[geometry]\n"
            "r_outer = 0.057\nr_stator = 0.047\n"
            "r_magnet = 0.046\nr_rotor = 0.043\n"
            'magnetization = "halbach"\n'
            "halbach_segments = 3\n"
            "halbach_angles_deg = [0.0, -45.0, -135.0]\n"
        )
        motor = load_motor(p)
        assert motor.magnetization == "halbach"
        assert motor.halbach_segments == 3
        assert motor.halbach_angles_deg == (0.0, -45.0, -135.0)

    def test_halbach_ideal_no_angles(self, tmp_path):
        p = tmp_path / "halbach_ideal.toml"
        p.write_text(
            "[circuit]\nn_p = 4\n"
            "[materials]\nB_rem = 1.2\n"
            "[geometry]\n"
            "r_outer = 0.06\nr_stator = 0.048\n"
            "r_magnet = 0.04\nr_rotor = 0.03\n"
            'magnetization = "halbach"\n'
            "halbach_segments = 4\n"
        )
        motor = load_motor(p)
        assert motor.magnetization == "halbach"
        assert motor.halbach_segments == 4
        assert motor.halbach_angles_deg is None

    def test_halbach_validation_error_includes_path(self, tmp_path):
        p = tmp_path / "bad_halbach.toml"
        p.write_text(
            "[circuit]\nn_p = 4\n"
            "[materials]\nB_rem = 1.2\n"
            "[geometry]\n"
            "r_outer = 0.06\nr_stator = 0.048\n"
            "r_magnet = 0.04\nr_rotor = 0.03\n"
            'magnetization = "halbach"\n'
        )
        with pytest.raises(ValueError, match="bad_halbach"):
            load_motor(p)

    def test_halbach_angles_length_mismatch_from_toml(self, tmp_path):
        p = tmp_path / "bad_angles.toml"
        p.write_text(
            "[circuit]\nn_p = 4\n"
            "[materials]\nB_rem = 1.2\n"
            "[geometry]\n"
            "r_outer = 0.06\nr_stator = 0.048\n"
            "r_magnet = 0.04\nr_rotor = 0.03\n"
            'magnetization = "halbach"\n'
            "halbach_segments = 3\n"
            "halbach_angles_deg = [0.0, -45.0]\n"
        )
        with pytest.raises(ValueError, match="length"):
            load_motor(p)

    def test_radial_default_unchanged(self, tmp_path):
        p = tmp_path / "radial.toml"
        p.write_text(
            "[circuit]\nn_p = 4\n"
            "[geometry]\n"
            "r_outer = 0.06\nr_stator = 0.048\n"
            "r_magnet = 0.04\nr_rotor = 0.03\n"
        )
        motor = load_motor(p)
        assert motor.magnetization == "radial"
        assert motor.halbach_segments is None
        assert motor.halbach_angles_deg is None

