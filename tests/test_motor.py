"""Tests for Motor and DriveParams."""

import pytest

from phasesweep.geometry import default_inrunner
from phasesweep.motor import DriveParams, Motor


class TestMotorConstruction:

    def test_required_only(self):
        geo = default_inrunner()
        m = Motor(name="minimal", geometry=geo, n_p=2)
        assert m.R_s is None
        assert m.L_d is None

    def test_all_fields(self):
        geo = default_inrunner()
        m = Motor(
            name="full", geometry=geo, n_p=2,
            R_s=0.2, L_d=4e-3, L_q=4e-3, psi_f=0.1, B_rem=0.5,
            J=0.002, N=50, k_w=0.966, L_stk=0.10, coils_series=1,
        )
        assert m.R_s == 0.2
        assert m.B_rem == 0.5

    def test_n_p_below_2_raises(self):
        with pytest.raises(ValueError, match="n_p"):
            Motor(name="bad", geometry=default_inrunner(), n_p=1)

    def test_R_s_negative_raises(self):
        with pytest.raises(ValueError, match="R_s"):
            Motor(name="bad", geometry=default_inrunner(), n_p=2, R_s=-1.0)

    def test_L_d_zero_raises(self):
        with pytest.raises(ValueError, match="L_d"):
            Motor(name="bad", geometry=default_inrunner(), n_p=2, L_d=0.0)

    def test_L_q_negative_raises(self):
        with pytest.raises(ValueError, match="L_q"):
            Motor(name="bad", geometry=default_inrunner(), n_p=2, L_q=-1e-3)

    def test_psi_f_zero_raises(self):
        with pytest.raises(ValueError, match="psi_f"):
            Motor(name="bad", geometry=default_inrunner(), n_p=2, psi_f=0.0)

    def test_k_w_above_1_raises(self):
        with pytest.raises(ValueError, match="k_w"):
            Motor(name="bad", geometry=default_inrunner(), n_p=2, k_w=1.5)


class TestMotorConfigId:

    def test_determinism(self):
        geo = default_inrunner()
        m1 = Motor(name="a", geometry=geo, n_p=2, R_s=0.2)
        m2 = Motor(name="a", geometry=geo, n_p=2, R_s=0.2)
        assert m1.config_id == m2.config_id

    def test_excludes_name(self):
        geo = default_inrunner()
        m1 = Motor(name="a", geometry=geo, n_p=2, R_s=0.2)
        m2 = Motor(name="b", geometry=geo, n_p=2, R_s=0.2)
        assert m1.config_id == m2.config_id

    def test_excludes_drive(self):
        geo = default_inrunner()
        m1 = Motor(name="a", geometry=geo, n_p=2, drive=DriveParams(U_DC=300.0))
        m2 = Motor(name="a", geometry=geo, n_p=2, drive=DriveParams(U_DC=600.0))
        assert m1.config_id == m2.config_id

    def test_none_fields_dont_contribute(self):
        geo = default_inrunner()
        m1 = Motor(name="a", geometry=geo, n_p=2)
        m2 = Motor(name="a", geometry=geo, n_p=2, R_s=0.2)
        assert m1.config_id != m2.config_id

    def test_is_12_chars(self):
        m = Motor(name="a", geometry=default_inrunner(), n_p=2)
        assert len(m.config_id) == 12


class TestDriveParams:

    def test_defaults(self):
        d = DriveParams()
        assert d.U_DC == 540.0

    def test_negative_U_DC_raises(self):
        with pytest.raises(ValueError, match="U_DC"):
            DriveParams(U_DC=-1.0)

    def test_zero_MAX_I_S_raises(self):
        with pytest.raises(ValueError, match="MAX_I_S"):
            DriveParams(MAX_I_S=0.0)


class TestSerialization:

    def test_roundtrip(self, default_motor):
        d = default_motor.to_dict()
        m2 = Motor.from_dict(d)
        assert m2.config_id == default_motor.config_id
        assert m2.name == default_motor.name
        assert m2.n_p == default_motor.n_p
        assert m2.R_s == default_motor.R_s
        assert m2.geometry.r_outer == default_motor.geometry.r_outer

    def test_roundtrip_minimal(self):
        geo = default_inrunner()
        m = Motor(name="min", geometry=geo, n_p=2)
        m2 = Motor.from_dict(m.to_dict())
        assert m2.config_id == m.config_id
        assert m2.R_s is None

    def test_roundtrip_all_fields(self):
        geo = default_inrunner()
        m = Motor(
            name="all", geometry=geo, n_p=4,
            R_s=3.6, L_d=8e-3, L_q=12e-3, psi_f=0.15, B_rem=1.1,
            J=0.005, N=100, k_w=0.933, L_stk=0.08, coils_series=1,
            mu_r_fe=500.0, mu_r_pm=1.08,
            drive=DriveParams(U_DC=300.0, MAX_I_S=10.0, W_REF=200.0),
        )
        m2 = Motor.from_dict(m.to_dict())
        assert m2.config_id == m.config_id
        for field in ("R_s", "L_d", "L_q", "psi_f", "B_rem", "J",
                       "N", "k_w", "L_stk", "mu_r_fe", "mu_r_pm"):
            assert getattr(m2, field) == getattr(m, field), f"{field} mismatch"
        assert m2.drive.U_DC == 300.0
        assert m2.drive.MAX_I_S == 10.0
        assert m2.drive.W_REF == 200.0
