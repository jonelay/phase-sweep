"""Tests for Motor and DriveParams."""

import pytest

from phasesweep.machines.geometry import default_inrunner
from phasesweep.machines.motor import DriveParams, InductionParams, Motor


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

    def test_psi_f_negative_raises(self):
        with pytest.raises(ValueError, match="psi_f"):
            Motor(name="bad", geometry=default_inrunner(), n_p=2, psi_f=-0.1)

    def test_psi_f_zero_allowed(self):
        # SynRM: reluctance-only machine has no PM flux linkage
        m = Motor(name="synrm", geometry=default_inrunner(), n_p=2, psi_f=0.0)
        assert m.psi_f == 0.0

    def test_inset_requires_partial_alpha_p(self):
        import dataclasses
        geo = dataclasses.replace(default_inrunner(), magnet_mount="inset")
        with pytest.raises(ValueError, match="inset.*alpha_p < 1"):
            Motor(name="bad", geometry=geo, n_p=2)  # alpha_p defaults to 1.0
        m = Motor(name="ok", geometry=geo, n_p=2, alpha_p=0.7)
        assert m.geometry.magnet_mount == "inset"

    def test_psi_f_zero_with_b_rem_raises(self):
        # A reluctance machine has no magnets: psi_f = 0 contradicts B_rem
        with pytest.raises(ValueError, match="contradicts"):
            Motor(name="bad", geometry=default_inrunner(), n_p=2,
                  psi_f=0.0, B_rem=1.2)

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


class TestKEndField:

    def _motor(self, **kw):
        return Motor(name="k", geometry=default_inrunner(), n_p=2, **kw)

    def test_default_none(self):
        assert self._motor().k_end is None

    def test_valid_values(self):
        assert self._motor(k_end=0.9252).k_end == 0.9252
        # > 1 is real under magnet overhang (flux concentration)
        assert self._motor(k_end=1.03).k_end == 1.03

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="k_end"):
            self._motor(k_end=0.0)

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError, match="k_end"):
            self._motor(k_end=1.5)

    def test_config_id_unchanged_when_none(self):
        # back-compat: existing run-IDs must not move
        assert self._motor().config_id == self._motor(k_end=None).config_id

    def test_config_id_hashes_k_end(self):
        assert (self._motor(k_end=0.9252).config_id
                != self._motor().config_id)
        assert (self._motor(k_end=0.9252).config_id
                != self._motor(k_end=0.95).config_id)

    def test_dict_roundtrip(self):
        m = self._motor(k_end=0.9252)
        m2 = Motor.from_dict(m.to_dict())
        assert m2.k_end == 0.9252
        assert m2.config_id == m.config_id


class TestDriveParams:

    def test_defaults_are_none(self):
        d = DriveParams()
        assert d.U_DC is None
        assert d.MAX_I_S is None
        assert d.W_REF is None
        assert d.I_LIMIT is None

    def test_negative_U_DC_raises(self):
        with pytest.raises(ValueError, match="U_DC"):
            DriveParams(U_DC=-1.0)

    def test_zero_MAX_I_S_raises(self):
        with pytest.raises(ValueError, match="MAX_I_S"):
            DriveParams(MAX_I_S=0.0)

    @pytest.mark.parametrize("field", ["U_DC", "MAX_I_S", "W_REF", "I_LIMIT"])
    def test_positive_only_fields_reject_nonpositive(self, field):
        for bad in (0.0, -5.0):
            with pytest.raises(ValueError, match=field):
                DriveParams(**{field: bad})


class TestMotorGuardSweep:
    """One parametrized sweep over every positive-only Motor field plus
    the range/cross-field guards the per-field tests above don't touch."""

    @pytest.mark.parametrize("field", [
        "B_rem", "J", "N", "L_stk", "I_rated", "coils_series", "r_th",
        "thermal_time_constant", "k_h", "k_e", "m_core", "B_core",
        "mu_r_fe", "mu_r_pm",
    ])
    def test_positive_only_field_rejects_nonpositive(self, field):
        for bad in (0.0, -1.0):
            with pytest.raises(ValueError, match=field):
                Motor(name="bad", geometry=None, n_p=2, **{field: bad})

    def test_alpha_fe_out_of_range_raises(self):
        for bad in (0.9, 3.1):
            with pytest.raises(ValueError, match="alpha_fe"):
                Motor(name="bad", geometry=None, n_p=2, alpha_fe=bad)

    def test_alpha_p_out_of_range_raises(self):
        for bad in (0.0, -0.1, 1.001):
            with pytest.raises(ValueError, match="alpha_p"):
                Motor(name="bad", geometry=None, n_p=2, alpha_p=bad)

    def test_alpha_Br_percent_trap_raises(self):
        # datasheet %/°C pasted raw (-0.12) and a positive slope both fail
        for bad in (-0.12, 0.001):
            with pytest.raises(ValueError, match="alpha_Br"):
                Motor(name="bad", geometry=None, n_p=2, alpha_Br=bad)

    def test_alpha_B_knee_out_of_range_raises(self):
        for bad in (-0.2, 0.2):
            with pytest.raises(ValueError, match="alpha_B_knee"):
                Motor(name="bad", geometry=None, n_p=2, alpha_B_knee=bad)

    def test_B_knee_at_or_above_B_rem_raises(self):
        with pytest.raises(ValueError, match="B_knee"):
            Motor(name="bad", geometry=None, n_p=2, B_rem=1.2, B_knee=1.2)

    def test_winding_temp_limit_at_or_below_ambient_raises(self):
        with pytest.raises(ValueError, match="winding_temp_limit"):
            Motor(name="bad", geometry=None, n_p=2,
                  winding_temp_limit=40.0, ambient_temp=40.0)

    @pytest.mark.parametrize("field,value", [
        ("psi_f", 0.1), ("B_rem", 1.2), ("L_d", 1e-3), ("L_q", 1e-3),
    ])
    def test_induction_rejects_synchronous_field(self, field, value):
        with pytest.raises(ValueError, match=field):
            Motor(name="bad", geometry=None, n_p=2,
                  induction=InductionParams(), **{field: value})


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


class TestGeometryOptional:
    """Datasheet/circuit-only motors carry no geometry."""

    def test_no_geometry_construction(self):
        m = Motor(name="ipm", geometry=None, n_p=2, R_s=0.5, psi_f=0.1)
        assert m.geometry is None

    def test_no_geometry_config_id_stable(self):
        m = Motor(name="ipm", geometry=None, n_p=2, R_s=0.5, psi_f=0.1)
        # Pinned: the None-geometry sentinel must not silently drift
        assert m.config_id == Motor(
            name="other", geometry=None, n_p=2, R_s=0.5, psi_f=0.1,
        ).config_id

    def test_no_geometry_roundtrip(self):
        m = Motor(name="ipm", geometry=None, n_p=2, R_s=0.5,
                  L_d=5e-3, L_q=1.05e-2, psi_f=0.1, I_rated=10.0)
        m2 = Motor.from_dict(m.to_dict())
        assert m2.geometry is None
        assert m2.config_id == m.config_id

    def test_geometry_present_config_id_unchanged(self):
        # Adding the geometry-optional + thermal fields must NOT perturb the
        # hash of an existing geometry-bearing motor (no run-ID regression).
        geo = default_inrunner()
        m = Motor(name="m", geometry=geo, n_p=2, R_s=0.2, psi_f=0.1)
        parts = [geo.config_id, "2", "R_s=0.2", "psi_f=0.1",
                 "alpha_p=1.0", "mu_r_fe=1000.0", "mu_r_pm=1.05"]
        import hashlib
        expected = hashlib.md5("|".join(parts).encode()).hexdigest()[:12]
        assert m.config_id == expected

    def test_field_factory_rejects_no_geometry(self):
        from phasesweep.solvers.analytical import prepare_analytical
        from phasesweep.solvers.fem_runner import prepare_fem
        m = Motor(name="ipm", geometry=None, n_p=2, B_rem=1.0)
        with pytest.raises(ValueError, match="geometry"):
            prepare_analytical(m)
        with pytest.raises(ValueError, match="geometry"):
            prepare_fem(m)


class TestThermalFields:
    """Optional thermal context fields."""

    def test_thermal_fields_optional(self):
        m = Motor(name="m", geometry=default_inrunner(), n_p=2)
        assert m.winding_temp_limit is None
        assert m.r_th is None
        assert m.insulation_class is None

    def test_thermal_fields_roundtrip(self):
        m = Motor(name="m", geometry=None, n_p=2, R_s=0.5, psi_f=0.1,
                  winding_temp_limit=155.0, ambient_temp=40.0, r_th=0.5,
                  insulation_class="F")
        m2 = Motor.from_dict(m.to_dict())
        assert m2.winding_temp_limit == 155.0
        assert m2.ambient_temp == 40.0
        assert m2.r_th == 0.5
        assert m2.insulation_class == "F"
        assert m2.config_id == m.config_id

    def test_thermal_fields_hashed(self):
        base = Motor(name="m", geometry=None, n_p=2, R_s=0.5, psi_f=0.1)
        hot = Motor(name="m", geometry=None, n_p=2, R_s=0.5, psi_f=0.1,
                    winding_temp_limit=180.0, ambient_temp=40.0, r_th=0.5)
        assert base.config_id != hot.config_id

    def test_insulation_class_not_hashed(self):
        a = Motor(name="m", geometry=None, n_p=2, R_s=0.5, psi_f=0.1,
                  insulation_class="F")
        b = Motor(name="m", geometry=None, n_p=2, R_s=0.5, psi_f=0.1,
                  insulation_class="H")
        assert a.config_id == b.config_id

    def test_bad_r_th(self):
        with pytest.raises(ValueError, match="r_th"):
            Motor(name="m", geometry=None, n_p=2, r_th=0.0)

    def test_temp_limit_below_ambient(self):
        with pytest.raises(ValueError, match="winding_temp_limit"):
            Motor(name="m", geometry=None, n_p=2,
                  winding_temp_limit=40.0, ambient_temp=50.0)


class TestHalbachFields:

    def test_halbach_construction(self):
        m = Motor(
            name="h", geometry=default_inrunner(), n_p=10,
            B_rem=1.35,
            magnetization="halbach", halbach_segments=3,
            halbach_angles_deg=(0.0, -45.0, -135.0),
        )
        assert m.magnetization == "halbach"
        assert m.halbach_segments == 3
        assert m.halbach_angles_deg == (0.0, -45.0, -135.0)

    def test_halbach_ideal_rule_no_angles(self):
        m = Motor(
            name="h", geometry=default_inrunner(), n_p=4,
            B_rem=1.2,
            magnetization="halbach", halbach_segments=4,
        )
        assert m.halbach_angles_deg is None

    def test_halbach_without_segments_raises(self):
        with pytest.raises(ValueError, match="halbach_segments"):
            Motor(
                name="h", geometry=default_inrunner(), n_p=4,
                B_rem=1.2,
                magnetization="halbach",
            )

    def test_halbach_segments_zero_raises(self):
        with pytest.raises(ValueError, match="halbach_segments"):
            Motor(
                name="h", geometry=default_inrunner(), n_p=4,
                B_rem=1.2,
                magnetization="halbach", halbach_segments=0,
            )

    def test_radial_with_segments_raises(self):
        with pytest.raises(ValueError, match="halbach_segments must be None"):
            Motor(
                name="h", geometry=default_inrunner(), n_p=4,
                B_rem=1.2,
                magnetization="radial", halbach_segments=3,
            )

    def test_radial_with_angles_raises(self):
        with pytest.raises(ValueError, match="halbach_angles_deg must be None"):
            Motor(
                name="h", geometry=default_inrunner(), n_p=4,
                B_rem=1.2,
                halbach_angles_deg=(0.0, -45.0),
            )

    def test_halbach_alpha_p_not_one_raises(self):
        with pytest.raises(ValueError, match="alpha_p=1.0"):
            Motor(
                name="h", geometry=default_inrunner(), n_p=4,
                B_rem=1.2,
                magnetization="halbach", halbach_segments=3,
                alpha_p=0.8,
            )

    def test_halbach_angles_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length"):
            Motor(
                name="h", geometry=default_inrunner(), n_p=4,
                B_rem=1.2,
                magnetization="halbach", halbach_segments=3,
                halbach_angles_deg=(0.0, -45.0),
            )

    def test_bad_magnetization_raises(self):
        with pytest.raises(ValueError, match="magnetization"):
            Motor(
                name="h", geometry=default_inrunner(), n_p=4,
                magnetization="parallel",
            )

    def test_halbach_segments_float_raises(self):
        with pytest.raises(ValueError, match="halbach_segments"):
            Motor(
                name="h", geometry=default_inrunner(), n_p=4,
                B_rem=1.2,
                magnetization="halbach", halbach_segments=2.5,
            )

    def test_halbach_segments_bool_raises(self):
        with pytest.raises(ValueError, match="halbach_segments"):
            Motor(
                name="h", geometry=default_inrunner(), n_p=4,
                B_rem=1.2,
                magnetization="halbach", halbach_segments=True,
            )

    def test_halbach_angles_inf_raises(self):
        import math
        with pytest.raises(ValueError, match="finite"):
            Motor(
                name="h", geometry=default_inrunner(), n_p=4,
                B_rem=1.2,
                magnetization="halbach", halbach_segments=3,
                halbach_angles_deg=(0.0, math.inf, -135.0),
            )

    def test_radial_config_id_unchanged(self):
        """Adding halbach fields must not perturb radial motor config_id."""
        geo = default_inrunner()
        m = Motor(name="m", geometry=geo, n_p=2, R_s=0.2, psi_f=0.1)
        import hashlib
        parts = [geo.config_id, "2", "R_s=0.2", "psi_f=0.1",
                 "alpha_p=1.0", "mu_r_fe=1000.0", "mu_r_pm=1.05"]
        expected = hashlib.md5("|".join(parts).encode()).hexdigest()[:12]
        assert m.config_id == expected

    def test_halbach_config_id_differs_from_radial(self):
        geo = default_inrunner()
        radial = Motor(name="m", geometry=geo, n_p=4, B_rem=1.2)
        halbach = Motor(
            name="m", geometry=geo, n_p=4, B_rem=1.2,
            magnetization="halbach", halbach_segments=3,
        )
        assert radial.config_id != halbach.config_id

    def test_halbach_config_id_with_vs_without_angles(self):
        geo = default_inrunner()
        ideal = Motor(
            name="m", geometry=geo, n_p=4, B_rem=1.2,
            magnetization="halbach", halbach_segments=3,
        )
        custom = Motor(
            name="m", geometry=geo, n_p=4, B_rem=1.2,
            magnetization="halbach", halbach_segments=3,
            halbach_angles_deg=(0.0, -60.0, -120.0),
        )
        assert ideal.config_id != custom.config_id

    def test_halbach_roundtrip(self):
        m = Motor(
            name="h", geometry=default_inrunner(), n_p=10,
            B_rem=1.35,
            magnetization="halbach", halbach_segments=3,
            halbach_angles_deg=(0.0, -45.0, -135.0),
        )
        m2 = Motor.from_dict(m.to_dict())
        assert m2.magnetization == "halbach"
        assert m2.halbach_segments == 3
        assert m2.halbach_angles_deg == (0.0, -45.0, -135.0)
        assert m2.config_id == m.config_id

    def test_halbach_roundtrip_ideal(self):
        m = Motor(
            name="h", geometry=default_inrunner(), n_p=4,
            B_rem=1.2,
            magnetization="halbach", halbach_segments=4,
        )
        m2 = Motor.from_dict(m.to_dict())
        assert m2.magnetization == "halbach"
        assert m2.halbach_segments == 4
        assert m2.halbach_angles_deg is None
        assert m2.config_id == m.config_id

    def test_radial_roundtrip_omits_halbach_keys(self):
        m = Motor(name="m", geometry=default_inrunner(), n_p=2)
        d = m.to_dict()
        assert "magnetization" not in d
        assert "halbach_segments" not in d
        assert "halbach_angles_deg" not in d
