"""Tests for Geometry dataclass and factory functions."""

import math

import pytest

from phasesweep.machines.geometry import (
    default_inrunner,
    geometry_from_toml,
    inrunner,
    outrunner,
)


class TestInrunnerConstruction:

    def test_valid_inrunner(self):
        geo = inrunner(r_outer=1.0, r_stator=0.70, r_magnet=0.64, r_rotor=0.30)
        assert geo.topology == "inrunner"
        assert geo.r_inner == 0.0

    def test_default_r_ag(self):
        geo = inrunner(r_outer=1.0, r_stator=0.70, r_magnet=0.64, r_rotor=0.30)
        assert geo.r_ag == pytest.approx((0.70 + 0.64) / 2)

    def test_custom_r_ag(self):
        geo = inrunner(r_outer=1.0, r_stator=0.70, r_magnet=0.64, r_rotor=0.30, r_ag=0.68)
        assert geo.r_ag == 0.68

    def test_radii_ordering_violation(self):
        with pytest.raises(ValueError, match="ordering violated"):
            inrunner(r_outer=0.5, r_stator=0.70, r_magnet=0.64, r_rotor=0.30)

    def test_r_ag_outside_airgap(self):
        with pytest.raises(ValueError, match="r_ag"):
            inrunner(r_outer=1.0, r_stator=0.70, r_magnet=0.64, r_rotor=0.30, r_ag=0.75)

    def test_frozen(self):
        geo = default_inrunner()
        with pytest.raises(AttributeError):
            geo.r_outer = 2.0  # type: ignore[misc]


class TestOutrunnerConstruction:

    def test_valid_outrunner(self):
        geo = outrunner(r_outer=0.80, r_rotor=0.70, r_magnet=0.64, r_stator=0.50, r_inner=0.10)
        assert geo.topology == "outrunner"

    def test_r_inner_zero_raises(self):
        with pytest.raises(ValueError, match="r_inner > 0"):
            outrunner(r_outer=0.80, r_rotor=0.70, r_magnet=0.64, r_stator=0.50, r_inner=0.0)

    def test_radii_ordering_violation(self):
        with pytest.raises(ValueError, match="ordering violated"):
            outrunner(r_outer=0.80, r_rotor=0.50, r_magnet=0.64, r_stator=0.60, r_inner=0.10)

    def test_r_ag_outside_airgap(self):
        with pytest.raises(ValueError, match="r_ag"):
            outrunner(r_outer=0.80, r_rotor=0.70, r_magnet=0.64, r_stator=0.50, r_inner=0.10, r_ag=0.40)


class TestBackIronThickness:

    def _base(self, **kw):
        return outrunner(r_outer=0.80, r_rotor=0.70, r_magnet=0.64,
                         r_stator=0.50, r_inner=0.10, **kw)

    def test_default_none(self):
        assert self._base().back_iron_thickness is None

    def test_valid_split(self):
        geo = self._base(back_iron_thickness=0.05)
        assert geo.back_iron_thickness == 0.05

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="back_iron_thickness"):
            self._base(back_iron_thickness=0.0)

    def test_exceeds_wall_raises(self):
        # wall = r_outer - r_rotor = 0.10
        with pytest.raises(ValueError, match="back_iron_thickness"):
            self._base(back_iron_thickness=0.15)

    def test_inrunner_raises(self):
        # the __post_init__ guard (factory doesn't expose the kwarg)
        from phasesweep.machines.geometry import Geometry
        with pytest.raises(ValueError, match="outrunner-only"):
            Geometry(topology="inrunner", r_outer=1.0, r_stator=0.70,
                     r_magnet=0.64, r_rotor=0.30, r_inner=0.0, r_ag=0.67,
                     back_iron_thickness=0.05)

    def test_config_id_unchanged_when_none(self):
        # back-compat: existing run-IDs must not move
        g_none = self._base()
        g_explicit_none = self._base(back_iron_thickness=None)
        assert g_none.config_id == g_explicit_none.config_id

    def test_config_id_hashes_thickness(self):
        g1 = self._base(back_iron_thickness=0.04)
        g2 = self._base(back_iron_thickness=0.05)
        assert g1.config_id != g2.config_id
        assert g1.config_id != self._base().config_id

    def test_from_toml(self):
        geo = geometry_from_toml(
            {"r_outer": 0.80, "r_rotor": 0.70, "r_magnet": 0.64,
             "r_stator": 0.50, "r_inner": 0.10, "back_iron_thickness": 0.05},
            "outrunner",
        )
        assert geo.back_iron_thickness == 0.05

    def test_from_toml_inrunner_rejected_loudly(self):
        # previously the key was silently dropped for inrunners
        with pytest.raises(ValueError, match="outrunner-only"):
            geometry_from_toml(
                {"r_outer": 0.05, "r_stator": 0.03, "r_magnet": 0.025,
                 "r_rotor": 0.02, "back_iron_thickness": 0.005},
                "inrunner",
            )

    def test_circle_spec_splits_yoke(self):
        from phasesweep.solvers.fem_field import _circle_spec
        plain = [n for _, n in _circle_spec(self._base())]
        assert plain == ["yoke", "pm", "airgap", "stator", "air"]
        split = _circle_spec(self._base(back_iron_thickness=0.05))
        names = [n for _, n in split]
        assert names == ["shell", "back_iron", "pm", "airgap", "stator", "air"]
        # split radius sits between r_rotor and r_outer
        radii = {n: r for r, n in split}
        assert radii["back_iron"] == pytest.approx(0.75)  # r_rotor + 0.05


class TestConfigId:

    def test_determinism(self):
        g1 = default_inrunner()
        g2 = default_inrunner()
        assert g1.config_id == g2.config_id

    def test_uniqueness(self):
        g1 = inrunner(r_outer=1.0, r_stator=0.70, r_magnet=0.64, r_rotor=0.30)
        g2 = inrunner(r_outer=1.0, r_stator=0.71, r_magnet=0.64, r_rotor=0.30)
        assert g1.config_id != g2.config_id

    def test_is_12_chars(self):
        assert len(default_inrunner().config_id) == 12


class TestSlotOpeningRatio:

    def test_derived_from_width(self):
        geo = inrunner(r_outer=1.0, r_stator=0.70, r_magnet=0.64, r_rotor=0.30,
                       n_slots=12, slot_depth=0.05, slot_opening_width=0.055)
        slot_pitch = 2 * math.pi * 0.70 / 12
        assert geo.slot_opening_ratio == pytest.approx(0.055 / slot_pitch)

    def test_zero_without_slots(self):
        geo = inrunner(r_outer=1.0, r_stator=0.70, r_magnet=0.64, r_rotor=0.30,
                       slot_opening_width=0.055)
        assert geo.slot_opening_ratio == 0.0

    def test_zero_without_width(self):
        geo = inrunner(r_outer=1.0, r_stator=0.70, r_magnet=0.64, r_rotor=0.30,
                       n_slots=12, slot_depth=0.05)
        assert geo.slot_opening_ratio == 0.0

    def test_config_id_hashes_width(self):
        kw = dict(r_outer=1.0, r_stator=0.70, r_magnet=0.64, r_rotor=0.30,
                  n_slots=12, slot_depth=0.05)
        g1 = inrunner(**kw, slot_opening_width=0.055)
        g2 = inrunner(**kw, slot_opening_width=0.056)
        assert g1.config_id != g2.config_id


class TestDefaultInrunner:

    def test_matches_legacy_constants(self):
        geo = default_inrunner()
        assert geo.r_outer == 1.0
        assert geo.r_stator == 0.70
        assert geo.r_magnet == 0.64
        assert geo.r_rotor == 0.30
        assert geo.r_inner == 0.0
        assert geo.r_ag == pytest.approx(0.67)


class TestGeometryFromToml:

    def test_spec_names(self):
        geo_dict = {
            "r_outer": 1.0, "r_stator": 0.70, "r_magnet": 0.64,
            "r_rotor": 0.30, "r_inner": 0.0,
        }
        geo = geometry_from_toml(geo_dict, "inrunner")
        assert geo.r_outer == 1.0

    def test_legacy_names(self):
        geo_dict = {
            "stator_od": 0.113, "stator_id": 0.0478, "rotor_od": 0.047,
            "magnet_thickness": 0.00435,
        }
        geo = geometry_from_toml(geo_dict, "inrunner")
        assert geo.r_outer == pytest.approx(0.113 / 2)
        assert geo.r_stator == pytest.approx(0.0478 / 2)
        assert geo.r_magnet == pytest.approx(0.047 / 2)
        assert geo.r_rotor == pytest.approx(0.047 / 2 - 0.00435)

    def test_legacy_names_outrunner_raises(self):
        geo_dict = {
            "stator_od": 0.113, "stator_id": 0.0478, "rotor_od": 0.047,
            "magnet_thickness": 0.00435,
        }
        with pytest.raises(ValueError, match="inrunner-only"):
            geometry_from_toml(geo_dict, "outrunner")

    def test_creator_toml_geometry(self):
        geo_dict = {
            "stator_od": 0.113, "stator_id": 0.0478, "rotor_od": 0.047,
            "magnet_thickness": 0.00435, "L_stk": 0.0301,
        }
        geo = geometry_from_toml(geo_dict, "inrunner")
        assert geo.topology == "inrunner"
        assert geo.r_outer > geo.r_stator > geo.r_magnet > geo.r_rotor

    def test_missing_fields_raises(self):
        with pytest.raises(ValueError, match="must contain"):
            geometry_from_toml({}, "inrunner")

    def test_outrunner_spec_names(self):
        geo_dict = {
            "r_outer": 0.80, "r_stator": 0.50, "r_magnet": 0.64,
            "r_rotor": 0.70, "r_inner": 0.10,
        }
        geo = geometry_from_toml(geo_dict, "outrunner")
        assert geo.topology == "outrunner"
        assert geo.r_inner == 0.10
        assert geo.r_outer > geo.r_rotor > geo.r_magnet > geo.r_stator > geo.r_inner

    def test_unknown_topology_raises(self):
        geo_dict = {
            "r_outer": 1.0, "r_stator": 0.70, "r_magnet": 0.64,
            "r_rotor": 0.30, "r_inner": 0.0,
        }
        with pytest.raises(ValueError, match="Unknown topology"):
            geometry_from_toml(geo_dict, "Outrunner")

    def test_outrunner_missing_r_inner_raises(self):
        geo_dict = {
            "r_outer": 0.80, "r_stator": 0.50, "r_magnet": 0.64,
            "r_rotor": 0.70,
        }
        with pytest.raises(ValueError, match="r_inner > 0"):
            geometry_from_toml(geo_dict, "outrunner")


class TestSlotInvariants:
    """Slot-field validation: unphysical slot configs must be rejected."""

    def _base(self, **kw):
        return dict(r_outer=1.0, r_stator=0.70, r_magnet=0.64, r_rotor=0.30) | kw

    def test_negative_slot_depth_raises(self):
        with pytest.raises(ValueError, match="slot_depth"):
            inrunner(**self._base(n_slots=9, slot_depth=-0.01))

    def test_slot_punch_through_inrunner_raises(self):
        with pytest.raises(ValueError, match="punches through"):
            inrunner(**self._base(n_slots=9, slot_depth=0.35))

    def test_slot_punch_through_outrunner_raises(self):
        with pytest.raises(ValueError, match="punches through"):
            outrunner(r_outer=0.80, r_rotor=0.70, r_magnet=0.64,
                      r_stator=0.50, r_inner=0.10, n_slots=9, slot_depth=0.45)

    def test_slot_width_ratio_bounds(self):
        with pytest.raises(ValueError, match="slot_width_ratio"):
            inrunner(**self._base(n_slots=9, slot_depth=0.1, slot_width_ratio=1.5))
        with pytest.raises(ValueError, match="slot_width_ratio"):
            inrunner(**self._base(n_slots=9, slot_depth=0.1, slot_width_ratio=0.0))

    def test_slot_opening_exceeds_pitch_raises(self):
        pitch = 2 * math.pi * 0.70 / 9
        with pytest.raises(ValueError, match="slot pitch"):
            inrunner(**self._base(n_slots=9, slot_depth=0.1,
                                  slot_opening_width=pitch * 1.1))

    def test_zero_slot_depth_with_slots_legal(self):
        geo = inrunner(**self._base(n_slots=9, slot_depth=0.0))
        assert geo.slot_opening_ratio == 0.0

    def test_smooth_bore_ignores_slot_fields(self):
        geo = inrunner(**self._base(n_slots=0, slot_depth=-1.0, slot_width_ratio=5.0))
        assert geo.n_slots == 0
