"""Tests for Geometry dataclass and factory functions."""

import math

import pytest

from phasesweep.geometry import (
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
