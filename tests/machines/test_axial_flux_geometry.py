"""Tests for AxialFluxGeometry dataclass and factory functions."""

import math
from pathlib import Path

import pytest

from phasesweep.machines.geometry import (
    AxialFluxGeometry,
    axial_flux,
    axial_flux_geometry_from_toml,
    geometry_from_toml,
)

REPO_ROOT = Path(__file__).parent.parent.parent


class TestAxialFluxConstruction:

    def test_valid_1r2s(self):
        geo = axial_flux(
            variant="1R2S", r_out=0.164, r_in=0.0985,
            airgap=0.0015, magnet_thickness=0.004,
            rotor_yoke=0.020, stator_yoke=0.020,
        )
        assert geo.variant == "1R2S"
        assert geo.r_mean == pytest.approx(0.1313, abs=1e-4)
        assert geo.active_length == pytest.approx(0.0655)
        assert geo.k_D == pytest.approx(0.0985 / 0.164)

    def test_valid_2r1s_nn(self):
        geo = axial_flux(
            variant="2R1S-NN", r_out=0.120, r_in=0.0695,
            airgap=0.002, magnet_thickness=0.012,
            rotor_yoke=0.009, stator_yoke=0.041,
        )
        assert geo.variant == "2R1S-NN"

    def test_valid_2r1s_ns(self):
        geo = axial_flux(
            variant="2R1S-NS", r_out=0.120, r_in=0.0695,
            airgap=0.002, magnet_thickness=0.012,
            rotor_yoke=0.009, stator_yoke=0.041,
        )
        assert geo.variant == "2R1S-NS"

    def test_valid_dssr(self):
        geo = axial_flux(
            variant="DSSR", r_out=0.1225, r_in=0.070,
            airgap=0.001, magnet_thickness=0.012,
            rotor_yoke=0.0, stator_yoke=0.020,
        )
        assert geo.variant == "DSSR"

    def test_valid_dssr_with_magnetic_rotor(self):
        geo = axial_flux(
            variant="DSSR", r_out=0.1225, r_in=0.070,
            airgap=0.001, magnet_thickness=0.012,
            rotor_yoke=0.008, stator_yoke=0.020,
        )
        assert geo.rotor_yoke == 0.008

    def test_valid_coreless(self):
        geo = axial_flux(
            variant="coreless", r_out=0.158, r_in=0.060,
            airgap=0.003, magnet_thickness=0.010,
            rotor_yoke=0.012, stator_yoke=0.0,
        )
        assert geo.variant == "coreless"
        assert geo.n_slots == 0
        assert geo.rotor_yoke == 0.012

class TestAxialFluxValidation:

    def test_r_in_ge_r_out_raises(self):
        with pytest.raises(ValueError, match="r_in.*must be < r_out"):
            axial_flux(
                variant="1R2S", r_out=0.10, r_in=0.15,
                airgap=0.001, magnet_thickness=0.004,
                rotor_yoke=0.020, stator_yoke=0.020,
            )

    def test_r_out_zero_raises(self):
        with pytest.raises(ValueError, match="r_out must be > 0"):
            axial_flux(
                variant="1R2S", r_out=0.0, r_in=0.05,
                airgap=0.001, magnet_thickness=0.004,
                rotor_yoke=0.020, stator_yoke=0.020,
            )

    def test_r_in_zero_raises(self):
        with pytest.raises(ValueError, match="r_in must be > 0"):
            axial_flux(
                variant="1R2S", r_out=0.164, r_in=0.0,
                airgap=0.001, magnet_thickness=0.004,
                rotor_yoke=0.020, stator_yoke=0.020,
            )

    def test_airgap_zero_raises(self):
        with pytest.raises(ValueError, match="airgap must be > 0"):
            axial_flux(
                variant="1R2S", r_out=0.164, r_in=0.0985,
                airgap=0.0, magnet_thickness=0.004,
                rotor_yoke=0.020, stator_yoke=0.020,
            )

    def test_magnet_thickness_zero_raises(self):
        with pytest.raises(ValueError, match="magnet_thickness must be > 0"):
            axial_flux(
                variant="1R2S", r_out=0.164, r_in=0.0985,
                airgap=0.001, magnet_thickness=0.0,
                rotor_yoke=0.020, stator_yoke=0.020,
            )

    def test_unknown_variant_raises(self):
        with pytest.raises(ValueError, match="Unknown axial-flux variant"):
            axial_flux(
                variant="TORUS", r_out=0.164, r_in=0.0985,
                airgap=0.001, magnet_thickness=0.004,
                rotor_yoke=0.020, stator_yoke=0.020,
            )

    def test_typo_variant_raises(self):
        with pytest.raises(ValueError, match="Unknown axial-flux variant"):
            axial_flux(
                variant="2R1S_NN", r_out=0.120, r_in=0.0695,
                airgap=0.002, magnet_thickness=0.012,
                rotor_yoke=0.009, stator_yoke=0.041,
            )

    def test_negative_rotor_yoke_raises(self):
        with pytest.raises(ValueError, match="rotor_yoke must be >= 0"):
            axial_flux(
                variant="DSSR", r_out=0.12, r_in=0.07,
                airgap=0.001, magnet_thickness=0.012,
                rotor_yoke=-0.001, stator_yoke=0.020,
            )

    def test_negative_stator_yoke_raises(self):
        with pytest.raises(ValueError, match="stator_yoke must be >= 0"):
            axial_flux(
                variant="1R2S", r_out=0.164, r_in=0.0985,
                airgap=0.001, magnet_thickness=0.004,
                rotor_yoke=0.020, stator_yoke=-0.001,
            )

    def test_negative_n_slots_raises(self):
        with pytest.raises(ValueError, match="n_slots must be >= 0"):
            axial_flux(
                variant="1R2S", r_out=0.164, r_in=0.0985,
                airgap=0.001, magnet_thickness=0.004,
                rotor_yoke=0.020, stator_yoke=0.020,
                n_slots=-1,
            )

    @pytest.mark.parametrize("field, val", [
        ("r_out", float("nan")),
        ("r_out", float("inf")),
        ("r_in", float("nan")),
        ("r_in", float("inf")),
        ("airgap", float("nan")),
        ("airgap", float("inf")),
        ("magnet_thickness", float("nan")),
        ("magnet_thickness", float("inf")),
        ("rotor_yoke", float("nan")),
        ("rotor_yoke", float("inf")),
        ("stator_yoke", float("nan")),
        ("stator_yoke", float("inf")),
    ])
    def test_nan_inf_fields_rejected(self, field, val):
        kw = dict(variant="DSSR", r_out=0.12, r_in=0.07,
                  airgap=0.001, magnet_thickness=0.012,
                  rotor_yoke=0.0, stator_yoke=0.020)
        kw[field] = val
        with pytest.raises(ValueError, match="finite"):
            axial_flux(**kw)

    def test_valid_variants_not_in_init(self):
        """_VALID_VARIANTS must be ClassVar, not a dataclass field."""
        import dataclasses
        field_names = [f.name for f in dataclasses.fields(AxialFluxGeometry)]
        assert "_VALID_VARIANTS" not in field_names

    def test_valid_variants_not_in_asdict(self):
        import dataclasses
        geo = axial_flux(
            variant="1R2S", r_out=0.164, r_in=0.0985,
            airgap=0.0015, magnet_thickness=0.004,
            rotor_yoke=0.020, stator_yoke=0.020,
        )
        d = dataclasses.asdict(geo)
        assert "_VALID_VARIANTS" not in d


class TestAxialFluxVariantConstraints:

    def test_coreless_stator_yoke_nonzero_raises(self):
        with pytest.raises(ValueError, match="coreless.*stator_yoke=0"):
            axial_flux(
                variant="coreless", r_out=0.158, r_in=0.060,
                airgap=0.003, magnet_thickness=0.010,
                rotor_yoke=0.012, stator_yoke=0.005,
            )

    def test_coreless_with_slots_raises(self):
        with pytest.raises(ValueError, match="coreless.*cannot have slotted"):
            axial_flux(
                variant="coreless", r_out=0.158, r_in=0.060,
                airgap=0.003, magnet_thickness=0.010,
                rotor_yoke=0.012, stator_yoke=0.0,
                n_slots=12,
            )

    def test_dssr_stator_yoke_zero_raises(self):
        with pytest.raises(ValueError, match="DSSR.*stator_yoke > 0"):
            axial_flux(
                variant="DSSR", r_out=0.1225, r_in=0.070,
                airgap=0.001, magnet_thickness=0.012,
                rotor_yoke=0.0, stator_yoke=0.0,
            )

    def test_1r2s_rotor_yoke_zero_raises(self):
        with pytest.raises(ValueError, match="1R2S.*rotor_yoke > 0"):
            axial_flux(
                variant="1R2S", r_out=0.164, r_in=0.0985,
                airgap=0.0015, magnet_thickness=0.004,
                rotor_yoke=0.0, stator_yoke=0.020,
            )

    def test_1r2s_stator_yoke_zero_raises(self):
        with pytest.raises(ValueError, match="1R2S.*stator_yoke > 0"):
            axial_flux(
                variant="1R2S", r_out=0.164, r_in=0.0985,
                airgap=0.0015, magnet_thickness=0.004,
                rotor_yoke=0.020, stator_yoke=0.0,
            )

    def test_2r1s_nn_rotor_yoke_zero_raises(self):
        with pytest.raises(ValueError, match="2R1S-NN.*rotor_yoke > 0"):
            axial_flux(
                variant="2R1S-NN", r_out=0.120, r_in=0.0695,
                airgap=0.002, magnet_thickness=0.012,
                rotor_yoke=0.0, stator_yoke=0.041,
            )

    def test_2r1s_nn_stator_yoke_zero_raises(self):
        with pytest.raises(ValueError, match="2R1S-NN.*stator_yoke > 0"):
            axial_flux(
                variant="2R1S-NN", r_out=0.120, r_in=0.0695,
                airgap=0.002, magnet_thickness=0.012,
                rotor_yoke=0.009, stator_yoke=0.0,
            )

    def test_2r1s_ns_rotor_yoke_zero_raises(self):
        with pytest.raises(ValueError, match="2R1S-NS.*rotor_yoke > 0"):
            axial_flux(
                variant="2R1S-NS", r_out=0.120, r_in=0.0695,
                airgap=0.002, magnet_thickness=0.012,
                rotor_yoke=0.0, stator_yoke=0.041,
            )


class TestAxialFluxSlots:

    def _slotted(self, **kw):
        defaults = dict(
            variant="1R2S", r_out=0.164, r_in=0.0985,
            airgap=0.0015, magnet_thickness=0.004,
            rotor_yoke=0.020, stator_yoke=0.020,
            n_slots=36, slot_depth=0.010,
        )
        return axial_flux(**(defaults | kw))

    def test_valid_slotted(self):
        geo = self._slotted()
        assert geo.n_slots == 36

    def test_slot_depth_exceeds_yoke_raises(self):
        with pytest.raises(ValueError, match="slot_depth.*exceeds stator_yoke"):
            self._slotted(slot_depth=0.025)  # > stator_yoke=0.020

    def test_slot_opening_exceeds_pitch_raises(self):
        geo = self._slotted()
        pitch = 2 * math.pi * geo.r_mean / 36
        with pytest.raises(ValueError, match="slot pitch"):
            self._slotted(slot_opening_width=pitch * 1.1)

    @pytest.mark.parametrize("field, val", [
        ("slot_depth", float("nan")),
        ("slot_depth", float("inf")),
        ("slot_opening_width", float("nan")),
        ("slot_opening_width", float("inf")),
    ])
    def test_nan_inf_slot_fields_rejected(self, field, val):
        with pytest.raises(ValueError, match="finite"):
            self._slotted(**{field: val})

    def test_slot_opening_ratio_derived(self):
        geo = self._slotted(slot_opening_width=0.005)
        pitch = 2 * math.pi * geo.r_mean / 36
        assert geo.slot_opening_ratio == pytest.approx(0.005 / pitch)


class TestAxialFluxConfigId:

    def _base(self, **kw):
        defaults = dict(
            variant="1R2S", r_out=0.164, r_in=0.0985,
            airgap=0.0015, magnet_thickness=0.004,
            rotor_yoke=0.020, stator_yoke=0.020,
        )
        return axial_flux(**(defaults | kw))

    def test_determinism(self):
        assert self._base().config_id == self._base().config_id

    def test_uniqueness_variant(self):
        g1 = axial_flux(
            variant="2R1S-NN", r_out=0.120, r_in=0.0695,
            airgap=0.002, magnet_thickness=0.012,
            rotor_yoke=0.009, stator_yoke=0.041,
        )
        g2 = axial_flux(
            variant="2R1S-NS", r_out=0.120, r_in=0.0695,
            airgap=0.002, magnet_thickness=0.012,
            rotor_yoke=0.009, stator_yoke=0.041,
        )
        assert g1.config_id != g2.config_id

    def test_uniqueness_dimension(self):
        g1 = self._base(airgap=0.0015)
        g2 = self._base(airgap=0.0020)
        assert g1.config_id != g2.config_id

    def test_is_12_chars(self):
        assert len(self._base().config_id) == 12

    def test_axial_hash_pinned_unslotted(self):
        assert self._base().config_id == "ce29a30f50f3"

    def test_axial_hash_pinned_slotted(self):
        geo = axial_flux(
            variant="2R1S-NN", r_out=0.120, r_in=0.0695,
            airgap=0.002, magnet_thickness=0.012,
            rotor_yoke=0.009, stator_yoke=0.041,
            n_slots=18, slot_depth=0.011,
        )
        assert geo.config_id == "35a18ec5d24a"


class TestAxialFluxFromToml:

    def test_mm_keys(self):
        geo = axial_flux_geometry_from_toml({
            "D_out_mm": 328.0, "D_in_mm": 197.0,
            "airgap_mm": 1.5, "magnet_thickness_mm": 4.0,
            "rotor_yoke_mm": 20.0, "stator_yoke_mm": 20.0,
            "variant": "1R2S",
        })
        assert geo.r_out == pytest.approx(0.164)
        assert geo.r_in == pytest.approx(0.0985)
        assert geo.airgap == pytest.approx(0.0015)
        assert geo.variant == "1R2S"

    def test_meter_keys(self):
        geo = axial_flux_geometry_from_toml({
            "r_out": 0.164, "r_in": 0.0985,
            "airgap": 0.0015, "magnet_thickness": 0.004,
            "rotor_yoke": 0.020, "stator_yoke": 0.020,
            "variant": "1R2S",
        })
        assert geo.r_out == pytest.approx(0.164)

    def test_missing_radii_raises(self):
        with pytest.raises(ValueError, match="must contain r_out"):
            axial_flux_geometry_from_toml({"airgap_mm": 1.5, "variant": "1R2S"})

    def test_slot_opening_ratio_conversion(self):
        geo = axial_flux_geometry_from_toml({
            "D_out_mm": 240.0, "D_in_mm": 139.0,
            "airgap_mm": 2.0, "magnet_thickness_mm": 12.0,
            "rotor_yoke_mm": 9.0, "stator_yoke_mm": 41.0,
            "variant": "2R1S-NN",
            "n_slots": 18, "slot_depth_mm": 11.0,
            "slot_opening_ratio": 0.3,
        })
        r_mean = (0.120 + 0.0695) / 2
        pitch = 2 * math.pi * r_mean / 18
        assert geo.slot_opening_width == pytest.approx(0.3 * pitch)


class TestGeometryFromTomlDispatch:

    def test_axial_dispatch(self):
        geo = geometry_from_toml({
            "r_out": 0.164, "r_in": 0.0985,
            "airgap": 0.0015, "magnet_thickness": 0.004,
            "rotor_yoke": 0.020, "stator_yoke": 0.020,
            "variant": "1R2S",
        }, "axial")
        assert isinstance(geo, AxialFluxGeometry)
        assert geo.variant == "1R2S"

    def test_inrunner_dispatch_unchanged(self):
        from phasesweep.machines.geometry import Geometry
        geo = geometry_from_toml({
            "r_outer": 1.0, "r_stator": 0.70, "r_magnet": 0.64,
            "r_rotor": 0.30,
        }, "inrunner")
        assert isinstance(geo, Geometry)

    def test_unknown_topology_raises(self):
        with pytest.raises(ValueError, match="Unknown topology"):
            geometry_from_toml({
                "r_outer": 1.0, "r_stator": 0.70, "r_magnet": 0.64,
                "r_rotor": 0.30,
            }, "pancake")


class TestRadialConfigIdUnchanged:
    """Axial-flux additions must not change existing radial config_id hashes."""

    def test_inrunner_hash_stable(self):
        from phasesweep.machines.geometry import inrunner
        geo = inrunner(r_outer=1.0, r_stator=0.70, r_magnet=0.64, r_rotor=0.30)
        assert geo.config_id == "9e9c111fa7c7"

    def test_outrunner_hash_stable(self):
        from phasesweep.machines.geometry import outrunner
        geo = outrunner(
            r_outer=0.80, r_rotor=0.70, r_magnet=0.64,
            r_stator=0.50, r_inner=0.10,
        )
        assert geo.config_id == "148c3b70ddaf"


class TestRadialSolverRejectsAxial:

    def test_prepare_analytical_rejects_axial(self):
        from phasesweep.machines.motor import Motor
        from phasesweep.solvers.analytical import prepare_analytical

        geo = axial_flux(
            variant="1R2S", r_out=0.164, r_in=0.0985,
            airgap=0.0015, magnet_thickness=0.004,
            rotor_yoke=0.020, stator_yoke=0.020,
        )
        motor = Motor(name="test_axial", geometry=geo, n_p=6, B_rem=1.1)
        with pytest.raises(ValueError, match="radial-flux Geometry.*AxialFluxGeometry"):
            prepare_analytical(motor)

    def test_prepare_fem_rejects_axial(self):
        from phasesweep.machines.motor import Motor
        from phasesweep.solvers.fem_runner import prepare_fem

        geo = axial_flux(
            variant="1R2S", r_out=0.164, r_in=0.0985,
            airgap=0.0015, magnet_thickness=0.004,
            rotor_yoke=0.020, stator_yoke=0.020,
        )
        motor = Motor(name="test_axial", geometry=geo, n_p=6, B_rem=1.1)
        with pytest.raises(ValueError, match="radial-flux Geometry.*AxialFluxGeometry"):
            prepare_fem(motor)


class TestMotorWithAxialGeometry:

    def test_motor_accepts_axial_geometry(self):
        from phasesweep.machines.motor import Motor

        geo = axial_flux(
            variant="1R2S", r_out=0.164, r_in=0.0985,
            airgap=0.0015, magnet_thickness=0.004,
            rotor_yoke=0.020, stator_yoke=0.020,
        )
        motor = Motor(name="parviainen", geometry=geo, n_p=6, psi_f=1.5831)
        assert motor.geometry is geo
        assert motor.config_id

    def test_motor_round_trip(self):
        from phasesweep.machines.motor import Motor

        geo = axial_flux(
            variant="2R1S-NN", r_out=0.120, r_in=0.0695,
            airgap=0.002, magnet_thickness=0.012,
            rotor_yoke=0.009, stator_yoke=0.041,
            n_slots=18, slot_depth=0.011,
        )
        motor = Motor(name="mahmoudi", geometry=geo, n_p=3, psi_f=0.3331)
        d = motor.to_dict()
        motor2 = Motor.from_dict(d)
        assert isinstance(motor2.geometry, AxialFluxGeometry)
        assert motor2.geometry.variant == "2R1S-NN"
        assert motor2.config_id == motor.config_id

    def test_motor_json_round_trip(self):
        import json

        from phasesweep.machines.motor import Motor

        geo = axial_flux(
            variant="1R2S", r_out=0.164, r_in=0.0985,
            airgap=0.0015, magnet_thickness=0.004,
            rotor_yoke=0.020, stator_yoke=0.020,
        )
        motor = Motor(name="test_json", geometry=geo, n_p=6, psi_f=1.5)
        d = motor.to_dict()
        js = json.dumps(d)
        motor2 = Motor.from_dict(json.loads(js))
        assert isinstance(motor2.geometry, AxialFluxGeometry)
        assert motor2.config_id == motor.config_id


class TestIntegrationLoadMotor:
    """Integration tests loading actual anchor TOMLs through the production path."""

    def test_load_parviainen(self):
        from phasesweep.machines.configs import load_motor
        m = load_motor(REPO_ROOT / "data/parviainen_afpm/parviainen_5kw_afpm.toml")
        assert isinstance(m.geometry, AxialFluxGeometry)
        assert m.geometry.variant == "1R2S"
        assert m.geometry.r_out == pytest.approx(0.164)
        assert m.geometry.r_in == pytest.approx(0.0985)
        assert m.n_p == 6

    def test_load_mahmoudi(self):
        from phasesweep.machines.configs import load_motor
        m = load_motor(REPO_ROOT / "data/mahmoudi_torus/mahmoudi_10kw_torus.toml")
        assert isinstance(m.geometry, AxialFluxGeometry)
        assert m.geometry.variant == "2R1S-NN"
        assert m.geometry.r_out == pytest.approx(0.120)
        assert m.geometry.n_slots == 18

    def test_load_emrax_circuit_only(self):
        from phasesweep.machines.configs import load_motor
        m = load_motor(REPO_ROOT / "data/emrax_228/emrax_228_mv.toml")
        assert m.geometry is None
