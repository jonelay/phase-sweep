"""Tests for model registry structure and analytical runner."""

import dataclasses

import pytest

from phasesweep.geometry import inrunner
from phasesweep.motor import DriveParams
from phasesweep.registry import MODEL_REGISTRY, ModelInfo, _run_analytical_impl
from phasesweep.sweep_types import RunConfig, compute_run_id
from tests.conftest import make_motor

# ---------------------------------------------------------------------------
# Registry structure
# ---------------------------------------------------------------------------

class TestRegistryStructure:

    def test_ten_entries(self):
        assert set(MODEL_REGISTRY.keys()) == {
            "analytical", "fem", "drive_sim", "rated_torque", "stall_torque",
            "backemf_capture", "inductance_test", "resistance_test",
            "torque_test", "airgap_flux_test",
        }

    def test_all_entries_are_model_info(self):
        for key, info in MODEL_REGISTRY.items():
            assert isinstance(info, ModelInfo), f"{key} is not ModelInfo"

    def test_all_computed_entries_have_fn(self):
        for key, info in MODEL_REGISTRY.items():
            if info.source == "computed":
                assert info.fn is not None, f"{key} missing fn"

    def test_all_entries_have_produces(self):
        for key, info in MODEL_REGISTRY.items():
            assert len(info.produces) > 0, f"{key} has empty produces"

    def test_all_entries_have_hash_fields(self):
        for key, info in MODEL_REGISTRY.items():
            assert isinstance(info.hash_fields, frozenset), f"{key} hash_fields not frozenset"

    def test_name_matches_key(self):
        for key, info in MODEL_REGISTRY.items():
            assert info.name == key

    def test_cost_values(self):
        assert MODEL_REGISTRY["analytical"].cost == "fast"
        assert MODEL_REGISTRY["fem"].cost == "slow"
        assert MODEL_REGISTRY["drive_sim"].cost == "medium"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:

    def test_analytical_valid_with_b_rem(self):
        motor = make_motor(B_rem=1.2, psi_f=None)
        MODEL_REGISTRY["analytical"].validate(motor)

    def test_analytical_valid_with_psi_f_derivation(self):
        motor = make_motor()
        MODEL_REGISTRY["analytical"].validate(motor)

    def test_analytical_rejects_missing_b_rem_and_derivation(self):
        motor = make_motor(psi_f=None)
        with pytest.raises(ValueError, match="cannot determine B_rem"):
            MODEL_REGISTRY["analytical"].validate(motor)

    def test_fem_valid(self):
        motor = make_motor()
        MODEL_REGISTRY["fem"].validate(motor)

    def test_drive_sim_valid(self):
        motor = make_motor()
        MODEL_REGISTRY["drive_sim"].validate(motor)

    def test_drive_sim_rejects_missing_fields(self):
        motor = make_motor(R_s=None, L_d=None)
        with pytest.raises(ValueError, match="R_s, L_d"):
            MODEL_REGISTRY["drive_sim"].validate(motor)


# ---------------------------------------------------------------------------
# Analytical runner
# ---------------------------------------------------------------------------

class TestAnalyticalRunner:

    def test_returns_expected_keys(self):
        rc = RunConfig(motor=make_motor(), model="analytical")
        result = _run_analytical_impl(rc)
        assert "theta_list" in result
        assert "B_r_list" in result
        assert "fundamental" in result
        assert "thd_pct" in result

    def test_fundamental_positive(self):
        rc = RunConfig(motor=make_motor(), model="analytical", n_theta=360)
        result = _run_analytical_impl(rc)
        assert result["fundamental"] > 0

    def test_theta_length_matches_n_theta(self):
        rc = RunConfig(motor=make_motor(), model="analytical", n_theta=180)
        result = _run_analytical_impl(rc)
        assert len(result["theta_list"]) == 180
        assert len(result["B_r_list"]) == 180

    def test_backemf_fundamental_present(self):
        rc = RunConfig(motor=make_motor(), model="analytical")
        result = _run_analytical_impl(rc)
        assert result["backemf_fundamental"] is not None
        assert result["backemf_fundamental"] > 0

    def test_backemf_none_without_psi_f_or_winding(self):
        motor = make_motor(psi_f=None, B_rem=0.5, N=None, k_w=None, L_stk=None)
        rc = RunConfig(motor=motor, model="analytical")
        result = _run_analytical_impl(rc)
        assert result["backemf_fundamental"] is None

    def test_backemf_carter_consistent_when_derived(self):
        # No explicit psi_f on a slotted geometry: back-EMF must come from
        # the same Carter-corrected field as flux_linkage_peak
        geo = inrunner(r_outer=1.0, r_stator=0.70, r_magnet=0.64, r_rotor=0.30,
                       n_slots=12, slot_depth=0.05, slot_opening_width=0.055)  # ratio ~0.15
        motor = make_motor(psi_f=None, B_rem=1.2, geometry=geo)
        result = _run_analytical_impl(RunConfig(motor=motor, model="analytical"))
        w_e = motor.drive.W_REF * motor.n_p
        assert result["backemf_fundamental"] == pytest.approx(
            w_e * result["flux_linkage_peak"], rel=1e-12)

    def test_backemf_explicit_psi_f_wins(self):
        motor = make_motor(psi_f=0.123)
        result = _run_analytical_impl(RunConfig(motor=motor, model="analytical"))
        w_e = motor.drive.W_REF * motor.n_p
        assert result["backemf_fundamental"] == pytest.approx(w_e * 0.123)

    def test_flux_linkage_roundtrip_slotted_explicit_psi_f(self):
        # psi_f → derived B_rem → flux_linkage_peak must round-trip exactly
        # on a slotted motor: the inversion and psi_f_carter use the same
        # Carter-adjusted field radii and original-bore winding formula
        # (was −2.06% before the S110 Carter-consistent inversion)
        from phasesweep.configs import load_motor
        from tests.conftest import REPO_ROOT
        motor = load_motor(REPO_ROOT / "motors/actuator_steel_rotor.toml")
        motor = dataclasses.replace(motor, psi_f=0.268e-3, B_rem=None)
        result = _run_analytical_impl(RunConfig(motor=motor, model="analytical"))
        assert result["flux_linkage_peak"] == pytest.approx(0.268e-3, rel=1e-9)

    def test_alpha_p_reduces_fundamental(self):
        from math import pi, sin
        rc_full = RunConfig(motor=make_motor(B_rem=1.2, psi_f=None, alpha_p=1.0), model="analytical")
        rc_part = RunConfig(motor=make_motor(B_rem=1.2, psi_f=None, alpha_p=0.75), model="analytical")
        B1_full = _run_analytical_impl(rc_full)["fundamental"]
        B1_part = _run_analytical_impl(rc_part)["fundamental"]
        assert B1_part < B1_full
        expected_ratio = sin(pi * 0.75 / 2)
        assert B1_part / B1_full == pytest.approx(expected_ratio, rel=1e-6)


# ---------------------------------------------------------------------------
# hash_fields integration with compute_run_id
# ---------------------------------------------------------------------------

class TestHashFieldsIntegration:

    def test_analytical_hash_ignores_fem_params(self):
        motor = make_motor()
        rc1 = RunConfig(motor=motor, model="analytical", maxh_fraction=0.05)
        rc2 = RunConfig(motor=motor, model="analytical", maxh_fraction=0.10)
        fields = MODEL_REGISTRY["analytical"].hash_fields
        assert compute_run_id(rc1, fields) == compute_run_id(rc2, fields)

    def test_fem_hash_sensitive_to_nonlinear(self):
        motor = make_motor()
        rc1 = RunConfig(motor=motor, model="fem", nonlinear=False)
        rc2 = RunConfig(motor=motor, model="fem", nonlinear=True)
        fields = MODEL_REGISTRY["fem"].hash_fields
        assert compute_run_id(rc1, fields) != compute_run_id(rc2, fields)

    def test_drive_sim_hash_sensitive_to_load(self):
        from phasesweep.sim import SimPlan
        motor = make_motor()
        base = SimPlan(
            load_torque=1.0, load_time=0.5, t_stop=1.0, speed_step_time=0.05,
            settle_threshold=0.05, ss_window=0.1, droop_window=0.1,
            accel_window=(0.05, 0.3),
            alpha_s=25.0, alpha_c=1257.0, T_s=125e-6, tau_m=0.1,
        )
        plan2 = dataclasses.replace(base, load_torque=2.0)
        rc1 = RunConfig(motor=motor, model="drive_sim", sim_plan=base)
        rc2 = RunConfig(motor=motor, model="drive_sim", sim_plan=plan2)
        fields = MODEL_REGISTRY["drive_sim"].hash_fields
        assert compute_run_id(rc1, fields) != compute_run_id(rc2, fields)

    def test_drive_sim_hash_ignores_fem_params(self):
        motor = make_motor()
        rc1 = RunConfig(motor=motor, model="drive_sim", maxh_fraction=0.05)
        rc2 = RunConfig(motor=motor, model="drive_sim", maxh_fraction=0.10)
        fields = MODEL_REGISTRY["drive_sim"].hash_fields
        assert compute_run_id(rc1, fields) == compute_run_id(rc2, fields)

    @pytest.mark.parametrize("model", ["drive_sim", "stall_torque", "analytical"])
    def test_drive_variant_changes_run_id(self, model):
        m1 = make_motor()
        m2 = make_motor(drive=DriveParams(U_DC=48.0, MAX_I_S=5.0,
                                          W_REF=500.0, I_LIMIT=3.0))
        rc1 = RunConfig(motor=m1, model=model)
        rc2 = RunConfig(motor=m2, model=model)
        fields = MODEL_REGISTRY[model].hash_fields
        assert compute_run_id(rc1, fields) != compute_run_id(rc2, fields)

    def test_fem_hash_ignores_drive_params(self):
        m1 = make_motor()
        m2 = make_motor(drive=DriveParams(U_DC=48.0, MAX_I_S=5.0,
                                          W_REF=500.0, I_LIMIT=3.0))
        rc1 = RunConfig(motor=m1, model="fem")
        rc2 = RunConfig(motor=m2, model="fem")
        fields = MODEL_REGISTRY["fem"].hash_fields
        assert compute_run_id(rc1, fields) == compute_run_id(rc2, fields)

    def test_model_version_changes_run_id(self, monkeypatch):
        # Model-code version busts the result cache: bumping it must
        # change the run ID with no input change (S110 convention fix)
        rc = RunConfig(motor=make_motor(), model="analytical")
        id_before = compute_run_id(rc)
        info = MODEL_REGISTRY["analytical"]
        monkeypatch.setitem(
            MODEL_REGISTRY, "analytical",
            dataclasses.replace(info, version=info.version + 1),
        )
        assert compute_run_id(rc) != id_before

    def test_computed_models_bumped_to_v2(self):
        # S110 square-wave convention changed outputs of every computed
        # model (derived psi_f / B_rem); their run IDs must not collide
        # with pre-fix cached results
        for key, info in MODEL_REGISTRY.items():
            if info.source == "computed":
                assert info.version >= 2, f"{key} still at version {info.version}"
