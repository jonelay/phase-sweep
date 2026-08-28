"""Tests for model registry structure and analytical runner."""

import dataclasses

import pytest

from phasesweep.machines.geometry import inrunner
from phasesweep.machines.motor import DriveParams
from phasesweep.registry import MODEL_REGISTRY, ModelInfo, _run_analytical_impl
from phasesweep.sweep_types import RunConfig, compute_run_id
from tests.conftest import make_motor, requires_sim

# ---------------------------------------------------------------------------
# Registry structure
# ---------------------------------------------------------------------------

class TestRegistryStructure:

    def test_registered_models(self):
        assert set(MODEL_REGISTRY.keys()) == {
            "analytical", "analytical_axial", "fem", "fem_axial",
            "drive_sim", "drive_sim_two_mass",
            "rated_torque", "stall_torque",
            "thermal_duty", "torque_speed", "iron_loss", "demag_screen",
            "cogging_torque",
            "backemf_capture", "inductance_test", "resistance_test",
            "torque_test", "airgap_flux_test", "iron_loss_test",
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

    def test_axial_validators_accept_axial_and_reject_radial(self):
        from tests.crossval.axial_fem_contract import load_parviainen_motor
        axial = load_parviainen_motor()
        MODEL_REGISTRY["analytical_axial"].validate(axial)
        MODEL_REGISTRY["fem_axial"].validate(axial)
        radial = make_motor(B_rem=1.2, psi_f=None)
        with pytest.raises(ValueError):
            MODEL_REGISTRY["analytical_axial"].validate(radial)
        with pytest.raises(ValueError):
            MODEL_REGISTRY["fem_axial"].validate(radial)


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

    def test_thd_pins_waveform_shape(self):
        """Value pin on the analytical v4 odd-harmonic series.

        Guards the n_p -> fundamental-bin wiring end to end, and is the only
        model-level check that would see a pure scale error in compute_thd.
        Independent of n_theta. Update on an analytical physics
        change, alongside the version bump.
        """
        rc = RunConfig(motor=make_motor(), model="analytical", n_theta=360)
        result = _run_analytical_impl(rc)
        assert result["thd_pct"] == pytest.approx(32.27, abs=0.05)

    def test_fundamental_positive(self):
        rc = RunConfig(motor=make_motor(), model="analytical", n_theta=360)
        result = _run_analytical_impl(rc)
        assert result["fundamental"] > 0

    # The next five tests kill specific mutmut survivors: the runner's
    # explicit arguments to zhu_howe_Br_series all coincide with the
    # series defaults at the standard test motor, so dropping any of
    # them passed silently until pinned off-default here.

    def test_outputs_match_produces(self):
        rc = RunConfig(motor=make_motor(B_rem=1.2, psi_f=None),
                       model="analytical", n_theta=360)
        assert set(_run_analytical_impl(rc)) == \
            MODEL_REGISTRY["analytical"].produces

    def test_samples_at_custom_r_ag(self):
        """geometry.r_ag defaults to midgap — which is also the series'
        internal default, so a dropped r_eval only shows on a custom
        sampling radius (fundamental decays toward the stator)."""
        base = make_motor(B_rem=1.2, psi_f=None)
        geo = base.geometry
        geo_hi = dataclasses.replace(
            geo, r_ag=geo.r_magnet + 0.85 * (geo.r_stator - geo.r_magnet))
        r_mid = _run_analytical_impl(
            RunConfig(motor=base, model="analytical", n_theta=360))
        r_hi = _run_analytical_impl(
            RunConfig(motor=make_motor(B_rem=1.2, psi_f=None,
                                       geometry=geo_hi),
                      model="analytical", n_theta=360))
        assert r_hi["fundamental"] == pytest.approx(0.85933, abs=1e-4)
        assert r_hi["fundamental"] < r_mid["fundamental"]

    def test_passes_mu_r_pm_through(self):
        """Off-default magnet permeability must reach the series (the
        default 1.05 masks a dropped kwarg)."""
        r = _run_analytical_impl(
            RunConfig(motor=make_motor(B_rem=1.2, psi_f=None, mu_r_pm=1.4),
                      model="analytical", n_theta=360))
        assert r["fundamental"] == pytest.approx(0.84163, abs=1e-4)

    def test_series_capped_below_nyquist(self):
        """At n_theta=12 / n_p=2 the uncapped series would alias order 10
        into the fundamental bin (0.936 instead of 0.889) — pin that the
        runner passes max_order = n_theta // 2."""
        r = _run_analytical_impl(
            RunConfig(motor=make_motor(B_rem=1.2, psi_f=None),
                      model="analytical", n_theta=12))
        assert r["fundamental"] == pytest.approx(0.88937, abs=1e-4)

    def test_k_end_heuristic_value_pinned(self):
        """The heuristic k_end feeds the reported (not applied) column:
        pin its value so the g + h_m/mu_r_pm effective-gap formula can't
        drift silently (regression pin at the standard test motor)."""
        r = _run_analytical_impl(
            RunConfig(motor=make_motor(B_rem=1.2, psi_f=None),
                      model="analytical", n_theta=360))
        assert r["k_end"] == pytest.approx(0.179354, abs=1e-5)
        assert r["k_end_source"] == "heuristic"

    def test_k_end_none_without_l_stk(self):
        """No L_stk -> no heuristic: the keys must be None, not ''."""
        r = _run_analytical_impl(
            RunConfig(motor=make_motor(B_rem=1.2, psi_f=None, L_stk=None),
                      model="analytical", n_theta=360))
        assert r["k_end"] is None
        assert r["k_end_source"] is None

    def test_theta_length_matches_n_theta(self):
        rc = RunConfig(motor=make_motor(), model="analytical", n_theta=180)
        result = _run_analytical_impl(rc)
        assert len(result["theta_list"]) == 180
        assert len(result["B_r_list"]) == 180

    def test_n_theta_boundary_rejected(self):
        # n_theta == 2*n_p + 1 passes the FFT-bin bound but makes the
        # series' max_order cutoff drop the k=1 term: previously a silent
        # all-zero waveform stored as OK
        motor = make_motor()
        rc = RunConfig(
            motor=motor, model="analytical", n_theta=2 * motor.n_p + 1)
        with pytest.raises(ValueError, match=r"2\*n_p \+ 2"):
            _run_analytical_impl(rc)
        ok = _run_analytical_impl(RunConfig(
            motor=motor, model="analytical", n_theta=2 * motor.n_p + 2))
        assert ok["fundamental"] > 0

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

    def test_k_end_scales_derived_flux_linkage(self):
        # Computed Motor.k_end multiplies the B_rem-derived flux linkage
        # and the derived back-EMF exactly.
        base = _run_analytical_impl(RunConfig(
            motor=make_motor(psi_f=None, B_rem=1.0), model="analytical"))
        corr = _run_analytical_impl(RunConfig(
            motor=make_motor(psi_f=None, B_rem=1.0, k_end=0.9),
            model="analytical"))
        assert corr["flux_linkage_peak"] == pytest.approx(
            0.9 * base["flux_linkage_peak"], rel=1e-12)
        assert corr["backemf_fundamental"] == pytest.approx(
            0.9 * base["backemf_fundamental"], rel=1e-12)
        assert corr["k_end"] == 0.9
        assert corr["k_end_source"] == "computed"
        assert corr["k_end_applied"] is True

    def test_k_end_not_applied_when_b_rem_derived_from_psi_f(self):
        # B_rem derived from a measured psi_f already includes end
        # effects — applying k_end would double-count (guard).
        base = _run_analytical_impl(RunConfig(
            motor=make_motor(), model="analytical"))  # psi_f=0.1, no B_rem
        corr = _run_analytical_impl(RunConfig(
            motor=make_motor(k_end=0.9), model="analytical"))
        assert corr["flux_linkage_peak"] == pytest.approx(
            base["flux_linkage_peak"], rel=1e-12)
        assert corr["backemf_fundamental"] == pytest.approx(
            base["backemf_fundamental"], rel=1e-12)
        assert corr["k_end_source"] == "computed"
        # a "computed" source with the correction suppressed must say so
        assert corr["k_end_applied"] is False

    def test_k_end_heuristic_reported_not_applied(self):
        # Without Motor.k_end the heuristic stays informational only.
        result = _run_analytical_impl(RunConfig(
            motor=make_motor(psi_f=None, B_rem=1.0), model="analytical"))
        assert result["k_end_source"] == "heuristic"
        assert result["k_end_applied"] is False
        assert 0 < result["k_end"] < 1

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
        # (was −2.06% before the Carter-consistent inversion)
        from phasesweep.machines.configs import load_motor
        from tests.conftest import REPO_ROOT
        motor = load_motor(REPO_ROOT / "motors/outrunner_14mm_steel.toml")
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

class TestAnalyticalAxialRunner:
    """Model-level pins for _run_analytical_axial_impl. The axial series
    itself is covered in test_analytical_axial.py; before these, the
    runner's own wiring (pole pitch, Carter input, leakage/winding
    chain, back-EMF paths, output keys) had no direct assertions — the
    mutmut sweep left 47 survivors here."""

    @pytest.fixture(scope="class")
    def parviainen(self):
        from tests.crossval.axial_fem_contract import load_parviainen_motor
        return load_parviainen_motor()

    @pytest.fixture(scope="class")
    def result(self, parviainen):
        rc = RunConfig(motor=parviainen, model="analytical_axial",
                       n_theta=360)
        from phasesweep.registry import _run_analytical_axial_impl
        return _run_analytical_axial_impl(rc)

    def test_outputs_match_produces(self, result):
        assert set(result) == MODEL_REGISTRY["analytical_axial"].produces

    def test_snapshot_pins(self, result):
        """Regression pins at the Parviainen anchor motor (analytical_axial
        v-current). Each pins one wiring leg: fundamental+peak the theta
        grid and series call, k_c the Carter input, k_sigma_pm the
        leakage factor, flux linkage the GAP_LINKAGE * leakage * winding
        transfer chain. Re-freeze deliberately on physics changes."""
        assert result["fundamental"] == pytest.approx(0.713666, abs=1e-5)
        assert result["peak_Br"] == pytest.approx(0.718797, abs=1e-5)
        assert result["thd_pct"] == pytest.approx(29.846, abs=0.01)
        assert result["k_c"] == pytest.approx(1.173937, abs=1e-5)
        assert result["k_sigma_pm"] == pytest.approx(0.920850, abs=1e-5)
        assert result["flux_linkage_peak"] == pytest.approx(1.581912,
                                                            abs=1e-5)

    def test_theta_grid(self, result):
        theta = result["theta_list"]
        assert len(theta) == 360
        assert theta[0] == 0.0
        assert theta[-1] == pytest.approx(2 * 3.14159265 * 359 / 360,
                                          abs=1e-6)

    def test_series_capped_below_nyquist(self, parviainen):
        """At n_theta=24 / n_p=6 the uncapped series aliases order 30
        onto the fundamental bin (0.71880 instead of 0.71367) — pin that
        the runner passes max_order = n_theta // 2."""
        from phasesweep.registry import _run_analytical_axial_impl
        r = _run_analytical_axial_impl(
            RunConfig(motor=parviainen, model="analytical_axial",
                      n_theta=24))
        assert r["fundamental"] == pytest.approx(0.713666, abs=1e-4)

    def test_n_theta_boundary_rejected(self, parviainen):
        from phasesweep.registry import _run_analytical_axial_impl
        with pytest.raises(ValueError, match="n_theta"):
            _run_analytical_axial_impl(
                RunConfig(motor=parviainen, model="analytical_axial",
                          n_theta=2 * parviainen.n_p + 1))

    def test_n_theta_boundary_accepted(self, parviainen):
        """The reject test pins only the failing side, and both mutations
        of the bound survive it. n_theta = 2*n_p + 2 is the first grid
        that resolves the fundamental, so it has to go through."""
        from phasesweep.registry import _run_analytical_axial_impl
        r = _run_analytical_axial_impl(
            RunConfig(motor=parviainen, model="analytical_axial",
                      n_theta=2 * parviainen.n_p + 2))
        assert r["fundamental"] == pytest.approx(0.713666, abs=1e-5)

    def test_mu_r_pm_reaches_the_series(self, parviainen):
        """The anchor motor's mu_r_pm is 1.05, which is also the series
        default — so dropping the passthrough is invisible at the anchor
        even though mu_r_pm still reaches Carter and the leakage factor.
        Pin an off-default value, the same trap the radial runner had."""
        from phasesweep.registry import _run_analytical_axial_impl
        hot = dataclasses.replace(parviainen, mu_r_pm=1.35)
        r = _run_analytical_axial_impl(
            RunConfig(motor=hot, model="analytical_axial", n_theta=360))
        assert r["fundamental"] == pytest.approx(0.648177, abs=1e-5)

    @pytest.mark.parametrize("missing", ["k_w", "coils_series"])
    def test_flux_linkage_needs_all_three_winding_fields(self, parviainen,
                                                         missing):
        """The winding guard is a three-way and. Drop any one field and
        there is no model-derived flux linkage, and no back-EMF from it —
        loosening the guard reaches None arithmetic instead."""
        from phasesweep.registry import _run_analytical_axial_impl
        m = dataclasses.replace(parviainen, psi_f=None,
                                drive=DriveParams(W_REF=100.0),
                                **{missing: None})
        r = _run_analytical_axial_impl(
            RunConfig(motor=m, model="analytical_axial", n_theta=360))
        assert r["flux_linkage_peak"] is None
        assert r["backemf_fundamental"] is None

    def test_backemf_psi_f_wins_over_flux_linkage(self, parviainen):
        """Both back-EMF paths pinned: explicit psi_f (949.86 = W_REF *
        n_p * psi_f) beats the model-derived flux linkage (949.147);
        with W_REF unset it stays None."""
        from phasesweep.registry import _run_analytical_axial_impl
        with_drive = dataclasses.replace(
            parviainen, drive=DriveParams(W_REF=100.0))
        r1 = _run_analytical_axial_impl(
            RunConfig(motor=with_drive, model="analytical_axial",
                      n_theta=360))
        assert r1["backemf_fundamental"] == pytest.approx(949.86, abs=0.01)
        no_psi = dataclasses.replace(with_drive, psi_f=None)
        r2 = _run_analytical_axial_impl(
            RunConfig(motor=no_psi, model="analytical_axial", n_theta=360))
        assert r2["backemf_fundamental"] == pytest.approx(949.147, abs=0.01)
        assert (_run_analytical_axial_impl(
            RunConfig(motor=parviainen, model="analytical_axial",
                      n_theta=360))["backemf_fundamental"] is None)


class TestHashFieldsIntegration:

    @requires_sim
    def test_every_runconfig_field_reaches_the_hash(self):
        # Guard against silent run-ID collisions: a RunConfig field added
        # without a matching entry in compute_run_id's solver_params dict
        # would silently drop out of hashing. An unregistered model skips
        # the hash_fields filter, so every solver param must move the ID.
        from phasesweep.simulation.params import TwoMassLoad
        from phasesweep.simulation.sim import SimPlan
        perturb = {
            "maxh_fraction": 0.123,
            "n_theta": 721,
            "nonlinear": True,
            "nonlinear_solver": "picard",
            "j_s": 123.0,
            "i_fault": 0.42,
            "sim_plan": SimPlan(
                load_torque=1.0, load_time=0.5, t_stop=2.0,
                speed_step_time=0.1, settle_threshold=0.05,
                ss_window=0.2, droop_window=0.3, accel_window=(0.1, 0.5),
                alpha_s=25.0, alpha_c=1250.0, T_s=125e-6, tau_m=0.02,
            ),
            "duty_profile": ((1.0, 2.0),),
            "dataset_id": "guard",
            "load_mech": TwoMassLoad(J_L=0.005, K_S=700.0, C_S=0.01),
            "rotation": 0.123,
            "n_planes": 5,
            "cogging_points": 24,
            "slip": 0.05,
        }
        solver_fields = {
            f.name for f in dataclasses.fields(RunConfig)
        } - {"motor", "model"}
        assert solver_fields == set(perturb), (
            "RunConfig fields changed: update this map AND the "
            "solver_params dict in compute_run_id"
        )
        base = RunConfig(motor=make_motor(), model="unregistered-model")
        id_base = compute_run_id(base)
        for name, value in perturb.items():
            rc = dataclasses.replace(base, **{name: value})
            assert compute_run_id(rc) != id_base, f"{name} not hashed"

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

    @requires_sim
    def test_drive_sim_hash_sensitive_to_load(self):
        from phasesweep.simulation.sim import SimPlan
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
        # change the run ID with no input change
        rc = RunConfig(motor=make_motor(), model="analytical")
        id_before = compute_run_id(rc)
        info = MODEL_REGISTRY["analytical"]
        monkeypatch.setitem(
            MODEL_REGISTRY, "analytical",
            dataclasses.replace(info, version=info.version + 1),
        )
        assert compute_run_id(rc) != id_before

    def test_computed_models_bumped_to_v2(self):
        # Square-wave convention changed outputs of every computed model
        # (derived psi_f / B_rem); their run IDs must not collide with
        # pre-fix cached results. Later models (thermal_duty) start at
        # v1 — they never had pre-v2 outputs.
        post_convention_fix = {
            "thermal_duty", "torque_speed", "iron_loss", "demag_screen",
            "drive_sim_two_mass", "cogging_torque", "analytical_axial",
            "fem_axial",
        }
        for key, info in MODEL_REGISTRY.items():
            if info.source == "computed" and key not in post_convention_fix:
                assert info.version >= 2, f"{key} still at version {info.version}"


def test_all_registry_entries_importable():
    """Verify every MODEL_REGISTRY entry's lazy imports resolve."""
    for key, info in MODEL_REGISTRY.items():
        assert info.fn is not None or info.source == "measured", (
            f"{key}: fn is None for a computed model"
        )
