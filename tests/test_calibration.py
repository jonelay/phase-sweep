"""Calibration framework tests.

Acceptance targets:
1. Recover effective B_rem 1.139 T on the actuator back-EMF data — the
   earlier manual calibration reproduced automatically.
2. Negative control — no single-param fit within published-parameter
   bounds closes the 0.92 MTPA/nameplate band.
3. Round-trip: calibrating against self-generated synthetic data
   recovers the perturbed truth.
Plus identifiability guards, the calibration record, the TOML writer,
and the CLI.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from math import sqrt

import pytest

from phasesweep.calibration import calibrate, write_motor_toml
from phasesweep.configs import load_motor
from phasesweep.measured import (
    MeasuredResult,
    MeasurementConditions,
    measured_run_result,
    validate_measured,
)
from phasesweep.motor import Motor
from phasesweep.perturbation import perturb_motor
from phasesweep.registry import MODEL_REGISTRY
from phasesweep.sweep_types import RunConfig, RunResult
from tests.conftest import REPO_ROOT

ACTUATOR_TOML = REPO_ROOT / "motors/actuator_steel_rotor.toml"
BACKEMF_JSON = REPO_ROOT / "data/actuator_steel_rotor/backemf_measured.json"
PSI_F_MEAS = 0.000268  # Wb, actuator back-EMF speed sweep


def _measured(
    motor: Motor,
    quantities: dict[str, float],
    *,
    test_type: str = "backemf_capture",
    tolerances: dict[str, float] | None = None,
    derived_params: tuple[str, ...] = (),
    dataset_id: str = "test_dataset",
    source: str = "measured",
) -> RunResult:
    data = MeasuredResult(
        motor_name=motor.name,
        test_type=test_type,
        conditions=MeasurementConditions(
            speed_rpm=10000, temperature_C=22.0, load_torque_Nm=0.0,
            date="2026-03-18", instrument="test",
        ),
        quantities=quantities,
        waveforms={}, uncertainty={}, source_file="test",
        tolerances=tolerances or {},
        source=source,
        derived_params=derived_params,
    )
    validate_measured(data)
    return measured_run_result(data, motor, dataset_id)


@pytest.fixture(scope="module")
def actuator():
    return load_motor(ACTUATOR_TOML)


@pytest.fixture(scope="module")
def actuator_backemf(actuator):
    """The committed import-format dataset from the speed sweep."""
    data = MeasuredResult.from_dict(json.loads(BACKEMF_JSON.read_text()))
    validate_measured(data)
    return measured_run_result(data, actuator, BACKEMF_JSON.stem)


def _awan():
    """Awan 2.2-kW IPM — geometry-less published-params motor."""
    return Motor(
        name="Awan 2.2-kW IPM", geometry=None, n_p=3,
        psi_f=0.545, L_d=0.036, L_q=0.051, R_s=3.5,
        I_rated=4.3 * sqrt(2),
    )


def _awan_nameplate(motor):
    return _measured(
        motor, {"tau_mtpa": 14.0},
        test_type="torque_test",
        tolerances={"tau_mtpa": 2.0},
        dataset_id="torque_rated",
        source="published",
    )


# ---------------------------------------------------------------------------
# Acceptance 1: recover effective B_rem 1.139 T
# ---------------------------------------------------------------------------

class TestRecoverEffectiveBrem:

    @pytest.fixture(scope="class")
    def result(self, actuator, actuator_backemf):
        return calibrate(
            actuator, [actuator_backemf],
            params=["B_rem"], quantities=["backemf_fundamental"],
        )

    def test_recovers_1p139(self, result):
        """The manual calibration, reproduced automatically."""
        fitted = result.record.params[0]
        assert fitted.param == "B_rem"
        assert fitted.initial == pytest.approx(1.45)
        assert fitted.final == pytest.approx(1.139, abs=0.005)

    def test_closed_within_tolerance(self, result):
        assert result.record.closed
        assert all(r["passed"] for r in result.record.residuals_after)

    def test_residual_before_is_the_plus27_gap(self, result):
        """The raw 2D over-prediction (+27%) is the starting residual."""
        before = result.record.residuals_before[0]
        assert 24.0 < before["rel_pct"] < 36.0

    def test_calibrated_motor_carries_new_brem(self, result):
        assert result.motor.B_rem == pytest.approx(1.139, abs=0.005)
        assert result.motor.psi_f is None  # invalidated for re-derivation

    def test_config_ids_differ_and_recorded(self, result, actuator):
        rec = result.record
        assert rec.source_config_id == actuator.config_id
        assert rec.calibrated_config_id == result.motor.config_id
        assert rec.source_config_id != rec.calibrated_config_id

    def test_record_metadata(self, result):
        rec = result.record
        assert rec.dataset_ids == ("backemf_measured",)
        assert rec.models == ("analytical",)
        assert rec.model_versions == {
            "analytical": MODEL_REGISTRY["analytical"].version}
        assert rec.optimizer["success"]
        assert rec.optimizer["cost_final"] < rec.optimizer["cost_initial"]

    def test_exactly_determined_warning(self, result):
        """1 param vs 1 row is an inversion — recorded, never silent."""
        assert any("exactly determined" in w for w in result.record.warnings)
        assert result.record.params[0].stderr is None


# ---------------------------------------------------------------------------
# CREATOR B_core: the hand back-solve, re-derived
# ---------------------------------------------------------------------------

_IRON_LOSS_JSON = REPO_ROOT / "data/creator_case_pmsm/iron_loss_noload.json"


@pytest.mark.skipif(not _IRON_LOSS_JSON.exists(), reason=(
    "CREATOR-derived JSONs not found — run "
    "python scripts/fetch_creator_dataset.py"
))
class TestCreatorBCore:
    """motors/creator_case_pmsm.toml ships a CALIBRATED B_core, back-solved
    by hand so p_fe matches the measured no-load loss. These tests re-derive
    it through calibrate() from the unfitted FEM-flux estimate, so the
    shipped number has a reproducible provenance path instead of a comment.

    The fit is not validation and never becomes it: it closes by
    construction on the one dataset any agreement test would use.
    """

    @pytest.fixture(scope="class")
    def shipped(self):
        return load_motor(REPO_ROOT / "motors/creator_case_pmsm.toml")

    @pytest.fixture(scope="class")
    def dataset(self):
        return MeasuredResult.from_dict(json.loads(_IRON_LOSS_JSON.read_text())), _IRON_LOSS_JSON.stem

    @pytest.fixture(scope="class")
    def result(self, shipped, dataset):
        from scripts.calibrate_creator_b_core import unfitted_b_core

        data, stem = dataset
        source = replace(shipped, B_core=unfitted_b_core(shipped))
        return calibrate(
            source, [measured_run_result(data, source, stem)],
            params=["B_core"], quantities=["p_fe"],
            bounds={"B_core": (-0.5, 2.0)},
        )

    def test_recovers_the_shipped_b_core(self, result, shipped):
        fitted = result.record.params[0]
        assert fitted.param == "B_core"
        assert fitted.initial == pytest.approx(0.273, abs=0.005)
        # 0.014% off the shipped 0.4974: the hand back-solve targeted the
        # measured curve interpolated to the model's rounded W_REF
        # (66.654 Hz), the framework targets the 66.667 Hz grid point.
        assert fitted.final == pytest.approx(shipped.B_core, rel=1e-3)
        assert result.motor.B_core == pytest.approx(shipped.B_core, rel=1e-3)

    def test_starting_residual_is_the_unfitted_shortfall(self, result):
        """The 2.7x gap the calibration papers over, as a residual."""
        assert result.record.residuals_before[0]["rel_pct"] == pytest.approx(
            63.4, abs=1.0)

    def test_closed_and_exactly_determined(self, result):
        rec = result.record
        assert rec.closed
        assert any("exactly determined" in w for w in rec.warnings)
        assert rec.params[0].stderr is None
        assert rec.models == ("iron_loss",)

    def test_committed_record_is_current(self, result):
        """The shipped record artifact tracks the live fit and the model
        version it was produced under — a bump must regenerate it. `initial`
        is pinned too, not just `final`: it carries the fem flux snapshot, so
        a fem bump that moves the airgap field fails here rather than leaving
        a stale starting point in a committed provenance artifact."""
        path = REPO_ROOT / "data/creator_case_pmsm/b_core_calibration.record.json"
        rec = json.loads(path.read_text())
        assert rec["params"][0]["final"] == pytest.approx(
            result.record.params[0].final, rel=1e-9)
        assert rec["params"][0]["initial"] == pytest.approx(
            result.record.params[0].initial, rel=1e-9)
        assert rec["model_versions"] == {
            "iron_loss": MODEL_REGISTRY["iron_loss"].version}
        assert rec["closed"]

    def test_untagged_by_design_and_the_tag_would_refuse(self, shipped, dataset):
        """The dataset carries no derived_params tag on purpose: the shipped
        B_core came from it, so the honest tag would refuse every fit and
        leave the calibration unrecorded. Pinned here so the exception stays
        a decision, not a drift."""
        data, stem = dataset
        assert data.derived_params == ()
        tagged = replace(data, derived_params=("B_core",))
        with pytest.raises(ValueError, match="circular fit refused"):
            calibrate(
                shipped, [measured_run_result(tagged, shipped, stem)],
                params=["B_core"], quantities=["p_fe"],
            )


# ---------------------------------------------------------------------------
# Acceptance 2: negative control — the 0.92 band
# ---------------------------------------------------------------------------

class TestNegativeControl092Band:
    """No single physical param within published-parameter uncertainty
    closes the MTPA/nameplate gap (0.92±0.01 across 3 anchors;
    saturated-L rules out a constant-L correction). A calibration
    framework that "fixed" this would be lying — assert it cannot.
    """

    @pytest.mark.parametrize("param,bound", [
        ("psi_f", (-0.05, 0.05)),
        ("L_d", (-0.15, 0.15)),
        ("L_q", (-0.15, 0.15)),
    ])
    def test_single_param_cannot_close_gap(self, param, bound):
        motor = _awan()
        result = calibrate(
            motor, [_awan_nameplate(motor)],
            params=[param], quantities=["tau_mtpa"],
            bounds={param: bound},
        )
        rec = result.record
        assert not rec.closed, (
            f"fitting {param} within {bound} must NOT close the 0.92 band"
        )
        assert not rec.residuals_after[0]["passed"]
        assert any("at a bound" in w for w in rec.warnings)


# ---------------------------------------------------------------------------
# Acceptance 3: synthetic round-trip
# ---------------------------------------------------------------------------

class TestSyntheticRoundTrip:

    def test_recovers_perturbed_truth(self, actuator):
        """Calibrating against self-generated data recovers the known delta."""
        truth = perturb_motor(actuator, "B_rem", -0.10)
        assert truth is not None
        info = MODEL_REGISTRY["analytical"]
        metrics = info.fn(RunConfig(motor=truth, model="analytical"))
        synthetic = _measured(
            actuator,
            {"backemf_fundamental": metrics["backemf_fundamental"]},
            tolerances={"backemf_fundamental": 0.5},
            dataset_id="synthetic",
        )
        result = calibrate(
            actuator, [synthetic],
            params=["B_rem"], quantities=["backemf_fundamental"],
        )
        fitted = result.record.params[0]
        assert fitted.delta == pytest.approx(-0.10, abs=1e-4)
        assert fitted.final == pytest.approx(truth.B_rem, rel=1e-4)
        assert result.record.closed


class TestMultiDatasetRows:
    """Residual rows are keyed per dataset: two datasets sharing a
    test_type and quantity must both constrain the fit (previously the
    second silently overwrote the first in _fit_rows)."""

    def _synthetic(self, actuator, delta, dataset_id):
        truth = perturb_motor(actuator, "B_rem", delta)
        assert truth is not None
        metrics = MODEL_REGISTRY["analytical"].fn(
            RunConfig(motor=truth, model="analytical"))
        return _measured(
            actuator,
            {"backemf_fundamental": metrics["backemf_fundamental"]},
            tolerances={"backemf_fundamental": 0.5},
            dataset_id=dataset_id,
        )

    def test_both_datasets_enter_the_fit(self, actuator):
        low = self._synthetic(actuator, -0.10, "sweep_low")
        high = self._synthetic(actuator, -0.06, "sweep_high")
        result = calibrate(
            actuator, [low, high],
            params=["B_rem"], quantities=["backemf_fundamental"],
        )
        rec = result.record
        assert len(rec.residuals_before) == 2
        assert {r["dataset"] for r in rec.residuals_before} == {
            "sweep_low", "sweep_high"}
        # the fit must land between the two single-dataset answers
        assert -0.10 < rec.params[0].delta < -0.06
        # two rows against one param: no exactly-determined warning
        assert not any("exactly determined" in w for w in rec.warnings)

    def test_duplicate_dataset_ids_deduped(self, actuator):
        a = self._synthetic(actuator, -0.10, "same_id")
        b = self._synthetic(actuator, -0.06, "same_id")
        result = calibrate(
            actuator, [a, b],
            params=["B_rem"], quantities=["backemf_fundamental"],
        )
        assert len(result.record.residuals_before) == 2


# ---------------------------------------------------------------------------
# Identifiability guards
# ---------------------------------------------------------------------------

class TestGuards:

    def test_circular_fit_refused(self, actuator):
        """Fitting a param against the dataset it was derived from is an echo."""
        motor = replace(actuator, psi_f=PSI_F_MEAS)
        backemf = PSI_F_MEAS * motor.drive.W_REF * motor.n_p
        meas = _measured(
            motor, {"backemf_fundamental": backemf},
            derived_params=("psi_f",),
        )
        with pytest.raises(ValueError, match="circular fit refused"):
            calibrate(motor, [meas], params=["psi_f"],
                      quantities=["backemf_fundamental"])

    def test_circular_extends_to_interderivable_pair(self, actuator):
        """psi_f derived from the dataset blocks fitting B_rem too."""
        backemf = PSI_F_MEAS * actuator.drive.W_REF * actuator.n_p
        meas = _measured(
            actuator, {"backemf_fundamental": backemf},
            derived_params=("psi_f",),
        )
        with pytest.raises(ValueError, match="circular fit refused"):
            calibrate(actuator, [meas], params=["B_rem"],
                      quantities=["backemf_fundamental"])

    @pytest.mark.parametrize("dataset,motor_toml", [
        ("data/etel_tmb/backemf_from_ku_0140_030_ra.json",
         "data/etel_tmb/etel_tmb0140_030_ra.toml"),
        ("data/rexroth_ms2n/backemf_from_ke.json",
         "data/rexroth_ms2n/rexroth_ms2n04_d0bhn.toml"),
        ("data/tecnotion_qtr/backemf_from_ke.json",
         "data/tecnotion_qtr/tecnotion_qtr_a_105_25_n.toml"),
    ])
    def test_fleet_emf_datasets_have_teeth(self, dataset, motor_toml):
        """The committed EMF-constant datasets refuse the psi_f echo fit —
        those TOML psi_f values were derived from these same constants."""
        motor = load_motor(REPO_ROOT / motor_toml)
        path = REPO_ROOT / dataset
        data = MeasuredResult.from_dict(json.loads(path.read_text()))
        validate_measured(data)
        meas = measured_run_result(data, motor, path.stem)
        with pytest.raises(ValueError, match="circular fit refused"):
            calibrate(motor, [meas], params=["psi_f"],
                      quantities=["backemf_fundamental"])

    def test_underdetermined_refused(self, actuator, actuator_backemf):
        with pytest.raises(ValueError, match="under-determined"):
            calibrate(actuator, [actuator_backemf],
                      params=["B_rem", "gap"],
                      quantities=["backemf_fundamental"])

    def test_psi_f_and_brem_together_refused(self, actuator, actuator_backemf):
        with pytest.raises(ValueError, match="interderivable"):
            calibrate(actuator, [actuator_backemf],
                      params=["psi_f", "B_rem"],
                      quantities=["backemf_fundamental"])

    def test_unknown_param_refused(self, actuator, actuator_backemf):
        with pytest.raises(ValueError, match="unknown fittable param"):
            calibrate(actuator, [actuator_backemf],
                      params=["n_p"], quantities=["backemf_fundamental"])

    def test_unset_param_refused(self, actuator_backemf, actuator):
        motor = replace(actuator, L_d=None, L_q=None)
        with pytest.raises(ValueError, match="not set"):
            calibrate(motor, [actuator_backemf],
                      params=["L_d"], quantities=["backemf_fundamental"])

    def test_zero_sensitivity_refused(self):
        """A param with no effect on the quantities must not ride along."""
        motor = _awan()
        with pytest.raises(ValueError, match="no sensitivity"):
            calibrate(motor, [_awan_nameplate(motor)],
                      params=["R_s"], quantities=["tau_mtpa"])

    def test_bound_for_unfitted_param_refused(self, actuator, actuator_backemf):
        with pytest.raises(ValueError, match="matches no fitted param"):
            calibrate(actuator, [actuator_backemf],
                      params=["B_rem"], quantities=["backemf_fundamental"],
                      bounds={"k_w": (-0.1, 0.1)})

    def test_inverted_bound_refused(self, actuator, actuator_backemf):
        with pytest.raises(ValueError, match="lower must be < upper"):
            calibrate(actuator, [actuator_backemf],
                      params=["B_rem"], quantities=["backemf_fundamental"],
                      bounds={"B_rem": (0.3, -0.3)})

    def test_bounds_excluding_zero_clamp_x0(self, actuator, actuator_backemf):
        """Bounds that exclude a zero delta must not crash scipy with an
        infeasible x0 — the guess is clamped and the clamp recorded."""
        result = calibrate(
            actuator, [actuator_backemf],
            params=["B_rem"], quantities=["backemf_fundamental"],
            bounds={"B_rem": (0.05, 0.3)},
        )
        rec = result.record
        assert any("clamped" in w for w in rec.warnings)
        assert rec.params[0].delta >= 0.05 - 1e-9
        assert not rec.closed  # the data wants B_rem lower, not higher

    def test_auto_models_never_pick_thermal_duty(self, actuator):
        """thermal_duty needs a duty_profile the calibration loop's
        RunConfig never carries — auto-selecting it aborted the run."""
        from phasesweep.calibration import _auto_models
        meas = RunResult(
            config=RunConfig(motor=actuator, model="backemf_capture",
                             dataset_id="d"),
            model="backemf_capture", status="OK",
            metrics={}, elapsed_s=0.0, source="measured",
        )
        models = _auto_models([meas], ("p_fe",))
        assert "thermal_duty" not in models
        assert "iron_loss" in models

    def test_model_valueerror_during_fit_is_rejection(
            self, actuator, actuator_backemf, monkeypatch):
        """A model raising ValueError on a perturbed motor mid-optimization
        must act like a bounds rejection, not kill the fit."""
        info = MODEL_REGISTRY["analytical"]
        orig_fn = info.fn

        def flaky(config):
            if config.motor.B_rem < 1.30:
                raise ValueError("solver out of domain")
            return orig_fn(config)

        monkeypatch.setitem(
            MODEL_REGISTRY, "analytical", replace(info, fn=flaky))
        result = calibrate(
            actuator, [actuator_backemf],
            params=["B_rem"], quantities=["backemf_fundamental"],
        )
        rec = result.record
        assert result.motor.B_rem >= 1.30  # settled outside the raising region
        assert not rec.closed

    def test_no_rows_refused(self, actuator, actuator_backemf):
        with pytest.raises(ValueError,
                           match=r"no fast computed model|no equality"):
            calibrate(actuator, [actuator_backemf],
                      params=["B_rem"], quantities=["nonexistent_quantity"])

    def test_collinear_sensitivities_warned(self, actuator):
        """k_w and B_rem both scale backemf and flux linkage linearly —
        the Jacobian columns are parallel and the fit must say so."""
        w_e = actuator.drive.W_REF * actuator.n_p
        metrics = {
            "backemf_fundamental": PSI_F_MEAS * w_e,
            "flux_linkage_peak": PSI_F_MEAS,
        }
        meas = RunResult(
            config=RunConfig(motor=actuator, model="backemf_capture",
                             dataset_id="collinear_test"),
            model="backemf_capture", status="OK",
            metrics=metrics, elapsed_s=0.0, source="measured",
        )
        result = calibrate(
            actuator, [meas],
            params=["B_rem", "k_w"],
            quantities=["backemf_fundamental", "flux_linkage_peak"],
            bounds={"B_rem": (-0.3, 0.3), "k_w": (-0.1, 0.1)},
        )
        assert any("near-collinear" in w for w in result.record.warnings)


# ---------------------------------------------------------------------------
# Calibrated TOML writer
# ---------------------------------------------------------------------------

class TestWriteMotorToml:

    @pytest.mark.parametrize("name", [
        "actuator_steel_rotor", "actuator_aluminum_rotor",
        "creator_case_pmsm", "belkhadir_outrunner",
        "deylami_fan", "lrk_outrunner",
    ])
    def test_round_trip_config_id(self, name, tmp_path):
        motor = load_motor(REPO_ROOT / f"motors/{name}.toml")
        out = tmp_path / "rt.toml"
        write_motor_toml(motor, out)
        assert load_motor(out).config_id == motor.config_id

    def test_refuses_overwrite(self, actuator, tmp_path):
        out = tmp_path / "cal.toml"
        write_motor_toml(actuator, out)
        with pytest.raises(FileExistsError):
            write_motor_toml(actuator, out)

    def test_header_written_as_comments(self, actuator, tmp_path):
        out = tmp_path / "cal.toml"
        write_motor_toml(actuator, out, header="line one\nline two")
        text = out.read_text()
        assert text.startswith("# line one\n# line two\n")

    def test_escapes_special_chars_in_strings(self, actuator, tmp_path):
        motor = replace(actuator, name='cal "v2" \\ backslash')
        out = tmp_path / "esc.toml"
        write_motor_toml(motor, out)
        assert load_motor(out).name == motor.name

    def test_record_sidecar_serializes(self, actuator, actuator_backemf, tmp_path):
        result = calibrate(
            actuator, [actuator_backemf],
            params=["B_rem"], quantities=["backemf_fundamental"],
        )
        path = tmp_path / "cal.record.json"
        result.record.save(path)
        raw = json.loads(path.read_text())
        assert raw["source_config_id"] == actuator.config_id
        assert raw["calibrated_config_id"] == result.motor.config_id
        assert raw["params"][0]["param"] == "B_rem"
        assert raw["closed"] is True
        assert raw["model_versions"]["analytical"] >= 4


# ---------------------------------------------------------------------------
# derived_params dataset tag
# ---------------------------------------------------------------------------

class TestDerivedParamsTag:

    def test_round_trips_through_dict(self):
        data = MeasuredResult(
            motor_name="m", test_type="backemf_capture",
            conditions=MeasurementConditions(0, 20, 0, "2026-01-01", "x"),
            quantities={"backemf_fundamental": 1.0},
            waveforms={}, uncertainty={}, source_file="x",
            derived_params=("psi_f",),
        )
        again = MeasuredResult.from_dict(data.to_dict())
        assert again.derived_params == ("psi_f",)

    def test_validate_rejects_non_motor_fields(self):
        data = MeasuredResult(
            motor_name="m", test_type="backemf_capture",
            conditions=MeasurementConditions(0, 20, 0, "2026-01-01", "x"),
            quantities={"backemf_fundamental": 1.0},
            waveforms={}, uncertainty={}, source_file="x",
            derived_params=("not_a_field",),
        )
        with pytest.raises(ValueError, match="not Motor fields"):
            validate_measured(data)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestCliCalibrate:

    @pytest.fixture
    def backemf_json(self):
        """The committed import-format dataset — the CLI runs off the repo."""
        return BACKEMF_JSON

    def _run(self, monkeypatch, argv):
        from phasesweep import cli_calibrate
        monkeypatch.setattr(sys, "argv", argv)
        cli_calibrate.main()

    def test_happy_path(self, monkeypatch, capsys, tmp_path, backemf_json):
        out = tmp_path / "actuator_calibrated.toml"
        self._run(monkeypatch, [
            "phasesweep-calibrate", str(ACTUATOR_TOML),
            "--data", str(backemf_json),
            "--params", "B_rem",
            "--quantities", "backemf_fundamental",
            "--out", str(out),
        ])
        stdout = capsys.readouterr().out
        assert "Closed within tolerance: True" in stdout
        assert out.exists()
        assert out.with_suffix(".record.json").exists()
        assert load_motor(out).B_rem == pytest.approx(1.139, abs=0.005)

    def test_refusal_exits_1(self, monkeypatch, capsys, tmp_path, backemf_json):
        with pytest.raises(SystemExit) as exc:
            self._run(monkeypatch, [
                "phasesweep-calibrate", str(ACTUATOR_TOML),
                "--data", str(backemf_json),
                "--params", "B_rem", "--params", "psi_f",
                "--quantities", "backemf_fundamental",
                "--out", str(tmp_path / "cal.toml"),
            ])
        assert exc.value.code == 1
        assert "Calibration refused" in capsys.readouterr().err

    def test_out_equal_to_source_refused(self, monkeypatch, capsys, backemf_json):
        with pytest.raises(SystemExit) as exc:
            self._run(monkeypatch, [
                "phasesweep-calibrate", str(ACTUATOR_TOML),
                "--data", str(backemf_json),
                "--params", "B_rem",
                "--quantities", "backemf_fundamental",
                "--out", str(ACTUATOR_TOML),
            ])
        assert exc.value.code == 1
        assert "must not be the source" in capsys.readouterr().err

    def test_bad_bound_spec_exits(self, monkeypatch, tmp_path, backemf_json):
        with pytest.raises(SystemExit):
            self._run(monkeypatch, [
                "phasesweep-calibrate", str(ACTUATOR_TOML),
                "--data", str(backemf_json),
                "--params", "B_rem",
                "--quantities", "backemf_fundamental",
                "--bound", "B_rem=oops",
                "--out", str(tmp_path / "cal.toml"),
            ])

    def test_bound_for_unfitted_param_exits(
            self, monkeypatch, capsys, tmp_path, backemf_json):
        with pytest.raises(SystemExit) as exc:
            self._run(monkeypatch, [
                "phasesweep-calibrate", str(ACTUATOR_TOML),
                "--data", str(backemf_json),
                "--params", "B_rem",
                "--quantities", "backemf_fundamental",
                "--bound", "k_w=-0.1:0.1",
                "--out", str(tmp_path / "cal.toml"),
            ])
        assert exc.value.code == 1
        assert "matches no fitted param" in capsys.readouterr().err

    def test_residual_pairs_align_by_key(self):
        """residuals_after can be shorter than residuals_before — pairing
        must match by key, not by position (a zip misaligned the report)."""
        from types import SimpleNamespace

        from phasesweep.cli_calibrate import _residual_pairs

        def row(dataset, rel):
            return {"dataset": dataset, "quantity": "q",
                    "measured_model": "m", "computed_model": "c",
                    "rel_pct": rel}

        b1, b2 = row("d1", 10.0), row("d2", 20.0)
        a2 = row("d2", 1.0)
        rec = SimpleNamespace(residuals_before=(b1, b2), residuals_after=(a2,))
        assert _residual_pairs(rec) == [(b1, None), (b2, a2)]
