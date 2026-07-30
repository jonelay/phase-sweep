"""Tests for measured data import."""

import json

import pytest

from phasesweep.measured import (
    BoundRef,
    CurveRef,
    KeyMapping,
    MeasuredResult,
    MeasurementConditions,
    import_measured,
    validate_measured,
)
from phasesweep.registry import MODEL_REGISTRY
from phasesweep.result_store import ResultStore
from tests.conftest import REPO_ROOT
from tests.conftest import make_motor as _make_motor


def _conditions(**kw):
    defaults = dict(speed_rpm=2000, temperature_C=22.0, load_torque_Nm=0.0,
                    date="2024-01-15", instrument="oscilloscope")
    defaults.update(kw)
    return MeasurementConditions(**defaults)


def _measured(test_type="backemf_capture", quantities=None, **kw):
    defaults = dict(
        motor_name="test",
        test_type=test_type,
        conditions=_conditions(),
        quantities=quantities or {"backemf_fundamental": 47.37},
        waveforms={},
        uncertainty={"backemf_fundamental": 0.5},
        source_file="test.csv",
    )
    defaults.update(kw)
    return MeasuredResult(**defaults)


# ---------------------------------------------------------------------------
# Serialization round-trips
# ---------------------------------------------------------------------------

class TestSerialization:

    def test_conditions_round_trip(self):
        tc = _conditions(notes="hello")
        assert MeasurementConditions.from_dict(tc.to_dict()) == tc

    def test_measured_result_round_trip(self):
        mr = _measured()
        assert MeasuredResult.from_dict(mr.to_dict()) == mr


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:

    def test_valid_backemf(self):
        validate_measured(_measured())

    def test_valid_inductance(self):
        validate_measured(_measured(
            test_type="inductance_test",
            quantities={"L_d": 4e-3, "L_q": 6e-3},
        ))

    def test_valid_resistance(self):
        validate_measured(_measured(
            test_type="resistance_test",
            quantities={"R_s": 0.2},
        ))

    def test_invalid_test_type(self):
        with pytest.raises(ValueError, match="unknown measured test_type"):
            validate_measured(_measured(test_type="bogus"))

    def test_computed_model_rejected(self):
        with pytest.raises(ValueError, match="unknown measured test_type"):
            validate_measured(_measured(test_type="analytical"))

    def test_invalid_quantity_key(self):
        with pytest.raises(ValueError, match="no resolution path"):
            validate_measured(_measured(quantities={"not_real": 1.0}))

    def test_empty_motor_name(self):
        with pytest.raises(ValueError, match="motor_name"):
            validate_measured(_measured(motor_name=""))

    def test_valid_published_source(self):
        validate_measured(_measured(source="published"))

    def test_descriptive_source_rejected(self):
        with pytest.raises(ValueError, match="source must be"):
            validate_measured(_measured(source="published (ANSYS FEM)"))


# ---------------------------------------------------------------------------
# Import path
# ---------------------------------------------------------------------------

class TestImport:

    def test_backemf_import(self, tmp_path):
        data = _measured()
        json_path = tmp_path / "measured.json"
        json_path.write_text(json.dumps(data.to_dict()))

        result = import_measured(json_path, _make_motor(), tmp_path / "out")
        assert result.source == "measured"
        assert result.status == "OK"
        assert result.model == "backemf_capture"
        assert result.metrics["backemf_fundamental"] == 47.37

    def test_inductance_import(self, tmp_path):
        data = _measured(
            test_type="inductance_test",
            quantities={"L_d": 4e-3, "L_q": 6e-3},
        )
        json_path = tmp_path / "measured.json"
        json_path.write_text(json.dumps(data.to_dict()))

        result = import_measured(json_path, _make_motor(), tmp_path / "out")
        assert result.metrics["L_d"] == 4e-3
        assert result.metrics["L_q"] == 6e-3

    def test_resistance_import(self, tmp_path):
        data = _measured(
            test_type="resistance_test",
            quantities={"R_s": 0.2},
        )
        json_path = tmp_path / "measured.json"
        json_path.write_text(json.dumps(data.to_dict()))

        result = import_measured(json_path, _make_motor(), tmp_path / "out")
        assert result.metrics["R_s"] == 0.2

    def test_result_store_round_trip(self, tmp_path):
        data = _measured()
        json_path = tmp_path / "measured.json"
        json_path.write_text(json.dumps(data.to_dict()))

        import_measured(json_path, _make_motor(), tmp_path / "out")

        store = ResultStore(tmp_path / "out")
        results = store.load_results(source="measured")
        assert len(results) == 1
        assert results[0].model == "backemf_capture"
        assert results[0].source == "measured"


# ---------------------------------------------------------------------------
# Registry entries
# ---------------------------------------------------------------------------

class TestMeasuredRegistry:

    @pytest.mark.parametrize("key", [
        "backemf_capture", "inductance_test", "resistance_test",
        "torque_test", "airgap_flux_test",
    ])
    def test_measured_entries_exist(self, key):
        info = MODEL_REGISTRY[key]
        assert info.source == "measured"
        assert info.cost == "none"
        assert info.fn is None
        assert info.validate is None


# ---------------------------------------------------------------------------
# CREATOR fixture
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (REPO_ROOT / "data/creator_case_pmsm/backemf_measured.json").exists(),
    reason="CREATOR-derived CSVs not found — run python scripts/fetch_creator_dataset.py",
)
class TestCreatorFixture:

    def test_fixture_loads(self):
        fixture = REPO_ROOT / "data/creator_case_pmsm/backemf_measured.json"
        raw = json.loads(fixture.read_text())
        data = MeasuredResult.from_dict(raw)
        assert data.motor_name == "CREATOR Case PMSM"
        assert data.test_type == "backemf_capture"
        assert data.quantities["backemf_fundamental"] == 47.37
        validate_measured(data)


# ---------------------------------------------------------------------------
# Committed import-format datasets
# ---------------------------------------------------------------------------

_EMF_CONSTANT_DATASETS = [
    "data/etel_tmb/backemf_from_ku_0140_030_ra.json",
    "data/etel_tmb/backemf_from_ku_0210_030_ta.json",
    "data/etel_tmb/backemf_from_ku_0290_030_ra.json",
    "data/etel_tmb/backemf_from_ku_0360_030_ta.json",
    "data/etel_tmb/backemf_from_ku_0450_030_va.json",
    "data/rexroth_ms2n/backemf_from_ke.json",
    "data/tecnotion_qtr/backemf_from_ke.json",
]


class TestCommittedDatasets:
    """Every data/ JSON with a registered measured test_type must import
    cleanly; the datasheet EMF-constant datasets must carry the psi_f
    derived_params tag (their TOML psi_f came from the same constants)."""

    @staticmethod
    def _import_format():
        for path in sorted((REPO_ROOT / "data").rglob("*.json")):
            raw = json.loads(path.read_text())
            if raw.get("test_type") in MODEL_REGISTRY:
                yield path, MeasuredResult.from_dict(raw)

    def test_all_import_format_datasets_validate(self):
        found = set()
        for path, data in self._import_format():
            validate_measured(data)
            found.add(str(path.relative_to(REPO_ROOT)))
        expected = set(_EMF_CONSTANT_DATASETS) | {
            "data/actuator_steel_rotor/backemf_measured.json"}
        assert expected <= found

    @pytest.mark.parametrize("rel", _EMF_CONSTANT_DATASETS)
    def test_emf_constant_datasets_tagged(self, rel):
        raw = json.loads((REPO_ROOT / rel).read_text())
        data = MeasuredResult.from_dict(raw)
        assert data.derived_params == ("psi_f",)
        assert data.source == "published"

    def test_actuator_dataset_untagged(self):
        """B_rem in the actuator TOML is the nominal grade value — fitting
        it against the sweep is the flagship calibration, not an echo."""
        raw = json.loads((
            REPO_ROOT / "data/actuator_steel_rotor/backemf_measured.json"
        ).read_text())
        assert MeasuredResult.from_dict(raw).derived_params == ()


# ---------------------------------------------------------------------------
# New fields round-trip
# ---------------------------------------------------------------------------

class TestNewFieldsRoundTrip:

    def test_tolerances_round_trip(self):
        mr = _measured(
            test_type="torque_test",
            quantities={"tau_rated": 0.1},
            tolerances={"tau_rated": 15.0},
            bound_compare={"tau_rated": BoundRef("tau_rated", "gte")},
        )
        d = mr.to_dict()
        mr2 = MeasuredResult.from_dict(d)
        assert mr2.tolerances == {"tau_rated": 15.0}
        assert mr2.bound_compare["tau_rated"].computed_key == "tau_rated"

    def test_curve_compare_round_trip(self):
        mr = _measured(
            test_type="torque_test",
            quantities={"gamma_opt_deg": 10.0},
            curve_compare={"gamma_opt_deg": CurveRef("I_curve", "gamma_curve_deg", at_x=0.2)},
        )
        d = mr.to_dict()
        mr2 = MeasuredResult.from_dict(d)
        assert mr2.curve_compare["gamma_opt_deg"].at_x == 0.2
        assert mr2.curve_compare["gamma_opt_deg"].extract == "interp"

    def test_key_mapping_round_trip(self):
        mr = _measured(
            test_type="airgap_flux_test",
            quantities={"backemf_peak": 67.5},
            key_mapping={"backemf_peak": KeyMapping("backemf_fundamental", "peak ≈ fundamental")},
        )
        d = mr.to_dict()
        mr2 = MeasuredResult.from_dict(d)
        assert mr2.key_mapping["backemf_peak"].computed_key == "backemf_fundamental"

    def test_source_published(self):
        mr = _measured(
            test_type="airgap_flux_test",
            quantities={"B_ag_peak": 0.776},
            source="published",
        )
        d = mr.to_dict()
        mr2 = MeasuredResult.from_dict(d)
        assert mr2.source == "published"

    def test_old_data_defaults(self):
        raw = {
            "motor_name": "test", "test_type": "backemf_capture",
            "conditions": {"speed_rpm": 0, "temperature_C": 22, "load_torque_Nm": 0,
                           "date": "2024-01-01", "instrument": "scope"},
            "quantities": {"backemf_fundamental": 47.0},
        }
        mr = MeasuredResult.from_dict(raw)
        assert mr.tolerances == {}
        assert mr.source == "measured"
        assert mr.curve_compare == {}
        assert mr.bound_compare == {}
        assert mr.key_mapping == {}


# ---------------------------------------------------------------------------
# Validation with comparison metadata
# ---------------------------------------------------------------------------

class TestValidationWithMetadata:

    def test_accepts_bound_compare_keys(self):
        mr = _measured(
            test_type="torque_test",
            quantities={"tau_rated": 0.1},
            bound_compare={"tau_rated": BoundRef("tau_rated", "gte")},
        )
        validate_measured(mr)

    def test_accepts_curve_compare_keys(self):
        mr = _measured(
            test_type="torque_test",
            quantities={"gamma_opt_deg": 10.0},
            curve_compare={"gamma_opt_deg": CurveRef("I_curve", "gamma_curve_deg", at_x=0.2)},
        )
        validate_measured(mr)

    def test_accepts_key_mapping_keys(self):
        mr = _measured(
            test_type="airgap_flux_test",
            quantities={"backemf_peak": 67.5},
            key_mapping={"backemf_peak": KeyMapping("backemf_fundamental", "note")},
        )
        validate_measured(mr)

    def test_rejects_orphan_keys(self):
        mr = _measured(
            test_type="torque_test",
            quantities={"tau_rated": 0.1, "orphan_key": 99.0},
            bound_compare={"tau_rated": BoundRef("tau_rated", "gte")},
        )
        with pytest.raises(ValueError, match="no resolution path"):
            validate_measured(mr)


# ---------------------------------------------------------------------------
# Import with metadata propagation
# ---------------------------------------------------------------------------

class TestImportWithMetadata:

    def test_import_propagates_source(self, tmp_path):
        mr = _measured(
            test_type="airgap_flux_test",
            quantities={"B_ag_peak": 0.776},
            source="published",
        )
        path = tmp_path / "test.json"
        path.write_text(json.dumps(mr.to_dict()))
        result = import_measured(path, _make_motor(), tmp_path / "out")
        assert result.source == "published"

    def test_import_propagates_tolerances(self, tmp_path):
        mr = _measured(
            test_type="torque_test",
            quantities={"tau_rated": 0.1},
            tolerances={"tau_rated": 15.0},
            bound_compare={"tau_rated": BoundRef("tau_rated", "gte")},
        )
        path = tmp_path / "test.json"
        path.write_text(json.dumps(mr.to_dict()))
        result = import_measured(path, _make_motor(), tmp_path / "out")
        assert result.tolerances == {"tau_rated": 15.0}

    def test_import_stores_comparison_metadata(self, tmp_path):
        mr = _measured(
            test_type="torque_test",
            quantities={"tau_rated": 0.1},
            bound_compare={"tau_rated": BoundRef("tau_rated", "gte")},
        )
        path = tmp_path / "test.json"
        path.write_text(json.dumps(mr.to_dict()))
        result = import_measured(path, _make_motor(), tmp_path / "out")
        assert "_bound_compare" in result.metrics
        assert result.metrics["_bound_compare"]["tau_rated"]["relation"] == "gte"


# ---------------------------------------------------------------------------
# Data file fixtures
# ---------------------------------------------------------------------------

class TestDataFixtures:

    def test_creator_torque_rated(self):
        raw = json.loads((REPO_ROOT / "data/creator_case_pmsm/torque_rated.json").read_text())
        data = MeasuredResult.from_dict(raw)
        validate_measured(data)
        assert data.quantities["tau_rated"] == 0.10
        assert "tau_rated" in data.bound_compare

    def test_creator_mtpa_angles(self):
        raw = json.loads((REPO_ROOT / "data/creator_case_pmsm/mtpa_angles.json").read_text())
        data = MeasuredResult.from_dict(raw)
        validate_measured(data)
        assert "gamma_opt_at_0p15" in data.curve_compare
        assert "gamma_opt_at_0p30" in data.curve_compare

    def test_belkhadir_airgap_flux(self):
        raw = json.loads((REPO_ROOT / "data/belkhadir_outrunner/airgap_flux_published.json").read_text())
        data = MeasuredResult.from_dict(raw)
        validate_measured(data)
        assert data.source == "published"
        assert "B_ag_fundamental" in data.key_mapping
        assert data.key_mapping["B_ag_fundamental"].computed_key == "fundamental"
        assert data.quantities["B_ag_fundamental"] == pytest.approx(0.9625)
        assert "backemf_peak" in data.key_mapping

    def test_awan_torque_rated(self):
        raw = json.loads((REPO_ROOT / "data/awan_ipm/torque_rated.json").read_text())
        data = MeasuredResult.from_dict(raw)
        validate_measured(data)
        assert data.source == "published"
        assert data.quantities["tau_rated"] == 14.0
