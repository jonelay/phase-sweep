"""Cross-validation: Toyota Prius 2004 IPM (50 kW traction motor).

Circuit-only motor — validates the stall torque and envelope models
against ORNL dynamometer data, not the field solver (no geometry).

The Prius is the most-reproduced IPM benchmark; its saliency
(L_q/L_d ~ 2.72) and the wide gap between characteristic current
(psi_f/L_d ~ 71 A peak) and peak current (250 A) make it the canonical
test case for IPM torque and field-weakening models.

Sources: ORNL/TM-2004/137 (teardown), TM-2004/185 rev. 2007 (back-EMF,
         locked-rotor torque), TM-2004/247 (dynamometer).
"""

import pytest

from phasesweep.machines.configs import load_motor
from phasesweep.registry import MODEL_REGISTRY
from phasesweep.sweep_types import RunConfig, RunResult
from phasesweep.validation.crossval import compare_results
from phasesweep.validation.measured import import_measured
from tests.conftest import REPO_ROOT

MOTOR_TOML = REPO_ROOT / "motors/prius_2004.toml"


@pytest.fixture(scope="module")
def prius():
    return load_motor(MOTOR_TOML)


@pytest.fixture(scope="module")
def stall_result(prius):
    rc = RunConfig(motor=prius, model="stall_torque")
    metrics = MODEL_REGISTRY["stall_torque"].fn(rc)
    return RunResult(
        config=rc, model="stall_torque", status="OK",
        metrics=metrics, elapsed_s=0.0, source="computed",
    )


class TestPriusStallTorque:

    def test_stall_torque_at_peak_current(self, stall_result):
        """MTPA at MAX_I_S (250 A peak) — pinned to model prediction.

        905 Nm over-predicts the ORNL dynamometer measurement (~340 Nm
        at 250 A) by ~166%. The linear model's constant L_q (6.15 mH,
        Kuptsov moderate-load) drops to ~3 mH at this current, so
        the reluctance torque contribution is massively inflated.
        Documented limitation — only low-current results are tight."""
        tau = stall_result.metrics["tau_stall"]
        assert tau == pytest.approx(904.5, abs=2.0)

    def test_mtpa_angle_reflects_saliency(self, stall_result):
        """MTPA angle at peak current should be well above zero — the
        Prius's ~40 deg puts about two-thirds of I_peak into the d-axis
        to exploit the reluctance torque."""
        gamma = stall_result.metrics["gamma_opt_deg"]
        assert 37.0 < gamma < 43.0

    def test_k_T_effective_includes_reluctance(self, stall_result):
        """Effective torque constant (Nm/A_peak at stall) should exceed
        the magnet-only k_T = 1.5*n_p*psi_f = 0.964 Nm/A_peak by a
        wide margin — the saliency nearly quadruples it at this current
        (because the constant L_q overstates reluctance contribution)."""
        k_T_mag = 1.5 * 4 * 0.1607
        k_T_eff = stall_result.metrics["k_T_effective"]
        assert k_T_eff > k_T_mag * 2.0
        assert k_T_eff == pytest.approx(3.618, abs=0.01)

    def test_characteristic_current_well_below_peak(self, prius):
        """psi_f/L_d << MAX_I_S: the motor reaches deep flux reversal
        at peak current (I_ch/I_peak ~ 0.28), which gives unlimited
        max speed in the linear model. This is the design intent for
        EV traction — wide constant-power speed range."""
        i_ch = prius.psi_f / prius.L_d
        assert i_ch < prius.drive.MAX_I_S * 0.4
        assert i_ch == pytest.approx(71.1, abs=0.5)


class TestPriusTorqueBound:

    def test_import_torque_rated(self, prius, tmp_path):
        result = import_measured(
            REPO_ROOT / "data/prius_2004/torque_rated.json",
            prius, tmp_path,
        )
        assert result.source == "published"
        assert "_bound_compare" in result.metrics

    def test_stall_torque_bound_via_crossval(self, prius, stall_result,
                                              tmp_path):
        """Model MTPA at MAX_I_S should exceed the ORNL dynamometer
        measurement (340 Nm at 250 A locked-rotor). Goes through the
        crossval compare_results path — the bound_compare in the JSON
        maps tau_rated to tau_stall (stall_torque model output).

        The bound passes by a wide margin (~166% over-prediction) because
        the constant-inductance model overstates reluctance torque at
        this current level. This is a one-sided bound, not a tight
        validation."""
        measured = import_measured(
            REPO_ROOT / "data/prius_2004/torque_rated.json",
            prius, tmp_path,
        )
        rows = compare_results(measured, stall_result)
        bound_rows = [r for r in rows if r.comparison_type == "bound"]
        assert len(bound_rows) == 1
        assert bound_rows[0].passed, (
            f"tau_stall={bound_rows[0].val_b:.1f} should be >= "
            f"ORNL measured={bound_rows[0].val_a:.1f}"
        )
