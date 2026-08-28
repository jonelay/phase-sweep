"""Contract tests for model run-time input metadata."""

from __future__ import annotations

import pytest

import phasesweep.registry as registry
from phasesweep.machines.geometry import inrunner
from phasesweep.registry import MODEL_REGISTRY
from phasesweep.sweep_types import RunConfig
from tests.conftest import make_motor, requires_fem, requires_sim

EXPECTED_NEEDS_RUNTIME = {
    "drive_sim": frozenset({"sim_plan"}),
    "drive_sim_two_mass": frozenset({"sim_plan", "load_mech"}),
    "thermal_duty": frozenset({"duty_profile"}),
    "demag_screen": frozenset({"i_fault"}),
}


def _applicable_motor(model: str):
    if model == "demag_screen":
        geometry = inrunner(
            r_outer=0.06,
            r_stator=0.048,
            r_magnet=0.04,
            r_rotor=0.03,
            n_slots=12,
            slot_depth=0.005,
        )
        return make_motor(geometry=geometry, B_rem=1.2, B_knee=0.3)
    return make_motor(I_rated=10.0)


def test_registry_declares_all_required_runtime_inputs() -> None:
    declared = {
        name: info.needs_runtime
        for name, info in MODEL_REGISTRY.items()
        if info.needs_runtime
    }
    assert declared == EXPECTED_NEEDS_RUNTIME


# demag_screen's runner imports the FEM solver module before its i_fault
# guard, so exercising it needs ngsolve even though no solve runs.
@pytest.mark.parametrize(
    "model",
    [
        pytest.param(m, marks=requires_fem)
        if m == "demag_screen"
        else pytest.param(m, marks=requires_sim)
        if m in {"drive_sim", "drive_sim_two_mass"}
        else m
        for m in EXPECTED_NEEDS_RUNTIME
    ],
)
def test_runtime_input_models_raise_when_fields_are_absent(
    model: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    motor = _applicable_motor(model)
    info = MODEL_REGISTRY[model]
    assert info.validate is not None
    info.validate(motor)
    assert info.fn is not None
    if model == "demag_screen":
        monkeypatch.setattr(registry, "_check_fem", lambda _name: None)
    with pytest.raises(ValueError):
        info.fn(RunConfig(motor=motor, model=model))
