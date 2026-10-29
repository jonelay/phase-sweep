"""motulator drive-simulation runner: RunConfig in, metrics dict out."""

from __future__ import annotations

from typing import Any

from phasesweep.sweep_types import RunConfig

__all__ = [
    "run_drive_sim",
]


def run_drive_sim(config: RunConfig) -> dict[str, Any]:
    """Run motulator simulation from RunConfig."""
    from phasesweep.simulation.params import prepare_drive_sim, uses_flux_map
    from phasesweep.simulation.sim import build_sim, extract_metrics, extract_waveforms

    params = prepare_drive_sim(config.motor)
    plan = config.sim_plan
    if plan is None:
        raise ValueError("RunConfig.sim_plan is required for drive_sim")
    two_mass = config.model == "drive_sim_two_mass"
    if two_mass and config.load_mech is None:
        raise ValueError(
            "RunConfig.load_mech is required for drive_sim_two_mass"
        )
    if not two_mass and config.load_mech is not None:
        raise ValueError(
            "RunConfig.load_mech is set but model is not "
            "drive_sim_two_mass — use model='drive_sim_two_mass'"
        )
    sim = build_sim(params, plan, load_mech=config.load_mech)
    res = sim.simulate(t_stop=plan.t_stop)
    assert params.drive.W_REF is not None
    metrics: dict[str, Any] = extract_metrics(
        res, plan=plan, w_ref=params.drive.W_REF, two_mass=two_mass,
    )
    metrics.update(extract_waveforms(res, two_mass=two_mass))
    metrics["machine_model"] = "flux_map" if uses_flux_map(params) else "constant_L"
    return metrics
