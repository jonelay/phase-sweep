"""Shape of a stored result record as written to results.jsonl.

Static contract for consumers that read phase-sweep stores as dicts
(downstream adapters, the dashboard). Only stdlib `typing`, so the
module imports without numpy. `record_vectors.json` pins real rows
against these shapes; see docs/result-store-contract.md §5.

Compound values (`geometry`, `induction`, `flux_map`, `thermal_network`)
stay `dict[str, Any]`: no external consumer reads inside them, and their
owners (`Geometry`, `InductionParams`, `FluxMap`, `ThermalNetwork`) define
the shape.

No `from __future__ import annotations` here: stringified annotations hide
`NotRequired` from TypedDict, so `__required_keys__` would list every key.
"""

import json
from importlib.resources import files
from typing import Any, Literal, NotRequired, TypedDict


class DriveDict(TypedDict, total=False):
    U_DC: float
    MAX_I_S: float
    W_REF: float
    I_LIMIT: float


class MotorDict(TypedDict):
    name: str
    n_p: int
    alpha_p: float
    mu_r_fe: float
    mu_r_pm: float

    drive: NotRequired[DriveDict]
    geometry: NotRequired[dict[str, Any]]
    induction: NotRequired[dict[str, Any]]
    flux_map: NotRequired[dict[str, Any]]
    thermal_network: NotRequired[dict[str, Any]]
    magnetization: NotRequired[str]
    halbach_segments: NotRequired[int | None]
    halbach_angles_deg: NotRequired[list[float]]
    bh_curve: NotRequired[list[list[float]]]

    R_s: NotRequired[float]
    L_d: NotRequired[float]
    L_q: NotRequired[float]
    psi_f: NotRequired[float]
    B_rem: NotRequired[float]
    J: NotRequired[float]
    N: NotRequired[int]
    k_w: NotRequired[float]
    L_stk: NotRequired[float]
    I_rated: NotRequired[float]
    coils_series: NotRequired[int]
    coil_pitch_slots: NotRequired[int]
    k_end: NotRequired[float]
    winding_temp_limit: NotRequired[float]
    ambient_temp: NotRequired[float]
    r_th: NotRequired[float]
    insulation_class: NotRequired[str]
    thermal_time_constant: NotRequired[float]
    alpha_Br: NotRequired[float]
    magnet_temp: NotRequired[float]
    B_knee: NotRequired[float]
    alpha_B_knee: NotRequired[float]
    k_h: NotRequired[float]
    k_e: NotRequired[float]
    alpha_fe: NotRequired[float]
    m_core: NotRequired[float]
    B_core: NotRequired[float]
    winding_temp_limit_basis: NotRequired[str]
    winding_temp_limit_source: NotRequired[str]
    magnet_temp_max: NotRequired[float]
    magnet_temp_max_basis: NotRequired[str]
    magnet_temp_max_source: NotRequired[str]


MOTOR_COMPOUND_KEYS: frozenset[str] = frozenset({
    "drive", "geometry", "induction", "flux_map", "thermal_network",
    "magnetization", "halbach_segments", "halbach_angles_deg", "bh_curve",
})


class RunConfigDict(TypedDict):
    motor: MotorDict
    model: str
    maxh_fraction: float
    n_theta: int
    nonlinear: bool
    j_s: float

    i_fault: NotRequired[float]
    sim_plan: NotRequired[dict[str, Any]]
    duty_profile: NotRequired[list[list[float]]]
    dataset_id: NotRequired[str]
    load_mech: NotRequired[dict[str, Any]]
    nonlinear_solver: NotRequired[str]
    rotation: NotRequired[float]
    cogging_points: NotRequired[int]
    n_planes: NotRequired[int]
    slip: NotRequired[float]
    w_mech: NotRequired[float]
    tau_ref: NotRequired[float]
    T_coolant: NotRequired[float]


class RunResultDict(TypedDict):
    config: RunConfigDict
    model: str
    # str, not Literal: stores hold provenance-qualified sources such as
    # "published (ANSYS FEM)".
    source: str
    motor_config_id: str
    status: Literal["OK", "TIMEOUT", "ERROR"]
    metrics: dict[str, Any] | None
    elapsed_s: float
    error_msg: str | None
    timestamp: str
    model_version: int | None
    schema_version: str

    tolerances: NotRequired[dict[str, float]]


class StoredRecord(RunResultDict):
    """A RunResultDict after sweepstore's envelope: `run_id` is stamped
    on save, but pre-envelope rows lack it."""

    run_id: NotRequired[str]


def load_record_vectors() -> dict[str, Any]:
    """Pinned real store rows with their expected round-trip shapes."""
    return json.loads(
        (files("phasesweep") / "_vectors" / "record_vectors.json").read_text()
    )
