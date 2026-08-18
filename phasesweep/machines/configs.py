from __future__ import annotations

import logging
import tomllib
from pathlib import Path
from typing import Any

from phasesweep.machines.geometry import geometry_from_toml
from phasesweep.machines.motor import DriveParams, Motor


def _require_section(raw: dict[str, Any], section: str, label: str) -> dict[str, Any]:
    if section not in raw:
        raise ValueError(f"{label}: missing required section [{section}]")
    return raw[section]


def _require_field(section_data: dict[str, Any], field: str, section: str, label: str) -> Any:
    if field not in section_data:
        raise ValueError(
            f"{label}: [{section}] missing required field '{field}'"
        )
    return section_data[field]


def _get_or_warn(section: dict[str, Any], field: str, default: Any, label: str) -> Any:
    if field not in section:
        logging.warning(
            "%s: missing [materials] %s — defaulting to %s", label, field, default,
        )
        return default
    return section[field]


def load_motor(path: str | Path) -> Motor:
    """Load a TOML motor definition file into a Motor dataclass.

    TOML values are SI (meters, ohms, henries, Wb, T, A) with two
    conveniences: `I_rated_rms` is converted to peak (× √2) and
    `slot_opening_ratio` to `slot_opening_width` at parse time.
    Raises ValueError naming the file, section, and field on invalid input.
    """
    path = Path(path)
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return motor_from_dict(raw, label=str(path), default_name=path.stem)


def motor_from_dict(raw: dict[str, Any], *, label: str, default_name: str) -> Motor:
    """Build a Motor from an already-parsed TOML structure. `label` names
    the source in error messages; `default_name` stands in for the file
    stem when [motor] has no name. The dict-level entry point lets the
    server validate a config-editor payload before writing TOML.
    """
    circuit = _require_section(raw, "circuit", label)
    n_p = _require_field(circuit, "n_p", "circuit", label)

    # `or {}` not `get(k, {})`: a JSON payload from the config editor can
    # carry an explicit null for a section, and the default only applies
    # when the key is absent (a present null crashed the PUT route).
    motor_raw = raw.get("motor") or {}
    winding = raw.get("winding") or {}
    geometry_raw = raw.get("geometry") or {}
    materials = raw.get("materials") or {}
    drive_raw = raw.get("drive") or {}
    thermal = raw.get("thermal") or {}
    iron = raw.get("iron") or {}

    # [geometry] is optional: datasheet/circuit-only motors run the circuit
    # tier (rated/stall torque, torque_speed, drive_sim, thermal_duty,
    # iron_loss) without it. The
    # field/FEM factories raise if a field solver is requested.
    geo = None
    if geometry_raw:
        topology = motor_raw.get("topology", "inrunner")
        if "n_slots" in winding:
            if "n_slots" in geometry_raw and geometry_raw["n_slots"] != winding["n_slots"]:
                raise ValueError(
                    f"{label}: n_slots mismatch: [geometry] has "
                    f"{geometry_raw['n_slots']}, [winding] has {winding['n_slots']}"
                )
            # Merge before building so a TOML slot_opening_ratio converts
            # to width using the [winding] slot count
            geometry_raw = {**geometry_raw, "n_slots": winding["n_slots"]}
        try:
            geo = geometry_from_toml(geometry_raw, topology)
        except (ValueError, KeyError) as e:
            raise ValueError(f"{label}: [geometry] {e}") from e

    try:
        drive = DriveParams(
            U_DC=drive_raw.get("U_DC"),
            MAX_I_S=drive_raw.get("MAX_I_S"),
            W_REF=drive_raw.get("W_REF"),
            I_LIMIT=drive_raw.get("I_LIMIT"),
        )
    except ValueError as e:
        raise ValueError(f"{label}: [drive] {e}") from e

    name = motor_raw.get("name", default_name)

    I_rated = circuit.get("I_rated")
    I_rated_rms = circuit.get("I_rated_rms")
    if I_rated is None and I_rated_rms is not None:
        from math import sqrt
        I_rated = I_rated_rms * sqrt(2)

    try:
        return Motor(
            name=name,
            geometry=geo,
            n_p=n_p,
            R_s=circuit.get("R_s"),
            L_d=circuit.get("L_d"),
            L_q=circuit.get("L_q"),
            psi_f=circuit.get("psi_f"),
            J=circuit.get("J"),
            B_rem=materials.get("B_rem"),
            alpha_p=geometry_raw.get("alpha_p", 1.0),
            mu_r_fe=_get_or_warn(materials, "mu_r_fe", 1000.0, label),
            mu_r_pm=materials.get("mu_r_pm", 1.05),
            N=winding.get("N"),
            k_w=winding.get("k_w"),
            L_stk=geometry_raw.get("L_stk"),
            I_rated=I_rated,
            coils_series=winding.get("coils_series"),
            winding_temp_limit=thermal.get("winding_temp_limit"),
            ambient_temp=thermal.get("ambient_temp"),
            r_th=thermal.get("r_th"),
            insulation_class=thermal.get("insulation_class"),
            thermal_time_constant=thermal.get("thermal_time_constant"),
            alpha_Br=materials.get("alpha_Br"),
            magnet_temp=thermal.get("magnet_temp"),
            B_knee=materials.get("B_knee"),
            alpha_B_knee=materials.get("alpha_B_knee"),
            k_h=iron.get("k_h"),
            k_e=iron.get("k_e"),
            alpha_fe=iron.get("alpha_fe"),
            m_core=iron.get("m_core"),
            B_core=iron.get("B_core"),
            drive=drive,
        )
    except ValueError as e:
        raise ValueError(f"{label}: {e}") from e


def load_motors(
    directory: str | Path = "motors", *, strict: bool = False
) -> dict[str, Motor]:
    """Load all .toml motor files from a directory.

    When strict is False (default), invalid files are skipped with a warning.
    When strict is True, the first validation error is raised.
    """
    results: dict[str, Motor] = {}
    for p in sorted(Path(directory).glob("*.toml")):
        try:
            motor = load_motor(p)
        except (KeyError, ValueError) as e:
            if strict:
                raise
            logging.warning("Skipping %s: %s", p.name, e)
            continue
        results[motor.name] = motor
    return results
