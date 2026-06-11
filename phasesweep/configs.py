from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import numpy as np

from phasesweep.geometry import geometry_from_toml
from phasesweep.motor import DriveParams as DriveParamsDC
from phasesweep.motor import Motor as MotorDC


def _require_section(raw: dict[str, Any], section: str, path: Path) -> dict[str, Any]:
    if section not in raw:
        raise ValueError(f"{path}: missing required section [{section}]")
    return raw[section]


def _require_field(section_data: dict[str, Any], field: str, section: str, path: Path) -> Any:
    if field not in section_data:
        raise ValueError(
            f"{path}: [{section}] missing required field '{field}'"
        )
    return section_data[field]


def load_motor(path: str | Path) -> MotorDC:
    """Load a TOML motor definition file into a Motor dataclass.

    TOML values are SI (meters, ohms, henries, Wb, T, A) with two
    conveniences: `I_rated_rms` is converted to peak (× √2) and
    `slot_opening_ratio` to `slot_opening_width` at parse time.
    Raises ValueError naming the file, section, and field on invalid input.
    """
    path = Path(path)
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    circuit = _require_section(raw, "circuit", path)
    n_p = _require_field(circuit, "n_p", "circuit", path)

    winding = raw.get("winding", {})
    geometry_raw = raw.get("geometry", {})
    materials = raw.get("materials", {})
    drive_raw = raw.get("drive", {})

    if not geometry_raw:
        raise ValueError(f"{path}: missing required section [geometry]")

    topology = raw.get("motor", {}).get("topology", "inrunner")
    if "n_slots" in winding:
        if "n_slots" in geometry_raw and geometry_raw["n_slots"] != winding["n_slots"]:
            raise ValueError(
                f"{path}: n_slots mismatch: [geometry] has "
                f"{geometry_raw['n_slots']}, [winding] has {winding['n_slots']}"
            )
        # Merge before building so a TOML slot_opening_ratio converts
        # to width using the [winding] slot count
        geometry_raw = {**geometry_raw, "n_slots": winding["n_slots"]}
    try:
        geo = geometry_from_toml(geometry_raw, topology)
    except (ValueError, KeyError) as e:
        raise ValueError(f"{path}: [geometry] {e}") from e

    try:
        drive = DriveParamsDC(
            U_DC=drive_raw.get("U_DC", 540.0),
            MAX_I_S=drive_raw.get("MAX_I_S", 20.0),
            W_REF=drive_raw.get("W_REF", 2 * np.pi * 50),
            I_LIMIT=drive_raw.get("I_LIMIT"),
        )
    except ValueError as e:
        raise ValueError(f"{path}: [drive] {e}") from e

    name = raw.get("motor", {}).get("name", path.stem)

    I_rated = circuit.get("I_rated")
    I_rated_rms = circuit.get("I_rated_rms")
    if I_rated is None and I_rated_rms is not None:
        from math import sqrt
        I_rated = I_rated_rms * sqrt(2)

    try:
        return MotorDC(
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
            mu_r_fe=materials.get("mu_r_fe", 1000.0),
            mu_r_pm=materials.get("mu_r_pm", 1.05),
            N=winding.get("N"),
            k_w=winding.get("k_w"),
            L_stk=geometry_raw.get("L_stk"),
            I_rated=I_rated,
            coils_series=winding.get("coils_series"),
            drive=drive,
        )
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from e


def load_motors(directory: str | Path = "motors") -> dict[str, MotorDC]:
    """Load all .toml motor files from a directory. Skips files that fail validation."""
    import logging
    results: dict[str, MotorDC] = {}
    for p in sorted(Path(directory).glob("*.toml")):
        try:
            motor = load_motor(p)
        except (KeyError, ValueError) as e:
            logging.warning("Skipping %s: %s", p.name, e)
            continue
        results[motor.name] = motor
    return results
