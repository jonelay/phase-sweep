from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from numpy.typing import NDArray


class MotorConfig(TypedDict):
    n_p: int
    R_s: float
    L_d: float
    L_q: float
    psi_f: float
    J: float
    n_slots: int
    j_s: float


class FullMotorConfig(MotorConfig, total=False):
    N: int
    k_w: float
    L_stk: float


class DriveParams(TypedDict):
    U_DC: float
    MAX_I_S: float
    W_REF: float


@dataclass
class Motor:
    config: FullMotorConfig
    drive: DriveParams
    name: str
    metadata: dict[str, Any] = field(default_factory=dict)


def load_motor(path: str | Path) -> Motor:
    """Load a TOML motor definition file."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    circuit = raw["circuit"]
    winding = raw["winding"]
    geometry = raw.get("geometry", {})
    rated = raw.get("rated", {})

    config: FullMotorConfig = {
        "n_p": circuit["n_p"],
        "R_s": circuit["R_s"],
        "L_d": circuit["L_d"],
        "L_q": circuit["L_q"],
        "psi_f": circuit["psi_f"],
        "J": circuit["J"],
        "n_slots": winding["n_slots"],
        "j_s": rated.get("j_s", 0.0),
        "N": winding["N"],
        "k_w": winding["k_w"],
        "L_stk": geometry.get("L_stk", 0.10),
    }

    drive_raw = raw.get("drive", {})
    drive: DriveParams = {
        "U_DC": drive_raw.get("U_DC", U_DC),
        "MAX_I_S": drive_raw.get("MAX_I_S", MAX_I_S),
        "W_REF": drive_raw.get("W_REF", W_REF),
    }

    metadata: dict[str, Any] = {}
    for section in ("geometry", "materials", "rated", "provenance"):
        if section in raw:
            metadata[section] = raw[section]

    return Motor(
        config=config,
        drive=drive,
        name=raw.get("motor", {}).get("name", Path(path).stem),
        metadata=metadata,
    )


def load_motors(directory: str | Path = "motors") -> dict[str, Motor]:
    """Load all .toml motor files from a directory."""
    results: dict[str, Motor] = {}
    for p in sorted(Path(directory).glob("*.toml")):
        motor = load_motor(p)
        results[motor.name] = motor
    return results


CONFIGS: dict[str, FullMotorConfig] = {
    "A: 4-pole SPMSM": FullMotorConfig(
        n_p=2,
        R_s=0.2,
        L_d=4e-3,
        L_q=4e-3,
        psi_f=0.1,
        J=0.002,
        n_slots=12,
        j_s=0.10,
        N=50,
        k_w=0.966,
        L_stk=0.10,
    ),
    "B: 4-pole IPMSM": FullMotorConfig(
        n_p=2,
        R_s=0.2,
        L_d=4e-3,
        L_q=6e-3,
        psi_f=0.1,
        J=0.002,
        n_slots=12,
        j_s=0.10,
        N=50,
        k_w=0.966,
        L_stk=0.10,
    ),
    "C: 8-pole SPMSM": FullMotorConfig(
        n_p=4,
        R_s=0.3,
        L_d=2e-3,
        L_q=2e-3,
        psi_f=0.08,
        J=0.002,
        n_slots=24,
        j_s=0.08,
        N=50,
        k_w=0.966,
        L_stk=0.10,
    ),
}

U_DC: float = 540
MAX_I_S: float = 20
T_STOP: float = 2.0
W_REF: float = 2 * np.pi * 50

SWEEP_BASE: dict[str, float | int] = dict(n_p=2, R_s=0.2, L_d=4e-3, J=0.002)
SWEEP_LOAD: float = 3.0
LOAD_T: float = 1.2
T_STOP_SWEEP: float = 1.8

PSI_F_VALS: NDArray[np.floating] = np.linspace(0.04, 0.22, 6)
RATIO_VALS: NDArray[np.floating] = np.linspace(1.0, 3.0, 6)
