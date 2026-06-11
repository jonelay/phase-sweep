"""Motor dataclass composing Geometry with electrical/winding/material fields.

Motor is the data model. All fields except name, geometry, and n_p
are optional (None means "not yet measured/specified").
"""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass
from math import pi
from typing import Any

from phasesweep.geometry import Geometry


@dataclass(frozen=True)
class DriveParams:
    """Drive/controller configuration.

    Field names are deliberately ALL_CAPS to map 1:1 onto motulator's
    config names (kept at 1.0 by decision).

      U_DC    — DC bus voltage (V)
      MAX_I_S — maximum stator current (A, peak)
      W_REF   — reference speed (MECHANICAL rad/s)
      I_LIMIT — optional drive current limit for stall torque (A, peak);
                falls back to MAX_I_S when None
    """

    U_DC: float = 540.0
    MAX_I_S: float = 20.0
    W_REF: float = 2 * pi * 50
    I_LIMIT: float | None = None

    def __post_init__(self) -> None:
        if self.U_DC <= 0:
            raise ValueError(f"U_DC must be > 0, got {self.U_DC}")
        if self.MAX_I_S <= 0:
            raise ValueError(f"MAX_I_S must be > 0, got {self.MAX_I_S}")
        if self.W_REF <= 0:
            raise ValueError(f"W_REF must be > 0, got {self.W_REF}")
        if self.I_LIMIT is not None and self.I_LIMIT <= 0:
            raise ValueError(f"I_LIMIT must be > 0, got {self.I_LIMIT}")


@dataclass(frozen=True)
class Motor:
    """A PMSM: Geometry plus electrical/winding/material parameters.

    Only name, geometry, and n_p are required; None means "not yet
    measured/specified" — the prepare_* factories raise ValueError when
    a solver needs a missing field. All currents and flux linkages are
    peak-valued (motulator convention).

      n_p          — pole pairs
      R_s          — stator phase resistance (Ω)
      L_d, L_q     — d/q-axis inductance (H)
      psi_f        — PM flux linkage per phase (Wb, peak)
      B_rem        — magnet remanence (T)
      J            — rotor inertia (kg·m²)
      N            — turns per coil
      k_w          — winding factor (dimensionless, (0, 1])
      L_stk        — stack axial length (m)
      I_rated      — rated continuous stator current (A, peak)
      coils_series — coils in series per phase (N_eff = N × coils_series)
      alpha_p      — magnet pole-arc / pole-pitch ratio ((0, 1])
      mu_r_fe      — relative permeability of iron (dimensionless)
      mu_r_pm      — magnet recoil permeability (dimensionless)
      drive        — DriveParams operating point
    """

    name: str
    geometry: Geometry
    n_p: int

    R_s: float | None = None
    L_d: float | None = None
    L_q: float | None = None
    psi_f: float | None = None
    B_rem: float | None = None
    J: float | None = None
    N: int | None = None
    k_w: float | None = None
    L_stk: float | None = None
    I_rated: float | None = None
    coils_series: int | None = None

    alpha_p: float = 1.0

    mu_r_fe: float = 1000.0
    mu_r_pm: float = 1.05

    drive: DriveParams = dataclasses.field(default_factory=DriveParams)

    def __post_init__(self) -> None:
        if self.n_p < 2:
            raise ValueError(f"n_p must be >= 2, got {self.n_p}")
        if self.R_s is not None and self.R_s < 0:
            raise ValueError(f"R_s must be >= 0, got {self.R_s}")
        if self.L_d is not None and self.L_d <= 0:
            raise ValueError(f"L_d must be > 0, got {self.L_d}")
        if self.L_q is not None and self.L_q <= 0:
            raise ValueError(f"L_q must be > 0, got {self.L_q}")
        if self.psi_f is not None and self.psi_f <= 0:
            raise ValueError(f"psi_f must be > 0, got {self.psi_f}")
        if self.B_rem is not None and self.B_rem <= 0:
            raise ValueError(f"B_rem must be > 0, got {self.B_rem}")
        if self.J is not None and self.J <= 0:
            raise ValueError(f"J must be > 0, got {self.J}")
        if self.N is not None and self.N <= 0:
            raise ValueError(f"N must be > 0, got {self.N}")
        if self.k_w is not None and not (0 < self.k_w <= 1):
            raise ValueError(f"k_w must be in (0, 1], got {self.k_w}")
        if self.L_stk is not None and self.L_stk <= 0:
            raise ValueError(f"L_stk must be > 0, got {self.L_stk}")
        if self.I_rated is not None and self.I_rated <= 0:
            raise ValueError(f"I_rated must be > 0, got {self.I_rated}")
        if self.coils_series is not None and self.coils_series <= 0:
            raise ValueError(f"coils_series must be > 0, got {self.coils_series}")
        if not (0 < self.alpha_p <= 1):
            raise ValueError(f"alpha_p must be in (0, 1], got {self.alpha_p}")
        if self.mu_r_fe <= 0:
            raise ValueError(f"mu_r_fe must be > 0, got {self.mu_r_fe}")
        if self.mu_r_pm <= 0:
            raise ValueError(f"mu_r_pm must be > 0, got {self.mu_r_pm}")

    @property
    def config_id(self) -> str:
        parts = [self.geometry.config_id, str(self.n_p)]
        for name in ("R_s", "L_d", "L_q", "psi_f", "B_rem", "J", "N", "k_w", "L_stk", "I_rated", "coils_series"):
            val = getattr(self, name)
            if val is not None:
                parts.append(f"{name}={val}")
        parts.append(f"alpha_p={self.alpha_p}")
        parts.append(f"mu_r_fe={self.mu_r_fe}")
        parts.append(f"mu_r_pm={self.mu_r_pm}")
        key = "|".join(parts)
        return hashlib.md5(key.encode()).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        geo = dataclasses.asdict(self.geometry)
        drive = dataclasses.asdict(self.drive)
        d: dict[str, Any] = {
            "name": self.name,
            "geometry": geo,
            "n_p": self.n_p,
            "alpha_p": self.alpha_p,
            "mu_r_fe": self.mu_r_fe,
            "mu_r_pm": self.mu_r_pm,
            "drive": drive,
        }
        for name in ("R_s", "L_d", "L_q", "psi_f", "B_rem", "J", "N", "k_w", "L_stk", "I_rated", "coils_series"):
            val = getattr(self, name)
            if val is not None:
                d[name] = val
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Motor:
        geo_d = d["geometry"]
        geo = Geometry(**geo_d)
        drive_d = d.get("drive", {})
        drive = DriveParams(**drive_d) if drive_d else DriveParams()
        kwargs: dict[str, Any] = {
            "name": d["name"],
            "geometry": geo,
            "n_p": d["n_p"],
            "alpha_p": d.get("alpha_p", 1.0),
            "mu_r_fe": d.get("mu_r_fe", 1000.0),
            "mu_r_pm": d.get("mu_r_pm", 1.05),
            "drive": drive,
        }
        for name in ("R_s", "L_d", "L_q", "psi_f", "B_rem", "J", "N", "k_w", "L_stk", "I_rated", "coils_series"):
            if name in d:
                kwargs[name] = d[name]
        return cls(**kwargs)
