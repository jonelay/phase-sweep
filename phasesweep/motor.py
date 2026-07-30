"""Motor dataclass composing Geometry with electrical/winding/material fields.

Motor is the data model. All fields except name, geometry, and n_p
are optional (None means "not yet measured/specified").
"""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass
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

    U_DC: float | None = None
    MAX_I_S: float | None = None
    W_REF: float | None = None
    I_LIMIT: float | None = None

    def __post_init__(self) -> None:
        if self.U_DC is not None and self.U_DC <= 0:
            raise ValueError(f"U_DC must be > 0, got {self.U_DC}")
        if self.MAX_I_S is not None and self.MAX_I_S <= 0:
            raise ValueError(f"MAX_I_S must be > 0, got {self.MAX_I_S}")
        if self.W_REF is not None and self.W_REF <= 0:
            raise ValueError(f"W_REF must be > 0, got {self.W_REF}")
        if self.I_LIMIT is not None and self.I_LIMIT <= 0:
            raise ValueError(f"I_LIMIT must be > 0, got {self.I_LIMIT}")


@dataclass(frozen=True)
class Motor:
    """A PMSM: Geometry plus electrical/winding/material parameters.

    name, geometry, and n_p are the only required arguments (geometry
    may be None — see below). None elsewhere means "not yet
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

    geometry is optional (None) for datasheet/circuit-only motors with no
    teardown: the field/FEM factories raise on missing geometry, while the
    circuit tier (rated/stall torque, torque_speed, drive_sim,
    thermal_duty, iron_loss) runs from d-q params alone.

    Optional thermal context (None = not yet specified) feeds the
    thermal_duty model's S1 budget:
      winding_temp_limit — max winding temperature (°C)
      ambient_temp       — ambient temperature (°C)
      r_th               — winding→ambient thermal resistance (K/W)
      insulation_class   — documentation tag (e.g. "F", "H")
      thermal_time_constant — winding thermal time constant τ_th (s);
                    enables the first-order transient march in
                    thermal_duty (datasheet τth, e.g. the ETEL TMB sheets)

    Optional magnet temperature derating, applied at the solver
    boundary — psi_f and B_rem stay at the 20 °C reference on the Motor:
      alpha_Br    — B_rem/psi_f fractional temperature coefficient (per K,
                    negative; e.g. NdFeB N42 ≈ -0.0012, SmCo ≈ -0.0003).
                    Datasheets print %/°C — divide by 100.
      magnet_temp — magnet operating temperature (°C)

    Optional demag knee inputs (TOML [materials] section) for
    the demag_screen model:
      B_knee       — demag knee flux density (T) at the 20 °C reference:
                     the magnet operating point B_m below which
                     irreversible demagnetization onsets. May be ≤ 0
                     (grades whose knee sits below zero at 20 °C).
      alpha_B_knee — absolute knee temperature slope dB_knee/dT (T/K).
                     Positive for NdFeB (knee rises when hot), negative
                     for ferrite (knee rises when cold). Absolute, not
                     fractional — the knee can cross zero.

    Optional lumped iron-loss inputs (TOML [iron] section) for
    the Bertotti iron_loss model — per-kg steel Steinmetz coefficients
    plus a single-B core approximation:
      k_h      — hysteresis coefficient (W·s/kg/T^alpha_fe)
      k_e      — classical eddy coefficient (W·s²/kg/T²)
      alpha_fe — Steinmetz flux-density exponent (~1.5-1.8 for Si steel)
      m_core   — core mass cycling at B_core (kg)
      B_core   — peak core flux density at the operating point (T)
    """

    name: str
    geometry: Geometry | None
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

    winding_temp_limit: float | None = None
    ambient_temp: float | None = None
    r_th: float | None = None
    insulation_class: str | None = None
    thermal_time_constant: float | None = None

    alpha_Br: float | None = None
    magnet_temp: float | None = None

    B_knee: float | None = None
    alpha_B_knee: float | None = None

    k_h: float | None = None
    k_e: float | None = None
    alpha_fe: float | None = None
    m_core: float | None = None
    B_core: float | None = None

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
        if self.r_th is not None and self.r_th <= 0:
            raise ValueError(f"r_th must be > 0, got {self.r_th}")
        if self.thermal_time_constant is not None and self.thermal_time_constant <= 0:
            raise ValueError(
                f"thermal_time_constant must be > 0, got {self.thermal_time_constant}"
            )
        if (
            self.winding_temp_limit is not None
            and self.ambient_temp is not None
            and self.winding_temp_limit <= self.ambient_temp
        ):
            raise ValueError(
                f"winding_temp_limit ({self.winding_temp_limit}) must exceed "
                f"ambient_temp ({self.ambient_temp})"
            )
        if self.alpha_Br is not None and not (-0.01 < self.alpha_Br < 0):
            raise ValueError(
                f"alpha_Br must be in (-0.01, 0) per K, got {self.alpha_Br} — "
                f"datasheets print %/°C (e.g. -0.12): divide by 100"
            )
        if (
            self.B_knee is not None
            and self.B_rem is not None
            and self.B_knee >= self.B_rem
        ):
            raise ValueError(
                f"B_knee ({self.B_knee}) must lie below B_rem ({self.B_rem})"
            )
        if self.alpha_B_knee is not None and not (-0.1 < self.alpha_B_knee < 0.1):
            raise ValueError(
                f"alpha_B_knee must be in (-0.1, 0.1) T/K, got "
                f"{self.alpha_B_knee} — it is an absolute slope dB_knee/dT, "
                f"not a fractional coefficient"
            )
        if self.k_h is not None and self.k_h <= 0:
            raise ValueError(f"k_h must be > 0, got {self.k_h}")
        if self.k_e is not None and self.k_e <= 0:
            raise ValueError(f"k_e must be > 0, got {self.k_e}")
        if self.alpha_fe is not None and not (1.0 <= self.alpha_fe <= 3.0):
            raise ValueError(
                f"alpha_fe must be in [1, 3], got {self.alpha_fe}")
        if self.m_core is not None and self.m_core <= 0:
            raise ValueError(f"m_core must be > 0, got {self.m_core}")
        if self.B_core is not None and self.B_core <= 0:
            raise ValueError(f"B_core must be > 0, got {self.B_core}")
        if not (0 < self.alpha_p <= 1):
            raise ValueError(f"alpha_p must be in (0, 1], got {self.alpha_p}")
        if self.mu_r_fe <= 0:
            raise ValueError(f"mu_r_fe must be > 0, got {self.mu_r_fe}")
        if self.mu_r_pm <= 0:
            raise ValueError(f"mu_r_pm must be > 0, got {self.mu_r_pm}")

    @property
    def config_id(self) -> str:
        geo_id = self.geometry.config_id if self.geometry is not None else "none"
        parts = [geo_id, str(self.n_p)]
        for name in (
            "R_s", "L_d", "L_q", "psi_f", "B_rem", "J", "N", "k_w", "L_stk",
            "I_rated", "coils_series",
            "winding_temp_limit", "ambient_temp", "r_th",
            "thermal_time_constant",
            "alpha_Br", "magnet_temp",
            "B_knee", "alpha_B_knee",
            "k_h", "k_e", "alpha_fe", "m_core", "B_core",
        ):
            val = getattr(self, name)
            if val is not None:
                parts.append(f"{name}={val}")
        parts.append(f"alpha_p={self.alpha_p}")
        parts.append(f"mu_r_fe={self.mu_r_fe}")
        parts.append(f"mu_r_pm={self.mu_r_pm}")
        key = "|".join(parts)
        return hashlib.md5(key.encode()).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        drive = {k: v for k, v in dataclasses.asdict(self.drive).items()
                 if v is not None}
        d: dict[str, Any] = {
            "name": self.name,
            "n_p": self.n_p,
            "alpha_p": self.alpha_p,
            "mu_r_fe": self.mu_r_fe,
            "mu_r_pm": self.mu_r_pm,
        }
        if drive:
            d["drive"] = drive
        if self.geometry is not None:
            d["geometry"] = dataclasses.asdict(self.geometry)
        for name in (
            "R_s", "L_d", "L_q", "psi_f", "B_rem", "J", "N", "k_w", "L_stk",
            "I_rated", "coils_series",
            "winding_temp_limit", "ambient_temp", "r_th", "insulation_class",
            "thermal_time_constant",
            "alpha_Br", "magnet_temp",
            "B_knee", "alpha_B_knee",
            "k_h", "k_e", "alpha_fe", "m_core", "B_core",
        ):
            val = getattr(self, name)
            if val is not None:
                d[name] = val
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Motor:
        geo_d = d.get("geometry")
        geo = Geometry(**geo_d) if geo_d else None
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
        for name in (
            "R_s", "L_d", "L_q", "psi_f", "B_rem", "J", "N", "k_w", "L_stk",
            "I_rated", "coils_series",
            "winding_temp_limit", "ambient_temp", "r_th", "insulation_class",
            "thermal_time_constant",
            "alpha_Br", "magnet_temp",
            "B_knee", "alpha_B_knee",
            "k_h", "k_e", "alpha_fe", "m_core", "B_core",
        ):
            if name in d:
                kwargs[name] = d[name]
        return cls(**kwargs)
