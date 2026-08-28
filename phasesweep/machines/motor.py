"""Motor dataclass composing Geometry with electrical/winding/material fields.

All fields except name, geometry, and n_p are optional (None means "not yet
measured/specified").
"""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass
from typing import Any

from phasesweep.machines.geometry import AxialFluxGeometry, Geometry


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


def _check_xy_table(
    name: str, table: tuple[tuple[float, float], ...] | None,
    *, y_positive: bool = True,
) -> None:
    """Validate an interpolation table: >= 2 rows of finite (x, y) pairs,
    x strictly increasing and > 0, y > 0 (when y_positive)."""
    import math

    if table is None:
        return
    if len(table) < 2:
        raise ValueError(f"{name} needs >= 2 rows, got {len(table)}")
    prev_x = -float("inf")
    for row in table:
        if len(row) != 2:
            raise ValueError(f"{name} rows must be (x, y) pairs, got {row}")
        x, y = row
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"{name} entries must be finite, got {row}")
        if x <= 0:
            # x = 0 would silently shadow the scalar reference that
            # anchors the interpolation at f/I = 0 (see _interp_anchored)
            raise ValueError(f"{name} x-values must be > 0, got {x}")
        if x <= prev_x:
            raise ValueError(
                f"{name} x-values must be strictly increasing, got {x} "
                f"after {prev_x}"
            )
        if y_positive and y <= 0:
            raise ValueError(f"{name} y-values must be > 0, got {y}")
        prev_x = x


@dataclass(frozen=True)
class InductionParams:
    """Squirrel-cage induction machine parameters (TOML [induction] section).

    A Motor with `induction` set is an induction machine: the synchronous
    d-q fields (psi_f, B_rem, L_d, L_q) must be None, and the
    synchronous-machine models reject it loudly. All rotor quantities
    are referred to the stator; inductances in H, resistances in Ω.

      R_2         — rotor resistance referred to the stator (Ω, at the
                    reference temperature and DC / low rotor frequency)
      L_m         — magnetizing inductance (H, rated-point reference)
      L_sigma_s   — stator leakage inductance (H)
      L_sigma_r   — rotor leakage inductance referred to the stator (H)
      n_bars      — cage bar count (documentation / future FEM tier)
      f_rated     — rated supply frequency (Hz)
      s_rated     — rated slip (dimensionless, (0, 1))
      V_LL_rated  — rated supply line-line voltage (V rms). Explicitly
                    line-line rms because IM datasheet voltage
                    conventions are the canonical trap (a naive plug-in
                    of a wye machine's L-L voltage as phase voltage
                    triples the apparent power).
      design_letter — NEMA design letter ("A"/"B"/"C"/"D"),
                    documentation + envelope-bound priors
      R_2_vs_f    — deep-bar table ((f_rotor_Hz, R_2_ohm), ...): rotor
                    resistance vs ROTOR frequency s·f. Real cages rise
                    steeply, so a constant R_2 gets the starting region
                    badly wrong.
      L_sigma_s_vs_f — stator leakage vs SUPPLY frequency table
      L_sigma_r_vs_f — rotor leakage vs ROTOR frequency table
      L_m_vs_i    — magnetizing saturation table
                    ((I_m_A_rms, L_m_H), ...): L_m vs magnetizing
                    branch current
    """

    R_2: float | None = None
    L_m: float | None = None
    L_sigma_s: float | None = None
    L_sigma_r: float | None = None
    n_bars: int | None = None
    f_rated: float | None = None
    s_rated: float | None = None
    V_LL_rated: float | None = None
    design_letter: str | None = None
    R_2_vs_f: tuple[tuple[float, float], ...] | None = None
    L_sigma_s_vs_f: tuple[tuple[float, float], ...] | None = None
    L_sigma_r_vs_f: tuple[tuple[float, float], ...] | None = None
    L_m_vs_i: tuple[tuple[float, float], ...] | None = None

    def __post_init__(self) -> None:
        # Normalize numeric types at construction (frozen dataclass, so
        # object.__setattr__): TOML/JSON round-trips may deliver 10 vs
        # 10.0, which compare equal but hash differently through the
        # config_id JSON serialization.
        for f in ("R_2", "L_m", "L_sigma_s", "L_sigma_r", "f_rated",
                  "s_rated", "V_LL_rated"):
            v = getattr(self, f)
            if v is not None:
                object.__setattr__(self, f, float(v))
        for f in ("R_2_vs_f", "L_sigma_s_vs_f", "L_sigma_r_vs_f",
                  "L_m_vs_i"):
            t = getattr(self, f)
            if t is not None:
                object.__setattr__(
                    self, f,
                    tuple(tuple(float(x) for x in row) for row in t),
                )
        if self.n_bars is not None:
            # int-normalize like the floats above: 28.0 vs 28 must not
            # split one machine into two config_ids
            if self.n_bars != int(self.n_bars):
                raise ValueError(
                    f"[induction] n_bars must be an integer, got {self.n_bars}")
            object.__setattr__(self, "n_bars", int(self.n_bars))
        for f in ("R_2", "L_m", "L_sigma_s", "L_sigma_r", "f_rated",
                  "V_LL_rated"):
            v = getattr(self, f)
            if v is not None and v <= 0:
                raise ValueError(f"[induction] {f} must be > 0, got {v}")
        if self.n_bars is not None and self.n_bars < 2:
            raise ValueError(
                f"[induction] n_bars must be >= 2, got {self.n_bars}")
        if self.s_rated is not None and not (0 < self.s_rated < 1):
            raise ValueError(
                f"[induction] s_rated must be in (0, 1), got {self.s_rated}")
        if (self.design_letter is not None
                and self.design_letter not in ("A", "B", "C", "D")):
            raise ValueError(
                f"[induction] design_letter must be one of A/B/C/D, "
                f"got {self.design_letter!r}")
        for name in ("R_2_vs_f", "L_sigma_s_vs_f", "L_sigma_r_vs_f",
                     "L_m_vs_i"):
            _check_xy_table(f"[induction] {name}", getattr(self, name))

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if v is None:
                continue
            if isinstance(v, tuple):
                d[f.name] = [list(row) for row in v]
            else:
                d[f.name] = v
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> InductionParams:
        kwargs: dict[str, Any] = {}
        for f in dataclasses.fields(cls):
            if f.name not in d:
                continue
            v = d[f.name]
            if f.name.endswith(("_vs_f", "_vs_i")) and v is not None:
                v = tuple((float(x), float(y)) for x, y in v)
            kwargs[f.name] = v
        return cls(**kwargs)


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
      k_end        — 3D axial end-effect factor on the winding flux
                     linkage (dimensionless, (0, ~1.05]; > 1 is real
                     under magnet overhang). A computed-and-committed
                     value from the offline wedge solve
                     (scripts/compute_k_end.py) with a provenance
                     record — NOT the uncalibrated analytical
                     end_effect_factor. None = no correction applied.
      alpha_p      — magnet pole-arc / pole-pitch ratio ((0, 1])
      mu_r_fe      — relative permeability of iron (dimensionless)
      mu_r_pm      — magnet recoil permeability (dimensionless)
      drive        — DriveParams operating point
      induction    — InductionParams for a squirrel-cage induction
                     machine (None = synchronous machine). Mutually
                     exclusive with the d-q synchronous fields psi_f,
                     B_rem, L_d, L_q — `motor.induction is not None` is
                     THE topology discriminator, and both model families
                     reject the other topology loudly.

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
    geometry: Geometry | AxialFluxGeometry | None
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
    k_end: float | None = None

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

    magnetization: str = "radial"
    halbach_segments: int | None = None
    halbach_angles_deg: tuple[float, ...] | None = None

    mu_r_fe: float = 1000.0
    mu_r_pm: float = 1.05

    drive: DriveParams = dataclasses.field(default_factory=DriveParams)

    induction: InductionParams | None = None

    def __post_init__(self) -> None:
        if self.n_p < 2:
            raise ValueError(f"n_p must be >= 2, got {self.n_p}")
        if self.induction is not None:
            pm_set = [f for f in ("psi_f", "B_rem", "L_d", "L_q")
                      if getattr(self, f) is not None]
            if pm_set:
                raise ValueError(
                    f"[induction] is set together with the synchronous d-q "
                    f"field(s) {', '.join(pm_set)} — a squirrel-cage "
                    f"induction machine has no PM flux linkage or d-q "
                    f"inductances. Remove one side."
                )
        if self.R_s is not None and self.R_s < 0:
            raise ValueError(f"R_s must be >= 0, got {self.R_s}")
        if self.L_d is not None and self.L_d <= 0:
            raise ValueError(f"L_d must be > 0, got {self.L_d}")
        if self.L_q is not None and self.L_q <= 0:
            raise ValueError(f"L_q must be > 0, got {self.L_q}")
        if self.psi_f is not None and self.psi_f < 0:
            raise ValueError(f"psi_f must be >= 0, got {self.psi_f}")
        if self.B_rem is not None and self.B_rem <= 0:
            raise ValueError(f"B_rem must be > 0, got {self.B_rem}")
        if self.psi_f == 0.0 and self.B_rem is not None:
            raise ValueError(
                f"psi_f = 0 (reluctance machine, no magnets) contradicts "
                f"B_rem={self.B_rem} — remove B_rem or set psi_f > 0"
            )
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
        if self.k_end is not None and not (0 < self.k_end <= 1.2):
            raise ValueError(
                f"k_end must be in (0, 1.2], got {self.k_end} — computed "
                f"end-effect factors land in (0, ~1.05] (fem_wedge)"
            )
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
        if (
            self.geometry is not None
            and hasattr(self.geometry, "magnet_mount")
            and self.geometry.magnet_mount == "inset"
            and self.alpha_p >= 1.0
        ):
            raise ValueError(
                "magnet_mount='inset' requires alpha_p < 1: the interpole "
                f"iron IS the inset feature (got alpha_p={self.alpha_p})"
            )
        if self.magnetization not in ("radial", "halbach"):
            raise ValueError(
                f"magnetization must be 'radial' or 'halbach', "
                f"got {self.magnetization!r}"
            )
        if self.magnetization == "halbach":
            if (
                self.halbach_segments is None
                or not isinstance(self.halbach_segments, int)
                or isinstance(self.halbach_segments, bool)
                or self.halbach_segments < 1
            ):
                raise ValueError(
                    f"halbach_segments must be >= 1 when magnetization='halbach', "
                    f"got {self.halbach_segments}"
                )
            if self.alpha_p != 1.0:
                raise ValueError(
                    f"Halbach magnetization requires alpha_p=1.0 (full-pitch "
                    f"magnet ring); partial-pitch Halbach arrays are not yet "
                    f"supported. Got alpha_p={self.alpha_p}"
                )
            if self.halbach_angles_deg is not None:
                import math
                if len(self.halbach_angles_deg) != self.halbach_segments:
                    raise ValueError(
                        f"halbach_angles_deg length ({len(self.halbach_angles_deg)}) "
                        f"must equal halbach_segments ({self.halbach_segments})"
                    )
                if not all(
                    isinstance(a, (int, float)) and math.isfinite(a)
                    for a in self.halbach_angles_deg
                ):
                    raise ValueError(
                        "halbach_angles_deg values must be finite numbers"
                    )
        else:
            if self.halbach_segments is not None:
                raise ValueError(
                    "halbach_segments must be None when magnetization='radial'"
                )
            if self.halbach_angles_deg is not None:
                raise ValueError(
                    "halbach_angles_deg must be None when magnetization='radial'"
                )
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
            "I_rated", "coils_series", "k_end",
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
        if self.magnetization != "radial":
            parts.append(f"magnetization={self.magnetization}")
            parts.append(f"halbach_segments={self.halbach_segments}")
            if self.halbach_angles_deg is not None:
                parts.append(f"halbach_angles_deg={self.halbach_angles_deg}")
        if self.induction is not None:
            # Append-only under the not-None rule: every synchronous
            # motor's hash is byte-identical to before. R_2/L_m/leakages
            # and their tables ARE machine properties (unlike [drive],
            # which is an operating point and hashes via compute_run_id);
            # slip and supply overrides go in RunConfig.
            import json
            parts.append(
                "induction="
                + json.dumps(self.induction.to_dict(), sort_keys=True)
            )
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
        if self.induction is not None:
            d["induction"] = self.induction.to_dict()
        if self.geometry is not None:
            d["geometry"] = dataclasses.asdict(self.geometry)
        if self.magnetization != "radial":
            d["magnetization"] = self.magnetization
            d["halbach_segments"] = self.halbach_segments
            if self.halbach_angles_deg is not None:
                d["halbach_angles_deg"] = list(self.halbach_angles_deg)
        for name in (
            "R_s", "L_d", "L_q", "psi_f", "B_rem", "J", "N", "k_w", "L_stk",
            "I_rated", "coils_series", "k_end",
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
        geo: Geometry | AxialFluxGeometry | None = None
        if geo_d:
            if "variant" in geo_d:
                geo = AxialFluxGeometry(**geo_d)
            else:
                geo = Geometry(**geo_d)
        drive_d = d.get("drive", {})
        drive = DriveParams(**drive_d) if drive_d else DriveParams()
        induction_d = d.get("induction")
        induction = (
            InductionParams.from_dict(induction_d) if induction_d else None
        )
        angles_raw = d.get("halbach_angles_deg")
        kwargs: dict[str, Any] = {
            "name": d["name"],
            "geometry": geo,
            "n_p": d["n_p"],
            "alpha_p": d.get("alpha_p", 1.0),
            "magnetization": d.get("magnetization", "radial"),
            "halbach_segments": d.get("halbach_segments"),
            "halbach_angles_deg": (
                tuple(angles_raw) if angles_raw is not None else None
            ),
            "mu_r_fe": d.get("mu_r_fe", 1000.0),
            "mu_r_pm": d.get("mu_r_pm", 1.05),
            "drive": drive,
            "induction": induction,
        }
        for name in (
            "R_s", "L_d", "L_q", "psi_f", "B_rem", "J", "N", "k_w", "L_stk",
            "I_rated", "coils_series", "k_end",
            "winding_temp_limit", "ambient_temp", "r_th", "insulation_class",
            "thermal_time_constant",
            "alpha_Br", "magnet_temp",
            "B_knee", "alpha_B_knee",
            "k_h", "k_e", "alpha_fe", "m_core", "B_core",
        ):
            if name in d:
                kwargs[name] = d[name]
        return cls(**kwargs)
