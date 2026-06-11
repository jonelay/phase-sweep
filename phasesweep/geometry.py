"""Geometry dataclass for PMSM cross-section definition.

Flat frozen dataclass in physical meters. Named radii map to Zhu & Howe 2002:
  r_stator = R_s (stator bore), r_magnet = R_m, r_rotor = R_r.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class Geometry:
    """PMSM cross-section in physical SI units (radii and widths in meters).

    Radii, outside-in for an inrunner (ordering enforced in __post_init__;
    outrunner reverses r_stator/r_magnet/r_rotor):
      r_outer   — outermost boundary / frame (m)
      r_stator  — stator bore at the airgap, Zhu & Howe R_s (m)
      r_magnet  — PM/airgap interface, Zhu & Howe R_m (m)
      r_rotor   — rotor iron behind the magnets, Zhu & Howe R_r (m)
      r_inner   — shaft OD or hollow-center ID (m)
      r_ag      — airgap sampling radius, default mid-gap (m)

    Slots (0/0.0 = smooth bore):
      n_slots            — stator slot count
      slot_depth         — radial slot depth (m)
      slot_width_ratio   — slot body width / slot pitch (dimensionless)
      slot_opening_width — slot throat width at the bore (m);
                           slot_opening_ratio is derived from it
    """

    topology: Literal["inrunner", "outrunner"]

    r_outer: float
    r_stator: float
    r_magnet: float
    r_rotor: float
    r_inner: float
    r_ag: float

    n_slots: int = 0
    slot_depth: float = 0.0
    slot_width_ratio: float = 0.6
    slot_opening_width: float = 0.0

    def __post_init__(self) -> None:
        if self.topology == "inrunner":
            if not (self.r_outer > self.r_stator > self.r_magnet > self.r_rotor >= self.r_inner >= 0):
                raise ValueError(
                    f"Inrunner radii ordering violated: "
                    f"r_outer={self.r_outer} > r_stator={self.r_stator} > "
                    f"r_magnet={self.r_magnet} > r_rotor={self.r_rotor} >= "
                    f"r_inner={self.r_inner} >= 0"
                )
            if not (self.r_stator > self.r_ag > self.r_magnet):
                raise ValueError(
                    f"r_ag={self.r_ag} must be in airgap "
                    f"(r_stator={self.r_stator}, r_magnet={self.r_magnet})"
                )
        elif self.topology == "outrunner":
            if self.r_inner <= 0:
                raise ValueError(
                    f"Outrunner requires r_inner > 0, got {self.r_inner}"
                )
            if not (self.r_outer > self.r_rotor > self.r_magnet > self.r_stator > self.r_inner > 0):
                raise ValueError(
                    f"Outrunner radii ordering violated: "
                    f"r_outer={self.r_outer} > r_rotor={self.r_rotor} > "
                    f"r_magnet={self.r_magnet} > r_stator={self.r_stator} > "
                    f"r_inner={self.r_inner} > 0"
                )
            if not (self.r_magnet > self.r_ag > self.r_stator):
                raise ValueError(
                    f"r_ag={self.r_ag} must be in airgap "
                    f"(r_magnet={self.r_magnet}, r_stator={self.r_stator})"
                )
        else:
            raise ValueError(f"Unknown topology: {self.topology!r}")

    @property
    def slot_opening_ratio(self) -> float:
        """Slot opening over slot pitch at the bore. Derived from
        slot_opening_width so it can never go stale against n_slots."""
        if self.n_slots <= 0 or self.slot_opening_width <= 0:
            return 0.0
        return self.slot_opening_width / (2 * math.pi * self.r_stator / self.n_slots)

    @property
    def config_id(self) -> str:
        key = (
            f"{self.topology}_{self.r_outer:.6f}_{self.r_stator:.6f}_"
            f"{self.r_magnet:.6f}_{self.r_rotor:.6f}_{self.r_inner:.6f}_"
            f"{self.r_ag:.6f}_{self.n_slots}_{self.slot_depth:.6f}_"
            f"{self.slot_width_ratio:.6f}_{self.slot_opening_width:.6f}"
        )
        return hashlib.md5(key.encode()).hexdigest()[:12]


def inrunner(
    *,
    r_outer: float,
    r_stator: float,
    r_magnet: float,
    r_rotor: float,
    r_inner: float = 0.0,
    r_ag: float | None = None,
    n_slots: int = 0,
    slot_depth: float = 0.0,
    slot_width_ratio: float = 0.6,
    slot_opening_width: float = 0.0,
) -> Geometry:
    """Build an inrunner Geometry (rotor inside stator). All lengths in
    meters; r_ag defaults to the airgap midpoint (r_stator + r_magnet)/2;
    r_inner defaults to 0.0 (solid shaft)."""
    if r_ag is None:
        r_ag = (r_stator + r_magnet) / 2
    return Geometry(
        topology="inrunner",
        r_outer=r_outer, r_stator=r_stator, r_magnet=r_magnet,
        r_rotor=r_rotor, r_inner=r_inner, r_ag=r_ag,
        n_slots=n_slots, slot_depth=slot_depth, slot_width_ratio=slot_width_ratio,
        slot_opening_width=slot_opening_width,
    )


def outrunner(
    *,
    r_outer: float,
    r_rotor: float,
    r_magnet: float,
    r_stator: float,
    r_inner: float,
    r_ag: float | None = None,
    n_slots: int = 0,
    slot_depth: float = 0.0,
    slot_width_ratio: float = 0.6,
    slot_opening_width: float = 0.0,
) -> Geometry:
    """Build an outrunner Geometry (stator inside rotating magnet ring).
    All lengths in meters; r_ag defaults to the airgap midpoint
    (r_magnet + r_stator)/2; r_inner must be > 0 (hollow stator bore)."""
    if r_ag is None:
        r_ag = (r_magnet + r_stator) / 2
    return Geometry(
        topology="outrunner",
        r_outer=r_outer, r_stator=r_stator, r_magnet=r_magnet,
        r_rotor=r_rotor, r_inner=r_inner, r_ag=r_ag,
        n_slots=n_slots, slot_depth=slot_depth, slot_width_ratio=slot_width_ratio,
        slot_opening_width=slot_opening_width,
    )


def default_inrunner() -> Geometry:
    return inrunner(
        r_outer=1.0, r_stator=0.70, r_magnet=0.64, r_rotor=0.30, r_inner=0.0,
    )


def geometry_from_toml(geo: dict[str, Any], topology: str = "inrunner") -> Geometry:
    """Build Geometry from a TOML [geometry] section.

    Supports spec names (r_outer, r_stator, ...) and legacy TOML names
    (stator_od, stator_id, rotor_od, magnet_thickness).
    """
    if "r_outer" in geo:
        r_outer = geo["r_outer"]
        r_stator = geo["r_stator"]
        r_magnet = geo["r_magnet"]
        r_rotor = geo["r_rotor"]
        r_inner = geo.get("r_inner", 0.0)
    elif "stator_od" in geo:
        r_outer = geo["stator_od"] / 2
        r_stator = geo["stator_id"] / 2
        r_magnet = geo["rotor_od"] / 2
        magnet_thickness = geo.get("magnet_thickness", 0.0)
        r_rotor = r_magnet - magnet_thickness
        r_inner = geo.get("r_inner", 0.0)
    else:
        raise ValueError("Geometry section must contain r_outer or stator_od")

    r_ag = geo.get("r_ag")
    n_slots = geo.get("n_slots", 0)
    slot_depth = geo.get("slot_depth", geo.get("slot_height", 0.0))
    slot_width_ratio = geo.get("slot_width_ratio", 0.6)
    slot_opening_width = geo.get("slot_opening_width", 0.0)
    if "slot_opening_width" not in geo and "slot_opening_ratio" in geo and n_slots > 0:
        slot_pitch = 2 * math.pi * r_stator / n_slots
        slot_opening_width = geo["slot_opening_ratio"] * slot_pitch

    if topology == "inrunner":
        factory = inrunner
    elif topology == "outrunner":
        factory = outrunner
    else:
        raise ValueError(f"Unknown topology {topology!r} (expected 'inrunner' or 'outrunner')")
    return factory(
        r_outer=r_outer, r_stator=r_stator, r_magnet=r_magnet,
        r_rotor=r_rotor, r_inner=r_inner, r_ag=r_ag,
        n_slots=n_slots, slot_depth=slot_depth, slot_width_ratio=slot_width_ratio,
        slot_opening_width=slot_opening_width,
    )
