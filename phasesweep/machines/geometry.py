"""Geometry dataclasses for PMSM cross-section definition.

Radial-flux: flat frozen dataclass in physical meters. Named radii map to
Zhu & Howe 2002: r_stator = R_s (stator bore), r_magnet = R_m, r_rotor = R_r.

Axial-flux: AxialFluxGeometry defines the annular active region plus the
axial stack (gaps, magnets, yokes). Separate class — different physical
parameterization, distinct config_id prefix, no inheritance from Geometry.
"""

from __future__ import annotations

import hashlib
import math
import numbers
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

__all__ = [
    "AxialFluxGeometry",
    "AxialFluxVariant",
    "Geometry",
    "axial_flux",
    "axial_flux_geometry_from_toml",
    "default_inrunner",
    "geometry_from_toml",
    "inrunner",
    "outrunner",
]


def int_normalize(obj: object, fields: tuple[str, ...], label: str) -> None:
    """Coerce integral count fields to int on a frozen dataclass.

    Counts are hashed into config_id as text, so 12.0 and 12 would split
    one machine into two ids. JSON editors and ``dataclasses.replace``
    legitimately produce integral floats; a non-integral count is an error.
    """
    for f in fields:
        v = getattr(obj, f)
        if v is None or type(v) is int:
            continue
        if isinstance(v, numbers.Integral) and not isinstance(v, bool):
            object.__setattr__(obj, f, int(v))
            continue
        if not (isinstance(v, float) and v.is_integer()):
            raise ValueError(f"{label} {f} must be an integer, got {v!r}")
        object.__setattr__(obj, f, int(v))


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

    Rotor back-iron (outrunner only):
      back_iron_thickness — radial thickness of the magnetic back-iron ring
                            (m); splits the yoke into iron ring + non-magnetic
                            shell. None = whole wall is solid iron (legacy).

    Magnet mounting (FEM only — the analytical solver gates on both):
      magnet_mount     — "surface" (interpole gaps are air, legacy) or
                         "inset" (magnets recessed in the rotor; interpole
                         gaps are rotor iron). Inset requires alpha_p < 1
                         (checked where alpha_p is known: Motor / FemParams).
      bridge_thickness — radial thickness (m) of an iron bridge ring
                         between the magnet faces and the airgap (retaining
                         ring cut from the PM annulus). Compatible with any
                         magnet_mount or magnetization (e.g. Halbach with a
                         retaining sleeve). None = magnet faces exposed to
                         the airgap.

    Squirrel-cage rotor (induction machines; n_bars = 0 means a PM rotor):
      n_bars               — rotor bar count. When > 0 the rotor carries
                             no PM annulus (r_magnet == r_rotor is the
                             rotor surface) and the geometry is inrunner
      bar_shape            — "rectangular" (radial-sided, area exactly
                             bar_width x bar_height) or "round" (circle
                             of diameter bar_width; bar_height must be 0)
      bar_width            — arc width at the bar mid-radius (m), or the
                             diameter for round bars
      bar_height           — radial extent of a rectangular bar (m)
      bar_bridge_thickness — iron between the rotor surface and the bar
                             top (m); 0 = bar flush with the surface
      bar_opening_width    — air notch through the bridge at the rotor
                             surface (m); 0 = closed slot. Requires a
                             bridge and must be narrower than the bar
      end_ring_axial_length, end_ring_radial_height — end-ring section
                             (m); not meshed in 2D, consumed by the
                             end-ring impedance correction. None = unknown
      skew_angle           — total mechanical bar-skew angle over the
                             stack (rad): the angular offset between a
                             bar's two ends. None = unskewed (a 2D solve
                             is exact in-plane). Consumed by the
                             multi-slice harmonic models; a TOML can give
                             skew_slots (stator slot pitches) instead

    Axial end geometry (3D k_end pipeline; not used by the 2D solvers):
      magnet_overhang — axial extension of the magnets per side beyond the
                        active stack L_stk (m). Consumed by the offline
                        wedge k_end solve (scripts/compute_k_end.py /
                        fem_wedge.compute_k_end) — the dominant lever on
                        the end-effect factor. 0.0 = measured co-extensive;
                        None = not measured.
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

    # Outrunner only: split the rotor yoke [r_rotor, r_outer] into a magnetic
    # back-iron ring of this radial thickness (iron) plus a non-magnetic shell
    # (mu_r=1) out to r_outer. None = the whole wall is solid iron (legacy).
    back_iron_thickness: float | None = None

    magnet_mount: Literal["surface", "inset"] = "surface"
    bridge_thickness: float | None = None

    magnet_overhang: float | None = None

    n_bars: int = 0
    bar_shape: Literal["rectangular", "round"] = "rectangular"
    bar_width: float = 0.0
    bar_height: float = 0.0
    bar_bridge_thickness: float = 0.0
    bar_opening_width: float = 0.0
    end_ring_axial_length: float | None = None
    end_ring_radial_height: float | None = None
    skew_angle: float | None = None

    def __post_init__(self) -> None:
        int_normalize(self, ("n_slots", "n_bars"), "[geometry]")
        if self.topology == "inrunner":
            # A cage rotor has no PM annulus: r_magnet == r_rotor is the
            # rotor surface. PM rotors keep the strict ordering.
            pm_ok = (self.r_magnet >= self.r_rotor if self.n_bars > 0
                     else self.r_magnet > self.r_rotor)
            if not (self.r_outer > self.r_stator > self.r_magnet
                    and pm_ok and self.r_rotor >= self.r_inner >= 0):
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

        if self.back_iron_thickness is not None:
            if self.topology != "outrunner":
                raise ValueError(
                    "back_iron_thickness is outrunner-only "
                    f"(topology={self.topology!r})"
                )
            wall = self.r_outer - self.r_rotor
            if not (0 < self.back_iron_thickness < wall):
                raise ValueError(
                    f"back_iron_thickness={self.back_iron_thickness} must be in "
                    f"(0, {wall:.6g}) m (r_outer - r_rotor); use None for a "
                    "fully solid yoke"
                )

        if self.magnet_mount not in ("surface", "inset"):
            raise ValueError(
                f"magnet_mount must be 'surface' or 'inset', "
                f"got {self.magnet_mount!r}"
            )
        if self.bridge_thickness is not None:
            pm_thickness = abs(self.r_magnet - self.r_rotor)
            if not (0 < self.bridge_thickness < pm_thickness):
                raise ValueError(
                    f"bridge_thickness={self.bridge_thickness} must be in "
                    f"(0, {pm_thickness:.6g}) m (the PM annulus radial "
                    "thickness); use None for exposed magnet faces"
                )

        if self.magnet_overhang is not None and not (
                math.isfinite(self.magnet_overhang) and self.magnet_overhang >= 0):
            raise ValueError(
                f"magnet_overhang={self.magnet_overhang} must be >= 0 and finite "
                "(axial extension per side; use None for not measured)"
            )

        if self.n_bars > 0:
            self._check_cage()
        elif self.n_bars < 0:
            raise ValueError(f"n_bars={self.n_bars} must be >= 0")
        elif any((self.bar_width, self.bar_height, self.bar_bridge_thickness,
                  self.bar_opening_width)):
            raise ValueError("bar_* fields require n_bars > 0")
        if self.skew_angle is not None:
            if self.n_bars <= 0:
                raise ValueError(
                    "skew_angle describes the cage bars (n_bars > 0); a PM "
                    "rotor must leave it unset"
                )
            if not (math.isfinite(self.skew_angle) and self.skew_angle > 0):
                raise ValueError(
                    f"skew_angle={self.skew_angle} must be > 0 and finite "
                    "(mechanical rad; use None for an unskewed cage)"
                )
            # Real skews are ~one slot pitch. Anything past two bar
            # pitches is almost certainly degrees entered as radians.
            if self.skew_angle > 4 * math.pi / self.n_bars:
                raise ValueError(
                    f"skew_angle={self.skew_angle} rad exceeds two bar "
                    f"pitches ({4 * math.pi / self.n_bars:.4g} rad for "
                    f"n_bars={self.n_bars}): skew_angle is mechanical "
                    "RADIANS over the stack — degrees entered here look "
                    "exactly like this"
                )
        for f in ("end_ring_axial_length", "end_ring_radial_height"):
            v = getattr(self, f)
            if v is not None and not (math.isfinite(v) and v > 0):
                raise ValueError(f"{f}={v} must be > 0 and finite (use None for unknown)")
        if (self.end_ring_radial_height is not None
                and self.end_ring_radial_height >= self.r_rotor):
            raise ValueError(
                f"end_ring_radial_height={self.end_ring_radial_height} must be "
                f"less than r_rotor={self.r_rotor}: the ring is measured inward "
                f"from the rotor surface, and a mean ring diameter of zero or "
                f"less makes the end-ring impedance meaningless")

        if self.n_slots > 0:
            if not (math.isfinite(self.slot_depth) and self.slot_depth >= 0):
                raise ValueError(f"slot_depth={self.slot_depth} must be >= 0 and finite")
            if self.slot_depth > 0:
                # Slots extend from the bore into the stator annulus:
                # inrunner [r_stator, r_stator + depth], outrunner
                # [r_stator - depth, r_stator].
                if self.topology == "inrunner":
                    yoke = self.r_outer - (self.r_stator + self.slot_depth)
                else:
                    yoke = (self.r_stator - self.slot_depth) - self.r_inner
                if yoke <= 0:
                    raise ValueError(
                        f"slot_depth={self.slot_depth} punches through the "
                        f"stator annulus (remaining yoke {yoke:.6g} m)"
                    )
            if not (0 < self.slot_width_ratio < 1):
                raise ValueError(
                    f"slot_width_ratio={self.slot_width_ratio} must be in (0, 1)"
                )
            if not (math.isfinite(self.slot_opening_width) and self.slot_opening_width >= 0):
                raise ValueError(
                    f"slot_opening_width={self.slot_opening_width} must be >= 0 and finite"
                )
            slot_pitch = 2 * math.pi * self.r_stator / self.n_slots
            if self.slot_opening_width >= slot_pitch:
                raise ValueError(
                    f"slot_opening_width={self.slot_opening_width} exceeds "
                    f"slot pitch {slot_pitch:.6g} m at the bore"
                )

    def _check_cage(self) -> None:
        if self.topology != "inrunner":
            raise ValueError("cage rotors (n_bars > 0) are inrunner-only")
        if self.n_bars < 2:
            raise ValueError(f"n_bars={self.n_bars} must be >= 2")
        if self.r_magnet != self.r_rotor:
            raise ValueError(
                f"a cage rotor has no PM annulus: r_magnet={self.r_magnet} "
                f"must equal r_rotor={self.r_rotor} (the rotor surface)"
            )
        for f, v in (("magnet_mount", "surface"), ("bridge_thickness", None),
                     ("magnet_overhang", None)):
            if getattr(self, f) != v:
                raise ValueError(f"{f} is PM-only; a cage rotor must leave it unset")
        if self.bar_shape not in ("rectangular", "round"):
            raise ValueError(
                f"bar_shape must be 'rectangular' or 'round', got {self.bar_shape!r}"
            )
        if not (math.isfinite(self.bar_width) and self.bar_width > 0):
            raise ValueError(f"bar_width={self.bar_width} must be > 0 and finite")
        if self.bar_shape == "round" and self.bar_height != 0.0:
            raise ValueError(
                "round bars take their diameter from bar_width; leave bar_height at 0"
            )
        if self.bar_shape == "rectangular" and not (
                math.isfinite(self.bar_height) and self.bar_height > 0):
            raise ValueError(f"bar_height={self.bar_height} must be > 0 and finite")
        if not (math.isfinite(self.bar_bridge_thickness) and self.bar_bridge_thickness >= 0):
            raise ValueError(
                f"bar_bridge_thickness={self.bar_bridge_thickness} must be >= 0 and finite"
            )
        _r_top, r_bottom = self.bar_radii
        if r_bottom <= self.r_inner:
            raise ValueError(
                f"bar depth {self.bar_bridge_thickness + self.bar_depth:.6g} m "
                f"reaches r_inner={self.r_inner} (rotor iron floor)"
            )
        if self.bar_arc_width >= 2 * math.pi / self.n_bars:
            raise ValueError(
                f"bar_width={self.bar_width} exceeds the bar pitch at the bar mid-radius"
            )
        if not (math.isfinite(self.bar_opening_width) and self.bar_opening_width >= 0):
            raise ValueError(
                f"bar_opening_width={self.bar_opening_width} must be >= 0 and finite"
            )
        if self.bar_opening_width > 0:
            if self.bar_bridge_thickness <= 0:
                raise ValueError(
                    "bar_opening_width needs bar_bridge_thickness > 0 to cut through"
                )
            if self.bar_opening_width >= self.bar_width:
                raise ValueError(
                    f"bar_opening_width={self.bar_opening_width} must be narrower "
                    f"than bar_width={self.bar_width}"
                )

    @property
    def bar_depth(self) -> float:
        """Radial extent of one bar (m): bar_height, or the diameter for
        round bars."""
        return self.bar_width if self.bar_shape == "round" else self.bar_height

    @property
    def bar_radii(self) -> tuple[float, float]:
        """(r_top, r_bottom) of the bar body (m), below the bridge."""
        r_top = self.r_rotor - self.bar_bridge_thickness
        return r_top, r_top - self.bar_depth

    @property
    def bar_arc_width(self) -> float:
        """Full angular width (rad) of a rectangular bar at its mid-radius,
        or of a round bar's diameter at its center."""
        r_top, r_bottom = self.bar_radii
        return self.bar_width / (0.5 * (r_top + r_bottom))

    @property
    def bar_area(self) -> float:
        """Cross-section of one bar (m^2), exact for both shapes."""
        if self.bar_shape == "round":
            return math.pi * self.bar_width**2 / 4
        return self.bar_width * self.bar_height

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
        # Append only when set so existing run-IDs / cache keys are unchanged.
        if self.back_iron_thickness is not None:
            key += f"_bi{self.back_iron_thickness:.6f}"
        if self.magnet_mount != "surface":
            key += f"_mm{self.magnet_mount}"
        if self.bridge_thickness is not None:
            key += f"_bt{self.bridge_thickness:.6f}"
        if self.magnet_overhang is not None:
            key += f"_mo{self.magnet_overhang:.6f}"
        if self.n_bars > 0:
            key += (f"_cb{self.n_bars}_{self.bar_shape}_{self.bar_width:.6f}_"
                    f"{self.bar_height:.6f}_{self.bar_bridge_thickness:.6f}_"
                    f"{self.bar_opening_width:.6f}")
        if self.end_ring_axial_length is not None:
            key += f"_erl{self.end_ring_axial_length:.6f}"
        if self.end_ring_radial_height is not None:
            key += f"_erh{self.end_ring_radial_height:.6f}"
        if self.skew_angle is not None:
            key += f"_sk{self.skew_angle:.6f}"
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
    magnet_mount: Literal["surface", "inset"] = "surface",
    bridge_thickness: float | None = None,
    magnet_overhang: float | None = None,
    **cage: Any,
) -> Geometry:
    """Build an inrunner Geometry (rotor inside stator). All lengths in
    meters; r_ag defaults to the airgap midpoint (r_stator + r_magnet)/2;
    r_inner defaults to 0.0 (solid shaft). Cage-rotor fields (n_bars,
    bar_*, end_ring_*) pass through as keywords."""
    if r_ag is None:
        r_ag = (r_stator + r_magnet) / 2
    return Geometry(
        topology="inrunner",
        r_outer=r_outer, r_stator=r_stator, r_magnet=r_magnet,
        r_rotor=r_rotor, r_inner=r_inner, r_ag=r_ag,
        n_slots=n_slots, slot_depth=slot_depth, slot_width_ratio=slot_width_ratio,
        slot_opening_width=slot_opening_width,
        magnet_mount=magnet_mount, bridge_thickness=bridge_thickness,
        magnet_overhang=magnet_overhang,
        **cage,
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
    back_iron_thickness: float | None = None,
    magnet_mount: Literal["surface", "inset"] = "surface",
    bridge_thickness: float | None = None,
    magnet_overhang: float | None = None,
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
        back_iron_thickness=back_iron_thickness,
        magnet_mount=magnet_mount, bridge_thickness=bridge_thickness,
        magnet_overhang=magnet_overhang,
    )


_CAGE_KEYS = (
    "n_bars", "bar_shape", "bar_width", "bar_height", "bar_bridge_thickness",
    "bar_opening_width", "end_ring_axial_length", "end_ring_radial_height",
    "skew_angle",
)


def default_inrunner() -> Geometry:
    return inrunner(
        r_outer=1.0, r_stator=0.70, r_magnet=0.64, r_rotor=0.30, r_inner=0.0,
    )


def geometry_from_toml(geo: dict[str, Any], topology: Literal["inrunner", "outrunner", "axial"] = "inrunner") -> Geometry | AxialFluxGeometry:
    """Build Geometry or AxialFluxGeometry from a TOML [geometry] section.

    Dispatches on topology: 'inrunner'/'outrunner' → radial Geometry,
    'axial' → AxialFluxGeometry. Radial path supports spec names
    (r_outer, r_stator, ...) and legacy TOML names (stator_od, stator_id,
    rotor_od, magnet_thickness).
    """
    if topology == "axial":
        return axial_flux_geometry_from_toml(geo)
    n_bars = geo.get("n_bars", 0)
    if "r_outer" in geo:
        r_outer = geo["r_outer"]
        r_stator = geo["r_stator"]
        r_rotor = geo["r_rotor"]
        # A cage rotor has no PM annulus: r_magnet defaults to the rotor
        # surface so the TOML does not repeat r_rotor.
        r_magnet = geo["r_magnet"] if (n_bars <= 0 or "r_magnet" in geo) else r_rotor
        r_inner = geo.get("r_inner", 0.0)
    elif "stator_od" in geo:
        # legacy OD/ID names encode the inrunner layout (stator outside,
        # magnet ring on the rotor OD) — meaningless for an outrunner
        if topology != "inrunner":
            raise ValueError(
                "legacy geometry names (stator_od/stator_id/rotor_od) are "
                "inrunner-only; use r_outer/r_stator/r_magnet/r_rotor for "
                f"topology={topology!r}"
            )
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

    kwargs = dict(
        r_outer=r_outer, r_stator=r_stator, r_magnet=r_magnet,
        r_rotor=r_rotor, r_inner=r_inner, r_ag=r_ag,
        n_slots=n_slots, slot_depth=slot_depth, slot_width_ratio=slot_width_ratio,
        slot_opening_width=slot_opening_width,
        magnet_mount=geo.get("magnet_mount", "surface"),
        bridge_thickness=geo.get("bridge_thickness"),
        magnet_overhang=geo.get("magnet_overhang"),
    )
    for f in _CAGE_KEYS:
        if f in geo:
            kwargs[f] = geo[f]
    if "skew_slots" in geo:
        # Bar skew in STATOR slot pitches (the usual datasheet form):
        # one slot pitch = 2π/n_slots mechanical rad.
        if "skew_angle" in geo:
            raise ValueError(
                "skew_angle and skew_slots are exclusive: give the skew "
                "either in mechanical radians or in stator slot pitches"
            )
        if n_slots <= 0:
            raise ValueError(
                "skew_slots needs n_slots > 0 (it is measured in stator "
                "slot pitches); give skew_angle in radians instead"
            )
        kwargs["skew_angle"] = geo["skew_slots"] * 2 * math.pi / n_slots
    factory: Callable[..., Geometry]
    if topology == "inrunner":
        if "back_iron_thickness" in geo:
            raise ValueError(
                "back_iron_thickness is outrunner-only (ring/shell yoke "
                "split); remove it from this inrunner [geometry] section"
            )
        factory = inrunner
    elif topology == "outrunner":
        if any(f in geo for f in _CAGE_KEYS):
            raise ValueError(
                "cage-rotor fields (n_bars, bar_*, end_ring_*) are inrunner-only"
            )
        factory = outrunner
        if "back_iron_thickness" in geo:
            kwargs["back_iron_thickness"] = geo["back_iron_thickness"]
    else:
        raise ValueError(f"Unknown topology {topology!r} (expected 'inrunner', 'outrunner', or 'axial')")
    return factory(**kwargs)


# ---------------------------------------------------------------------------
# Axial-flux geometry
# ---------------------------------------------------------------------------

AxialFluxVariant = Literal[
    "1R2S",       # 1 rotor, 2 stators (Parviainen TORUS)
    "2R1S-NN",    # 2 rotors, 1 stator, north-north (Mahmoudi TORUS-NN)
    "2R1S-NS",    # 2 rotors, 1 stator, north-south
    "DSSR",       # double-stator single-rotor (Mlot, Abdelli)
    "coreless",   # 2 rotors, coreless stator (Kappatou)
]


@dataclass(frozen=True)
class AxialFluxGeometry:
    """Axial-flux PMSM geometry in physical SI units (meters).

    Annular active region:
      r_out  — active outer radius (m)
      r_in   — active inner radius (m)

    Axial stack (per side, measured from the air-gap plane outward):
      airgap            — mechanical air gap per stator/rotor side (m)
      magnet_thickness  — PM axial thickness (m)
      rotor_yoke        — rotor back-iron axial thickness (m);
                          >= 0 (0 for yokeless rotor; > 0 typical)
      stator_yoke       — stator yoke axial thickness (m);
                          0 for coreless stator

    Slots (same convention as radial Geometry; evaluated at r_mean):
      n_slots            — stator slot count (per stator)
      slot_depth         — axial slot depth (m)
      slot_width_ratio   — slot body width / slot pitch at r_mean
      slot_opening_width — slot opening width at r_mean (m)

    variant — axial-flux magnetic circuit topology
    """

    variant: AxialFluxVariant

    r_out: float
    r_in: float

    airgap: float
    magnet_thickness: float
    rotor_yoke: float
    stator_yoke: float

    n_slots: int = 0
    slot_depth: float = 0.0
    slot_width_ratio: float = 0.6
    slot_opening_width: float = 0.0

    _VALID_VARIANTS: ClassVar[frozenset[str]] = frozenset(
        {"1R2S", "2R1S-NN", "2R1S-NS", "DSSR", "coreless"}
    )

    def __post_init__(self) -> None:
        int_normalize(self, ("n_slots",), "[geometry]")
        if self.variant not in self._VALID_VARIANTS:
            raise ValueError(
                f"Unknown axial-flux variant {self.variant!r} — "
                f"expected one of {sorted(self._VALID_VARIANTS)}"
            )
        if not (math.isfinite(self.r_out) and self.r_out > 0):
            raise ValueError(f"r_out must be > 0 and finite, got {self.r_out}")
        if not (math.isfinite(self.r_in) and self.r_in > 0):
            raise ValueError(f"r_in must be > 0 and finite, got {self.r_in}")
        if self.r_in >= self.r_out:
            raise ValueError(
                f"r_in={self.r_in} must be < r_out={self.r_out}"
            )
        if not (math.isfinite(self.airgap) and self.airgap > 0):
            raise ValueError(f"airgap must be > 0 and finite, got {self.airgap}")
        if not (math.isfinite(self.magnet_thickness) and self.magnet_thickness > 0):
            raise ValueError(
                f"magnet_thickness must be > 0 and finite, got {self.magnet_thickness}"
            )
        if not (math.isfinite(self.rotor_yoke) and self.rotor_yoke >= 0):
            raise ValueError(
                f"rotor_yoke must be >= 0 and finite, got {self.rotor_yoke}"
            )
        if not (math.isfinite(self.stator_yoke) and self.stator_yoke >= 0):
            raise ValueError(
                f"stator_yoke must be >= 0 and finite, got {self.stator_yoke}"
            )

        if self.variant == "coreless":
            if self.stator_yoke > 0:
                raise ValueError(
                    f"coreless variant requires stator_yoke=0, "
                    f"got {self.stator_yoke}"
                )
        elif self.variant == "DSSR":
            if self.stator_yoke <= 0:
                raise ValueError(
                    f"DSSR variant requires stator_yoke > 0, "
                    f"got {self.stator_yoke}"
                )
        elif self.variant in ("1R2S", "2R1S-NN", "2R1S-NS"):
            if self.rotor_yoke <= 0:
                raise ValueError(
                    f"{self.variant} variant requires rotor_yoke > 0, "
                    f"got {self.rotor_yoke}"
                )
            if self.stator_yoke <= 0:
                raise ValueError(
                    f"{self.variant} variant requires stator_yoke > 0, "
                    f"got {self.stator_yoke}"
                )

        if self.n_slots < 0:
            raise ValueError(f"n_slots must be >= 0, got {self.n_slots}")

        if self.n_slots > 0:
            if self.variant == "coreless":
                raise ValueError(
                    "coreless variant cannot have slotted stator "
                    f"(n_slots={self.n_slots})"
                )
            if not (math.isfinite(self.slot_depth) and self.slot_depth >= 0):
                raise ValueError(
                    f"slot_depth={self.slot_depth} must be >= 0 and finite"
                )
            if self.slot_depth > 0 and self.slot_depth >= self.stator_yoke:
                raise ValueError(
                    f"slot_depth={self.slot_depth} exceeds stator_yoke="
                    f"{self.stator_yoke} (no remaining yoke)"
                )
            if not (0 < self.slot_width_ratio < 1):
                raise ValueError(
                    f"slot_width_ratio={self.slot_width_ratio} must be in (0, 1)"
                )
            if not (math.isfinite(self.slot_opening_width) and self.slot_opening_width >= 0):
                raise ValueError(
                    f"slot_opening_width={self.slot_opening_width} must be >= 0 and finite"
                )
            slot_pitch = 2 * math.pi * self.r_mean / self.n_slots
            if self.slot_opening_width >= slot_pitch:
                raise ValueError(
                    f"slot_opening_width={self.slot_opening_width} exceeds "
                    f"slot pitch {slot_pitch:.6g} m at r_mean"
                )

    @property
    def r_mean(self) -> float:
        return (self.r_out + self.r_in) / 2

    @property
    def active_length(self) -> float:
        return self.r_out - self.r_in

    @property
    def k_D(self) -> float:
        return self.r_in / self.r_out

    @property
    def slot_opening_ratio(self) -> float:
        if self.n_slots <= 0 or self.slot_opening_width <= 0:
            return 0.0
        return self.slot_opening_width / (
            2 * math.pi * self.r_mean / self.n_slots
        )

    @property
    def config_id(self) -> str:
        key = (
            f"ax_{self.variant}_{self.r_out:.6f}_{self.r_in:.6f}_"
            f"{self.airgap:.6f}_{self.magnet_thickness:.6f}_"
            f"{self.rotor_yoke:.6f}_{self.stator_yoke:.6f}_"
            f"{self.n_slots}_{self.slot_depth:.6f}_"
            f"{self.slot_width_ratio:.6f}_{self.slot_opening_width:.6f}"
        )
        return hashlib.md5(key.encode()).hexdigest()[:12]


def axial_flux(
    *,
    variant: AxialFluxVariant,
    r_out: float,
    r_in: float,
    airgap: float,
    magnet_thickness: float,
    rotor_yoke: float = 0.0,
    stator_yoke: float = 0.0,
    n_slots: int = 0,
    slot_depth: float = 0.0,
    slot_width_ratio: float = 0.6,
    slot_opening_width: float = 0.0,
) -> AxialFluxGeometry:
    """Build an AxialFluxGeometry. All lengths in meters."""
    return AxialFluxGeometry(
        variant=variant,
        r_out=r_out, r_in=r_in,
        airgap=airgap, magnet_thickness=magnet_thickness,
        rotor_yoke=rotor_yoke, stator_yoke=stator_yoke,
        n_slots=n_slots, slot_depth=slot_depth,
        slot_width_ratio=slot_width_ratio,
        slot_opening_width=slot_opening_width,
    )


def axial_flux_geometry_from_toml(geo: dict[str, Any]) -> AxialFluxGeometry:
    """Build AxialFluxGeometry from a TOML [geometry] section.

    Accepts lengths in meters (r_out, r_in, airgap, ...) or millimeters
    via _mm suffixed keys (D_out_mm/D_in_mm → r_out/r_in as radius).
    """
    if "r_out" in geo:
        r_out = geo["r_out"]
        r_in = geo["r_in"]
    elif "D_out_mm" in geo:
        r_out = geo["D_out_mm"] / 2000
        r_in = geo["D_in_mm"] / 2000
    else:
        raise ValueError(
            "Axial [geometry] must contain r_out/r_in or D_out_mm/D_in_mm"
        )

    def _mm_or_m(key_mm: str, key_m: str, default: float = 0.0) -> float:
        if key_m in geo:
            return geo[key_m]
        if key_mm in geo:
            return geo[key_mm] / 1000
        return default

    airgap = _mm_or_m("airgap_mm", "airgap")
    magnet_thickness = _mm_or_m("magnet_thickness_mm", "magnet_thickness")
    rotor_yoke = _mm_or_m("rotor_yoke_mm", "rotor_yoke")
    stator_yoke = _mm_or_m("stator_yoke_mm", "stator_yoke")
    slot_depth = _mm_or_m("slot_depth_mm", "slot_depth")
    slot_opening_width = _mm_or_m(
        "slot_opening_width_mm", "slot_opening_width",
    )

    variant: AxialFluxVariant = geo.get("variant", "1R2S")

    n_slots = geo.get("n_slots", 0)
    slot_width_ratio = geo.get("slot_width_ratio", 0.6)

    if (
        "slot_opening_width" not in geo
        and "slot_opening_width_mm" not in geo
        and "slot_opening_ratio" in geo
        and n_slots > 0
    ):
        r_mean = (r_out + r_in) / 2
        slot_pitch = 2 * math.pi * r_mean / n_slots
        slot_opening_width = geo["slot_opening_ratio"] * slot_pitch

    return axial_flux(
        variant=variant,
        r_out=r_out, r_in=r_in,
        airgap=airgap, magnet_thickness=magnet_thickness,
        rotor_yoke=rotor_yoke, stator_yoke=stator_yoke,
        n_slots=n_slots, slot_depth=slot_depth,
        slot_width_ratio=slot_width_ratio,
        slot_opening_width=slot_opening_width,
    )
