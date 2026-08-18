"""Derive 14 mm outrunner geometry from physical measurements.

Takes caliper/micrometer/gauge-pin measurements and computes the radii
and parameters used in the `outrunner_14mm_steel` motor TOML. This is the
provenance for the `# from geometry.py` notes in that file: run it to
reproduce r_outer/r_rotor/r_magnet/r_stator/r_inner, the air gap, and
alpha_p from the raw bench measurements.

Outrunner radial stack (outside -> inside):
    rotor shell wall | magnets | air gap | stator teeth | stator core

Self-contained (stdlib only). Run directly to print the steel rotor derivation:
    python derive_geometry.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RotorMeasurements:
    """Raw measurements for one rotor variant."""
    name: str
    rotor_od_mm: float          # micrometer, outer shell
    wall_thickness_mm: float    # caliper avg, shell wall
    magnet_thickness_mm: float  # caliper avg, radial
    gauge_pin_in: float | None = None  # largest pin that fits magnet bore


@dataclass(frozen=True)
class StatorMeasurements:
    """Raw stator measurements."""
    stator_od_mm: float         # OD of stator lamination (tooth tips)
    stator_id_mm: float         # bore
    gauge_od_in: float | None = None  # gauge measurement of stator OD


@dataclass(frozen=True)
class DerivedGeometry:
    """All derived values in mm (display) and m (for TOML/phase-sweep)."""
    # Radii in mm
    r_outer_mm: float       # rotor shell outer radius
    r_rotor_mm: float       # rotor shell inner surface (magnet mounting surface)
    r_magnet_mm: float      # magnet inner face (air gap boundary, rotor side)
    r_stator_mm: float      # stator OD / 2 (air gap boundary, stator side)
    r_inner_mm: float       # stator bore / 2

    air_gap_mm: float       # r_magnet - r_stator
    alpha_p: float          # magnet arc fraction of pole pitch

    # For cross-checking
    magnet_bore_dia_mm: float  # 2 * r_magnet = diameter at magnet face

    @property
    def r_outer(self) -> float:
        return self.r_outer_mm / 1000

    @property
    def r_rotor(self) -> float:
        return self.r_rotor_mm / 1000

    @property
    def r_magnet(self) -> float:
        return self.r_magnet_mm / 1000

    @property
    def r_stator(self) -> float:
        return self.r_stator_mm / 1000

    @property
    def r_inner(self) -> float:
        return self.r_inner_mm / 1000

    @property
    def air_gap(self) -> float:
        return self.air_gap_mm / 1000


def derive_outrunner_geometry(
    rotor: RotorMeasurements,
    stator: StatorMeasurements,
    n_poles: int,
    magnet_width_mm: float,
) -> DerivedGeometry:
    """Derive outrunner geometry from physical measurements.

    Parameters
    ----------
    rotor : RotorMeasurements
        Rotor shell and magnet measurements.
    stator : StatorMeasurements
        Stator lamination measurements.
    n_poles : int
        Number of magnetic poles (not pole pairs).
    magnet_width_mm : float
        Circumferential width of one magnet (flat/chord dimension, mm).
    """
    if n_poles < 2 or n_poles % 2 != 0:
        raise ValueError(f"n_poles must be even and >= 2, got {n_poles}")
    if rotor.wall_thickness_mm <= 0:
        raise ValueError(f"wall_thickness_mm must be > 0, got {rotor.wall_thickness_mm}")
    if rotor.magnet_thickness_mm <= 0:
        raise ValueError(f"magnet_thickness_mm must be > 0, got {rotor.magnet_thickness_mm}")
    if stator.stator_od_mm <= 0:
        raise ValueError(f"stator_od_mm must be > 0, got {stator.stator_od_mm}")
    if stator.stator_id_mm <= 0:
        raise ValueError(f"stator_id_mm must be > 0, got {stator.stator_id_mm}")
    if stator.stator_id_mm >= stator.stator_od_mm:
        raise ValueError(
            f"stator_id_mm ({stator.stator_id_mm}) must be < "
            f"stator_od_mm ({stator.stator_od_mm})"
        )
    if magnet_width_mm <= 0:
        raise ValueError(f"magnet_width_mm must be > 0, got {magnet_width_mm}")

    # Radial stack: shell OD -> shell ID -> magnet face -> air gap -> stator OD -> stator ID
    r_outer = rotor.rotor_od_mm / 2
    r_rotor = r_outer - rotor.wall_thickness_mm
    r_magnet = r_rotor - rotor.magnet_thickness_mm
    r_stator = stator.stator_od_mm / 2
    r_inner = stator.stator_id_mm / 2

    air_gap = r_magnet - r_stator
    if air_gap <= 0:
        raise ValueError(
            f"Negative air gap: r_magnet={r_magnet:.3f} mm <= "
            f"r_stator={r_stator:.3f} mm"
        )

    magnet_bore_dia = 2 * r_magnet

    # alpha_p: magnet arc / pole pitch, computed at the magnet mounting surface
    # Magnets sit on r_rotor (cup ID). Pole pitch arc at that radius:
    pole_pitch_mm = math.pi * (2 * r_rotor) / n_poles

    # magnet_width_mm is the chord (flat). Convert to arc at r_rotor:
    half_chord = magnet_width_mm / 2
    if half_chord >= r_rotor:
        raise ValueError(
            f"Magnet width ({magnet_width_mm} mm) exceeds rotor diameter"
        )
    magnet_arc_mm = 2 * r_rotor * math.asin(half_chord / r_rotor)

    alpha_p = magnet_arc_mm / pole_pitch_mm

    return DerivedGeometry(
        r_outer_mm=r_outer,
        r_rotor_mm=r_rotor,
        r_magnet_mm=r_magnet,
        r_stator_mm=r_stator,
        r_inner_mm=r_inner,
        air_gap_mm=air_gap,
        alpha_p=alpha_p,
        magnet_bore_dia_mm=magnet_bore_dia,
    )


def gauge_pin_to_diameter_mm(gauge_pin_in: float) -> float:
    """Convert gauge pin diameter from inches to mm."""
    return gauge_pin_in * 25.4


# Bench measurements (caliper/micrometer averages of 3, gauge pins).
# See motor_parameters.md.
STEEL_ROTOR = RotorMeasurements(
    name="Steel rotor",
    rotor_od_mm=19.33,
    wall_thickness_mm=0.90,
    magnet_thickness_mm=1.287,
    gauge_pin_in=0.589,
)

STATOR = StatorMeasurements(
    stator_od_mm=13.9,
    stator_id_mm=6.0,
    gauge_od_in=0.5535,
)

N_POLES = 12
MAGNET_WIDTH_MM = 2.9


if __name__ == "__main__":
    geo = derive_outrunner_geometry(STEEL_ROTOR, STATOR, N_POLES, MAGNET_WIDTH_MM)
    print(f"\n{'=' * 60}")
    print(f"  {STEEL_ROTOR.name}")
    print(f"{'=' * 60}")
    print(f"  r_outer  (shell OD/2):    {geo.r_outer_mm:.3f} mm")
    print(f"  r_rotor  (cup ID/2):      {geo.r_rotor_mm:.3f} mm")
    print(f"  r_magnet (magnet face/2): {geo.r_magnet_mm:.3f} mm")
    print(f"  r_stator (stator OD/2):   {geo.r_stator_mm:.3f} mm")
    print(f"  r_inner  (stator ID/2):   {geo.r_inner_mm:.3f} mm")
    print(f"  air gap:                  {geo.air_gap_mm:.3f} mm")
    print(f"  alpha_p:                  {geo.alpha_p:.4f}")
    print(f"  magnet bore diameter:     {geo.magnet_bore_dia_mm:.3f} mm")
