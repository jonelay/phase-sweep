from __future__ import annotations

import dataclasses
import difflib
import logging
import math
import tomllib
from pathlib import Path
from typing import Any

from phasesweep.machines.geometry import (
    _CAGE_KEYS,
    AxialFluxGeometry,
    Geometry,
    geometry_from_toml,
)
from phasesweep.machines.motor import (
    DriveParams,
    FluxMap,
    InductionParams,
    Motor,
    ThermalNetwork,
)

__all__ = [
    "load_motor",
    "load_motors",
    "motor_from_dict",
    "motor_to_toml_dict",
]

# Keys each loader section consumes. Anything else in these sections is a
# hard error: a misspelled key that silently loads as "not set" is worse
# than a refusal. Tables the loader does not own ([rated], [provenance],
# ...) stay free-form for documentation.
_DESCRIPTIVE_KEYS: dict[str, frozenset[str]] = {
    # accepted and not read — documentary labels, kept next to the data
    "motor": frozenset({"type"}),
    "winding": frozenset({
        "layers", "connection", "q", "turns_per_phase", "turns_per_tooth",
    }),
    "materials": frozenset({"magnet_type", "steel_grade", "rotor_yoke_material"}),
}
_SECTION_KEYS: dict[str, frozenset[str]] = {
    "motor": frozenset({"name", "topology"}) | _DESCRIPTIVE_KEYS["motor"],
    "circuit": frozenset({
        "n_p", "R_s", "L_d", "L_q", "psi_f", "J", "I_rated", "I_rated_rms",
    }),
    "winding": frozenset({"N", "k_w", "coils_series", "coil_pitch_slots", "n_slots"})
    | _DESCRIPTIVE_KEYS["winding"],
    "materials": frozenset({
        "B_rem", "mu_r_fe", "mu_r_pm", "alpha_Br", "B_knee", "alpha_B_knee",
        "bh_curve",
    }) | _DESCRIPTIVE_KEYS["materials"],
    "drive": frozenset({"U_DC", "MAX_I_S", "W_REF", "I_LIMIT"}),
    "thermal": frozenset({
        "winding_temp_limit", "ambient_temp", "r_th", "insulation_class",
        "thermal_time_constant", "magnet_temp",
        "winding_temp_limit_basis", "winding_temp_limit_source",
        "magnet_temp_max", "magnet_temp_max_basis", "magnet_temp_max_source",
    }),
    "iron": frozenset({"k_h", "k_e", "alpha_fe", "m_core", "B_core"}),
    "induction": frozenset(f.name for f in dataclasses.fields(InductionParams)),
    "flux_map": frozenset(f.name for f in dataclasses.fields(FluxMap)),
    "thermal_network": frozenset(f.name for f in dataclasses.fields(ThermalNetwork)),
}
_GEOMETRY_COMMON = frozenset({
    "L_stk", "k_end", "alpha_p", "magnetization", "halbach_segments",
    "halbach_angles_deg", "n_slots", "slot_depth", "slot_width_ratio",
    "slot_opening_width", "slot_opening_ratio", "magnet_thickness",
})
_GEOMETRY_KEYS: dict[str, frozenset[str]] = {
    "inrunner": _GEOMETRY_COMMON | {
        "r_outer", "r_stator", "r_magnet", "r_rotor", "r_inner", "r_ag",
        "stator_od", "stator_id", "rotor_od", "slot_height",
        "magnet_mount", "bridge_thickness", "magnet_overhang",
        "skew_slots",
        *_CAGE_KEYS,
    },
    "outrunner": _GEOMETRY_COMMON | {
        "r_outer", "r_stator", "r_magnet", "r_rotor", "r_inner", "r_ag",
        "magnet_mount", "bridge_thickness", "magnet_overhang",
        "back_iron_thickness",
    },
    "axial": _GEOMETRY_COMMON | {
        "r_out", "r_in", "D_out_mm", "D_in_mm", "airgap", "airgap_mm",
        "magnet_thickness_mm", "rotor_yoke", "rotor_yoke_mm",
        "stator_yoke", "stator_yoke_mm", "slot_depth_mm",
        "slot_opening_width_mm", "variant",
    },
}
_FREE_SECTIONS = frozenset({
    "rated", "provenance", "validation", "envelope", "electrical",
    "columns", "magnets", "mass",
})
_NEAR_MISS_CUTOFF = 0.8


def _reject_unknown_keys(raw: dict[str, Any], label: str) -> None:
    topology = (raw.get("motor") or {}).get("topology", "inrunner")
    for section, data in raw.items():
        if section in _FREE_SECTIONS:
            continue
        if section == "geometry":
            allowed = _GEOMETRY_KEYS.get(topology, _GEOMETRY_KEYS["inrunner"])
        elif section in _SECTION_KEYS:
            allowed = _SECTION_KEYS[section]
        else:
            near = difflib.get_close_matches(
                section, [*_SECTION_KEYS, "geometry"], n=1, cutoff=_NEAR_MISS_CUTOFF,
            )
            hint = f" (did you mean [{near[0]}]?)" if near else ""
            raise ValueError(
                f"{label}: unknown section [{section}]{hint}; documentation "
                "tables belong under [provenance]"
            )
        if not data:
            continue
        unknown = [k for k in data if k not in allowed]
        if not unknown:
            continue
        near = difflib.get_close_matches(unknown[0], allowed, n=1, cutoff=_NEAR_MISS_CUTOFF)
        hint = f" (did you mean '{near[0]}'?)" if near else ""
        raise ValueError(
            f"{label}: [{section}] unknown field(s) "
            f"{', '.join(repr(k) for k in unknown)}{hint}; datasheet extras "
            "belong under [provenance]"
        )


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


def _get_or_warn(
    section: dict[str, Any], field: str, default: Any, label: str,
    section_name: str = "materials",
) -> Any:
    if field not in section:
        logging.warning(
            "%s: missing [%s] %s — defaulting to %s",
            label, section_name, field, default,
        )
        return default
    return section[field]


def load_motor(path: str | Path) -> Motor:
    """Load a TOML motor definition file into a Motor dataclass.

    TOML values are SI (meters, ohms, henries, Wb, T, A) with two
    conveniences: `I_rated_rms` is converted to peak (× √2) and
    `slot_opening_ratio` to `slot_opening_width` at parse time.
    Raises ValueError naming the file, section, and field on invalid input,
    including keys the loader does not consume (see docs/motor-toml.md).
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

    Every malformed input raises ValueError: a wrong-typed value (most
    often a quoted number) surfaces as a TypeError deep in validation, so
    it is re-raised naming the offending field where one can be found.
    """
    try:
        return _motor_from_dict(raw, label=label, default_name=default_name)
    except TypeError as e:
        quoted = _quoted_numbers(raw)
        hint = (f"; quoted number(s): {', '.join(quoted)}" if quoted
                else "; check for a value of the wrong type")
        raise ValueError(f"{label}: {e}{hint}") from e


def _quoted_numbers(raw: dict[str, Any]) -> list[str]:
    """`[section] key = 'value'` for string values that parse as numbers."""
    out: list[str] = []
    for section, table in raw.items():
        if not isinstance(table, dict):
            continue
        for k, v in table.items():
            if isinstance(v, str):
                try:
                    float(v)
                except ValueError:
                    continue
                out.append(f"[{section}] {k} = {v!r}")
    return out


def _motor_from_dict(raw: dict[str, Any], *, label: str, default_name: str) -> Motor:
    _reject_unknown_keys(raw, label)
    circuit = _require_section(raw, "circuit", label)
    n_p = _require_field(circuit, "n_p", "circuit", label)

    # `or {}` not `get(k, {})`: a JSON payload from the config editor can
    # carry an explicit null for a section, and the default only applies
    # when the key is absent (a present null crashed the PUT route).
    motor_raw = raw.get("motor") or {}
    winding = raw.get("winding") or {}
    # [winding] connection is documentary everywhere else, but every model
    # that takes a line voltage divides it by sqrt(3) (see
    # im_fem_harmonic.prepare_im_fem_harmonic). Accepting "delta" silently
    # would put the phase voltage out by that factor.
    conn = winding.get("connection")
    if conn is not None and str(conn).strip().lower() not in ("wye", "star", "y"):
        raise ValueError(
            f"{label}: [winding] connection={conn!r} is not supported. The "
            f"models assume a wye/star connection and derive the phase "
            f"voltage as V_LL/sqrt(3); a delta machine needs its own "
            f"derivation before it can be described here")
    geometry_raw = raw.get("geometry") or {}
    materials = raw.get("materials") or {}
    drive_raw = raw.get("drive") or {}
    thermal = raw.get("thermal") or {}
    iron = raw.get("iron") or {}
    induction_raw = raw.get("induction") or {}
    flux_map_raw = raw.get("flux_map") or {}
    thermal_network_raw = raw.get("thermal_network") or {}

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
        if "n_bars" in induction_raw and "n_bars" in geometry_raw \
                and geometry_raw["n_bars"] != induction_raw["n_bars"]:
            raise ValueError(
                f"{label}: n_bars mismatch: [geometry] has "
                f"{geometry_raw['n_bars']}, [induction] has {induction_raw['n_bars']}"
            )
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

    induction = None
    if induction_raw:
        try:
            induction = InductionParams.from_dict(induction_raw)
        except (ValueError, TypeError) as e:
            raise ValueError(f"{label}: [induction] {e}") from e

    flux_map = None
    if flux_map_raw:
        try:
            flux_map = FluxMap.from_dict(flux_map_raw)
        except (ValueError, TypeError, KeyError) as e:
            raise ValueError(f"{label}: [flux_map] {e}") from e

    thermal_network = None
    if thermal_network_raw:
        try:
            thermal_network = ThermalNetwork.from_dict(thermal_network_raw)
        except (ValueError, TypeError) as e:
            raise ValueError(f"{label}: [thermal_network] {e}") from e

    name = motor_raw.get("name", default_name)

    I_rated = circuit.get("I_rated")
    I_rated_rms = circuit.get("I_rated_rms")
    if I_rated is not None and I_rated_rms is not None:
        raise ValueError(
            f"{label}: [circuit] set either I_rated (peak) or I_rated_rms, not both"
        )
    if I_rated_rms is not None:
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
            # Full-pitch default is silent only where it is not a guess:
            # no geometry, a cage rotor, or Halbach (which requires 1.0)
            alpha_p=(_get_or_warn(geometry_raw, "alpha_p", 1.0, label, "geometry")
                     if geometry_raw and induction is None
                     and geometry_raw.get("magnetization") != "halbach"
                     else geometry_raw.get("alpha_p", 1.0)),
            magnetization=geometry_raw.get("magnetization", "radial"),
            halbach_segments=geometry_raw.get("halbach_segments"),
            halbach_angles_deg=(
                tuple(geometry_raw["halbach_angles_deg"])
                if "halbach_angles_deg" in geometry_raw else None
            ),
            # No missing-mu_r_fe warning for induction machines: the
            # circuit tier never reads it (a PM/FEM default only)
            mu_r_fe=(materials.get("mu_r_fe", 1000.0) if induction is not None
                     else _get_or_warn(materials, "mu_r_fe", 1000.0, label)),
            mu_r_pm=materials.get("mu_r_pm", 1.05),
            bh_curve=(tuple(tuple(r) for r in materials["bh_curve"])
                      if materials.get("bh_curve") is not None else None),
            N=winding.get("N"),
            k_w=winding.get("k_w"),
            L_stk=geometry_raw.get("L_stk"),
            k_end=geometry_raw.get("k_end"),
            I_rated=I_rated,
            coils_series=winding.get("coils_series"),
            coil_pitch_slots=winding.get("coil_pitch_slots"),
            winding_temp_limit=thermal.get("winding_temp_limit"),
            ambient_temp=thermal.get("ambient_temp"),
            r_th=thermal.get("r_th"),
            insulation_class=thermal.get("insulation_class"),
            thermal_time_constant=thermal.get("thermal_time_constant"),
            alpha_Br=materials.get("alpha_Br"),
            magnet_temp=thermal.get("magnet_temp"),
            winding_temp_limit_basis=thermal.get("winding_temp_limit_basis"),
            winding_temp_limit_source=thermal.get("winding_temp_limit_source"),
            magnet_temp_max=thermal.get("magnet_temp_max"),
            magnet_temp_max_basis=thermal.get("magnet_temp_max_basis"),
            magnet_temp_max_source=thermal.get("magnet_temp_max_source"),
            B_knee=materials.get("B_knee"),
            alpha_B_knee=materials.get("alpha_B_knee"),
            k_h=iron.get("k_h"),
            k_e=iron.get("k_e"),
            alpha_fe=iron.get("alpha_fe"),
            m_core=iron.get("m_core"),
            B_core=iron.get("B_core"),
            drive=drive,
            induction=induction,
            flux_map=flux_map,
            thermal_network=thermal_network,
        )
    except ValueError as e:
        raise ValueError(f"{label}: {e}") from e


# Motor scalar fields by the TOML section the loader reads them from —
# the writer's half of the schema. [geometry] also carries the motor-level
# geometry scalars below; every other Motor field is a nested section.
_MOTOR_TOML_SCALARS: dict[str, tuple[str, ...]] = {
    "circuit": ("n_p", "R_s", "L_d", "L_q", "psi_f", "J", "I_rated"),
    "winding": ("N", "k_w", "coils_series", "coil_pitch_slots"),
    "materials": ("B_rem", "mu_r_fe", "mu_r_pm", "alpha_Br", "B_knee",
                  "alpha_B_knee", "bh_curve"),
    "thermal": ("winding_temp_limit", "ambient_temp", "r_th",
                "insulation_class", "thermal_time_constant", "magnet_temp",
                "winding_temp_limit_basis", "winding_temp_limit_source",
                "magnet_temp_max", "magnet_temp_max_basis",
                "magnet_temp_max_source"),
    "iron": ("k_h", "k_e", "alpha_fe", "m_core", "B_core"),
}
_MOTOR_GEOMETRY_SCALARS = (
    "alpha_p", "L_stk", "k_end", "magnetization", "halbach_segments",
    "halbach_angles_deg",
)


def _toml_plain(v: Any) -> Any:
    if isinstance(v, tuple):
        return [_toml_plain(x) for x in v]
    return v


def _geometry_toml(geo: Geometry | AxialFluxGeometry) -> dict[str, Any]:
    """Non-default geometry fields under their TOML names. Defaults are
    left out because the outrunner and axial sections reject cage keys."""
    out: dict[str, Any] = {}
    for f in dataclasses.fields(geo):
        if f.name == "topology":
            continue
        v = getattr(geo, f.name)
        if v is None or (f.default is not dataclasses.MISSING and v == f.default):
            continue
        if f.name == "skew_angle":
            # TOML takes skew in stator slot pitches; round so a value that
            # came from skew_slots reproduces it exactly
            out["skew_slots"] = round(v * geo.n_slots / (2 * math.pi), 12)
            continue
        out[f.name] = v
    return out


def motor_to_toml_dict(motor: Motor) -> dict[str, dict[str, Any]]:
    """The load_motor TOML structure for *motor*: ``motor_from_dict`` of
    the result rebuilds an equal Motor with the same config_id.

    Raises ValueError for a geometry-less motor that sets a field the
    loader only reads from [geometry] (L_stk, k_end, alpha_p ≠ 1, Halbach).
    """
    geo = motor.geometry
    raw: dict[str, dict[str, Any]] = {"motor": {"name": motor.name}}
    if isinstance(geo, Geometry):
        raw["motor"]["topology"] = geo.topology
    elif isinstance(geo, AxialFluxGeometry):
        raw["motor"]["topology"] = "axial"
    for section in ("circuit", "winding"):
        raw[section] = {k: _toml_plain(getattr(motor, k))
                        for k in _MOTOR_TOML_SCALARS[section]}
    geo_scalars = {k: _toml_plain(getattr(motor, k)) for k in _MOTOR_GEOMETRY_SCALARS}
    if motor.magnetization == "radial":
        geo_scalars["magnetization"] = None
    if geo is not None:
        raw["geometry"] = {**_geometry_toml(geo), **geo_scalars}
    else:
        stranded = [k for k, v in geo_scalars.items()
                    if v is not None and not (k == "alpha_p" and v == 1.0)]
        if stranded:
            raise ValueError(
                f"Motor '{motor.name}': {', '.join(stranded)} set without a "
                f"geometry; the TOML loader reads them from [geometry] only")
    for section in ("materials", "thermal", "iron"):
        raw[section] = {k: _toml_plain(getattr(motor, k))
                        for k in _MOTOR_TOML_SCALARS[section]}
    raw["drive"] = dataclasses.asdict(motor.drive)
    if motor.induction is not None:
        raw["induction"] = motor.induction.to_dict()
    if motor.flux_map is not None:
        raw["flux_map"] = motor.flux_map.to_dict()
    if motor.thermal_network is not None:
        raw["thermal_network"] = motor.thermal_network.to_dict()
    for section in list(raw):
        raw[section] = {k: v for k, v in raw[section].items() if v is not None}
        if not raw[section]:
            del raw[section]
    return raw


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
        except (KeyError, ValueError, TypeError) as e:
            if strict:
                raise
            logging.warning("Skipping %s: %s", p.name, e)
            continue
        results[motor.name] = motor
    return results
