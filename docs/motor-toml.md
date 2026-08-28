# Motor TOML Format

Motor definition files (`motors/*.toml`) are loaded by
`phasesweep.load_motor(path)` into a `Motor` dataclass. All values are
SI: meters, ohms, henries, webers, tesla, amperes, rad/s. Two
conveniences are converted at parse time: `I_rated_rms` → peak (× √2)
and `slot_opening_ratio` → `slot_opening_width`.

Only `[circuit] n_p` is required; `name` defaults to the filename stem.
`[geometry]` is optional: datasheet/circuit-only motors omit it and run
the circuit-tier models (rated/stall torque, torque_speed, drive_sim,
thermal_duty, iron_loss); the field solvers require it. Everything else
is optional too — a missing field means "not yet measured/specified",
and the `prepare_*` solver factories raise a ValueError naming the
missing fields when a solver actually needs them.

## Sections

### `[motor]` — optional

| Field | Type | Notes |
|-------|------|-------|
| `name` | str | Defaults to the filename stem |
| `topology` | str | `"inrunner"` (default) or `"outrunner"` |
| `type` | str | Free-form label (e.g. `"SPMSM"`); not parsed |

### `[circuit]` — required (`n_p` only)

| Field | Units | Notes |
|-------|-------|-------|
| `n_p` | — | Pole pairs. **Required.** Must be ≥ 2 |
| `R_s` | Ω | Stator phase resistance |
| `L_d`, `L_q` | H | d/q-axis inductance |
| `psi_f` | Wb (peak) | PM flux linkage per phase. See [`psi_f`/`B_rem` authority](#psi_fb_rem-authority) before supplying both |
| `J` | kg·m² | Rotor inertia |
| `I_rated` | A (peak) | Rated continuous current |
| `I_rated_rms` | A (RMS) | Convenience: converted to peak ×√2. `I_rated` wins if both given |

### `[geometry]` — optional (required by the field solvers)

Spec names (preferred):

| Field | Units | Notes |
|-------|-------|-------|
| `r_outer` | m | Outermost boundary (frame) |
| `r_stator` | m | Stator bore at the airgap |
| `r_magnet` | m | PM/airgap interface |
| `r_rotor` | m | Rotor iron behind magnets |
| `r_inner` | m | Shaft OD / hollow-center ID (default 0.0; must be > 0 for outrunner) |
| `r_ag` | m | Airgap sampling radius (default: mid-gap) |
| `L_stk` | m | Stack axial length (maps to `Motor.L_stk`) |
| `n_slots` | — | Slot count (0 = smooth bore) |
| `slot_depth` | m | Radial slot depth (legacy alias: `slot_height`) |
| `slot_width_ratio` | — | Slot body width / slot pitch (default 0.6) |
| `slot_opening_width` | m | Slot throat width at the bore (Carter input) |
| `slot_opening_ratio` | — | Convenience: converted to width via slot pitch at the bore; width wins if both given |
| `alpha_p` | — | Magnet pole-arc / pole-pitch ratio, (0, 1] (maps to `Motor.alpha_p`, default 1.0) |
| `magnetization` | — | `"radial"` (default) or `"halbach"` — magnetization pattern. Halbach requires `alpha_p = 1.0` |
| `halbach_segments` | — | Segments per pole (κ_m) for Halbach arrays. Required when `magnetization = "halbach"` |
| `halbach_angles_deg` | deg | Per-segment local magnetization angles (length = `halbach_segments`). Omit for the ideal Halbach rule |
| `back_iron_thickness` | m | **Outrunner only** — magnetic back-iron ring inside the rotor wall, splitting it into ring + non-magnetic shell. Must be in (0, wall thickness); setting it on an inrunner raises |
| `magnet_mount` | — | `"surface"` (default, interpole gaps are air) or `"inset"` (magnets recessed in the rotor; interpole gaps are rotor iron). FEM only, requires `alpha_p < 1` and `nonlinear = true`; the analytical solver rejects inset rotors, and psi_f/B_rem must both be explicit (the radial derivation ignores the interpole iron shunt) |
| `bridge_thickness` | m | Iron bridge ring between the magnet faces and the airgap (retaining ring cut from the PM annulus). Compatible with any `magnet_mount` or `magnetization` (e.g. Halbach with a retaining sleeve). FEM only, requires `nonlinear = true`; the analytical solver rejects it. Must be in (0, magnet radial thickness); omit for exposed magnet faces |
| `magnet_overhang` | m | Axial extension of the magnets per side beyond `L_stk` (maps to `Geometry.magnet_overhang`). ≥ 0; `0.0` = measured co-extensive, omit = not measured. Input to the offline 3D wedge k_end solve (`scripts/compute_k_end.py`) — the dominant lever on the end-effect factor. Not used by the 2D solvers |
| `k_end` | — | 3D axial end-effect factor on the winding flux linkage (maps to `Motor.k_end`), accepted in (0, 1.2]. Values above 1 are possible under magnet overhang. A **computed** value from `scripts/compute_k_end.py` — commit its provenance record (`data/<motor>/k_end.record.json`) alongside. Applied by the `analytical` model to the B_rem-derived `flux_linkage_peak`/`backemf_fundamental` only (never when psi_f is the input — a measured psi_f already includes end effects) |

Legacy diameter names are also accepted: `stator_od`, `stator_id`,
`rotor_od`, `magnet_thickness` (halved/derived to radii). One of
`r_outer` or `stator_od` must be present.

Radii ordering is validated at load:
inrunner `r_outer > r_stator > r_magnet > r_rotor >= r_inner >= 0`;
outrunner reverses the middle three.

### `[winding]` — optional

| Field | Units | Notes |
|-------|-------|-------|
| `N` | — | Turns per coil (NOT per phase) |
| `k_w` | — | Winding factor, (0, 1] |
| `coils_series` | — | Coils in series per phase. Required when N and k_w are set (no auto-derivation): `N_eff = N × coils_series` |
| `n_slots` | — | Accepted here too; must match `[geometry]` if both given |

### `[materials]` — optional

| Field | Units | Default | Notes |
|-------|-------|---------|-------|
| `B_rem` | T | — | Magnet remanence. Derivable from `psi_f` (and vice versa) by the solver factories when winding params exist; see [`psi_f`/`B_rem` authority](#psi_fb_rem-authority) |
| `mu_r_fe` | — | 1000.0 | Iron relative permeability (linear FEM) |
| `mu_r_pm` | — | 1.05 | Magnet recoil permeability |
| `alpha_Br` | 1/K | — | Fractional remanence temperature coefficient, validated to (−0.01, 0); applied against `[thermal] magnet_temp` |
| `B_knee` | T | — | Demagnetization knee at the 20 °C reference (`demag_screen`) |
| `alpha_B_knee` | T/K | — | Knee temperature slope (absolute, not fractional — the knee can cross zero). Required together with `alpha_Br` when `magnet_temp` is set |

### `[thermal]` — optional

| Field | Units | Notes |
|-------|-------|-------|
| `winding_temp_limit` | °C | Insulation limit; with `r_th` and `ambient_temp` this sets the `thermal_resistance` S1 budget |
| `ambient_temp` | °C | Ambient for the temperature-rise budget |
| `r_th` | K/W | Winding-to-ambient thermal resistance |
| `insulation_class` | str | Free-form label (e.g. `"F"`); not parsed into a limit |
| `thermal_time_constant` | s | Winding thermal time constant for the transient duty march |
| `magnet_temp` | °C | Magnet operating temperature for `psi_f`/`B_knee` derating |

### `[iron]` — optional (required by `iron_loss`)

Per-kg Steinmetz coefficients for the lumped two-term Bertotti model,
plus the core mass and the single peak flux density the whole mass is
assumed to cycle at. All five are needed together — `prepare_iron_loss`
names the missing ones. Fit `k_h`/`k_e`/`alpha_fe` from a
multi-frequency specific-loss table with `phasesweep.models.iron_loss.fit_bertotti`.

| Field | Units | Notes |
|-------|-------|-------|
| `k_h` | W/(kg·Hz·T^α) | Hysteresis coefficient |
| `k_e` | W/(kg·Hz²·T²) | Classical eddy coefficient |
| `alpha_fe` | — | Steinmetz exponent, > 1 |
| `m_core` | kg | Core mass cycling at `B_core` |
| `B_core` | T (peak) | Effective lumped core flux density |

`B_core` is an *effective* lump, not a field value the geometry
predicts: real machines split B between teeth and yoke, and the
single-B model has no way to represent that. It can be derived from
flux and geometry or calibrated against a measured loss curve — record
which, in the file, since the two carry very different claims.
`motors/creator_case_pmsm.toml` is the calibrated case and documents
both numbers; the fit itself runs through the calibration framework
(`scripts/calibrate_creator_b_core.py`) and leaves a record next to the
dataset, the adjacent calibration record makes the calibrated value
reproducible.

With `[iron]` set, `thermal_duty` consumes the same `p_fe` against its
S1 budget (added to both the consumption and the `rated_current`
budget), so populating this section changes that model's output too.

### `[drive]` — optional

All four default to unset — there are no fallback bus/current/speed
values. A solver that needs one raises a `ValueError` naming the motor,
the model, and the missing field.

| Field | Units | Notes |
|-------|-------|-------|
| `U_DC` | V | DC bus voltage |
| `MAX_I_S` | A (peak) | Controller current limit |
| `W_REF` | rad/s (mechanical) | Reference speed |
| `I_LIMIT` | A (peak) | Stall-torque drive limit; falls back to `MAX_I_S` |

### Other sections

Sections not listed above (`[rated]`, `[provenance]`) and unrecognized
fields inside the listed ones (datasheet extras like `tooth_width`) are
ignored by `load_motor` — they are documentation for humans, not
inputs.

## `psi_f`/`B_rem` authority

`psi_f` (circuit tier) and `B_rem` (geometry/materials tier) encode the
same magnet through different models, so supplying both couples two
sources of truth. Choose one as authoritative by design intent:

- **Geometry-led** (a design study, no bench data): supply `B_rem` and
  omit `psi_f`; the solver factories derive flux linkage from magnet,
  geometry, and winding inputs.
- **Measurement-led** (a real motor with a back-EMF measurement): supply
  the measured `psi_f`; it is authoritative for the circuit models. An
  explicit `B_rem` then participates only in field calculations and is a
  separate model assumption.

Supplying both is legal. When the `B_rem` implied by `psi_f` disagrees
with the explicit value by more than 50%, every run that resolves them
warns (the threshold is loose because the simplified radial derivation
has real model error; inset and Halbach rotors are exempt because the
derivation does not apply). The warning means the two tiers describe
different magnets - resolve it by picking an evidence path, not by
tuning one value to silence the other. `scripts/check_motor.py` reports
the derived value, the disagreement, and the remediation for one motor.

## Minimal example

```toml
[motor]
name = "minimal-demo"

[circuit]
n_p = 4

[geometry]
r_outer  = 0.050
r_stator = 0.035
r_magnet = 0.032
r_rotor  = 0.025
```

This is enough for the analytical and FEM field solvers once `B_rem`
is added under `[materials]`. For a fully-populated example see
`motors/creator_case_pmsm.toml`.

## Errors

`load_motor` raises `ValueError` naming the file, section, and field:

```
ValueError: motors/foo.toml: [circuit] missing required field 'n_p'
```

`load_motors(directory)` loads every `*.toml` in a directory and
skips (with a warning) any file that fails validation.
