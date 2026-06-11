# Motor TOML Format

Motor definition files (`motors/*.toml`) are loaded by
`phasesweep.load_motor(path)` into a `Motor` dataclass. All values are
SI: meters, ohms, henries, webers, tesla, amperes, rad/s. Two
conveniences are converted at parse time: `I_rated_rms` → peak (× √2)
and `slot_opening_ratio` → `slot_opening_width`.

Only three things are required: `[circuit] n_p`, a `[geometry]`
section, and (optionally) a name. Everything else is optional — a
missing field means "not yet measured/specified", and the `prepare_*`
solver factories raise a ValueError naming the missing fields when a
solver actually needs them.

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
| `psi_f` | Wb (peak) | PM flux linkage per phase |
| `J` | kg·m² | Rotor inertia |
| `I_rated` | A (peak) | Rated continuous current |
| `I_rated_rms` | A (RMS) | Convenience: converted to peak ×√2. `I_rated` wins if both given |

### `[geometry]` — required

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
| `B_rem` | T | — | Magnet remanence. Derivable from `psi_f` (and vice versa) by the solver factories when winding params exist |
| `mu_r_fe` | — | 1000.0 | Iron relative permeability (linear FEM) |
| `mu_r_pm` | — | 1.05 | Magnet recoil permeability |

### `[drive]` — optional

| Field | Units | Default | Notes |
|-------|-------|---------|-------|
| `U_DC` | V | 540.0 | DC bus voltage |
| `MAX_I_S` | A (peak) | 20.0 | Controller current limit |
| `W_REF` | rad/s (mechanical) | 2π·50 | Reference speed |
| `I_LIMIT` | A (peak) | None | Stall-torque drive limit; falls back to MAX_I_S |

### Other sections

Unknown sections and fields (`[rated]`, `[provenance]`, datasheet
extras like `tooth_width`) are ignored by `load_motor` — they are
documentation for humans, not inputs.

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
