# Motor TOML Format

Motor definition files (`motors/*.toml`) are loaded by
`phasesweep.load_motor(path)` into a `Motor` dataclass. All values are
SI: meters, ohms, henries, webers, tesla, amperes, rad/s. Two
conveniences are converted at parse time: `I_rated_rms` → peak (× √2)
and `slot_opening_ratio` → `slot_opening_width`.

Only `[circuit] n_p` is required; `name` defaults to the filename stem.
`[geometry]` is optional: datasheet/circuit-only motors omit it and run
the circuit-tier models; the field solvers require it. Other fields can be
omitted until a model needs them: the `prepare_*` solver factories raise a
ValueError naming missing inputs.

## Sections

### `[motor]` — optional

| Field | Type | Notes |
|-------|------|-------|
| `name` | str | Defaults to the filename stem |
| `topology` | str | `"inrunner"` (default), `"outrunner"`, or `"axial"` (axial-flux; selects the [axial `[geometry]` keys](#geometry--axial-topology)) |
| `type` | str | Descriptive label (e.g. `"SPMSM"`); accepted, not read |

### `[circuit]` — required (`n_p` only)

| Field | Units | Notes |
|-------|-------|-------|
| `n_p` | — | Pole pairs. **Required.** Must be ≥ 1. The PM field and torque models (`analytical`, `fem`, `rated_torque`, …) additionally require ≥ 2; the axial-flux solvers (`analytical_axial`, `fem_axial`) and the induction tier accept n_p = 1 |
| `R_s` | Ω | Stator phase resistance |
| `L_d`, `L_q` | H | d/q-axis inductance |
| `psi_f` | Wb (peak) | PM flux linkage per phase. See [`psi_f`/`B_rem` authority](#psi_fb_rem-authority) before supplying both |
| `J` | kg·m² | Rotor inertia |
| `I_rated` | A (peak) | Rated continuous current |
| `I_rated_rms` | A (RMS) | Convenience: converted to peak ×√2. Setting both forms is an error |

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
| `L_stk` | m | Stack axial length |
| `n_slots` | — | Slot count (0 = smooth bore) |
| `slot_depth` | m | Radial slot depth (legacy alias: `slot_height`) |
| `slot_width_ratio` | — | Slot body width / slot pitch (default 0.6) |
| `slot_opening_width` | m | Slot throat width at the bore (Carter input) |
| `slot_opening_ratio` | — | Convenience: converted to width via slot pitch at the bore; width wins if both given |
| `alpha_p` | — | Magnet pole-arc / pole-pitch ratio, (0, 1] (default 1.0; a PM rotor without it loads with a warning) |
| `magnetization` | — | `"radial"` (default) or `"halbach"`. Halbach requires `alpha_p = 1.0` |
| `halbach_segments` | — | Segments per pole (κ_m). Required when `magnetization = "halbach"` |
| `halbach_angles_deg` | deg | Per-segment local magnetization angles (length = `halbach_segments`). Omit for the ideal Halbach rule |
| `back_iron_thickness` | m | **Outrunner only.** Magnetic back-iron ring inside the rotor wall, in (0, wall thickness); the rest of the wall is non-magnetic |
| `magnet_mount` | — | `"surface"` (default, interpole gaps are air) or `"inset"` (interpole gaps are rotor iron). See *FEM-only rotor features* |
| `bridge_thickness` | m | Iron bridge ring between the magnet faces and the airgap, in (0, magnet radial thickness). See *FEM-only rotor features* |
| `magnet_overhang` | m | Axial magnet extension per side beyond `L_stk`, ≥ 0. `0.0` = measured co-extensive, omit = not measured. See *End effects* |
| `k_end` | — | 3D axial end-effect factor on the winding flux linkage, (0, 1.2]. See *End effects* |
| `n_bars` | — | **Squirrel-cage rotor**: bar count, ≥ 2. See *Cage rotors* |
| `bar_shape` | — | `"rectangular"` (default; area `bar_width × bar_height`) or `"round"` (diameter `bar_width`; `bar_height` must be 0) |
| `bar_width` | m | Arc width at the bar mid-radius (rectangular) or the diameter (round) |
| `bar_height` | m | Radial extent of a rectangular bar |
| `bar_bridge_thickness` | m | Iron between the rotor surface and the bar top (default 0 = flush) |
| `bar_opening_width` | m | Air notch through the bridge at the rotor surface (default 0 = closed slot). Needs a bridge; narrower than the bar |
| `end_ring_axial_length`, `end_ring_radial_height` | m | End-ring section (not meshed; end-ring impedance input). Required by the `im_bar_impedance` and `im_fem_*` models |
| `skew_angle` | rad | Total **mechanical** bar skew over the stack (cage rotors only). See *Cage rotors* |
| `skew_slots` | — | Skew in **stator slot pitches** (`skew_angle = skew_slots · 2π/n_slots`). Needs `n_slots`; exclusive with `skew_angle` |

Legacy diameter names are also accepted: `stator_od`, `stator_id`,
`rotor_od`, `magnet_thickness` (halved/derived to radii). One of
`r_outer` or `stator_od` must be present.

Radii ordering is validated at load:
inrunner `r_outer > r_stator > r_magnet > r_rotor >= r_inner >= 0`;
outrunner reverses the middle three.

**FEM-only rotor features.** `magnet_mount = "inset"` and
`bridge_thickness` need the FEM model with `nonlinear = true` (the
interpole iron and the bridge saturate by design); the analytical solver
rejects both. Inset also requires `alpha_p < 1` and explicit `psi_f` and
`B_rem`, because the radial derivation between them ignores the interpole
iron shunt. A bridge composes with any `magnet_mount` or `magnetization`.

**End effects.** `magnet_overhang` is the main input to the offline 3D
wedge solve (`scripts/compute_k_end.py`) and is not read by the 2D
solvers. `k_end` is that solve's computed output; commit its
provenance record (`data/<motor>/k_end.record.json`) alongside. The
`analytical` model applies it to the `B_rem`-derived `flux_linkage_peak`
and `backemf_fundamental` only, never to a supplied `psi_f` (a measured
flux linkage already includes end effects). Values above 1 occur under
magnet overhang.

**Cage rotors.** `n_bars > 0` makes the rotor a cage: no PM annulus
(`r_magnet` defaults to `r_rotor`), inrunner only, `[induction]` required,
and the PM-only fields (`magnet_mount`, `bridge_thickness`,
`magnet_overhang`) must stay unset. It is cross-checked against
`[induction] n_bars` when both are given. `skew_angle` turns the
`im_fem_*` models into decoupled multi-slice averages, is rejected
together with `nonlinear = true`, and is capped at two bar pitches to
catch degrees given as radians. See
`data/creator_case_im/creator_case_im.toml` for a transcribed example.

### `[geometry]` — axial topology

With `topology = "axial"` the section builds an `AxialFluxGeometry`
(`machines/geometry.py`) for `analytical_axial` and `fem_axial`. Lengths are
in meters, or in millimeters through the `_mm` keys; a meter key wins when
both are given.

| Field | Units | Notes |
|-------|-------|-------|
| `r_out`, `r_in` | m | Active outer/inner radius. **Required** (or `D_out_mm`, `D_in_mm` as diameters in mm). `0 < r_in < r_out` |
| `airgap` / `airgap_mm` | m / mm | Mechanical air gap per stator/rotor side. **Required**, > 0 |
| `magnet_thickness` / `magnet_thickness_mm` | m / mm | Magnet axial thickness. **Required**, > 0 |
| `rotor_yoke` / `rotor_yoke_mm` | m / mm | Rotor back-iron axial thickness, ≥ 0 (default 0). Must be > 0 for `1R2S`, `2R1S-NN`, `2R1S-NS` |
| `stator_yoke` / `stator_yoke_mm` | m / mm | Stator yoke axial thickness, ≥ 0 (default 0). Must be > 0 for `1R2S`, `2R1S-NN`, `2R1S-NS`, `DSSR`; must be 0 for `coreless` |
| `variant` | — | `"1R2S"` (default), `"2R1S-NN"`, `"2R1S-NS"`, `"DSSR"`, or `"coreless"` |
| `n_slots` | — | Slot count per stator (default 0 = slotless). `coreless` rejects slots |
| `slot_depth` / `slot_depth_mm` | m / mm | Axial slot depth; must be less than `stator_yoke` |
| `slot_width_ratio` | — | Slot body width / slot pitch at the mean radius, in (0, 1) (default 0.6) |
| `slot_opening_width` / `slot_opening_width_mm` | m / mm | Slot opening at the mean radius; must be below the slot pitch |
| `slot_opening_ratio` | — | Convenience: converted to width via the slot pitch at the mean radius; width wins if both given |

`alpha_p`, `magnetization`, `L_stk`, and `k_end` are accepted as in the
radial table. See `data/parviainen_afpm/parviainen_5kw_afpm.toml` for an
example.

### `[winding]` — optional

| Field | Units | Notes |
|-------|-------|-------|
| `N` | — | Turns per coil (NOT per phase) |
| `k_w` | — | Winding factor, (0, 1] |
| `coils_series` | — | Coils in series per phase. Required when N and k_w are set (no auto-derivation): `N_eff = N × coils_series` |
| `coil_pitch_slots` | — | Coil span in slots, for the per-slot winding layout of the induction FEM models (`im_fem_harmonic`, `im_fem_locked_rotor`). Default: full pitch `n_slots / (2 n_p)`. The layout's fundamental winding factor must agree with `k_w` |
| `n_slots` | — | Accepted here too; must match `[geometry]` if both given |
| `connection` | — | `"wye"`, `"star"`, or `"y"` only; any other value (e.g. `"delta"`) is rejected at load, because the models derive phase voltage as V_LL/√3 |
| `layers`, `q`, `turns_per_phase`, `turns_per_tooth` | — | Descriptive winding labels; accepted, not read |

### `[materials]` — optional

| Field | Units | Default | Notes |
|-------|-------|---------|-------|
| `B_rem` | T | — | Magnet remanence. Derivable from `psi_f` (and vice versa) by the solver factories when winding params exist; see [`psi_f`/`B_rem` authority](#psi_fb_rem-authority) |
| `mu_r_fe` | — | 1000.0 | Iron relative permeability (linear FEM) |
| `bh_curve` | (A/m, T) | — | Lamination B-H curve as `[[H, B], ...]` rows, strictly increasing in both; the origin is implied. Read by the induction FEM models (`im_fem_harmonic`, `im_fem_locked_rotor`) when a run sets `nonlinear`; the PM FEM solvers keep the built-in generic curve. Absent: the generic curve |
| `mu_r_pm` | — | 1.05 | Magnet recoil permeability |
| `alpha_Br` | 1/K | — | Fractional remanence temperature coefficient, validated to (−0.01, 0); applied against `[thermal] magnet_temp` |
| `B_knee` | T | — | Demagnetization knee at the 20 °C reference (`demag_screen`) |
| `alpha_B_knee` | T/K | — | Knee temperature slope (absolute, not fractional; the knee can cross zero). Required together with `alpha_Br` when `magnet_temp` is set |
| `magnet_type`, `steel_grade`, `rotor_yoke_material` | str | — | Descriptive material labels; accepted, not read |

### `[thermal]` — optional

| Field | Units | Notes |
|-------|-------|-------|
| `winding_temp_limit` | °C | Insulation limit; every circuit model runs `R_s` at this temperature (`stall_torque`, `torque_speed`, `operating_point`, `thermal_duty`, `drive_sim`); with `r_th` and `ambient_temp` it also sets the `thermal_resistance` S1 budget. `motor_thermal` runs `R_s` at its solved winding temperature instead and reports headroom against this limit |
| `ambient_temp` | °C | Ambient for the temperature-rise budget |
| `r_th` | K/W | Winding-to-ambient thermal resistance |
| `insulation_class` | str | Free-form label (e.g. `"F"`); not parsed into a limit |
| `thermal_time_constant` | s | Winding thermal time constant for the transient duty march |
| `magnet_temp` | °C | Magnet operating temperature; every circuit model derates `psi_f` (and the `[flux_map]` surface) to it, `demag_screen` also `B_knee`. Requires `alpha_Br`. The field solvers do not derate (they solve at the 20 °C `B_rem`), although it still enters their run IDs |
| `magnet_temp_max` | °C | Magnet grade maximum operating temperature: a limit for `motor_thermal` headroom, never a derating temperature. `motor_thermal` solves the magnet node itself and ignores `magnet_temp`; the other circuit models derate to `magnet_temp` and do not check this limit |
| `winding_temp_limit_basis`, `magnet_temp_max_basis` | str | How the bounded temperature is measured: `"thermometer"`, `"resistance"`, `"ETD"` (IEC 60034-1) or `"hot_spot"`. Requires the limit |
| `winding_temp_limit_source`, `magnet_temp_max_source` | str | `"published"`, `"measured"`, `"computed"`, `"fitted"` or `"assumed"`. Requires the limit. `TempLimit.note` has no TOML field: it is set only on a per-run override |

### `[iron]` — optional (required by `iron_loss`)

Per-kg Steinmetz coefficients for the lumped two-term Bertotti model,
plus the core mass and the single peak flux density the whole mass is
assumed to cycle at. All five are needed together; `prepare_iron_loss`
names the missing ones. Fit `k_h`/`k_e`/`alpha_fe` from a
multi-frequency specific-loss table with `phasesweep.models.iron_loss.fit_bertotti`.

| Field | Units | Notes |
|-------|-------|-------|
| `k_h` | W/(kg·Hz·T^α) | Hysteresis coefficient |
| `k_e` | W/(kg·Hz²·T²) | Classical eddy coefficient |
| `alpha_fe` | — | Steinmetz exponent, > 1 |
| `m_core` | kg | Core mass cycling at `B_core` |
| `B_core` | T (peak) | Effective lumped core flux density |

`B_core` is an effective model input. The single-B model cannot represent
different flux densities in the teeth and yoke. Record whether `B_core` was
estimated from flux and geometry or fitted to a measured loss curve.
`motors/creator_case_pmsm.toml` shows the calibrated case.
`scripts/calibrate_creator_b_core.py` performs the fit and writes a
reproducible calibration record beside the dataset.
`data/creator_case_im/creator_case_im.toml` shows the derived case.

For an induction machine the block is the **stator** core: `iron_loss` runs
at `[induction] f_rated`, and `[drive] W_REF` is not consulted. `s_rated`
is optional there: when set, the rotor frequency is reported as an extra
`f_rotor_Hz` output (not lumped).

With `[iron]` set, `thermal_duty` consumes the same `p_fe` against its
S1 budget (added to both the consumption and the `rated_current`
budget), so populating this section changes that model's output too.

### `[induction]` — optional (squirrel-cage induction machines)

Setting this section makes the motor an induction machine: the
synchronous d-q fields (`psi_f`, `B_rem`, `L_d`, `L_q`) and `[flux_map]`
must then be absent (the loader raises on the contradiction), the `im_*`
models become available, and every synchronous-machine model rejects the
motor. All rotor quantities are referred to the stator.

| Field | Units | Notes |
|-------|-------|-------|
| `R_2` | Ω | Rotor resistance referred to the stator (DC / low-rotor-frequency reference) |
| `L_m` | H | Magnetizing inductance (rated-point reference) |
| `L_sigma_s` | H | Stator leakage inductance |
| `L_sigma_r` | H | Rotor leakage inductance referred to the stator |
| `n_bars` | — | Cage bar count (≥ 2). Must agree with `[geometry] n_bars` when a cage geometry is given |
| `sigma_bar` | S/m | Cage bar conductivity at the reference temperature (the harmonic FEM tier's bar material; the circuit tier does not read it) |
| `f_rated` | Hz | Rated supply frequency |
| `s_rated` | — | Rated slip, (0, 1) |
| `V_LL_rated` | V (RMS, line-line) | Rated supply voltage, line-line RMS. Check the datasheet's convention before transcribing |
| `design_letter` | — | NEMA design letter `"A"`/`"B"`/`"C"`/`"D"` (documentation / envelope priors) |
| `R_2_vs_f` | [[Hz, Ω], …] | Deep-bar table: rotor resistance vs **rotor** frequency `s·f`. The scalar `R_2` anchors f = 0; interpolation clamps past the last row |
| `L_sigma_s_vs_f` | [[Hz, H], …] | Stator leakage vs **supply** frequency, same anchoring |
| `L_sigma_r_vs_f` | [[Hz, H], …] | Rotor leakage vs **rotor** frequency, same anchoring |
| `L_m_vs_i` | [[A rms, H], …] | Magnetizing saturation: L_m vs magnetizing-branch RMS current, resolved by fixed-point iteration |

Tables must have ≥ 2 rows with strictly increasing x > 0 (the scalar
reference anchors x = 0). 2-pole (3000/3600 rpm) machines are accepted
(`n_p = 1`). The `im_equivalent_circuit` model solves the per-phase
circuit at `RunConfig.slip` (required, no default); `im_torque_slip`
sweeps slip at
the rated supply. Voltage and current outputs are per-phase RMS
(suffixed `_rms`), not the peak convention the PM circuit tier uses;
power and torque are whole-machine totals. See
`data/creator_case_im/creator_case_im.toml` for a fully-annotated
example including the convention traps. The four scalars are fittable
through the calibration workflow and appear in the validation-report
sensitivity grids; each perturbation scales its table with it.

If the machine arrives as test reports rather than a parameter set,
`phasesweep.models.im_circuit.identify_from_tests()` derives this
section's scalars from the standard DC / no-load / locked-rotor tests
(IEEE-112 short form); identifying at several locked-rotor frequencies
builds the deep-bar tables point by point. With the `fem` extra, a
cage `[geometry]` (bars and end-ring section), `sigma_bar` and a
`[winding]` section, `im_bar_impedance` predicts the deep-bar tables
from the field and reports them beside the measured ones; it never
replaces them.

### `[flux_map]` — optional (saturated synchronous machines)

Tabulated ψ_d(i_d, i_q) and ψ_q(i_d, i_q) surface: the full nonlinear
flux linkage that the constant `psi_f`/`L_d`/`L_q` triple linearizes.
Coexists with those `[circuit]` fields (they remain the linearized
operating point) but not with `[induction]`. When present,
`rated_torque` and `stall_torque` find the MTPA point numerically on the
surface (`mtpa_source = "flux_map"`) instead of the Morimoto quadratic;
`k_T` stays the zero-current magnet constant. `torque_speed` runs its
field-weakening search on the surface (`envelope_source = "flux_map"`),
`operating_point` solves its torque demand on it (`op_source =
"flux_map"`), and `drive_sim` builds motulator's saturated machine model from it
(`machine_model = "flux_map"`); plant and controller both see the
nonlinear surface. Each of these falls back to the constant-L path when
the grid does not cover the drive current limit. With a map in use,
`L_d`/`L_q` only set the timing plan and the SynRM/reverse-salient
guards; `drive_sim` additionally rejects surfaces that are non-salient or
reverse-salient at the current limit (motulator's MTPA search cannot
bracket them). With `[thermal] magnet_temp` set, every
consumer sees the map shifted to that temperature (ψ_d nodes moved by
ψ_f · `alpha_Br` · (T − `magnet_temp_ref_C`), ψ_q unchanged). Without
a `psi_f`, the scalar models take the map origin ψ_d(0, 0) referred to
20 °C (divided by 1 + `alpha_Br` · (`magnet_temp_ref_C` − 20)) when
`alpha_Br` is set. Without `alpha_Br` the origin is used as is, so a map
extracted hot gives the scalar models its extraction-temperature flux,
not the 20 °C value. All values are peak.

| Field | Units | Notes |
|-------|-------|-------|
| `i_d_A` | A (peak) | d-axis grid, strictly increasing, last value ≤ 0 (demagnetizing convention) |
| `i_q_A` | A (peak) | q-axis grid, strictly increasing, first value ≥ 0 (motoring convention) |
| `psi_d_Wb` | [[Wb, …], …] | ψ_d map, layout `[i_q_idx][i_d_idx]` (one row per `i_q` value) |
| `psi_q_Wb` | [[Wb, …], …] | ψ_q map, same layout |
| `magnet_temp_ref_C` | °C | optional: magnet temperature the map was extracted at; default 20 °C (the `alpha_Br` scalar-derating reference). Read by every map consumer when `magnet_temp` is set; unset maps keep their `config_id` |

The origin (i_d = 0, i_q = 0) should be a grid node so the map's own
ψ_f = ψ_d(0, 0) is defined; with `psi_f` also set, the two must agree
within 5% (a warning otherwise). Interpolation is cubic (≥ 4 points per
axis). `rated_torque` and `torque_speed` evaluate with bounds_error=True
(out-of-range currents are clipped or rejected by the caller). The
drive-sim model extends the one-quadrant grid by d-axis symmetry to
i_q < 0 and continues it linearly beyond the edges, so transients that
leave the grid stay bounded. See `motors/bmw_i3.toml` for a 7×7 example.

### `[thermal_network]` — optional (required by `motor_thermal`)

Three-node lumped thermal network (winding W, iron Fe, magnet/rotor M)
with the coolant as sink. Conductances in W/K, capacitances in J/K, all
six required and > 0 (a connected network with finite time constants).
The sink temperature is not a machine property: `motor_thermal` reads it
from `RunConfig.T_coolant` beside the `w_mech` / `tau_ref` demand. At
steady state only `G_WFe` and `G_FeS` matter (`T_M = T_Fe`); `G_FeM` and
the capacitances shape the transient functions. Record where each value
came from: a fitted `G_FeS` absorbs every loss the model does not carry.

| Field | Units | Notes |
|-------|-------|-------|
| `G_WFe` | W/K | Winding to iron (slot liner + impregnation) |
| `G_FeS` | W/K | Iron to coolant (yoke + jacket) |
| `G_FeM` | W/K | Iron to rotor across the air gap; invisible at steady state |
| `C_W`, `C_Fe`, `C_M` | J/K | Node heat capacities (specific heat × mass) |
| `winding_basis`, `magnet_basis` | str | Optional basis of the fitted node temperature (same vocabulary as the `[thermal]` limit bases). A limit on a different basis is rejected; unset is unchecked. Provenance only: not hashed into `config_id` |

### `[drive]` — optional

All four default to unset; there are no fallback bus/current/speed
values. A solver that needs one raises a `ValueError` naming the motor,
the model, and the missing field.

| Field | Units | Notes |
|-------|-------|-------|
| `U_DC` | V | DC bus voltage |
| `MAX_I_S` | A (peak) | Controller current limit |
| `W_REF` | rad/s (mechanical) | Reference speed |
| `I_LIMIT` | A (peak) | Stall-torque drive limit; falls back to `MAX_I_S` |

### Other sections

Every field inside the sections above must be one the loader reads (or
one of the descriptive labels marked "accepted, not read"); any other
key is an error, so a misspelled input cannot silently load as "not
set". The `[geometry]` field set depends on `[motor] topology`.

Tables the loader does not own are free-form documentation and are never
read: `[rated]`, `[provenance]`, `[validation]`, `[envelope]`,
`[electrical]`, `[columns]`, `[magnets]`, `[mass]`. Datasheet dimensions
the model does not use (tooth width, slot opening height, ...) belong in a
`[provenance.geometry]` sub-table, not in `[geometry]`. Any other
top-level table name is an error; a near-miss of a loader section
(`[thermall]`, `[circut]`) names the probable intended section.

## `psi_f`/`B_rem` authority

`psi_f` (circuit tier) and `B_rem` (geometry/materials tier) describe the
same magnet through different models. Choose the authoritative input
according to the evidence available:

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
tuning one value to silence the other. `phasesweep-check-motor` reports
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
ValueError: motors/foo.toml: [circuit] unknown field(s) 'pis_f' (did you mean 'psi_f'?); datasheet extras belong under [provenance]
ValueError: motors/foo.toml: unknown section [thermall] (did you mean [thermal]?)
```

`load_motors(directory)` loads every `*.toml` in a directory and
skips (with a warning) any file that fails validation.
