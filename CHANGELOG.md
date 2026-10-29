# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/).

Performance figures (timings, × speedups) are one-off measurements on the
stated development host at release time; they are not test-pinned and will
vary. Accuracy figures marked "measured, not pinned" were likewise observed
at release; the tests gate looser committed thresholds, not these values.

## [Unreleased]

## [0.6.0] - 2026-10-31

Model versions: a model whose version changes below re-keys its cached
runs. Additive outputs, append-only input fields (existing `config_id`s
hold) and guards that only reject bad input do not bump
`ModelInfo.version`; rows cached before them lack the new keys.

### Added
- **Induction-machine circuit tier**. `Motor.induction`
  (`InductionParams`, TOML `[induction]`) is the topology discriminator,
  mutually exclusive with the synchronous d-q fields and hashed
  append-only. `im_equivalent_circuit` solves the per-phase IEEE
  steady-state circuit at `RunConfig.slip` (no default, motoring range);
  `im_torque_slip` sweeps a fixed slip grid with breakdown, starting and
  rated-point extracts plus the constant-parameter Thevenin closed form.
  Both apply deep-bar `R_2_vs_f` and leakage tables (keyed on rotor vs
  supply frequency) and the `L_m_vs_i` saturation fixed point. Outputs
  are per-phase rms (`_rms` suffix), not the PM tier's peak convention;
  `eta_circuit` counts circuit losses only. An optional `[induction] R_fe`
  core-loss shunt across L_m reports `p_fe = 3|V_m|²/R_fe`, `R_fe_used`
  and the Boldea ratio `r_fe_over_x_m`; with R_fe beside measured
  `R_2(f)` tables, iron and stray loss may be counted twice. Synchronous
  `prepare_*` factories reject `[induction]` motors and vice versa;
  `perturb_motor` accepts the four induction scalars. `[winding]
  connection` is validated (every phase voltage is V_LL/√3).
- **Induction parameter identification**.
  `im_circuit.identify_from_tests()` recovers the circuit from DC,
  no-load and locked-rotor tests (IEEE-112 short form), with
  `separate_no_load_losses()` for the friction-and-windage split. On the
  CREATOR IM dataset: deep-bar R_2(f) within 9 %, leakage tables within
  5.5 %, L_m on the published saturation curve within 1.5 %, P_fw within
  8 %, and the raw-tests circuit within +2.6 % torque / +3.7 % current
  of the published-parameter circuit at nameplate slip (pinned). The
  Jurkovic MIT teaching note is the independent algebra fixture (~1 %).
- **CREATOR Case IM validation anchor**
  (`data/creator_case_im/creator_case_im.toml`). Stator current within
  2.5 % at all 45 measured locked-rotor points (10–50 Hz); torque within
  3.5 % at each frequency's top-current point, the residual pinned as a
  0.01–0.03 Nm core-loss-branch deficit. The constant-parameter circuit
  under-predicts locked-rotor torque ~30 % (pinned), which the deep-bar
  table earns. The 20 °C dataset resistances over-predict nameplate
  torque by +34 % (pinned temperature-trap finding); current at the
  nameplate-torque slip agrees within 8 %. The 31 Nm published maximum
  is a power limit, so breakdown is bounded one-sided. Tests skip without
  the external dataset.
- Induction circuit parameters (`R_2`, `L_m`, `L_sigma_s`, `L_sigma_r`)
  are fittable in `calibrate()` and the calibration CLI; the calibrated
  TOML round-trips `[induction]`. Validation-report sensitivity grids
  carry them on a `τ_rated_im` track, and the report accepts
  geometry-less motors and includes the CREATOR induction machine.
- `iron_loss` accepts induction machines as a stator-core lump at
  `f_rated` (no `W_REF`; `f_rotor_Hz` when `s_rated` is set). The CREATOR
  induction machine ships a derived `[iron]` block whose ~3.9× shortfall
  against the measured no-load separation is pinned.
- **Squirrel-cage FEM.** `[geometry] n_bars`, `bar_*`, `end_ring_*` and
  `[induction] sigma_bar` describe a cage; `fem_mesh.get_or_build_cage_mesh`
  meshes it with one region per bar, cached apart from the PM meshes
  (PM geometries hash and mesh as before). A cage whose bar count
  divides the pole-pair count, an end ring reaching the rotor radius, or
  an end-ring section too large for Alger's permeance is rejected. The
  CREATOR induction machine ships its transcribed geometry and
  `[winding]` block.
- Time-harmonic eddy-current solver (`solvers/fem_harmonic.py`): complex
  A_z with linear materials, a stator-frame velocity term for solid
  rotors, and per-bar axial-voltage unknowns closed by a cage circuit
  (prescribed currents, ideal ring, or end-ring segment impedance),
  eliminated by static condensation (36/28 production solve 255 s →
  13 s). Verified against Field's deep-bar solution, the bar/ring circuit
  identities and TEAM Problem 30a through an in-repo exact series
  solution (`solvers/team30a.py`, tables in `data/team30a/`). The
  magnetostatic PM solver is untouched.
- `im_bar_impedance` model: predicted deep-bar `R_2_list` /
  `L_sigma_r_list` vs rotor frequency, in the `[induction]` table shape,
  from a Bloch-periodic single-slot-pitch harmonic solve with the lumped
  end-ring segment; differential leakage reported separately. The
  circuit models keep reading the motor's measured tables.
- `im_fem_harmonic` (v3) and `im_fem_locked_rotor` (v3): full-machine 2D
  time-harmonic solves on the cage mesh with a per-slot double-layer
  winding (`[winding] coil_pitch_slots`, default full pitch; the layout
  must reproduce `k_w`). Both require the end-ring section. The first
  reports the operating point at `RunConfig.slip` and the rated supply
  (per-phase rms current, power factor, Maxwell-stress `tau` beside
  `tau_power`, loss split, `B_mag_grid`); the second runs virtual
  blocked-rotor tests at the motor's table frequencies through
  `im_circuit.identify_locked_rotor` and reports the linear-iron no-load
  current as a labeled diagnostic. One function owns the stator
  circuit: the 2D internal impedance Z_int = jωψ/I in series with
  `R_1 + jωL_ew`, shunted by `[induction] R_fe` when set. L_ew is a
  Boldea end-winding leakage (end-turn ratio π/2 default; CREATOR
  1.42 mH, closing ~50 % of the 2.8 mH leakage deficit at 50 Hz) the 2D
  section cannot see; it is in every terminal quantity and in the
  identified `L_sigma_s` after the IEEE-112 split, but not in L_m. With
  R_fe, terminal quantities include the shunt and air-gap quantities
  scale with the FEM-branch current; `p_in = p_cu_s + p_fe + p_ag`.
  CREATOR: locked-rotor current over-predicted 3–5 % (10 Hz) to 16–17 %
  (50 Hz), torque at the top-current points within ±10 %, identified
  `R_2(f)` 11–32 % under the published table with the same +26 % rise
  `im_bar_impedance` finds — pinned as findings, the circuit anchor
  untouched.
- Saturation under harmonic excitation (`RunConfig.nonlinear`) for both
  `im_fem_*` models. `[materials] bh_curve` carries a per-motor B-H curve
  (`fem_materials.BHCurve`, PCHIP; a curve ending above 3 T is rejected
  as swapped columns), append-only in `config_id`; motors without one
  keep the generic curve. Each iron element takes the energy-equivalent
  reluctivity ⟨ν|B|²⟩/⟨|B|²⟩ over its elliptic flux locus; the fixed
  point iterates on the current the stator circuit sends through the
  FEM branch, steps in log space with a per-element trust region, and
  raises on non-convergence. `im_fem_locked_rotor` adds a virtual
  no-load voltage sweep reporting L_m(I_m). CREATOR: the saturated
  no-load current sits on the measured reactive current at 0.5 and 1.1
  of rated voltage and up to 18 % under it in between, a gap a stacking
  factor of ~0.94 closes (not pinned); at rated slip saturation is a 2 % correction,
  at standstill it lifts current 30 % and torque 76 % — pinned as
  findings.
- Rotor-bar skew (`Geometry.skew_angle`, mechanical rad; TOML
  alternative `skew_slots`) for the `im_fem_*` models: five decoupled 2D
  slices with impedance, torque and powers averaged and the phase spread
  and worst bar current taken as maxima. This captures
  rotor-position-dependent slotting effects, not the classical
  sin(x)/x coupling (no axial bar-current continuity). Outputs
  `skew_model`, `n_skew_slices`; unskewed machines take the single-solve
  path unchanged; skew with `nonlinear = true` is rejected.
- `operating_point` model (v2): minimum-current (i_d, i_q) for
  a `(w_mech, tau_ref)` demand on `RunConfig` (hashed only when set).
  Constant-L MTPA with three-stage constant-torque field weakening (pole
  floor, MTPV, bisection to the voltage limit), or a bracketed contour
  search on the `[flux_map]` surface when it covers the current limit.
  Outputs dq currents, voltages and flux linkages (`psi_d`, `psi_q`),
  `m_index = 2·u_mag/U_DC`, power factor, regime, copper and shaft
  power, and the surface used. `R_s` is taken at `winding_temp`
  (default the motor's `winding_temp_limit`), PM flux at `magnet_temp`,
  on the map through `flux_map_at_magnet_temp` against the optional
  `[flux_map] magnet_temp_ref_C` (20 °C default).
  `cold_operating_point_params(motor, winding_temp, magnet_temp)` sets
  both explicitly. Not submittable as a server job (no job param sets
  the demand); use the Python API.
- `motor_thermal` model (v2): three-node lumped network
  (winding, iron, magnet; `[thermal_network]`, hashed append-only)
  coupled to `operating_point` as a fixed point — copper loss at the
  winding temperature, magnet flux at the magnet temperature, iron loss
  scaled by `|psi_s|/psi_ref`. Coolant temperature is
  `RunConfig.T_coolant`. Temperature limits (`[thermal]`
  `magnet_temp_max`, basis/source tags) give `headroom_winding_K`,
  `headroom_magnet_K`, `binding_limit` and `within_s1`; exceeding a limit
  is a result, not an error. `coupled_steady(..., limits=..., U_DC=...)`
  overrides limits and drive bus per run. Failures raise
  `InfeasibleDemand`, `ThermalRunaway` or `NoConvergence` (all
  `CoupledSolveError`, a `ValueError`) with the iterate state; the type
  is diagnostic and can depend on `relax` / `max_iter`. The n-node
  functions (`steady_temps`, `segment_map`, `periodic_start`,
  `periodic_peak`) reduce to `thermal_duty.transient_peak` on one node.
  Not submittable as a server job.
- `iron_loss.iron_loss_at(...)`: the lumped Bertotti loss at a
  mechanical speed and stator flux ratio.
- **`[flux_map]`** (`FluxMap`): tabulated ψ_d(i_d, i_q), ψ_q(i_d, i_q)
  for saturated synchronous machines, append-only in `config_id`.
  `rated_torque` and `stall_torque` run a numeric MTPA on the surface
  where it covers the current circle (`flux_map.map_covers`) and report
  `mtpa_source` (`stall_torque` also `mtpa_source_electromagnetic`). A
  map extracted hot (`magnet_temp_ref_C`) is referred to 20 °C before
  derating. `drive_sim` / `drive_sim_two_mass` run on the surface when it
  covers the drive current limit (`make_dq_maps` → motulator
  `SaturatedSynchronousMachinePars`), output `machine_model`, and fall
  back to constant L otherwise; non-salient or reverse-salient surfaces
  are rejected (motulator's MTPA bracket returns a zero-torque
  sentinel).
- **BMW i3 2016 IPM anchor data.** The Gierczynski 7×7 flux map: numeric
  MTPA within 1 % of the source torque sheet at (−396, 404) A where the
  secant-L MTPA over-predicts ~8 %; drive-sim steady state on the map
  MTPA to 0.5 % in current (measured, not pinned). The map publishes no
  extraction temperature, so hot-magnet peak torque is a bracket: at
  100 °C magnets the envelope is 243 N·m against a 20 °C reference and
  the 250 N·m nameplate holds against a 100 °C reference (pinned). An
  `[iron]` block fitted to the M-19 29 Ga catalog core-loss table
  (50–2000 Hz, B ≤ 1.5 T, ~5 % median residual; the CC BY-NC-SA table is
  cited, not redistributed) with ORNL core mass and an uncalibrated
  geometric `B_core` (`iron_loss` runs on the i3 given a `W_REF`). A `[thermal_network]` block whose `G_FeS` is
  fitted through the coupled solve to the digitized Burress 2017
  continuous test at 7000 rpm (assumed 60 °C coolant, 55–63 °C bracket
  recorded, residuals about ±4 K) — a pin, not a validation;
  `scripts/check_i3_thermal_transient.py` marches the block through the
  whole trace. These blocks re-key every i3 run ID.
- `torque_speed` emits the envelope currents `i_d_curve_peak` /
  `i_q_curve_peak` (and `_cont`).
- `analytical`, `analytical_axial` and `fem_axial` emit `backemf_source`
  (`"psi_f"`, `"computed"` or `None`): which input set
  `backemf_fundamental`.
- `phasesweep-check-motor`: staged configuration checks with an optional
  `--fem quick` analytical/FEM cross-check (one-process pool, 300 s
  deadline; a solver exception or timeout is a failed check).
- **Record schema and vectors.** `phasesweep.record_schema` exports
  TypedDicts for the stored-record envelope (`MotorDict`,
  `RunConfigDict`, `RunResultDict`, `StoredRecord`), and
  `_vectors/record_vectors.json` freezes real rows so `to_dict()` drift
  is caught (`scripts/gen_record_vectors.py` regenerates them). With
  `docs/result-store-contract.md` it is the documented consumer
  contract.

### Changed
- **License changed from LGPL-2.1-only to Apache-2.0** (`LICENSE`,
  `pyproject.toml`, `README.md`, `CONTRIBUTING.md`, and the statements
  about our own transcription files). Third-party terms are unchanged:
  NGSolve stays LGPL-2.1 as an optional dependency, the CREATOR PMSM and
  IM datasets stay CC BY-NC-ND 4.0 and CC BY-NC 4.0, and the Pyleecan
  geometry keeps its attribution in `data/prius_2004/NOTICE`. No
  `config_id` changes.
- **One temperature rule for every circuit model** — `torque_speed` v5,
  `rated_torque` v6, `stall_torque` v6, `drive_sim` v4,
  `drive_sim_two_mass` v2: `R_s` at `winding_temp_limit`, `psi_f` and the
  `[flux_map]` surface at `magnet_temp`, through one shared
  `solver_params.derate` and `flux_map.derated_flux_map`. `thermal_duty`
  and `demag_screen` gain the same range guard (derated `psi_f` > 0,
  `R_s` ≥ 0). Motors without the thermal fields are output-identical.
  With them: the BMW i3 at 100 °C magnets reports 243 N·m peak from
  `rated_torque`, `stall_torque`, `torque_speed` and the
  `operating_point` envelope alike (was 258 N·m from the first three);
  the ETEL TMB sheets (130 °C coil limit) run at +43 % `R_s`, base speed
  at In 3–9 % lower than cold, and the TMB0450 VA refuses `torque_speed` at its
  peak current — the hot resistive drop 353 V exceeds U_DC/√3 = 346 V
  (pinned; the sheet's Ip is a short-duty rating).
- The gap-thickness sweep axis is `SweepAxis("airgap", ...)`; the old
  `"r_ag"` raises (`r_ag` is the mid-gap sampling radius, never swept).
  The axis moves the stator surface and keeps the magnet fixed, as
  `perturb_motor` does; it used to split the change between bore and
  magnet face, changing magnet thickness. Points on that axis get new
  run IDs. The `proportional` strategy also scales
  `back_iron_thickness`, `bridge_thickness`, the cage bar dimensions and
  `end_ring_radial_height`; axial lengths stay fixed with `L_stk`.
- **Motor TOML keys are validated.** `load_motor` and the dashboard
  config editor reject an unknown key inside a loader section (with a
  near-miss hint, `pis_f` → `psi_f`) and any unknown top-level section
  (`[loss] k_h = ...` used to load with `k_h` unset). Descriptive labels
  (`[motor] type`, `[winding] layers/connection/q/turns_per_phase/
  turns_per_tooth`, `[materials] magnet_type/steel_grade/
  rotor_yoke_material`) stay accepted; `[rated]`, `[provenance]`,
  `[validation]`, `[envelope]`, `[electrical]`, `[columns]`, `[magnets]`
  and `[mass]` stay free-form. Setting both `I_rated` and `I_rated_rms`
  is an error. **Migration:** move datasheet dimensions the model does
  not use from `[geometry]` into `[provenance.geometry]`, and `topology`
  into `[motor]` (see `docs/motor-toml.md`, "Other sections").
- **ResultStore wraps a vendored `sweepstore`** (`phasesweep._sweepstore`,
  sweepstore 0.1.1; staleness vectors ship with it). Public API and run
  IDs unchanged; the store stays in the core install and `[store]` is an
  empty extra. `load_slim` trusts a stored `run_id` (rows carrying
  `run_id` and `motor_config_id` skip re-hashing; legacy rows
  recompute), which also fixes an `ImportError` loading a `drive_sim`
  row without `motulator`. `load_results(current_only=True)` applies the
  same superseded-version rule; the validation summary,
  `phasesweep-crossval` and the single-result route share one
  `record_is_current` helper.
- **Server generic layer from a vendored `view-sweep-server` 0.3.0**
  (`phasesweep._view_sweep_server`): `ConnectionManager` and `/ws`,
  local-only Host/Origin middleware, logging, NaN-safe JSON, and the WS
  protocol TypedDicts (now in `phasesweep._view_sweep_server.protocol`,
  a private API; `phasesweep.server.protocol` is removed).
  `phasesweep.server.app` keeps `ServerSettings`/`create_app`. Tauri
  origins are accepted by default; WS `cancel` frames reach
  `JobManager.cancel`; a malformed subscribe frame is dropped whole. The
  `/view-sweep/` dashboard assets ship in the wheel, and the `server`
  extra adds only FastAPI, uvicorn, structlog and tomli-w.
- Dashboard migrated onto view-sweep (served at `/view-sweep/`): theme,
  plots, chrome, composite export and data providers come from
  view-sweep; `style.css` keeps only domain rules on `--vs-` tokens.
  Persisted `ps-*` localStorage keys are unchanged; topbar controls are
  one right-aligned group.
- Cross-validation: when both datasets carry a tolerance for a quantity,
  the tighter one applies (was the wider); no shipped verdict changes.
  Curve comparisons reject a `curve_x` that is not strictly ascending.
- A PM `[geometry]` without `alpha_p` logs a warning when it defaults to
  full pitch (cage rotors and Halbach arrays stay silent).
- `geo_parallel.run_sweep(..., fem_nonlinear=True)` documents the
  saturating solve a sweep forced for `"fem"`; `False` gives linear iron.
- `Motor` accepts `n_p = 1` (2-pole induction machines); the PM field and
  torque models keep their `n_p >= 2` guard.
- The analytical runner raises `RuntimeError` if |r·B_r1| at mid-gap is
  below its value at the iron surface (a Laplace-solution invariant).
- One-sided saliency (`L_d` without `L_q` or vice versa) is a
  `ValueError` in `rated_torque`, `stall_torque`, `thermal_duty` and the
  dashboard's derived rated torque instead of silently running the
  non-salient form.
- `arkkio_torque` rejects `r_outer <= r_inner`; `demag_screen` rejects
  `i_fault <= 0` and non-finite values.
- `fem_wedge` and `fem_axial_3d` are marked research code (offline
  studies outside the public API); their tests carry the `slow` marker.
- `numpy>=2.0` is declared.

### Removed
- `run_fem_safe`, `run_cogging_safe`, `run_sim_safe` and the
  `python -m` entry points of `solvers.fem_runner`, `solvers.cogging`
  and `simulation.sim_runner`. The registry runners are the public
  `run_fem`, `run_cogging` and `run_drive_sim`; timeouts belong to the
  job pool.
- The `PHASESWEEP_MESH_CACHE` environment fallback; use
  `set_disk_cache_dir`.
- `ResultStore.get_stats`, `ResultStore.mark_pending` and the in-memory
  index cache.
- `ModelInfo.fundamental_only` (no model has used it since analytical
  v4).
- `fem_mesh._build_geometry` / `_build_slotted_geometry`; use
  `_build_motor_geometry`.

### Fixed
- Missing-extra errors no longer say `pip install phasesweep[...]` (the
  PyPI name belongs to an unrelated project): they read
  `pip install -e ".[<extra>]"` from a source checkout and name the
  missing module. `phasesweep-server` without the extra exits with one
  line, and `phasesweep.server` modules raise an `ImportError` naming the
  extra.
- `execute_parallel` / `run_sweep` with two or more workers applied a
  hidden 600 s pool deadline when no `timeout_s` was given, so long FEM
  sweeps timed out only in parallel. No `timeout_s` now means no
  deadline. With `workers=1` and a `timeout_s`, a one-process pool now
  enforces the deadline (the server's sub-task budget applies at
  `--workers 1`).
- `phasesweep-crossval --strict` passed when nothing was compared; it now
  exits 1 with "no comparisons were made".
- `RunConfig` validates every field: non-bool `nonlinear` (`"false"` ran
  nonlinear), unknown `nonlinear_solver` (`"Newton"` ran Picard in
  `fem_axial`), non-integer counts, non-finite `maxh_fraction`, `j_s`,
  `rotation`, `i_fault`, `slip`, and non-finite or zero-duration
  `duty_profile` segments raise; integral floats normalize to `int`. The
  server parses string params and returns 400 for unparseable values.
  Only already-mistyped inputs get new run IDs.
- Non-finite inputs are rejected by `RatedTorqueParams`,
  `StallTorqueParams`, `ThermalDutyParams` (a NaN duty segment stored
  `p_cu_avg = nan` with status OK), `IronLossParams`, and `Motor`
  (`ambient_temp`, `B_knee`, `alpha_B_knee`, `winding_temp_limit`,
  `magnet_temp`).
- Measured-data import validates values: a string, boolean, NaN or
  infinite quantity, uncertainty, tolerance or numeric condition raises
  `MeasuredFormatError` (a `ValueError`), and `POST /api/measured/{motor}`
  returns 400 before saving. String quantities used to drop out of
  comparison and string tolerances made the validation route 500.
- `BoundRef.relation` and `CurveRef.extract` are checked on construction
  (an unknown relation compared as `lt`, an unknown extract as RMS;
  `extract = "interp"` without `at_x` now raises).
- `RunResult.from_dict` and `ResultStore.load_slim` validate `status` and
  `source`; a legacy descriptive source (`"published (ANSYS FEM)"`) reads
  as `published`.
- A quoted number in a motor TOML (`R_s = "0.2"`) raised a bare
  `TypeError` that stopped server startup; it is a `ValueError` naming
  the field, and both loaders skip the file with a warning.
- `perturb_motor` OD and gap rebuilt the geometry through the builders,
  dropping `magnet_mount`, the bridge, `magnet_overhang`, back iron,
  cage, skew and a custom `r_ag` (CREATOR ψ_f 0.1144 → 0.1223 Wb at zero
  delta; i3 0.0436 → 0.0859 Wb). Only the radii change now. OD/gap return
  `None` (logged) where ψ_f cannot be re-derived from geometry (inset,
  bridged, Halbach, `[flux_map]`), and a `[flux_map]` motor refuses
  `L_stk`, `B_rem` and `k_w`; calibration and sensitivity show n/a
  instead of a wrong number.
- `scripts/populate_cache.py --verify` reported NaN or infinite metrics
  as matching (and `inf` as `-inf`); it now reports them, and flags a
  row whose stored config no longer hashes to its `run_id`.
- Server jobs: an executor exception left sub-tasks `pending` and the job
  reported the wrong outcome; those sub-tasks are now `ERROR` with the
  message. Sub-tasks canceled before starting no longer persist an
  `ERROR` row. Server sweeps no longer force `nonlinear=True` for
  `cogging_torque` (one grid got two sets of run IDs).
- Stored `drive_sim` rows failed to load without the `[sim]` extra;
  `SimPlan` now lives in `simulation.params`.
- `write_motor_toml` dropped `[flux_map]`, `[thermal_network]`, the
  Halbach fields, skew and the temperature-limit tags, so a calibrated
  TOML hashed differently from its record's `calibrated_config_id`. It
  now writes from `motor_to_toml_dict`, which every shipped motor
  round-trips. A geometry-less motor setting `L_stk`, `k_end`, `alpha_p`
  or Halbach fields raises.
- `produces` now lists every key a model may emit (`mtpa_source`,
  `f_rotor_Hz` and the `fem` nonlinear diagnostics were missing), so
  calibration and measured import can select them.
- Integral count fields (`n_slots`, `n_bars`, `n_p`, `N`,
  `coils_series`, `coil_pitch_slots`) are coerced to `int`, so `12.0`
  hashes like `12`; a non-integral count raises.
- `end_effect_factor` and `end_effect_factor_pole_pitch` return the
  documented limit 0 at `L_stk = 0` (was 1.0; unreachable from `Motor`).
- `resolve_psi_f` / `resolve_b_rem` errors name the missing geometry as
  well as missing winding inputs.
- `analytical_axial` raises on a zigzag leakage factor `k_zz <= 0`
  instead of multiplying it into `flux_linkage_peak`.
- `fem` runner: a B_r fundamental below 1e-9 of the peak is a failed
  solve (slot-harmonic percentages divided by a stand-in 1.0).
- `solve_field_fem` rejects `n_p < 1`, an `n_slots` override that
  disagrees with the geometry while `slot_opening_width` is set, and an
  interpole gap below the 0.5 mm OCC limit for surface magnets (it solved
  a full ring behind a warning; `info["arcs_collapsed"]` is removed). No
  shipped motor is below the limit.

## [0.5.0] - 2026-08-30

### Added
- **`scripts/check_motor.py` - staged motor check with explicit evidence
  levels.** Structural checks, `psi_f`/`B_rem` authority report, model
  applicability, fast-model execution with finite-output checks, and
  optional `--fem quick` analytical/FEM cross-check. Five-dimension status
  summary (CONFIG / EXECUTION / CONSISTENCY / CROSS-CHECK / MEASUREMENT).
  Exit code is nonzero only for objective failures; engineering concerns
  stay warnings.
- Hypothetical radial-SPM starter (`motors/examples/`) for the new-motor
  workflow.
- **Axial end-effect correction.** `Geometry.magnet_overhang` and
  `Motor.k_end` drive an offline 3D wedge calculation
  (`scripts/compute_k_end.py`). The correction applies only to
  `B_rem`-derived flux linkage, since measured `psi_f` already includes
  end effects. Existing motors are unchanged.
- **Newton solver for nonlinear 3D wedge and 2D FEM.** Falls back to
  Picard on stall and agrees to ~1e-6. Becomes the default with the v8
  BH-spline change below.
- **Elmer cross-code verification** (`scripts/elmer_radial/`): six
  radial FEM cases checked against ElmerFEM, with B₁ agreement
  0.07-0.21%. Pinned results run in the core test lane without
  requiring Elmer or NGSolve.
- **Halbach motor with iron retaining sleeve**
  (`halbach_nwpu_12kw_sleeved.toml`): a hypothetical variant for
  exploring sleeve effects. Nonlinear FEM reduces B₁ from 0.988 T to
  0.769 T (~22%). No measured data available, so this is not a
  validation anchor.
- **Batch torque-speed envelope.** `envelope_curve_batch` computes N
  envelopes in one broadcast pass, giving the sweep and server paths
  access to batched dispatch. Outputs are byte-identical to the serial
  path.
- **psi_f = 0 (synchronous reluctance machines).** Circuit models now
  support zero flux linkage. The L_d > L_q convention is required for
  drive_sim; field solvers reject psi_f = 0 since there is no PM field
  to solve. Validated against the Jilin 5.5 kW SynRM anchor.
- **Inset-magnet FEM rotors.** `magnet_mount = "inset"` with optional
  `bridge_thickness`. Nonlinear FEM is mandatory because the
  interpole/bridge leakage shunt saturates by design. CREATOR inset
  model closes FEM psi_f from +8% to +3% vs published 0.1144 Wb.
- **Halbach array magnetization in FEM.** Per-segment constant
  magnetization following the Zhang 2021 formulation. Supports both
  inrunner and outrunner topologies; the analytical solver gates
  Halbach (radial-only model).
- **`fem_axial` model (v1).** Multi-plane quasi-3D axial-flux FEM,
  open circuit only. Verified against the analytical layer model,
  periodic identification, and Elmer (measured, not pinned).
- **`analytical_axial` model (v1).** Quasi-3D Cartesian layer solver
  (Parviainen 2005 / Chung 2002) for iron-backed axial-flux variants;
  coreless is rejected. Parviainen 5 kW anchor: back-EMF -0.07% vs
  measured.
- `fem_axial_3d` - 3D scalar-potential edge-leakage study (not
  registered as a model). The analytical radial-edge leakage factor
  under-predicts FEM leakage (~1.9-2.0% vs 1.07%).
- `scripts/elmer_axial/` - Elmer verification for axial FEM, with a
  pinned-result test that runs in the core CI lane.
- **M800-50A iron-loss validation scaffold.** Nine-frequency loss
  tables from the CREATOR IM dataset (DOI 10.3217/kh3v7-dhn98). Global
  50-1000 Hz fit: ~18% median error, ~30% under-prediction at mains
  frequency, ~24% machine-level eddy share at 50 Hz. Tests skip in
  checkouts without the external dataset.

### Changed
- `ModelInfo.needs_runtime` declares which `RunConfig` fields a model
  needs beyond the motor.
- New-motor documentation: `psi_f`/`B_rem` authority, FEM excitation,
  test lane distinction.
- README validation language: CREATOR back-EMF identified as a
  consistency check, not independent validation.
- Public guides and release notes tightened; no model-limit or
  configuration changes.
- **Nonlinear FEM: monotone BH spline + Newton default (fem v7 → v8;
  fem_axial, cogging_torque, demag_screen v1 → v2).** PCHIP spline
  replaces piecewise-linear B-H interpolation, and Newton replaces
  Picard as the default nonlinear solver. Converged B₁ shifts by at
  most 0.03%; linear solves are unchanged.
  `RunConfig.nonlinear_solver` selects Picard per-run.
- **`fem_field` split into four modules:** `fem_mesh.py`,
  `fem_materials.py`, `fem_sampling.py`, and `fem_field.py`. No
  behavior change; public re-exports are preserved so existing imports
  continue to work.
- **`solver_params` split:** each Params dataclass moved next to its
  model's run function. The public API is unchanged; only direct
  imports from `phasesweep.solver_params` of moved names need updating.
- **FEM evaluation loops vectorized.** Mesh-point location is now
  batched and the MKL availability probe is cached per process. No
  output changes.
- **`torque_speed` grid thinned 601 → 151 (v2 → v3).** The coarser
  grid produces negligible output shift and enables batched dispatch.
- **`torque_speed` vectorized over speed axis (v1 → v2).** The full
  speed grid is now one set of array operations, with negligible output
  shift.
- **`bridge_thickness` decoupled from inset mount.** The iron retaining
  ring now works with any magnet_mount or magnetization. Nonlinear FEM
  remains mandatory regardless of mount type.
- **CREATOR TOML promoted to `magnet_mount = "inset"`.** The published
  saliency (L_q/L_d = 1.6) confirms interpole iron. Analytical crossval
  tests removed; FEM psi_f agreement improves from +8% to +3%.

### Fixed
- sdist no longer ships Elmer working files (hatchling does not honor
  nested `.gitignore` files).
- README quick-start uses the Belkhadir outrunner instead of CREATOR,
  which the analytical solver now rejects after the inset-mount
  promotion.
- `solve_field_fem` validates inputs before importing ngsolve, so
  missing-FEM installs get a `ValueError` instead of
  `ModuleNotFoundError`.
- Sweep jobs on axial-flux motors raise a clear error instead of
  crashing inside radial-only grid generation.
- motulator pinned to `==0.7.3` (0.7.4+ breaks `build_sim`).
- `torque_speed` rejects non-finite inputs instead of silently
  returning a zero-torque envelope.
- Rotated meshes now persist to the disk cache with LRU pruning
  (`PHASESWEEP_MESH_CACHE_MB`, default 512).
- Full-ring rotors (α_p ≥ 1) share one mesh across all rotor angles
  instead of rebuilding per position.
- `run_cogging_safe` forwards the parent's mesh cache directory to
  subprocesses.
- Server worker now round-trips `rotation`, `cogging_points`, and
  `n_planes` through the job config instead of dropping them.
- Calibration auto-model selection skips models that do not validate
  for the motor's topology.

## [0.4.0] - 2026-08-18

### Changed
- **Extras split.** Core install (`pip install phasesweep`) requires
  only numpy + scipy. FEM, drive simulation, plotting, and the server
  are optional extras (`[fem]`, `[sim]`, `[viz]`, `[server]`). Install
  everything with `pip install "phasesweep[all]"`. Missing extras raise
  `ImportError` with an install hint. **Migration:** existing installs
  should switch to `"phasesweep[all]"`.
- CI now runs three lanes: full (`[all]`), core-only, and a scheduled
  Tier-1 FEM job, with strict-skip gating to catch silent coverage
  erosion.

### Fixed
- `RunConfig.from_dict` no longer crashes in a core-only install
  (SimPlan import deferred).
- `test_analytical.py` vestigial `importorskip("ngsolve")` removed -
  it was silently skipping ~385 pure-analytical tests.

### Added
- **Toyota Prius 2004 IPM benchmark.** Circuit-only motor (50 kW,
  8-pole, V-shape NdFeB) with ORNL dynamometer validation: stall torque
  bound (340 Nm at 250 A), torque-speed envelope anchors (base speed,
  CPSR, power at 6000 rpm). Includes Pyleecan geometry (parked for
  future IPM ingest).
- **`cogging_torque` model (v1).** Sweeps `solve_field_fem` over one
  cogging period at open circuit, computing Maxwell-stress torque via
  Arkkio's radial-averaging method. `RunConfig.rotation` and
  `cogging_points` control the sweep.
- **`drive_sim_two_mass` model (v1).** Two-mass mechanical system
  (motor + load inertia coupled through a torsional shaft) using
  motulator's `TwoMassMechanicalSystem`. Anchor: Saarakkala & Hinkkanen
  2015 closed-form resonance/antiresonance (84.2 / 59.6 Hz).
- CREATOR cogging anchor migrated from raw `maxwell_stress_torque` to
  the model API.
- Cogging offset-shift noise diagnostic quantifies per-angle remeshing
  noise floor (<5% dominant-order stability).
- Server timeout scales with `cogging_points` for cogging jobs.
- `scripts/nitsche_rotation_spike.py` - mesh-deformation vs remeshing
  comparison for the rotation R2 decision.
- **JMT 1806 2400KV dataset-integrity tests.** Convention
  self-consistency and catalog-KV cross-check (0.5%) for the third
  measured back-EMF dataset. Model cross-validation still needs caliper
  geometry.

## [0.3.0] - 2026-07-30

### Added
- **CREATOR `B_core` re-derived through the calibration framework.** The
  no-load loss curve is now a registered dataset and the calibration
  script fits `B_core` from the unfitted FEM-flux estimate, recovering
  the previous hand-derived value to 0.014%. The dataset is deliberately
  untagged for the circularity guard since the shipped `B_core` was
  derived from it. CREATOR still does not validate `iron_loss`.
- **`docs/architecture.md`** - data-flow spine, three-tier hashing,
  model-coverage matrix, and claim tables.
- **`[iron]` section on CREATOR TOML** - first `iron_loss` coverage in
  `motors/`. `B_core` is calibrated to the measured no-load loss, not
  predicted from FEM flux; the unfitted FEM estimate runs 2.7x light
  against measurement. Both values are pinned.
- `[thermal]`, `[iron]`, `back_iron_thickness`, and magnet/demag
  `[materials]` fields documented in `docs/motor-toml.md`. These were
  parsed by `load_motor` but absent from the format doc.
- `iron_loss` added to `populate_cache.py`'s fast-model set, making it
  part of the `--verify` coverage.
- **`populate_cache.py --verify`** recomputes every current-version
  record and compares. `--verify-fem` includes FEM; `--rtol` sets the
  tolerance; `--prune-mismatched` drops failures for recomputation.
- **`--require-model`** makes `--verify` a physics-change gate by
  exiting 1 on zero coverage for the named model.
- Clicking a completed job card re-plots its results and switches to
  that motor.
- **Dashboard color palettes.** Selectable series palettes (Okabe-Ito
  default, Tol, Seaborn, Tableau) and perceptual |B| ramps (Viridis,
  Cividis, Magma). All theme-aware.
- **Dashboard measurement cursors, panel reorder, composite PNG
  export.** A/B cursors with per-series stats on result panels.
  Drag-to-reorder within a tab. Topbar PNG export with provenance
  stamps.
- **Dashboard keyboard shortcuts + linked highlight.** `1`-`4` switch
  tabs, `j`/`k` step configs, `?` shows the overlay. Hovering a result
  in one panel highlights it across all panels. ARIA tablist.
- **Result version stamp + staleness filter.** Every stored result
  carries a `model_version` stamp. Server read paths serve a record only
  when its stamp matches the live registry, so a physics version bump
  invalidates stale results.
- **Dashboard nameplate, stat tiles, provenance stamps.** Active config
  summary tiles and per-result provenance lines (`model vN · timestamp ·
  result_id`).
- **Tabbed dashboard layout.** Four tabs: Results, Sweep, Validation,
  Editor. Panel tab assignment persists per browser.
- **Dark theme.** Topbar toggle, persisted per browser, applied before
  stylesheet load. Chart styling reads CSS custom properties.
- **Editor ergonomics.** Geometry lengths display in mm (saved as
  meters). Only user-edited fields write back, so display rounding
  cannot drift untouched values. Muted text darkened to WCAG AA 4.5:1.
- **Sweep plot + run history.** Metric-vs-axis plot over sweep rows,
  CSV export, sidebar run-history panel.
- **Dashboard calibration workflow.** Import measured data, see
  disagreement, edit params, re-run, save calibrated TOML - end to end.
  Datasets attach by motor name so they follow parameter edits; the
  validation panel renders an echo warning for non-independent
  agreement.
- **Comparison workflows.** Model comparison panel with crossval
  quantities and measured-data markers. Validation summary matrix with
  tolerance-tier coloring. Ctrl/cmd-click multi-config overlay.
- **Result panels.** Harmonic spectrum, cross-section |B| heatmap, sim
  waveforms, and sortable sweep table. New additive model outputs:
  `B_mag_grid` (FEM) and downsampled sim traces.
- **Config editor + sweep builder.** Read-only anchors in `motors/`,
  editable user configs in `user_configs/`. The editor renders TOML
  sections with typed inputs and passes unknown sections through
  untouched. Grid sweep builder over SweepAxis fields.
- **Local server.** `phasesweep-server`: FastAPI + uvicorn with a job
  manager over the existing worker pool. REST API for jobs, results,
  configs, and measured import. WebSocket for live progress. Cache
  check on submit serves already-computed results from the store.
- **Dashboard skeleton.** Vanilla-JS panel system, job queue panel,
  B_r waveform with multi-config overlay, config selector. Plotly
  vendored (no CDN). WebSocket reconnect with exponential backoff.
- **Committed calibration datasets.** 14 mm outrunner back-EMF ships as
  an import-format dataset. Datasheet EMF-constant datasets for ETEL,
  Rexroth, and Tecnotion motors, each tagged `derived_params` so the
  circularity guard refuses echo fits.
- **Calibration framework.** `phasesweep-calibrate` CLI: bounded least
  squares over crossval residuals with identifiability guards (circular
  fits refused, under-determined fits refused). Output: calibrated TOML
  plus a `CalibrationRecord` JSON sidecar. Acceptance test recovers
  effective B_rem 1.139 T from the 14 mm outrunner back-EMF data.
- **mypy static type checking** passes clean and runs in CI.
- **Kollmorgen AKM44H multi-voltage envelope anchor** - four DC bus
  voltages (160-640 Vdc). First anchor generalizing the voltage-set
  rated-speed class beyond ETEL. All corners voltage-feasible; model
  margin +18..+46% above rated speed.
- **ABB BSM50N-233 two-voltage envelope anchor** - rated speed exactly
  proportional to supply voltage. Model margin +25% at both buses.
- **Tecnotion QTR two-voltage loaded-corner anchor** - the only anchor
  whose published corners carry load. Model margin +9-11% above
  published speeds. Tightest non-ETEL agreement.
- **Beckhoff AM8051-E four-voltage envelope anchor** - four mains
  voltages with a √2·U_N rectifier assumption. Four-vendor envelope
  class (n=5) now spans 48-679 V, 4-22 poles; all published corners
  voltage-feasible.
- **UW/ORNL 6-kW FSCW SPM measured-envelope anchor** - first measured
  envelope anchor, exercising the field-weakening branch. FW current
  placement agrees to <1% vs measured. CPSR = 6 reproduced.
- **Rexroth MS2N04 precision-corner anchor** - the manual explicitly
  states the voltage-set corner mechanism. Tightest catalog-vendor
  utilization in the registry (0.96).
- **FEM current-sheet phase (`sheet_phase`).** The armature sheet
  generalizes to an arbitrary electrical angle β. β = 0 is the existing
  q-axis sheet (no output change); β = ±π/2 gives a pure d-axis MMF.
  Composes with `rotation` for arbitrary rotor-aligned loading. First
  step of the demag screen.
- **`demag_screen` model (v1).** FEM demagnetization screen: magnet
  field plus a demagnetizing d-axis current sheet at `RunConfig.i_fault`.
  Reports `B_m_min`, `margin`, and `frac_below_knee` against the knee at
  magnet temperature. New `[materials]` inputs `B_knee` and
  `alpha_B_knee`. CREATOR clears the drive limit at +0.24 T margin
  (bound anchor; no measured demag data exists).
- **Magnet operating-point sampler** (`sample_magnet_Bm`,
  `demag_margin`). Samples the magnet operating point on a cell-centered
  polar grid inside the magnet arcs. At α_p = 1 the minimum pins to
  pole-boundary corners (over-conservative; documented).
- **`thermal_duty` iron-loss coupling.** When `[iron]` fields are set,
  lumped Bertotti p_fe joins the duty consumption and the S1 budget on
  both paths. Motors without `[iron]` are unchanged.
- **`thermal_duty` first-order thermal transient.** Optional
  `Motor.thermal_time_constant` enables a steady-periodic first-order
  winding-rise march. A profile can now fail `within_s1_transient`
  while passing the cycle-average `within_s1`. Motors without the
  field are unchanged.
- **FEM Maxwell-stress torque.** Armature runs (`j_s ≠ 0`) report
  `tau_maxwell` from the gap-circle integral. Additive output.
- **Cogging torque sweep + CREATOR waveform anchor.**
  `solve_field_fem(..., rotation=φ)` rotates the magnet pattern. CREATOR
  peak-to-peak 72.3 mN-m vs measured 73.4 (-1.5%), dominant order 12
  in both.
- **j_s / phase-current mapping + FEM k_T cross-check.**
  `j_s_from_phase_current` / `phase_current_from_j_s` set the FEM sheet
  amplitude from winding turns and slot geometry.
  `maxwell_interaction_torque_order` extracts the order-n_p synchronous
  mean torque. CREATOR FEM mean torque at rated current lands within
  ~2% of the published 0.10 N-m.
- **Envelope checks outside ETEL.** Kollmorgen and CREATOR catalog
  points verified voltage-feasible. Neither catalog rates max speed at
  the voltage corner, so the ETEL `nm` agreement class does not
  generalize.
- **`torque_speed` model.** Torque-speed envelope from the steady-state
  dq circuit under current and voltage limits. Exact MTPA below base
  speed, field-weakening/MTPV above via deterministic search. Both
  saliency signs, R_s included. Reports envelope curve, base speed,
  max speed, and peak power.
- **ETEL TMB family anchors n=2 → n=5.** Three additional frames join
  the fleet. Thermal-resistance budget validated across 22-88 poles
  (hot copper loss vs published Pc within 2%). First torque-speed
  envelope anchor class: datasheet nm reproduced +2.0..2.4%.
- **`iron_loss` model.** Lumped Bertotti two-term iron loss at the
  reference-speed electrical frequency. In-repo M250-35A fit: ~7%
  median over 50-1000 Hz. CREATOR no-load curve confirms the model
  form and pins the machine's eddy share (~11% at 233 Hz).
- **Kollmorgen GOLDLINE two-ambient family (n=25).** All B-series
  columns transcribed. The family is rated to one winding design
  temperature through the thermal-resistance relation the model uses.
  A transcription guard caught a B-402-A datasheet misprint.
- **`thermal_duty` model.** Copper-loss S1 budget over a torque-time
  profile. R_s derated to the winding operating temperature; without
  derating, cold R_s understates hot S1 loss by ~50% on the
  thermal-resistance path. Pure circuit model, no geometry required.
- **Geometry-optional `Motor`.** `Motor.geometry` is now
  `Geometry | None`, so circuit-only motors load without placeholder
  radii. Field solvers raise if geometry is missing.
- **Optional thermal fields on `Motor`.** `winding_temp_limit`,
  `ambient_temp`, `r_th`, and `insulation_class` from `[thermal]` TOML.
- **Magnet temperature derating.** Optional `alpha_Br` and `magnet_temp`
  derate psi_f in `prepare_thermal_duty`, raising the MTPA current per
  segment. `magnet_temp` without `alpha_Br` raises since there is no
  safe default.
- `Geometry.back_iron_thickness` (outrunner only) splits the rotor yoke
  into a magnetic back-iron ring and a non-magnetic shell.
- `back_iron_thickness` geometry sweep axis and scoping study script.
- **`k_T_effective` output** on `rated_torque` and `stall_torque`:
  effective torque per ampere at MTPA. The magnet-only `k_T` understates
  torque per ampere for salient machines.
- **ETEL TMB reverse-salient anchors.** First real machines exercising
  the γ < 0 MTPA branch. Catalog Kt reproduced within ~1%. First
  external validation of the thermal-resistance budget path: Pc within
  2%, continuous rating within 3%.
- **Du 2021 CLF-RSPMSM literature anchor.** FEM-only reverse-salient
  design study grounding the sign of the γ < 0 branch. Torque
  reproduced within 1.4%.
- `sh_upper_pct` FEM metric: upper slot-harmonic sideband (additive).
- `phasesweep-crossval --strict`: exit 1 unless every comparison passes.
- `execute_generic`/`execute_parallel` accept `timeout_s`.
- **ETEL TMB stall-torque bound anchors.** Linear-magnetics stall torque
  bounds the datasheet Tp from above (+41% / +102%, pinned).

### Changed
- **FEM interpole-gap width measured at the inner PM-annulus radius (fem
  v6 → v7).** The gap is a radial wedge narrowest at the inner radius,
  where OCC fails to cut the sliver. Both topologies now use the inner
  radius. Mesh disk-cache prefix bumped `mesh_v5 → mesh_v6`.
- **fem v6 - interpole-gap mesh rule enforced per-face.** Gaps in the
  0.5-1.0 mm band now carry a local maxh = gap/2 with no floor. Mesh
  prefix bumped to `mesh_v4`.
- **Sub-0.5 mm interpole gaps warn on arc collapse.** Previously the FEM
  silently solved α_p = 1.0 while the analytical model kept the
  requested value.
- **Dashboard table upgrades.** Column chooser, min/max cell shading,
  status/model/motor filters on run history, relative timestamps.
- **Dashboard app frame, density pass, sidebar rail, status bar.**
  Fixed topbar/nameplate/status-bar with independently scrolling main
  area and sidebar. Tighter spacing, one-line config rows, collapsible
  sidebar rail.
- **Dashboard config switch replaces plots** instead of overlaying.
  Comparing is explicit via ctrl-click.
- **Analytical odd-harmonic space harmonics (v4).** `B_r_list` is now a
  true multi-harmonic waveform with non-zero `thd_pct` and `peak_Br`.
  The fundamental bin and psi_f/B_rem derivations are unchanged.
- **`build_sim` rejects reverse-salient machines** (motulator 0.7.3's
  MTPA reference is wrong for L_d > L_q). Circuit models handle both
  saliency signs correctly.

### Fixed
- `docs/motor-toml.md` no longer advertises removed `[drive]` defaults.
- numpy scalars no longer serialize to their repr in the result store
  (`bool("False")` was reading back as True).
- **THD and wedge k_end value pins.** `compute_thd` had no value pin
  (scaling by 1.10 passed 185 tests). Wedge `k_end` was bounded
  0.5-1.0, wide enough to accept a 0.75x error. Both now have
  two-sided pins.
- CI now installs `--extra server` (41 API tests were silently
  skipping). Strict-skip gate catches future silent skips.
- Demag screen assertions tightened from one-sided to two-sided value
  pins.
- `test_outrunner_14mm_crossval` no longer swallows a broken script
  import.
- sdist no longer ships the third-party CREATOR dataset (20.5 MB,
  not redistributable).
- **QC audit fixes.** Server: null TOML sections return 400 instead of
  500. Docs: stale FEM defaults and pre-convention back-EMF values
  corrected. Tests: harmonic-series parity pins added. Citations: M_rn
  Eq 6a → 7a; IEEE TIA attribution → Aalto dissertation.
- **Release-audit hardening.** Server: Host/Origin validation, job drain
  on daemon thread, malformed payloads return 400. Dashboard:
  stale-response guards, LRU result cache. Calibration: inverted bounds
  refused, solver errors treated as rejected steps. FEM: overhang
  air-box cap error, injective cache filenames (prefix → `mesh_v5`).
- Analytical model rejects `n_theta = 2·n_p + 1` (produced an all-zero
  waveform).
- Inrunner TOMLs reject `back_iron_thickness` (outrunner-only field).
- Calibration residual rows keyed per dataset (duplicate datasets
  previously overwrote each other).
- Validation matrix dedupes to the latest result per model/source/dataset
  instead of widening on re-runs.
- Unknown sweep-axis fields fail at submission instead of producing a
  misleading "empty grid" error.
- NaN metrics no longer 500 the results API (served as `null`).
- `j_s` with zero-depth slots now raises instead of silently producing
  an open-circuit result.
- **Wedge Picard non-convergence now raises.** The fixed relaxation
  oscillated on saturating end-regions, so previously published
  nonlinear k_end values were from non-converged fields. Adaptive
  relaxation ported from the 2D solver. Converged nonlinear k_end is
  0.9252 vs the previously published 0.951.
- Parallel pool-timeout now correctly reports never-started jobs with
  `elapsed_s = 0`.
- **Reverse-salient MTPA corrected.** `rated_torque`, `stall_torque`,
  and `thermal_duty` previously clamped γ to 0 for L_d > L_q machines.
  The Morimoto + root is the global MTPA optimum for both saliency
  signs. Model versions bumped.

## [0.2.0] - 2026-06-11

### BREAKING - magnetization convention change

Both solvers now model **square-wave radial magnetization** (uniformly
magnetized arcs with per-pole sign alternation), and `B_rem` means
physical remanence. The previous sinusoidal-ring model made `B_rem` an
effective sinusoidal amplitude rather than a datasheet value.

Consequences:

- All fundamental-linked outputs of motors with a derived psi_f shift
  +27% (x4/π). Motors with an explicit `psi_f` are unaffected.
- FEM and analytical now agree on the fundamental to <1% at all pole-arc
  ratios; the previous spread was a convention mismatch.
- Derived B_rem from psi_f is Carter-consistent; the slotted
  psi_f → flux_linkage_peak round-trip is now exact.
- All pre-0.2.0 cached computed results are invalidated (model version
  bumped to 2). Re-import measured JSONs as well.

### Added
- Rated-torque model (MTPA at rated current) and locked-rotor
  stall-torque model.
- Measured-data import and cross-validation (`phasesweep-import`,
  `phasesweep-crossval`).
- Geometry sweeps and perturbation/sensitivity analysis with parallel
  execution.
- FEM geometry fidelity: discrete magnet arcs, trapezoidal slots with
  separate opening width, Carter-factor correction.
- Outrunner topology support in the FEM solver.
- `SimPlan`: drive-simulation timing derived from motor physics.
- Back-EMF validation pipeline against oscilloscope captures.
- Disk-persistent mesh cache for FEM subprocess workers.
- Public API in `phasesweep/__init__.py` with `__version__` and
  `py.typed`.

### Fixed
- FEM Dirichlet outer boundary was silently skipped.
- Carter factor used slot body width instead of slot opening width at
  the bore (over-correcting by ~15%).
- Drive-sim reference speed treated as mechanical, not electrical.
- Drive parameters included in run-ID hashing.
- Analytical flux linkage evaluated at midgap instead of bore, inflating
  derived values by ~1% on small-gap motors. Now evaluates at bore;
  explicit psi_f round-trips exactly.
- Disk mesh cache safe under parallel workers and recovers from corrupt
  entries.
- FEM armature source uses the motor's winding factor instead of a
  hard-coded 0.966 default.
- Validation report no longer discards failed runs before counting them.

### Changed
- `Geometry` stores physical `slot_opening_width`; `slot_opening_ratio`
  is derived. TOML files may specify either. Cached results for slotted
  motors are invalidated.
- Former `stall_torque` renamed to `rated_torque`; the current
  `stall_torque` is a new locked-rotor model.
- Solver inputs validated through typed param factories.
- drive_sim run IDs hash the full `SimPlan`.
- Measured imports carry a `dataset_id` so repeat captures no longer
  collide.

## [0.1.0] - 2026-03-01

Initial release: Motor/Geometry type hierarchy, model registry
(analytical, FEM, drive simulation), computation pipeline with crash-safe
JSONL result store, Zhu & Howe analytical solver, NGSolve 2D
magnetostatic FEM, motulator drive simulation, validation framework.
