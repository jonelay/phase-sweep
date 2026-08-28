# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/).

Performance figures (timings, × speedups) are one-off measurements on the
stated development host at release time; they are not test-pinned and will
vary. Accuracy figures marked "measured, not pinned" were likewise observed
at release; the tests gate looser committed thresholds, not these values.

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
