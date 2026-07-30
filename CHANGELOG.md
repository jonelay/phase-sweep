# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/).

## [0.3.0] - 2026-07-30

### Added

- **Torque-speed envelope model** — steady-state dq circuit under current
  and voltage limits: exact MTPA below base speed, field-weakening / MTPV
  above via deterministic search on the voltage boundary. Reports envelope
  curve, base speed, max speed, and peak power at both drive and
  continuous current limits.
- **Iron-loss model** — lumped Bertotti two-term iron loss at the
  reference-speed electrical frequency. Per-kg Steinmetz coefficients
  fitted from multi-frequency specific-loss tables; new optional `[iron]`
  TOML section.
- **Thermal-duty model** — iron-loss coupling (copper + iron on both
  budget paths), first-order thermal transient with optional winding time
  constant, `sustainable_duty_fraction`, and peak winding temperature.
- **Calibration framework** — bounded least-squares calibration over
  cross-validation residuals with identifiability guards: circular-fit
  refusal via `derived_params` provenance tags, under-determined and
  zero-sensitivity detection. CLI: `phasesweep-calibrate`. Output:
  calibrated motor TOML + JSON calibration record.
- **FEM demag screen** — magnet field plus demagnetizing d-axis current
  sheet at a specified fault current, reporting minimum magnet field,
  margin against knee, and area fraction below knee.
- **Cogging torque** — FEM rotor-sweep via Maxwell-stress torque over one
  cogging period. Rotation parameter on `RunConfig`; per-angle mesh cache.
- **Maxwell-stress torque** — post-processor on solved FEM fields;
  interaction-torque order extraction for k_T cross-checks.
- **Current-sheet phase** — armature sheet generalized to arbitrary
  electrical angle (d-axis, q-axis, or intermediate).
- **Local server** — FastAPI + uvicorn behind a `server` optional extra.
  Job manager over the existing parallel worker pool; REST API for jobs,
  results, configs, measured imports; WebSocket progress broadcast.
- **Browser dashboard** — vanilla-JS panel system served at `/`:
  B_r waveform with multi-config overlay, harmonic spectrum (client-side
  DFT), cross-section |B| heatmap, sim waveforms, sortable sweep table,
  model comparison, validation summary. Config editor with sweep builder,
  motor dropdown, model selector. Dark theme, keyboard shortcuts,
  drag-to-reorder panels, composite PNG export, A/B measurement cursors.
  Multiple color palettes (Okabe-Ito default). Tabbed layout with
  persistent panel state.
- **Result version stamp** — every stored result carries a
  `model_version` stamp; reads filter against the live registry version.
  RunResult schema v2.1.
- **Validation anchors** — multi-vendor motor database with datasheet
  back-EMF, rated-torque, and torque-speed envelope cross-validation
  covering a range of topologies, pole counts, and voltage classes.
- **mypy** — static type checking passes clean and runs in CI.

### Changed

- `populate_cache.py` gains `--verify` (recompute-and-compare) and
  `--require-model` for physics-change gating.

## [0.2.0] - 2026-06-11

### BREAKING — magnetization convention change

`B_rem` now means **physical remanence**, and both solvers model
**square-wave radial magnetization** (uniformly magnetized arcs with
per-pole sign alternation). The analytical source coefficient gains the
square-wave Fourier factor 4/π; the FEM source is uniform |M_r| per pole
instead of cosine-modulated. All fundamental-linked outputs of motors
with a derived `psi_f` shift +27% (×4/π). FEM and analytical now agree
on the fundamental to <1% at all pole-arc ratios. Every computed model
carries a `version` (now 2) hashed into run IDs — all pre-0.2.0 cached
results are invalidated.

### Added

- Rated-torque model (MTPA at rated current) and locked-rotor
  stall-torque model
- Measured-data import (`phasesweep-import`) and cross-validation CLI
  (`phasesweep-crossval`)
- Geometry sweeps and perturbation/sensitivity analysis with parallel
  execution
- FEM geometry fidelity: discrete magnet arcs, trapezoidal slots with
  separate opening width, Carter-factor correction
- Outrunner topology support in the FEM solver
- Drive-simulation timing derived from motor physics (`SimPlan`)
- Back-EMF validation pipeline against oscilloscope captures
- Disk-persistent mesh cache for FEM subprocess workers
- Curated public API with `__version__`; `py.typed` marker

### Fixed

- FEM Dirichlet outer boundary now actually applied
- Carter factor uses the slot opening width at the bore, not the slot
  body width (was over-correcting by ~15%)
- Drive-sim reference speed treated as mechanical, not electrical
- Analytical flux linkage evaluated at the bore instead of midgap;
  explicit `psi_f` now round-trips exactly
- Mesh cache safe under parallel workers; recovers from corrupt entries
- FEM armature source uses the motor's winding factor instead of a
  hard-coded default

### Changed

- `Geometry` stores physical `slot_opening_width` (meters);
  `slot_opening_ratio` is a derived property
- Solver inputs validated through typed param factories
- Run-ID hashing includes drive parameters and full `SimPlan`
- Measured imports carry a `dataset_id` in the run-ID hash

## [0.1.0] - 2026-03-01

Initial public release: Motor/Geometry type hierarchy, model registry
(analytical, FEM, drive simulation), computation pipeline with crash-safe
JSONL result store, Zhu & Howe analytical solver, NGSolve 2D magnetostatic
FEM, motulator drive simulation, validation framework.
