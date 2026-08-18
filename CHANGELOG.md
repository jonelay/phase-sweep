# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/).

## [0.4.0] - 2026-08-18

### Changed
- **Extras split — heavy dependencies are now optional.** Core install
 (`pip install phasesweep`) requires only numpy + scipy. FEM (ngsolve),
 drive simulation (motulator), plotting (matplotlib), and the server
 (fastapi/uvicorn) are now optional extras: `[fem]`, `[sim]`, `[viz]`,
 `[server]`. Install everything with `pip install "phasesweep[all]"`.
 Missing extras raise `ImportError` with an actionable install hint.
 **Migration:** existing installs that relied on all deps being in core
 should switch to `pip install "phasesweep[all]"`.
- CI now runs two lanes: full (`[all]`) and core-only, with strict-skip
 gating to catch silent coverage erosion.

### Fixed
- `RunConfig.from_dict` no longer crashes when loading results in a
 core-only install (SimPlan import deferred into the sim_plan branch).
- `test_analytical.py` vestigial `importorskip("ngsolve")` removed —
 was silently skipping ~385 pure-analytical tests in no-ngsolve environments.

### Added
- **Toyota Prius 2004 IPM benchmark — reference test case for IPM torque
 and field-weakening models.** Circuit-only motor (50 kW, 8-pole,
 V-shape NdFeB) with ORNL dynamometer validation: stall torque bound
 (340 Nm at 250 A, TM-2004/185 rev. 2007 p. 7), torque-speed envelope
 anchors (base speed, CPSR, power at 6000 rpm). Includes Pyleecan
 machine-readable geometry (Apache-2.0, parked for future IPM ingest).
 11 tests. README updated: Prius leads V&V section and quick-start.
- **`cogging_torque` model — rotor-sweep cogging torque via Arkkio's
 radial-averaging method.** New registry key (v1): sweeps
 `solve_field_fem` over one cogging period (2pi/lcm(n_slots, 2*n_p))
 at `j_s=0`, computing the Maxwell-stress torque averaged over
 multiple radii in the gap annulus. `RunConfig.rotation` (mechanical
 angle, rad) threads through to FEM with an identity-preserving
 default (0.0 — no existing run IDs move, no version bump).
 `RunConfig.cogging_points` sets angular resolution (default 12
 per period). Outputs: `rotation_list`, `tau_cogging_list` (per-m),
 `tau_cogging_pp`, `dominant_order` (FFT), `n_cogging_periods`.
 `cogging_angles()` helper in harmonics.py. Subprocess runner
 `run_cogging_safe` with timeout scaled by angle count.
- **`drive_sim_two_mass` model — two-mass mechanical system for drive
 simulation.** New registry key (v1) using motulator's
 `TwoMassMechanicalSystem`: motor + load inertia coupled through a
 torsional shaft (stiffness K_S, damping C_S). `TwoMassLoad` dataclass
 on `RunConfig.load_mech` (conditional hashing — no existing run IDs
 move). Controllers see J_M + J_L with speed-loop bandwidth clamped
 below the antiresonance. Additive outputs: `w_L_list`, `tau_S_list`,
 `tau_S_peak`. Anchor: Saarakkala & Hinkkanen 2015 closed-form
 resonance/antiresonance (84.2 / 59.6 Hz).
- **CREATOR cogging anchor migrated to model API** — the measured-cogging
 validation tests (`test_cogging_equilibria_at_aligned_positions`,
 `test_cogging_waveform_vs_measured`) now run through `_run_cogging_impl`
 / Arkkio torque instead of the raw `maxwell_stress_torque` recipe.
 Module-scoped `creator_cogging` fixture shares a single 12-angle sweep
 across both tests. Old `_solve_cogging_torque` helper removed.
- **Cogging offset-shift noise diagnostic** — `test_cogging_offset_shift_stability`
 re-runs the cogging sweep at a half-sample grid shift and checks
 dominant-order amplitude stability (<5%), quantifying the per-angle
 remeshing noise floor.
- **Server timeout scaling for cogging** — `JobManager._run_job` now
 multiplies the subtask timeout by `cogging_points` for `cogging_torque`
 subtasks, preventing production-mesh cogging sweeps from timing out.
- **Rotation R2 Nitsche spike** — `scripts/nitsche_rotation_spike.py`
 evaluates mesh-deformation rotation vs per-angle remeshing for the R2
 decision gate.
- **JMT 1806 2400KV dataset-integrity tests** — the 12N14P outrunner
 back-EMF sweep (third measured dataset) is now pinned by the
 suite: Ke/ψf/Kt/KV convention self-consistency, catalog-KV cross-check
 (0.5%), per-point Ke flatness, and slip-recovery bookkeeping. Model
 cross-validation still needs caliper geometry.

## [0.3.0] - 2026-07-30

### Added
- **CREATOR's `B_core` re-derived through the calibration framework.**
 The value shipped in `motors/creator_case_pmsm.toml`
 was back-solved by hand with nothing but a comment behind it. The
 no-load loss curve is now a registered dataset
 (`data/creator_case_pmsm/iron_loss_noload.json`, new `iron_loss_test`
 measured type), `scripts/calibrate_creator_b_core.py` fits it from the
 unfitted FEM-flux estimate, and the record sits next to the dataset.
 The fit recovers the hand value to **0.014%** — it targets the measured
 66.667 Hz grid point where the hand calculation interpolated the curve
 to the model's rounded `W_REF`. The framework's own
 "exactly determined — no redundancy" warning now carries the caveat
 that used to live only in prose. `B_core` joins `PerturbParam` and
 `FITTABLE_PARAMS` as an iron-tier scalar.
 The dataset is deliberately **untagged** for the circularity guard,
 the one exception in the fleet: the shipped `B_core` did come from it,
 so the honest `derived_params` tag would refuse the very fit that
 produces the record. The echo it would have caught is named in the
 dataset, and pinned as a test instead. CREATOR still
 does not validate `iron_loss`.

- **`docs/architecture.md`** — data-flow spine, three-tier hashing,
 model-coverage matrix, and claim tables (`docs/diagrams/README.md`).
- **`[iron]` section on `motors/creator_case_pmsm.toml`** — the first
 `iron_loss` coverage in `motors/`. `k_h`/`k_e`/`alpha_fe` are the
 in-repo M250-35A global fit; `m_core` (1.2516 kg) is geometric, from
 the paper's Table 12 dimensions and the dataset's steel density.
 `B_core` is **calibrated,
 not predicted**: the dataset gives no core flux density, and it is
 back-solved so `p_fe` matches the measured no-load loss at the rated
 point — exact there, 3.5% median over the 16-point curve, +23.6% at
 233 Hz. CREATOR therefore cannot validate `iron_loss`; it is fit to
 the same curve any agreement test would check against. The unfitted
 alternative is recorded alongside it: FEM per-pole flux (188.1 µWb)
 gives an effective 0.273 T and predicts 0.183 W, so the single-B lump
 runs **2.7× light** against measurement — the build-factor,
 rotational-loss and tooth-harmonic gap that sits outside the model's
 documented scope. Both numbers are pinned in `tests/models/test_iron_loss.py`.
- **`[thermal]`, `[iron]`, `back_iron_thickness`, and the magnet/demag
 `[materials]` fields documented in `docs/motor-toml.md`** — closing
 the missing-sections item from the 2026-07-15 wide code audit. All of
 them are parsed by `load_motor` but were absent from the format doc,
 whose "other sections are ignored" note therefore read as a claim
 that they are inert.
- **`iron_loss` added to `populate_cache.py`'s fast-model set** — it is
 a closed form over already-validated params, so populating it also
 makes it verifiable (the `--verify` set is derived from the same
 list), and the store gains a record instead of a permanent gap. The
 test that used `iron_loss` as its example of a model `--verify`
 cannot check now names `demag_screen`, and asserts on the refusal
 message rather than the exit code — the old assertion would have kept
 passing via the zero-coverage path.
- **`populate_cache.py --verify` — recompute-and-compare for the result
 store.** The model-version stamp catches superseded physics, but not a
 record whose metrics disagree with the version they claim. `--verify`
 recomputes every current-version computed record and compares
 (`--verify-fem` includes FEM; `--rtol` sets the tolerance;
 `--prune-mismatched` drops the failures so a later run recomputes).
 `--verify` detects records whose metrics do not match their stamped
 model version.
- **`--require-model` makes `--verify` a physics-change gate.** Run before
 a version bump, `--verify` answers the question the checklist asks by
 hand — did this edit move outputs? — across every motor in the store.
 But it only sees records that exist, so a store with no record for the
 edited model passed vacuously. `--require-model MODEL` (repeatable or
 comma-separated) exits 1 on zero coverage; `--require-model fem`
 implies `--verify-fem`; a model outside verify's set is refused rather
 than reported as covered. The verify summary now also lists records
 checked per model.


- **Click a completed job card to re-plot its output** — clicking a completed Jobs card switches the active
 config to the job's motor and replays that job's specific results to
 the panels (waveforms + sweep table), routing directly so it re-plots
 even after the overlay was cleared. The nameplate, sidebar selection,
 and submit form all follow the job's motor.

- **Dashboard color palettes** — two topbar
 dropdowns, a second persisted axis orthogonal to light/dark. The
 **series palette** re-hues every compare-set slot at once: **Okabe-Ito**
 (new default, colorblind-safe), **Tol vibrant**, **Tol bright**,
 **Seaborn colorblind**, and **Tableau 10** — each slot-ordered
 blue/teal-first with its early slots contrast-checked on both themes.
 The **|B| ramp** adds the perceptual maps **Viridis**, **Cividis**, and
 **Magma** alongside the accent ramp; all are theme-aware — the
 perceptual ramps lift their floor on the dark panel so low field stays
 legible on the near-black surface.
 `SERIES` is mutated in place so a switch re-hues chip/rail dots and
 panel traces together while the slot→config binding holds; overlay
 panels re-read their slot at draw time. The light/dark control moves
 next to the palette selectors and now shows a text label (Light/Dark)
 rather than a sun/moon glyph. No new deps — palettes are static lists
 in `panels/plot_theme.js`.

- **Dashboard measurement cursors, panel reorder, composite PNG export** — A/B cursors on
 the same-axis result panels (B_r waveform, sim waveforms): a header
 **Cursors** toggle drops two draggable vertical markers and a stats
 readout — A, B, Δ, and per-series min/max/peak-to-peak/mean/σ over
 the window; per-panel (heterogeneous x-axes rule out a global
 cursor), transient, source-labeled. Panels **drag-to-reorder**
 within a tab (header handle; order persists per browser) — the grid
 areas recompute from the spliced order, no hand-authored strings
 move. Topbar **⤓ PNG** renders every visible plot on the active tab
 to one canvas with a title/timestamp header and each panel's
 provenance stamp burned in — a report figure attributable to its
 physics version. Drops the "no export to PDF/PNG" limitation.

- **Dashboard keyboard layer + linked highlight** — global shortcuts: `1`–`4` switch tab,
 `j`/`k` step through configs, `?` opens a shortcut overlay, Esc
 closes it (else restores a maximized panel). Bare keys no-op while
 typing in any input/textarea/select/contenteditable — the config
 editor is a wall of text inputs. The tab bar becomes a real ARIA
 tablist: roving tabindex, `aria-controls`/`aria-labelledby` pairing,
 arrow keys move selection. New `app.highlightResult(id, origin)`
 bus: hovering a result in one panel highlights it in the others — the
 waveform fattens the matching trace, the spectrum dims non-matching
 bars, sweep table and run history tint the matching row.

- **Result version stamp + staleness filter** — every stored result now carries a
 `model_version` stamp written unconditionally at save (v1 included,
 so an unstamped record can't pass as a v1 one); RunResult schema
 bumps to **v2.1**. Both server read paths (`load_slim`, which the
 job cache check rides on, and `GET /api/results/{id}`) serve a
 record only when its stamp matches the live registry — a missing
 stamp filters exactly like a mismatch. Previously a physics version
 bump did not invalidate already-stored results: run IDs are
 recomputed at the current registry version, so superseded-physics
 records re-keyed to fresh ids and served as cache hits. Existing
 stores' run histories appear empty and recompute on demand —
 intended behavior under the version discipline.

- **Dashboard nameplate, stat tiles, provenance stamps** — new nameplate strip
 under the topbar names the active config with `pole pairs ·
 topology · peak B̂_r · fundamental · THD` tiles fed by the routed
 full results (prefers fem, labels the source model); it subsumes
 the topbar active-config span. Sim-waveform caption metrics become
 a compact KPI tile row (shared `statStrip()` widget). Result panels
 gain a muted provenance line — `model vN · timestamp ·
 result_id[:8]` — so report figures stay attributable to the physics
 version that produced them; overlay panels stamp the latest routed
 result with the trace count.

- **Tabbed dashboard layout** — the main
 area's single auto-fit grid of 8 panels is replaced by four tabs:
 Results (B_r waveform, harmonic spectrum, cross-section, sim
 waveforms) | Sweep | Validation (model comparison + summary, so the
 matrix-cell jump stays on-screen) | Editor. A panel's tab is its
 registry `slot`; the active tab persists per browser. Plots built
 while their tab is hidden re-fit on reveal. With its own tab, the
 Config Editor no longer needs a collapsed-by-default start.

- **Dark theme** — topbar
 toggle, persisted per browser and applied before the stylesheet
 loads (no flash). Both palettes are CSS custom properties; plotly
 chart styling reads `--chart-*` roles at layout-build time and live
 plots re-render on switch via a new optional panel `retheme()` hook.
 Trace series colors are theme-invariant (never re-hued); the
 sequential |B| heatmap ramp gets a dark variant that runs
 dark→bright.

- **Editor ergonomics** — config-editor geometry lengths (and sweep-builder
 start/stop) now display and accept **mm**, matching the plots, while
 the saved TOML stays in meters; only user-edited fields are written
 back, so display rounding can never drift an untouched value through
 a load/save round trip. Field labels get tooltips. Panel collapse
 state persists per browser and panels can declare a collapsed
 first-visit default. Polish: terminal jobs drop their
 progress bar, button casing unified to sentence case, muted text
 (`--text-muted` and the plotly axis gray) darkened to meet WCAG AA
 4.5:1.

- **Sweep value + run history** — the sweep comparison panel now leads with a
 metric-vs-axis plot over its accumulated rows (x = SweepAxis
 geometry field, auto-picked as the column with the most distinct
 values; y = any table metric; one series per motor·model) and gains
 CSV export (geometry in mm, metrics unrounded, current sort order),
 a fixed-width sort-arrow slot (no more column shift on sort), and a
 right-edge fade when columns are clipped by horizontal scroll. New
 sidebar run-history panel lists the most recent stored results
 (newest first, with timestamps) so a page reload no longer loses
 track of what the persistent store already holds; clicking a row
 routes the result to the main panels. `GET /api/results` slim rows
 now carry `motor_name` and `timestamp`.

- **Dashboard calibration workflow** — the manual calibration loop
 (import measured → see disagreement → edit params → re-run → save
 calibrated TOML) now works end-to-end in the dashboard. Measured
 and published results attach by Motor NAME instead of config_id, so
 datasets follow the motor through parameter edits while computed
 results stay pinned to their exact parameter set (`?motor=` results
 filter and `/api/validation/{motor}`). The validation payload
 carries per-dataset `derived_params` tags and the Validation
 Summary panel renders an echo warning: agreement with the dataset a
 param was derived from is not independent validation. The jobs-form
 motor dropdown now picks up configs saved from the editor.

- **Comparison workflows** — model comparison panel (crossval
 quantities across every model run on the active config, measured
 data as markers with error bars from the dataset's `_uncertainty`,
 delta-vs-reference in the legend), validation summary panel
 (model-pair × quantity agreement matrix colored against the
 tolerance tiers, diagnosis line, cell click jumps to the comparison
 panel), and ctrl/cmd-click multi-config overlay in the sidebar
 config list. New endpoint `GET /api/validation/{motor}` serves the
 validate-job crossval summary on demand.

- **Result panels** —
 four dashboard panels: harmonic spectrum (client-side DFT of
 `B_r_list`, electrical-order x-axis so pole counts align, overlay
 mode), cross-section |B| heatmap (plotly heatmap with material
 boundary circles; the spec's Three.js option was dropped and the
 vendored plotly bundle swapped basic→cartesian, ~1.4 MB), sim
 waveforms (torque / speed / |i_s| stacked subplots with sim-plan
 step markers and the w_ref line), and a sortable sweep comparison
 table (SweepAxis geometry columns + scalar metrics; clicking a row
 routes the full result to the other panels; on job completion the
 full result-id list is fetched past the capped live-routing path).
 Feeding them, two **additive** model outputs (no version bumps —
 existing values unchanged; cached results merely lack the new keys
 and the panels show a re-run hint): `fem` gains `B_mag_grid` /
 `grid_coords_list` (|B| rasterised on a 120×120 Cartesian grid,
 ~120 KB per result) and `drive_sim` gains downsampled
 `t_list` / `w_M_list` / `tau_M_list` / `i_s_abs_list` traces.

- **Config editor + sweep builder** — the dashboard gains a config-editor panel over a new
 server write path. Configs now come in two tiers: the read-only
 anchor fleet in `motors/` and an editable user tier in
 `user_configs/` (`--user-configs-dir`), loaded together at startup
 (anchor stems can never be shadowed). New endpoints:
 `GET /api/configs/{name}/raw` returns the parsed TOML structure for
 lossless round-tripping, `PUT /api/configs/{name}` validates the
 body by building the Motor (`configs.motor_from_dict`, extracted
 from `load_motor`), writes TOML via `tomli-w` (added to the
 `server` extra), and hot-loads the Motor so jobs see it
 immediately; a PUT against an anchor name returns 409.
 The editor renders the TOML sections with typed, unit-labeled
 inputs, keeps unknown fields editable, passes unknown sections
 (e.g. `[validation]`) through untouched, and saves anchors under a
 new name. Model selector with registry cost badges (one checked
 model submits that job; several submit one `validate` job) and a
 grid sweep builder over the SweepAxis fields. Also fixed: fully-cached jobs complete inside the POST response
 before the client can subscribe to the WebSocket — submit now
 routes the result IDs from the response itself.
- **Local server** —
 `phasesweep-server` (or `python -m phasesweep.server`): FastAPI +
 uvicorn behind a new `server` optional extra (fastapi, uvicorn,
 structlog). Job manager wraps the existing
 `parallel.execute_parallel` worker pool (serial FIFO, one pool at a
 time): single-point jobs dispatch by any computed registry key,
 `sweep` fans out over `geo_sweep` grids, `validate` runs the
 runnable model set and returns the crossval diagnosis. Cache check
 on submit serves already-computed run IDs from the ResultStore
 (full hits complete in the POST response; partial hits run only the
 missing sub-tasks). REST: jobs (submit/list/detail/cancel), results
 (query by motor/model/source + full-metrics detail), configs
 (read-only, keyed by TOML filename stem), measured import (same
 code route as `phasesweep-import`). WebSocket `/ws` broadcasts
 job_progress/job_complete/job_failed/job_cancelled to subscribed
 clients. structlog lifecycle logging (console renderer for dev,
 `--log-json` for production). `parallel.execute_generic` gains a
 cooperative `cancel` event: pending sub-tasks return CANCELLED,
 running ones complete.
- **Dashboard skeleton** —
 the server now serves a browser dashboard at `/`: vanilla-JS panel
 system (one file in `panels/` + one registry line per visualization),
 job queue panel (submit/cancel, status badges, per-job progress
 bars), B_r waveform panel with multi-config overlay (fixed-slot
 series colors, up to 8 traces), config selector that loads the
 latest stored waveform per model. WebSocket reconnect with
 exponential backoff (1 s doubling to 30 s cap) and job-list
 reconcile across server restarts. Plotly vendored as a pinned
 minified build (`plotly-basic-3.6.0.min.js`) — no CDN, works
 offline. New `GET /api/models` returns the submittable job types
 from the model registry.
- **Committed calibration datasets + fleet `derived_params` tags** — the 14 mm outrunner back-EMF speed sweep now ships as an
 import-format dataset (`data/outrunner_14mm_steel/backemf_measured.json`),
 so the flagship calibration (`phasesweep-calibrate … --params B_rem`)
 runs off the repo instead of hand-built JSON; the calibration test
 fixtures and CLI tests exercise the committed file. Datasheet
 EMF-constant datasets added for the fleet motors whose TOML `psi_f`
 was derived from those same constants — ETEL TMB ×5 (Ku), Rexroth
 MS2N04 (KE), Tecnotion QTR (Ke) — each tagged
 `derived_params = ["psi_f"]`, so the circularity guard refuses the
 psi_f/B_rem echo fit fleet-wide (verified via CLI: exit 1,
 "circular fit refused"). A sweep test validates every committed
 import-format dataset in `data/`.
- **Calibration framework** —
 `phasesweep.calibration.calibrate()` and the `phasesweep-calibrate`
 CLI: bounded least squares (`scipy.optimize.least_squares`, scipy now
 an explicit dependency) over cross-validation residuals weighted by
 per-dataset tolerance, parameter changes routed through
 `perturb_motor` (which gains the circuit set: `psi_f`, `R_s`, `L_d`,
 `L_q` — usable on geometry-less datasheet motors). Identifiability
 guards refuse circular fits (new dataset-side `derived_params`
 provenance tag, psi_f/B_rem as one equivalence class),
 under-determined and zero-sensitivity fits; near-collinear
 sensitivities and at-bound params are reported, exactly-determined
 inversions warned. Output: calibrated Motor written to a NEW TOML
 (new exact-round-trip `write_motor_toml` serializer; overwriting
 refused) plus a `CalibrationRecord` JSON sidecar (config_ids, fitted
 params with indicative uncertainty, residuals before/after, model
 versions, optimizer metadata). Acceptance tests pin the manual
 calibration (effective B_rem 1.139 T recovered automatically from the
 14 mm outrunner back-EMF data) and the negative control: no single-param fit
 within published-parameter bounds closes the 0.92 MTPA/nameplate band.
 Class-level correction factors remain a deliberate non-goal.
- **mypy static type checking** — non-strict `mypy
 phasesweep/` passes clean and runs in CI alongside ruff. Optional-field
 validation in the solver-param factories now narrows types explicitly;
 `winding_transfer` raises a proper `ValueError` (instead of `TypeError`)
 when geometry, `k_w`, or `L_stk` is missing. No output changes.
- **Kollmorgen AKM44H multi-voltage envelope anchor**
 (`data/kollmorgen_akm44h/`) — first anchor with the same winding's
 rated point published at four DC bus voltages (160/320/560/640 Vdc →
 1000/2500/4500/5500 rpm): the progression scales with the bus and the
 model's no-load limit at each lower bus falls short of the next bus's
 rated speed, so the rated speeds are voltage-set by construction —
 the generalization evidence the ETEL-only corner class lacked. All
 four rated points voltage-feasible; model corner +18..+46% above
 `Nrtd` (pinned per corner, margin shrinking with bus voltage —
 Kollmorgen sizes with drive/system headroom, utilization 0.71–0.85).
 ψ_f from `Ke` (line-line rms, 25 °C); the −5.7% `Kt` gap is the
 catalog's documented hot/cold split (`Km` cross-check +0.07%).
- **ABB BSM50N-233 two-voltage envelope anchor**
 (`data/abb_bsm50n/`) — rated speed printed at two voltages with
 7500/4000 = 300/160 exactly: rated speed proportional to supply, the
 sharpest voltage-set evidence in the registry. Model no-load limit an
 identical +25.0% above rated speed at both buses (zero spread); still
 +12/+17% above even at the full hot stall current (pinned;
 published-point utilization 0.86–0.90). Cleanest conventions surveyed
 (all parameters line-to-line WYE, `Ke` in peak and rms, cold/hot split
 enumerated with the 0.90 N-series hot coefficient printed — which
 closes the stall row to −3%).
- **Tecnotion QTR-A-105-25-N two-voltage loaded-corner anchor**
 (`data/tecnotion_qtr/`) — the only anchor whose two
 published corners carry load (max speed at continuous torque, 240 rpm
 @ 48 Vdc / 3625 rpm @ 325 Vdc; the two-voltage analog of ETEL's nb).
 Printed `Pc` closes with hot R to +1.6% (cold −24%), licensing the
 hot-R reconstruction; model corner a consistent +9.0/+11.0% above the
 published speeds at both buses (pinned — tightest non-ETEL margins;
 cold R would miss the IR-dominated 48 V corner by +52%). ψ_f from
 peak ph-ph `Ke` closes `Kt` to −0.6%; documented saturation flag
 (`Ic = 0.70·Ip` vs L valid `I < 0.6·Ip`).
- **Beckhoff AM8051-E four-voltage envelope anchor**
 (`data/beckhoff_am8051/`) — S1 nominal point tabulated at four mains
 voltages (115/230/400/480 VAC → 500/1400/2500/3000 rpm), reconstructed
 under a documented `U_DC = √2·U_N` rectifier assumption (unique to
 this anchor). Model corner +11..+16% above `Nn` at the upper mains,
 +35% at the 115 V outlier; ceiling discriminator holds at both lower
 mains. Largest documented hot/cold split in the registry: `KT` is hot
 by the manual's definition (−14.4% vs cold-`KE` ψ_f; `M0 = I0·KT`
 closes to −0.4%). The four-vendor envelope class (Kollmorgen, ABB,
 Tecnotion, Beckhoff + ETEL n=5) now spans 48–679 V buses, 4–22 poles:
 published corners are uniformly voltage-feasible with vendor margin;
 ETEL remains the only tight-agreement vendor.
- **UW/ORNL 6-kW FSCW SPM measured-envelope anchor**
 (`data/ornl_fscw/`, ORNL/TM-2005/183 Table 7) — first MEASURED
 envelope anchor and the only one exercising the field-weakening
 branch: four points to 5× base speed with per-point measured phase
 voltage, d/q currents, and shaft torque. Corner MTPA +5.2% (pinned);
 FW envelope one-sided +15..18% above measured shaft torque (lossless
 optimum vs shaft); **FW current placement is an agreement anchor**
 (model `i_d = −ψ/L` = −26.8 Arms vs measured −26.9 at 2000 rpm,
 <1%) — pins L = 1.3 mH over the report's conflicting 1.03 mH
 bookkeeping; CPSR = 6 design goal reproduced (6.9 kW at 4800 rpm).
- **Rexroth MS2N04-D0BHN precision-corner envelope anchor**
 (`data/rexroth_ms2n/`) — the only anchor whose manual states the
 corner mechanism in prose ("Rated speed is determined by the DC bus
 voltage UZK1", R911347583 Ed.07 §6.3) and prints it as precision data
 (non-round 2040 rpm vs the winding code's 2000 ± 250 nominal, ±5%
 tolerance class). No numeric DC bus printed, so the corner is pinned
 under both rectifier conventions: model corner +4.6% above nN at
 1.35·U_LL = 540 V and +9.9% at √2·U_LL; printed-corner utilization
 0.958/0.915, the tightest catalog-vendor corner in the registry.
 ψ_f from `KE` closes the printed cold `Km` to −0.44%; the rated pair
 `MN/IN` is a 100 K hot rating (−12.2% vs cold `Km`).
- **`solve_field_fem` current-sheet phase (`sheet_phase`)** — the
 armature sheet generalizes to `J_z = −j_s·k_w·cos(n_p·θ − β)`
 (β in electrical radians). `β = 0` is the existing q-axis sheet
 (bit-identical, no version bump); `β = ±π/2` is a pure d-axis MMF —
 the order-n_p interaction torque nulls to ~1e-4 of the q-axis value,
 with `+π/2` demagnetizing and `−π/2` magnetizing on both topologies.
 Composes with `rotation`: `sheet_phase = n_p·φ + γ` holds the sheet
 at a fixed electrical angle from the rotated magnet axis. The
 j_s ↔ phase-current mapping is phase-invariant (uniform slot comb).
 First step of the demag screen; mesh cache unaffected (source-term
 change only).
- **`demag_screen` registry model (v1)** — FEM demagnetization
 screen: magnet field plus a pure demagnetizing d-axis current sheet at
 `RunConfig.i_fault` (peak phase current, required — deliberately no
 drive-limit fallback), reporting `B_m_min` / `margin` /
 `frac_below_knee` against the knee at the magnet temperature. New
 optional `[materials]` inputs `B_knee` (T at 20 °C, may be ≤ 0) and
 `alpha_B_knee` (absolute T/K slope); with `magnet_temp` set, both
 `alpha_Br` and `alpha_B_knee` are required so remanence and knee stay
 at one temperature (input-schema rule — no version bumps). CREATOR
 TOML gains the data-backed `B_knee = 0.0` (its shipped ferrite demag
 curve is linear to B = 0), which rotates cached CREATOR run IDs
 (recompute to identical values); the screen clears the drive limit at
 +0.24 T margin — a bound anchor, no measured demag data exists.
- **Magnet operating-point sampler (`sample_magnet_Bm`, `demag_margin`)**
 — samples `B_m = sign(M_r)·B_r` (the magnet operating point in its
 magnetization direction) on a cell-centered polar grid inside the
 magnet arcs, with area weights; `demag_margin` reports the worst
 point, margin against a caller-supplied `B_knee`, and the area
 fraction below it. Step 2 of the demag screen (additive — existing
 outputs unchanged). At α_p = 1 the ideal square-wave pole transition
 pins the minimum to pole-boundary corners (over-conservative there;
 documented); with discrete arcs the minimum tracks the armature
 d-axis field.

- **`thermal_duty` iron-loss coupling** — when all five `[iron]` fields
 are set, the lumped Bertotti `p_fe` at `W_REF` (the same number
 `iron_loss` reports) joins the duty consumption on both budget paths
 and joins the `rated_current` budget (`P_S1 = 1.5·R_s·I_rated² + p_fe`
 — the nameplate S1 point dissipates copper AND iron, so a duty at
 exactly that point still reads `over_budget_ratio = 1`; below it the
 iron-aware ratio correctly reports smaller headroom).
 `sustainable_duty_fraction` treats `p_fe` as spent at every duty
 fraction (speed, not torque, sets it). New outputs `p_fe` /
 `p_total_avg`; partial `[iron]` sets are a loud error. Motors without
 `[iron]` are bit-identical (no version bump — input-schema rule);
 `W_REF` joins the `thermal_duty` hash fields, so existing cached
 thermal_duty run IDs rotate (recompute to identical values).
- **`thermal_duty` first-order thermal transient** — optional
 `Motor.thermal_time_constant` (s, `[thermal]` TOML; the ETEL TMB
 datasheet τth, now carried by all five ETEL anchors) enables an exact
 steady-periodic first-order march of the normalized winding rise over
 the duty cycle (closed-form periodic solution, peak at segment
 endpoints — no integration error). New outputs `transient_peak_ratio`,
 `within_s1_transient`, and `winding_temp_peak` (°C, thermal_resistance
 path) convert the documented "cycle-average criterion assumes a fast
 duty cycle" caveat into a computed check: a profile can now fail
 `within_s1_transient` while passing `within_s1`. τ ≫ cycle recovers
 the average criterion; segments ≫ τ recover the per-segment peak.
 Unset motors are bit-identical (input-schema rule, no version bump).
- **FEM Maxwell-stress torque** — `maxwell_stress_torque` post-processor
 on the solved A_z (`τ = (L_stk·r²/μ0)·∮B_r·B_θ dθ` on a gap circle);
 `fem` armature runs (`j_s ≠ 0`) report additive `tau_maxwell_per_m` /
 `tau_maxwell` (rotor torque, topology sign-corrected). Verified by
 contour independence across the gap, exact affinity in `j_s`, and a
 zero-current noise floor well below the interaction torque. No version
 bump (additive outputs).
- **Cogging torque sweep + CREATOR waveform anchor** —
 `solve_field_fem(..., rotation=φ)` rotates the magnet pattern (arcs +
 per-pole magnetization sign) by a mechanical angle; the stator stays
 fixed. Sweeping one cogging period (2π/lcm(n_slots, 2·n_p)) with
 `j_s = 0` and `maxwell_stress_torque` per position yields the cogging
 waveform — the repo's first waveform-level FEM torque anchor: CREATOR
 peak-to-peak 72.3 mN·m vs measured 73.4 (−1.5%; FEM peak +1.3% on the
 paper's 0.0357 N·m scalar), dominant order 12 in both. Per-order
 amplitudes are not calibrated (square-wave magnetization shifts
 energy from order 24 into 12, ≈1.5×/≈0.5× measured — the back-EMF
 5th-harmonic idealization signature). `rotation = 0`
 is bit-identical to the previous solver and keeps legacy mesh-cache
 filenames (no version bump); each nonzero angle caches its own mesh.
- **j_s ↔ phase-current mapping + FEM k_T cross-check** —
 `solver_params.j_s_from_phase_current` / `phase_current_from_j_s` set
 the FEM sheet amplitude from winding turns and slot-face geometry
 (`j_s = 3·N_eff·Î/(π·S)`, `S = fem_field.slot_source_moment` — the
 slot faces' radial moment per unit angle, opening + body for stepped
 slots; `k_w` cancels). New `fem_field.maxwell_interaction_torque_order`
 extracts the order-n_p Maxwell cross term between magnet-only and
 armature-only solves — the synchronous mean torque, separated from the
 position-locked slot-harmonic ripple the static integral carries
 (comb sidebands `m·n_slots ± n_p` meeting magnet harmonics; ~ −25% of
 the mean on CREATOR's 6s/4p at the solved position). Closure: the
 order-n_p interaction reproduces the winding formula to −5% on the
 test inrunner (one-sided: slot openings Carter-widen the armature
 gap), and the CREATOR FEM mean torque at rated current lands within
 ~2% of the published 0.10 N·m rated point. No version bump (pure
 additions; the solver source term is unchanged).

- **Envelope checks outside ETEL** — Kollmorgen B-104-B rated
 point (7500 rpm, 1.57 N·m at Ic) verified voltage-feasible (corner 24%
 above N Max); CREATOR catalog points all inside the envelope, its
 7050 rpm max speed identified as exactly the 235 Hz inverter frequency
 limit. Negative finding: neither catalog rates max speed at the
 voltage corner — the ETEL `nm` agreement class does not generalize.
 Saturated-L sensitivity quantified: `nb` moves only 0.3–0.6% per 1% of
 L and the effect is d-axis-only.

- **`torque_speed` model** — torque-speed envelope from the
 steady-state dq circuit under current and voltage limits
 (`U_max = U_DC/√3`, SVPWM linear region): exact MTPA below base speed
 (identical to `rated_torque` at the same current), field-weakening /
 MTPV above via a deterministic search on the voltage boundary; both
 saliency signs, `R_s` included exactly. Reports the envelope curve,
 base speed, max speed (`None` when the characteristic current
 `psi_f/L_d` is within the drive limit — unbounded CPSR), and peak
 power, at the drive current limit (`I_LIMIT`/`MAX_I_S`) and, when
 `I_rated` is set, at the continuous rating. New `TorqueSpeedParams` +
 `prepare_torque_speed`; datasheet torque-speed curves become a new
 validation anchor class.
- **ETEL TMB family anchors n=2 → n=5** — TMB+0140-030 RA, TMB+0290-030 RA,
 TMB+0450-030 VA join 0210/0360 (`data/etel_tmb/`, ψf-from-Ku pattern,
 all cross-checked against Isc within the datasheet rounding budget).
 Water-cooled `thermal_resistance`-path validation across 22–88 poles:
 hot copper loss vs published `Pc` within 2% (n=5), `ΔT/r_th` budget a
 consistent 2.5–3.6% above it. Stall bound + pinned overprediction band
 extended (+41% to +222%, tracking datasheet saturation); the 0450 VA is
 the first anchor where `saturation_warning` fires (Ip/Ic = 3.65).
 **First torque-speed envelope anchor class**: datasheet `nm` reproduced
 +2.0..2.4% (tight n=5 band — ETEL's voltage margin), base speed `nb` at
 rated current under-predicted 1–11% one-sided (linear-inductance
 saturation blind spot, pinned per anchor), rated points
 voltage-feasible on all frames.
- **`iron_loss` model** — lumped Bertotti two-term iron loss
 `P = m_core·(k_h·f·B^α + k_e·f²·B²)` at the reference-speed electrical
 frequency; per-kg steel Steinmetz coefficients + core mass + single
 peak core B as new optional Motor fields (`[iron]` TOML section, hashed
 only when set). `fit_bertotti` fits coefficients from multi-frequency
 specific-loss tables; the in-repo M250-35A characterization fits to
 k_h = 0.0153, k_e = 6.35e-5, α = 1.62 (~7% median, 50–1000 Hz,
 quick-reference points within 20%). The CREATOR measured no-load curve
 confirms the A·f + C·f² form (R² > 0.997) and pins a finding: the
 machine's eddy share (~11% at 233 Hz) is well below the catalog-steel
 prediction at any plausible core B — first quantified step against the
 documented ~14% copper-only optimism of the thermal_duty screen.
- **Kollmorgen GOLDLINE two-ambient family (n=25)** — all B-series
 columns from the B-104-B anchor's own datasheet transcribed to
 `data/kollmorgen_goldline/two_ambient_columns.toml` (10x/20x/40x
 frames). The datasheet-only absolute rating-temperature estimate
 (`Ic` + `Rm` + `Rth` + copper derating) lands at 154.6–158.4 °C on all
 25 columns — the family is rated to one winding design temperature
 through exactly the relation the `thermal_resistance` budget path
 models; air-cooled r_th consistency extended from n=1 to n=25.
 A per-column `Kt = √3·Kb` transcription guard caught a B-402-A
 datasheet misprint (corrected with a note).
- **`thermal_duty` model** — copper-loss S1 budget over a torque-time profile
 for IPM/control application work. Maps each torque segment to the minimum
 MTPA current, sums 3-phase copper loss (`1.5·R_s·I_peak²`), and reports
 time-averaged loss, peak loss, over-budget ratio, and sustainable duty
 fraction vs. an S1 continuous budget. The budget comes from a winding
 thermal resistance (`r_th` + temps) when available, else from `I_rated`
 (which already encodes the manufacturer's S1 thermal limit as a current).
 Copper loss is evaluated with `R_s` derated to the winding operating
 temperature (`R_s(T) = R_s(20 C)·[1 + 0.00393·(T − 20)]`, worst-case at
 `winding_temp_limit`) — without it the cold datasheet `R_s` understates the
 hot S1 loss by ~50% on the thermal-resistance budget path (it cancels on
 the rated-current path). Pure circuit model — no geometry required.
 New `ThermalDutyParams` +
 `prepare_thermal_duty`; `RunConfig.duty_profile` carries the profile and is
 hashed into the run-ID.
- **Geometry-optional `Motor`** — `Motor.geometry` is now `Geometry | None`,
 so datasheet/circuit-only motors (rated/stall torque, drive_sim,
 thermal_duty) load without fabricating placeholder radii. The field/FEM
 factories raise if a field solver is requested without geometry. `[geometry]`
 becomes optional in TOML. Geometry-bearing motors keep byte-identical
 `config_id`s (no run-ID regression).
- **Optional thermal fields on `Motor`** — `winding_temp_limit`, `ambient_temp`,
 `r_th` (numeric, hashed), and `insulation_class` (documentation, not hashed);
 loaded from a `[thermal]` TOML section. `None` preserves prior `config_id`s.
- **Magnet temperature derating** — optional `Motor.alpha_Br`
 (grade-specific B_rem/psi_f temperature coefficient, per K; `[materials]`
 in TOML) and `Motor.magnet_temp` (°C; `[thermal]`). When both are set,
 `prepare_thermal_duty` derates
 `psi_f(T) = psi_f(20 C)·[1 + alpha_Br·(T_magnet − 20)]`, raising the MTPA
 current per torque segment — unlike the R_s derating this does not cancel
 on the rated-current budget path, so both duty verdicts get less
 optimistic. `magnet_temp` without `alpha_Br` raises (grade-specific, no
 safe default); `alpha_Br` is validated to (−0.01, 0) to catch the %/°C
 trap. Both fields hash into `config_id`; unset fields preserve prior
 run-IDs and outputs bit-identically, so no model version bump.
- `Geometry.back_iron_thickness` (outrunner only): splits the rotor yoke into
 a magnetic back-iron ring + a non-magnetic (`mu_r=1`) shell, so a thin steel
 ring saturates under nonlinear FEM instead of the legacy "whole wall is iron"
 model. Opt-in; `None` preserves prior behavior and leaves `config_id` — hence
 run-IDs, mesh cache, and the captured reference field — unchanged.
- `back_iron_thickness` geometry sweep axis for studying airgap flux vs
 ring thickness (thin ring saturates under nonlinear FEM).
- **`k_T_effective` output** (`rated_torque`, `stall_torque`) — effective
 torque per ampere at MTPA (`τ / I_s`, peak convention). The reported `k_T`
 is magnet-only and understates torque per ampere for salient machines
 (~5% for a 1.6-saliency benchmark motor at rated current).
- **ETEL TMB reverse-salient anchors** (`data/etel_tmb/`, two frames: 44- and
 66-pole water-cooled torque motors, L_d > L_q from the manufacturer
 datasheets). First real machines exercising the reverse-salient MTPA branch
 (γ < 0): catalog Kt reproduced by `k_T_effective` within ~1%, ψf derivation
 cross-checked against the datasheet short-circuit current (< 0.2%). Also the
 **first external validation of the `thermal_resistance` budget path**: hot
 copper loss at the datasheet Ic matches the published dissipation Pc within
 2%, and the ΔT/r_th budget reproduces the manufacturer's continuous rating
 within 3% on both frames.
- **Du 2021 CLF-RSPMSM literature anchor** (`data/du2021_clf_rspmsm/`, DOI
 10.30941/cestems.2021.00020) — FEM-only reverse-salient design study
 (L_d/L_q = 1.08, 8-pole, 10 kW) grounding the sign and operating point of
 the γ < 0 MTPA branch: paper's published optimum −5° / i_d = +4.4 A
 (magnetizing), linear model −8.2° / +7.1 A on the same side with a flat
 optimum (0.16% torque difference), and torque reproduced within 1.4% using
 the paper's load-dependent flux linkage.
- **`sh_upper_pct` FEM metric** — upper slot-harmonic sideband (order
 Q + n_p, % of fundamental). Slot harmonics come in pairs Q ± n_p; the
 existing `sh_pct` reports only the lower sideband. Additive output key —
 no version bump; cached results simply lack the new key.
- **`phasesweep-crossval --strict`** — exit code 1 unless every comparison
 passes, so the crossval CLI can gate CI.
- `execute_generic`/`execute_parallel` accept `timeout_s` to override the
 whole-pool 600 s deadline (which does not scale with job count).
- **ETEL TMB stall-torque bound anchors** — first datasheet anchors for
 `stall_torque`: linear-magnetics stall torque bounds the datasheet peak
 torque Tp from above on both frames (+41% / +102%, pinned), and the
 reverse-salient γ < 0 branch is exercised at stall. Also pins a documented
 heuristic blind spot: both frames saturate hard at Ip while
 `I_stall/I_rated` sits below the strict 3.0 `saturation_warning`
 threshold.

### Changed

- **FEM interpole-gap width now measured at the inner PM-annulus radius
 (fem model v6 → v7).** `_interpole_gap_width` evaluated the gap
 arc-length at the *outer* PM radius on both topologies, but the gap is
 a radial wedge that is narrowest at the *inner* radius — where OCC
 fails to cut the sliver (the 0.5 mm arc-collapse threshold) and where
 the ≥2-element `pm_gap` refinement bites. Both now use the inner
 radius (`r_rotor` inrunner, `r_magnet` outrunner). Effect: the gap
 mesh is finer for every α_p < 1 run; on mm-scale arc motors, where the
 gap/2 term binds the global maxh, this shifts the fundamental slightly
 (reference B_r arrays moved 6e-5 T Belkhadir / 2e-4 T Deylami, ≤0.05%
 of peak) and near-threshold runs may now collapse to the full ring
 where they previously built arcs. Unit-scale α_p agreement is
 unchanged (+0.16 % at α_p = 0.7, same as before the fix), isolating
 the effect to the clamp-binding motors. Mesh disk-cache prefix bumped
 `mesh_v5 → mesh_v6`. Also adds an info-level log line when the arc
 clamp overrides the global maxh.

- **fem v6 — interpole-gap mesh rule enforced per-face**
 — gaps in the 0.5–1.0 mm band were meshed with fewer than 2 elements
 across (the global maxh keeps a 0.5 mm floor); the `pm_gap` faces now
 carry a local maxh = gap/2 with no floor. FEM outputs shift only for
 geometries in that band (none of the six in-repo motors; down-scaled
 sweep points can enter it). Mesh disk cache prefix bumped to
 `mesh_v4`.

- **Sub-0.5 mm interpole gaps now warn on arc collapse** — below the
 OCC arc-cut limit the FEM silently solved alpha_p = 1.0 while the
 analytical model kept the requested alpha_p (up to +7–17% silent FEM
 discontinuities in down-scaling geometry sweeps). The solver now logs
 a warning and sets `info["arcs_collapsed"]`; the collapse behavior
 itself is unchanged.

- **Dashboard table upgrades — column chooser, min/max shading,
 history filters** — the sweep
 table drops the silent 8-metric column cap: every discovered metric
 is a column, and a header-slot **column chooser** (persisted per
 browser) hides the ones you don't want, with an "N columns hidden"
 caption so the filter is never silent. CSV export still writes every
 metric, not just the shown ones. Rows whose shown metric columns are
 all empty (e.g. a drive_sim row under analytical columns) hide by
 default behind a toggle. Each metric column now shades its **min and
 max** cell (direction-neutral, not a pass/fail verdict). Run history
 gains a **status/model/motor filter row**, **relative timestamps**
 ("4d ago", absolute on hover), a full-text row tooltip, and a "show
 more" reveal past the first 30 rows. Sidebar Jobs rows get the same
 content treatment — "motor · type" title, relative timestamp, and
 the raw job id moved to the row tooltip instead of the always-on
 line.

- **Dashboard app frame, density pass, sidebar rail, status bar** — the page becomes an app
 frame: topbar, nameplate, and a new bottom status bar (connection
 ● · running-job count · last-result time) are fixed rows, with the
 main area and sidebar scrolling independently; main-area panels
 tile edge-to-edge with hairline seams instead of floating cards.
 Density: 13px base type with tabular numerals, tighter panel
 chrome and table padding. Sidebar: one-line config rows
 (compare-color dot · key · meta, ✎ marks user configs), a
 persisted collapse-to-rail toggle, and job/history rows flattened
 onto the sidebar surface. Primary buttons (Submit/Save/Run) go
 accent-outline — saturated fills stay reserved for data/state.
 The connection-lost banner now latches until reconnect, and the
 completion toast moves bottom-center in inverted colors.

- **Dashboard config switch now replaces the plots**
 — a plain config click clears the waveform/spectrum/
 cross-section/sim panels before routing the new config's results;
 previously the old config's traces lingered (and often dominated
 the y-range) until "clear overlay". Comparing stays explicit via
 ctrl-click, and de-selecting an overlay now removes its traces by
 rebuilding from the remaining marks. The sweep table still
 accumulates across configs by design. Also new: a topbar-level
 banner when the WebSocket drops ("job progress may be stale"), an
 eviction notice when the 8-trace overlay pool drops its oldest
 trace, and a tone legend (≤ tol / ≤ 2× tol / > 2× tol) plus
 click-to-pin cell detail on the validation matrix.

- **Analytical odd-harmonic space harmonics (`analytical` v4)** —
 `zhu_howe_Br_series` sums the odd harmonics of the square-wave
 magnetization (order `m = k·n_p`, coefficient `(4/kπ)·sin(kπα_p/2)`)
 through the per-order Zhu/Howe/Chan transfer function, radii-normalized
 for high-order conditioning (validated against the audited per-harmonic
 Eq 17 sum). The analytical `B_r_list` becomes a true waveform (flat-top
 below the Gibbs-overshooting fundamental), `thd_pct` becomes non-zero,
 and `peak_Br` is added; the fundamental bin and all psi_f/B_rem
 derivations are unchanged. `fundamental_only` no longer set on any
 registry model — waveform-shape comparisons (THD, peak, curve
 max/min) now run for real against the analytical model (e.g. the
 published ANSYS peak comparison that used to be skipped now reads
 −25.5%, same direction as FEM's −16%).

- **`build_sim` rejects reverse-salient machines (L_d > L_q)** — motulator
 0.7.3's MTPA current reference is silently wrong for them (zero-torque
 sentinel below ψf/(L_d − L_q), demagnetizing negative-torque root above
 it). `drive_sim` raises with a clear message until the upstream fix lands;
 the phase-sweep circuit models (`rated_torque`/`stall_torque`/
 `thermal_duty`) handle reverse saliency correctly and are unaffected.

### Fixed
- **`docs/motor-toml.md` no longer advertises removed `[drive]`
 defaults.** The table still listed `U_DC = 540.0`, `MAX_I_S = 20.0`,
 and `W_REF = 2π·50` as fallbacks. All four drive fields default to
 unset — a solver that needs one raises naming the motor, the model,
 and the field — so the doc was promising a silent default where the
 code raises.
- **numpy scalars no longer serialize to their repr in the result
 store.** `ResultStore.save` fell back to `str()` for anything the JSON
 encoder rejected, so a `np.bool_` metric was stored as the string
 `"True"`/`"False"` — and `bool("False")` reads back **True**. numpy
 scalars now unwrap to their Python value and arrays to lists. Two
 `thermal_duty` records in the local store held a stringified
 `within_s1`.


- **Test-audit residue — value pins for THD and wedge `k_end`.** Closes the
 two findings the test audit left open. No physics outputs change.
 - `compute_thd` had no value-pinning assertion anywhere: scaling its
 return by 1.10 passed 185 tests, because every numeric THD assertion
 was a one-sided bound on measured harmonics and both models call the
 same function, so crossval was blind too. Added an exact pin from
 hand-built amplitudes (50 %, plus DC-exclusion and scale-invariance),
 a textbook square-wave pin through the FFT path (48.34 % =
 √(π²/8 − 1)), and an analytical-runner pin (32.27 %) that also guards
 the `n_p` → fundamental-bin wiring end to end. All three catch the
 mutation.
 - Wedge `k_end` was asserted as `0.5 < k_end < 1.0` at five sites, wide
 enough to accept a 0.75× scale error — the same blind spot that let
 the original wedge study's silently non-converged nonlinear column
 pass review. Replaced
 with two-sided pins at the measured fast-config values (0.9428 linear,
 0.8986 nonlinear, 0.9626 tight-gap, 0.9866 with overhang), ±0.02 for
 mesh/solver headroom. The 0.75× mutation is now caught.
 - Pointless-test cleanup: the two published Creator torque checks now
 route through `magnet_torque_constant` / `mtpa_torque` instead of
 reimplementing the torque equation inline, so the datasheet numbers
 test library code (verified bit-identical to the old inline form); two
 subsumed 14 mm outrunner assertions — over-prediction direction and a "within
 50 %" sanity bound, both implied by the existing [20 %, 40 %] band —
 were dropped.

- **CI installed no `server` extra, so 41 API tests silently skipped.**
 `uv sync --extra test --extra dev` uninstalls `fastapi`, and
 `tests/test_server.py` opens with `pytest.importorskip("fastapi")` — the
 whole API/dashboard tier dropped out of CI while the run reported green,
 including the regression test for the null-section 500 above. CI now
 installs `--extra server`, and a `PHASESWEEP_STRICT_SKIP` gate in
 `tests/conftest.py` fails the session on any skip whose reason is not in
 an explicit allowlist (the two gitignored third-party datasets).
 Collection-time skips reach the gate, so a module-level `importorskip`
 cannot hide again.

- **Demag screen asserted only that the margin was positive.** Every
 assertion on the CREATOR anchor was one-sided, an equality against a
 constant, or monotonic in fault current — all preserved by a constant
 bias. A +0.15 T optimistic error (60% of the anchor's own margin) passed
 the whole file, whether injected into the returned `margin` or upstream
 in `B_m`. Adds a two-sided value pin at the frozen screen settings, plus
 the `margin`/`B_m_min` identity and a remanence upper bound.

- **`test_outrunner_14mm_crossval` swallowed a broken script import.** A
 `try/except ImportError` around `scripts/backemf_validation` turned a
 breakage into 10 silently skipped tests of 19, with the run still
 reporting success. The import is now hard.

- **sdist shipped a third-party dataset.** Hatchling does not honor the
 nested `data/creator_case_pmsm/.gitignore`, so the CREATOR measurement
 dataset (DOI 10.3217/sns1d-77m43) was packaged into the source
 distribution — 20.5 MB, and not ours to redistribute. Excluded
 explicitly; the tarball drops to 3.3 MB with no tracked file lost.

- **QC audit — server crash, spec drift, and citation errors.**
 No physics outputs change.
 - *Server:* `PUT /api/configs/{name}` returned **500** when any optional
 TOML section was JSON `null` — `raw.get(section, {})` yields `None`
 for a present-but-null key, and every downstream `.get` raised
 `AttributeError`. Seven of the eight sections were affected. Now
 `raw.get(section) or {}`, so a null section is a 400 like any other
 invalid payload. Found by API fuzzing (`scripts/fuzz_api.py`, new).
 - *FEM solver docs:* `r_stator`/`r_magnet`/`r_rotor` still shown
 as defaulted keywords — they had since become required. Back-EMF
 results were pre-convention-fix values (analytical +8.1% /
 FEM +19.3% / B_rem 1.341 T); the current, published figures are
 **+27.3% / +27.4% / 1.139 T**, and the analytical-vs-FEM "bracket"
 those old numbers implied was an artifact of the sinusoidal source.
 - *Model framework docs:* `DriveParams` documented with silent physical
 defaults (540 V / 20 A / 2π·50) and unconditional validation; all
 four fields are `float | None = None` with conditional checks.
 `RunConfig` block was missing `i_fault` and `dataset_id`.
 - *Tests:* two cases added to `TestZhuHoweSeries` pinning the harmonic
 series' odd-only parity and its `sin(kπα_p/2)` pole-arc scaling at
 α_p ≠ 1 — properties that mutation testing showed were covered only
 indirectly, via published-anchor crossval tests.
 - *Citations:* the magnetization Fourier coefficient `M_rn` is Zhu,
 Howe & Chan (2002) **Eq 7a**, not Eq 6a (6a is the series definition);
 corrected at 7 sites. `docs/references.md` attributed dissertation
 equation and section numbers (2.14, 2.7–2.8, §5.1.2) to the 2018 IEEE
 TIA paper — they belong to the **2019 Aalto doctoral dissertation**;
 bibliography re-pointed and "Awan (2018)" corrected.

- **Release-audit backlog — hardening sweep across server, dashboard,
 calibration, and FEM.** No physics outputs change. Known limitation:
 interpole-gap guard width is evaluated at the outer PM-annulus radius
 instead of the inner; fix deferred to a version-bump release.
 - *Server:* Host/Origin validation middleware (closes DNS-rebinding /
 CSRF against the localhost server; non-browser clients without an
 Origin header are unaffected); crossval-matrix dedupe now prefers OK
 records over newer TIMEOUT/ERROR ones; job drain moved to a daemon
 thread so Ctrl-C exits promptly; malformed `duty_profile`/`sim_plan`
 in a submit body return 400 instead of 500 (and no longer masquerade
 as a 404 unknown-motor); duplicate run_ids in one submission collapse
 to a single sub-task; WebSocket frames are validated and undecodable
 frames close 1003.
 - *Dashboard:* config-editor ignores stale out-of-order load responses;
 a failed boot-time config load now surfaces in the sidebar + toast
 instead of a blank page; `resultCache` is LRU-capped at 256;
 `dropMember`/`clearOverlays` bump `routeEpoch` so in-flight fetches
 can no longer resurrect dropped compare-set members.
 - *Calibration:* `--bound` for a param not being fitted is refused
 (was silently ignored), inverted bounds are refused; an infeasible
 zero initial guess is clamped into bounds with a record warning;
 `fit_bertotti` rejects non-finite/non-positive loss tables with a
 clear error instead of a LAPACK crash; before/after residuals are
 paired by key (a plain zip could misalign when a fit dropped rows);
 TOML string values are escaped; auto-model selection skips
 `thermal_duty` (needs a duty profile the calibration loop never
 carries); a solver `ValueError` on a perturbed motor mid-fit is
 treated as a rejected step instead of killing the fit.
 - *FEM:* loud error when magnet/yoke overhang reaches the end-effect
 air-box cap (was silently clamped fringe field); rotated (cogging)
 meshes no longer persist to disk — one file per angle grew the cache
 without bound; mesh-cache filenames use injective float reprs
 (6-sig-fig rounding could serve the wrong cached mesh), cache prefix
 bumped to `mesh_v5` — stale v3/v4 files purged from `output/`.

- **Analytical model rejects `n_theta = 2·n_p + 1`** — that single
 value passed the FFT-bin bound (`n_theta > 2*n_p`) but made the
 series' anti-alias cutoff (`max_order = n_theta // 2`) drop the k=1
 term, silently returning an all-zero waveform stored as an OK
 result. The guard is now `n_theta >= 2*n_p + 2` and raises before
 the solve.

- **Inrunner TOMLs reject `back_iron_thickness` loudly** — the
 outrunner-only ring/shell yoke split was silently dropped when the
 key appeared in an inrunner `[geometry]` section (direct `Geometry`
 construction already raised); `geometry_from_toml` now raises the
 same `ValueError`.

- **Calibration residual rows are keyed per dataset** — two measured
 datasets sharing a test_type and quantity previously overwrote each
 other in the fit (only the last one constrained the optimizer while
 `CalibrationRecord.dataset_ids` listed both). Rows now carry a
 unique per-dataset label, every dataset contributes residuals, and
 the record's `residuals_before`/`residuals_after` entries name
 their dataset.

- **Validation matrix no longer widens on duplicate/stale runs** — `validation_summary`
 fed `compare_all` every stored result for the motor, and
 `load_results` is deliberately unfiltered, so a re-run or a
 superseded-physics record of one model produced `analytical ↔
 analytical` 0.0% self-pair columns and doubled cross columns. The
 summary endpoint now dedupes to the latest result per
 `(model, source, dataset_id)` — the newest current run wins,
 superseded-version records drop out. The CLI
 `compare_all`/`diagnose` keep their every-pair semantics; the dedupe
 is at the endpoint only. The validation matrix also gains the
 sweep-table horizontal scroll-shadow so clipped pair columns are
 visible.

- **Unknown sweep-axis fields now fail loudly** — submitting a
 sweep with a field that isn't a sweep axis (e.g. `r_magnet`) used
 to fail per-combination inside grid generation and surface as the
 misleading "sweep grid is empty (every combination produced an
 invalid geometry)". `SweepAxis` now validates `field` and
 `strategy` at construction and the job API returns the allowed
 list.

- **NaN metrics 500'd the results API** — starlette's JSONResponse
 refuses non-finite floats, so any stored result with a NaN metric
 (e.g. drive_sim `t_settle` when the speed never enters the settle
 band) made `GET /api/results/{id}` — and job endpoints carrying a
 validate summary — return 500. Non-finite floats are now served as
 `null`.
- **`j_s` silently ignored without slot faces** — `solve_field_fem` with
 `j_s ≠ 0` and `n_slots > 0` but `slot_depth = 0` built a mesh with no
 slot regions, integrating the armature source over nothing; the field
 came back identical to open-circuit with no warning. Now a loud
 `ValueError`.
- **Wedge Picard non-convergence now raises** —
 `_picard_solve_wedge` silently returned the last iterate after
 exhausting its iteration budget; it now raises `RuntimeError` like the
 2D `_picard_solve`. Adding the raise exposed that the fixed
 relax = 0.3 oscillates on the saturating end-region (raw step pinned
 at ~0.9 for 40 iterations at the test settings) — the previously
 published nonlinear k_end numbers came from such silently
 non-converged fields. The 2D
 solver's adaptive relaxation (damp on raw-step growth,
 applied-step cap, recover on decay) is now ported to the wedge, with
 the same 200-iteration budget and `picard_tol`/`picard_n_iter` exposed.
 Production re-run of the headline point (0.25 mm overhang): linear
 k_end reproduces 0.9694, but the CONVERGED nonlinear value is 0.9252
 vs the previously published 0.951 — end-region saturation costs ~4.6%
 below linear, not ~2%, moving the 14 mm outrunner best estimate to
 k_end ≈ 0.93. A full nonlinear column re-run confirms this is
 a column shift, not a one-cell anomaly: converged nonlinear sits a
 uniform ~4.3–4.4 pt below linear at every overhang (0.00/0.25/0.50/
 1.00 mm → 0.8985/0.9252/0.9412/0.9582), all linear cells reproduce the
 published table exactly.

- **Parallel pool-timeout misreporting** — when the whole-pool deadline
 fired, jobs that never started were recorded as `TIMEOUT` with the full
 deadline as their elapsed time, indistinguishable from jobs that actually
 ran. Never-started jobs now report `elapsed_s = 0` and a "not started
 before pool deadline" message (best-effort: the executor's prefetched
 call queue, workers + 1 items, is uncancellable and still reports as
 running).

- **Reverse-salient (L_d > L_q) MTPA** — `rated_torque`, `stall_torque`, and
 `thermal_duty` previously degraded reverse-salient machines to the
 non-salient `k_T` mapping (γ clamped to 0), under-predicting torque and
 over-predicting thermal current. The Morimoto + root is the global MTPA
 optimum for both saliency signs (γ < 0 = magnetizing i_d; brute-force
 verified); reverse saliency now takes the same MTPA path as normal
 saliency. Model versions bumped: `rated_torque`/`stall_torque` v3→v4,
 `thermal_duty` v2→v3 (run-IDs change; no shipped motor exercises the
 reverse-salient path).

## [0.2.0] - 2026-06-11

### BREAKING — magnetization convention change

`B_rem` now means **physical remanence**, and both solvers model
**square-wave radial magnetization** (uniformly magnetized arcs with
per-pole sign alternation). Previously both solvers implemented a
sinusoidally magnetized ring, which made `B_rem` an effective sinusoidal
amplitude rather than a datasheet value. The analytical source coefficient
gains the square-wave Fourier factor 4/π (fundamental
M_1 = (4/π)(B_rem/μ0)·sin(πα_p/2), Zhu & Howe Eq. 6a); the FEM source is
uniform |M_r| per pole instead of cosine-modulated.

Consequences:

- **All fundamental-linked outputs of motors with a derived psi_f shift
 +27%** (×4/π): B_g1, flux_linkage_peak, back-EMF, Ke, Kt, torque
 predictions. Motors with an explicit `psi_f` are unaffected.
- FEM and analytical now agree on the fundamental to <1% at all pole-arc
 ratios (the previous FEM-vs-analytical spread was a convention
 mismatch, not solver error).
- Validation deltas changed accordingly: the 14 mm outrunner reference motor
 reads ~+38% raw vs measured back-EMF (was a coincidental +8%); the
 attribution spans magnetization profile, 3D end leakage, and winding
 factor.
- Derived B_rem from psi_f is Carter-consistent; the slotted
 psi_f → flux_linkage_peak round-trip is now exact.
- Every computed model carries a `version` (now 2) hashed into run IDs —
 all pre-0.2.0 cached computed results are invalidated and will re-run.
 Re-import measured JSONs as well (dataset_id hashing below).

### Added

- Rated-torque model (MTPA operating point at rated current) and a
 locked-rotor stall-torque model
- Measured-data import and cross-validation across motors
 (`phasesweep-import`, `phasesweep-crossval`)
- Geometry sweeps and perturbation/sensitivity analysis with parallel
 execution
- FEM geometry fidelity: discrete magnet arcs (`alpha_p < 1`), trapezoidal
 slots with separate opening width, Carter-factor correction
- Outrunner topology support in the FEM solver
- `SimPlan`: drive-simulation timing derived from motor physics
 (`plan_sim`), plus torque-control mode (`plan_torque_sim`)
- Back-EMF validation pipeline against oscilloscope captures
- Disk-persistent mesh cache for FEM subprocess workers
- Curated public API in `phasesweep/__init__.py` with `__version__`;
 `py.typed` marker for downstream type checkers

### Fixed

- FEM Dirichlet outer boundary is now actually applied (previously the
 homogeneous boundary condition was silently skipped)
- Carter factor uses the slot opening width at the bore, not the slot
 body width (was over-correcting by ~15%)
- Drive-sim reference speed treated as mechanical, not electrical
- Drive parameters (DC link, current limits, reference speed) included in
 run-ID hashing
- Analytical flux linkage evaluated the air-gap field at the midgap but
 applied the bore winding formula, inflating derived `flux_linkage_peak`
 and back-EMF by ~g_eff/(2·r_bore) (≈1% on small-gap motors). The new
 `solver_params.psi_f_carter` evaluates at the bore; an explicit psi_f
 now round-trips exactly. Derived back-EMF comparisons shift ~−1%.
- Disk mesh cache is safe under parallel workers (unique temp names) and
 recovers from corrupt cache entries instead of crashing
- FEM armature source uses the motor's winding factor instead of a
 hard-coded 0.966 default (affects loaded runs only, `j_s != 0`)
- Validation report no longer discards failed model runs before counting
 them (the error count was always zero)

### Changed

- `Geometry` now stores the physical `slot_opening_width` (meters);
 `slot_opening_ratio` is a derived property and can no longer go stale
 against `n_slots`. TOML files may specify either (width wins when both
 are present). Geometry `config_id` hashes the width, so cached results
 and meshes for slotted motors are invalidated.
- The former `stall_torque` model was renamed to `rated_torque` (it
 computes MTPA at `I_rated`); the current `stall_torque` is a new
 locked-rotor model
- Solver inputs validated through typed param factories
 (`prepare_analytical`, `prepare_fem`, `prepare_drive_sim`,
 `prepare_rated_torque`, `prepare_stall_torque`)
- drive_sim run IDs hash the full `SimPlan` (controller tuning and
 extraction windows included, not just load/timing) — existing cached
 drive_sim results re-run once
- Measured imports carry a `dataset_id` (source filename) in the run-ID
 hash so repeat captures of the same motor/test no longer collide;
 measurement conditions, waveforms, and uncertainty are persisted into
 result metrics. Re-import measured JSONs to pick up the new IDs.

## [0.1.0] - 2026-03-01

Initial public release: Motor/Geometry type hierarchy, model registry
(analytical, FEM, drive simulation), computation pipeline with crash-safe
JSONL result store, Zhu & Howe analytical solver, NGSolve 2D magnetostatic
FEM, motulator drive simulation, validation framework.
