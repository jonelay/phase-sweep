# phase-sweep

PMSM and induction-machine simulation for stator/rotor design exploration. Integrates
with [motulator](https://github.com/Aalto-Electric-Drives/motulator) and
[NGSolve](https://ngsolve.org/) through optional dependencies.

![phasesweep dashboard — CREATOR Case PMSM results](docs/images/dashboard.png)

## Project status

The project is under active development on `main`. Released versions are
tagged; the [CHANGELOG](CHANGELOG.md) owns the release history and records
what is in each version, including work on `main` ahead of the latest tag.

## What it does

phase-sweep reads a motor definition from TOML and runs the models that its
inputs support. A motor with geometry can use the analytical and FEM field
solvers; one with only circuit parameters can still produce torque-speed
envelopes, thermal estimates, and motulator drive simulations. The bundled
motors (`motors/` plus the validation anchors under `data/`) cover radial
and axial-flux machines, including SPM, IPM, outrunner, SynRM, Halbach,
and inset-magnet examples. Published and measured data are
included where licensing permits.

Sweeps can run in parallel and store their results in an append-only JSONL
file. A local browser dashboard runs jobs, plots stored results, and compares
them with imported measurements.

## Features

**Field solvers**
- Zhu & Howe (2002) closed-form air-gap field — inrunner and outrunner
- NGSolve 2D FEM — smooth-bore and slotted stators (OCC boolean geometry, exact arc-bounded slots), linear or nonlinear iron (damped Newton by default, with Picard startup/fallback), armature reaction, Maxwell-stress torque
- Cogging torque via FEM rotation sweep; demagnetization screen at a specified fault current
- Axial-flux machines: quasi-3D layer model (Parviainen) and multi-plane 2D Cartesian FEM, sharing the same TOML/registry/result-store pipeline
- Spatial-FFT harmonic decomposition (optional CuPy GPU path, falls back to NumPy)
- Independent-code verification: the radial and axial FEM solvers are cross-checked against [ElmerFEM](https://www.elmerfem.org/) using a different code, mesh, basis, and nonlinear linearization — see the [validation report](docs/validation-report.md)

**Circuit-parameter models**
- MTPA rated torque, stall torque, torque-speed envelope with field weakening (continuous and peak)
- Thermal-duty screen — copper-loss S1 budget over a torque-time duty profile, R_s derated to winding temperature. Deliberately optimistic (no AC/iron/magnet losses), so a failing verdict is decisive and a passing one is not
- Bertotti two-term iron loss
- Squirrel-cage induction machines: per-phase equivalent circuit and torque-slip curve with deep-bar `R_2(f)` and magnetizing-saturation tables, checked against the CREATOR IM open benchmark's measured locked-rotor sweep (the circuit parameters are the benchmark's own fits to that sweep, so this checks the circuit implementation rather than independently validating it), plus IEEE-112-style parameter identification from raw DC / no-load / locked-rotor test data (`identify_from_tests`), verified end to end against the same benchmark's published circuit parameters. Synchronous and induction model families reject each other's motors loudly

**Drive simulation**
- motulator current-vector control transients, single-mass or two-mass (compliant shaft) load

**Sweeps and validation**
- Geometry and parameter sweeps with parallel execution, batched model dispatch, and crash-safe JSONL result storage
- Perturbation-based sensitivity analysis (OD, gap, L_stk, B_rem, ...)
- Cross-validation framework: import measured data, compare against computed results across motors
- Bounded multi-parameter calibration against measured quantities

**Dashboard**
- Local FastAPI server + browser dashboard: queue solver jobs, watch progress live, plot stored waveforms and envelopes, edit motor configs

**Documentation:**
[motor TOML format](docs/motor-toml.md) |
[architecture](docs/architecture.md) |
[validation report](docs/validation-report.md) |
[result-store contract](docs/result-store-contract.md) |
[glossary](docs/glossary.md) |
[references](docs/references.md)

## Scope and limitations

The registered field solvers are two-dimensional; the axial models combine
multiple two-dimensional planes. Three-dimensional effects enter through
offline correction studies rather than the normal run path.

Saturation enters the PMSM circuit models only through a user-supplied
`[flux_map]` (consumed by `rated_torque`, `stall_torque`, `torque_speed`,
`operating_point`, `motor_thermal`, and `drive_sim`), and the induction
circuit models only through a user-supplied magnetizing table
(`[induction]` `L_m_vs_i`); no model predicts a saturated inductance from
geometry. AC winding losses and magnet eddy-current
losses are not implemented.

## Install

Requires Python 3.12+.

```bash
git clone https://github.com/jonelay/phase-sweep.git
cd phase-sweep

# Full install (all solvers + server + plotting)
pip install -e ".[all,test]"

# Or core only (circuit-parameter models and result store, no FEM/sim/plots)
pip install -e .

# Test suite (adds pytest)
pip install -e ".[test]"
```

Optional extras:

| Extra | What it adds |
|-------|-------------|
| `[fem]` | NGSolve FEM field solver |
| `[sim]` | motulator drive simulation |
| `[viz]` | matplotlib plotting |
| `[server]` | FastAPI local server + dashboard |
| `[all]` | All of the above |
| `[gpu]` | GPU harmonics (requires CUDA 12.x) |
| `[mkl]` | Intel MKL BLAS backend for NGSolve |
| `[test]` | pytest + plugins |

NGSolve provides Python packages for supported Linux, macOS, and Windows
environments. See the [NGSolve installation guide](https://ngsolve.org/installation.html)
for current platform details.
The circuit-parameter models and non-FEM CLI paths do not require NGSolve.

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv venv .venv --python 3.12
uv sync --extra all --extra test
```

Run the tests:

```bash
uv run pytest -m "not fem and not slow"  # core CI lane
uv run pytest                            # all tests available in this environment
```

## Quick start

Compute the torque-speed envelope of the Prius 2004 IPM:

```python
from phasesweep import MODEL_REGISTRY, RunConfig, load_motor

motor = load_motor("motors/prius_2004.toml")
cfg = RunConfig(motor=motor, model="torque_speed")
envelope = MODEL_REGISTRY["torque_speed"].fn(cfg)
print(f"Peak torque: {envelope['tau_curve_peak'][0]:.0f} Nm")
print(f"Peak power:  {envelope['p_max_peak']/1000:.0f} kW")
```

Both numbers are constant-inductance over-predictions — see the
[saturation caveat](#torque-speed-envelope-prius-2004-ipm) below.

Compute the air-gap field of a motor with full geometry — analytically,
then with the 2D FEM solver, and compare:

```python
motor = load_motor("motors/belkhadir_outrunner.toml")

cfg = RunConfig(motor=motor, model="analytical", n_theta=720)
metrics = MODEL_REGISTRY["analytical"].fn(cfg)
print(f"Analytical B1: {metrics['fundamental']:.3f} T")
print(f"Back-EMF:      {metrics['backemf_fundamental']:.1f} V")

cfg = RunConfig(motor=motor, model="fem", n_theta=720)
metrics = MODEL_REGISTRY["fem"].fn(cfg)
print(f"FEM B1:        {metrics['fundamental']:.3f} T")
```

(The analytical solver covers surface-mounted magnets with air between
the arcs; inset rotors like the bundled CREATOR motor are FEM-only and
are rejected with a pointer to the `fem` model.)

Launch the dashboard (needs the `[server]` extra):

```bash
phasesweep-server         # serves http://127.0.0.1:8000
```

It reads `motors/*.toml` and the result store in `output/` relative to
the working directory (the shipped motor definitions live in the source
repository, not in the wheel; `--motors-dir` points elsewhere), queues
solver jobs across worker processes, streams progress over WebSocket,
and plots stored waveforms, harmonics, and envelopes in the browser.
Downstream consumers of the store read rows against
`phasesweep.record_schema` and the
[result-store contract](docs/result-store-contract.md).

## CLI tools

| Command | Purpose |
|---------|---------|
| `phasesweep-check-motor` | Check a motor definition file for structural validity, input consistency, and model execution |
| `phasesweep-import` | Import measured data (JSON) into the result store |
| `phasesweep-crossval` | Compare computed vs measured results across motors |
| `phasesweep-calibrate` | Fit motor parameters to measured quantities |
| `phasesweep-server` | Local FastAPI server + browser dashboard |

Each takes `--help`.

## Available models

`RunConfig(motor=..., model=<key>)` dispatches through `MODEL_REGISTRY`.
Each entry declares what it produces and which motor fields it needs;
a motor missing a required field fails validation before any solve.

| Key | Cost | Computes | Requires |
|-----|------|----------|----------|
| `analytical` | fast | Air-gap B_r waveform, fundamental, THD, back-EMF, flux linkage | geometry, magnet (`B_rem`, `alpha_p`, `mu_r_pm`) |
| `fem` | slow | No-load B_r waveform, \|B\| field grid, fundamental, peak iron flux density; Maxwell-stress torque only when armature excitation (`j_s`) is supplied | geometry, magnet, `mu_r_fe`; winding/current info to derive a loaded `j_s` |
| `analytical_axial` | fast | Air-gap B_z waveform, fundamental, THD, flux linkage (quasi-3D layer model) | axial geometry, magnet |
| `fem_axial` | slow | B_z waveform and fundamental from multi-plane Cartesian FEM | axial geometry, magnet, `mu_r_fe` |
| `cogging_torque` | slow | Peak-to-peak cogging torque, dominant harmonic order | geometry, magnet, `mu_r_fe` |
| `demag_screen` | slow | Minimum magnet flux density vs knee at fault current, demag margin | geometry, magnet, winding (`N`, `k_w`, `coils_series`), `B_knee`; `i_fault` per run |
| `drive_sim` | medium | Speed/torque/current transients, settling time, droop | `R_s`, `L_d`, `L_q`, `psi_f`, `J` |
| `drive_sim_two_mass` | medium | `drive_sim` with a two-mass load; adds shaft torque | same, plus load params |
| `rated_torque` | fast | MTPA torque and angle at rated current, torque constant | `psi_f`, `I_rated` |
| `stall_torque` | fast | Voltage/current-limited stall torque | `psi_f`, `R_s`, drive limits |
| `torque_speed` | fast | Continuous + peak envelope, base speed, field-weakening range | `R_s`, `L_d`, `L_q`, `psi_f`, drive limits |
| `operating_point` | fast | Minimum-current (i_d, i_q), dq voltages, modulation index, power factor, regime and powers for a speed + torque demand; solves on `[flux_map]` when present | `R_s`, `L_d`, `L_q`, `psi_f`, drive limits; `w_mech`, `tau_ref` per run |
| `thermal_duty` | fast | S1 verdict, copper loss, winding temperature over a duty profile | `psi_f`, `R_s`, duty profile |
| `motor_thermal` | fast | Steady winding / iron / magnet temperatures and losses of a three-node network coupled to the operating point (copper at the winding temperature, magnet flux at the magnet temperature, iron loss following the stator flux) | `[thermal_network]`, `[iron]`, `alpha_Br`, `R_s`, `L_d`, `L_q`, `psi_f`, drive limits; `w_mech`, `tau_ref`, `T_coolant` per run |
| `iron_loss` | fast | Hysteresis + eddy core loss at a given speed | `[iron]` section (`k_h`, `k_e`, `alpha_fe`, `m_core`, `B_core`) |
| `im_equivalent_circuit` | fast | Induction-machine circuit at one slip: torque, currents, power factor, losses | `[induction]` section, `R_s`; `slip` per run |
| `im_torque_slip` | fast | Torque-slip sweep: breakdown, starting point, rated point (deep-bar `R_2(f)` aware) | `[induction]` section, `R_s` |
| `im_bar_impedance` | medium | Predicted deep-bar `R_2(f)` / `L_sigma_r(f)` tables from a periodic single-slot-pitch harmonic FEM solve, end ring included | cage `[geometry]` (bars, end ring), `sigma_bar`, winding (`N`, `k_w`, `coils_series`) |
| `im_fem_harmonic` | slow | Full-machine time-harmonic FEM at one slip: current, power factor, torque, loss split, field grid; linear iron, or saturating iron by an energy-equivalent fixed point with `nonlinear` (`[materials] bh_curve`) | cage `[geometry]` with stator slots, `sigma_bar`, `V_LL_rated`, `f_rated`, `R_s`, winding (`N`, `k_w`, `coils_series`, `coil_pitch_slots`); `slip` per run |
| `im_fem_locked_rotor` | slow | Virtual blocked-rotor tests identified into `R_2(f)` / leakage tables the IEEE-112 way; no-load voltage sweep identified into `L_m(I_m)` with `nonlinear` (linear-iron value otherwise) | same as `im_fem_harmonic` |

The `fem` quick-start call above is a no-load field solve and does not
return loaded torque. See
[docs/add-a-radial-pmsm.md](docs/add-a-radial-pmsm.md) for the staged
new-motor workflow.

The registry keys `backemf_capture`, `inductance_test`,
`resistance_test`, `torque_test`, `iron_loss_test`, and `airgap_flux_test`
are measured-record types with no compute function — they are imported
with `phasesweep-import` and anchor the cross-validation comparisons.

## Motor definition

Motors are TOML files. Only a pole-pair count is required (`name`
defaults to the filename stem);
each model declares which further fields it needs, so a circuit-only
motor (no geometry) runs the torque and envelope models, and a full
geometry unlocks the field solvers. Minimal example:

```toml
[motor]
name = "example"
type = "SPMSM"

[circuit]
n_p = 4
R_s = 0.05       # Ohm per phase
L_d = 0.0002     # H
L_q = 0.0002     # H
psi_f = 0.012    # Wb per-phase peak

[drive]
U_DC = 48.0      # V
MAX_I_S = 30.0   # A peak
```

Sections, fields, units, and defaults are documented in
[docs/motor-toml.md](docs/motor-toml.md). The bundled [`motors/`](motors/)
files are worked examples with provenance comments. To add a new motor and
evaluate it, follow [docs/add-a-radial-pmsm.md](docs/add-a-radial-pmsm.md)
and `phasesweep-check-motor`.

## Verification and validation

### Torque-speed envelope (Prius 2004 IPM)

The Prius 2004 case exercises stall torque, MTPA angle, and the
torque-speed envelope for a 50 kW, 8-pole, V-shape NdFeB traction motor.
Circuit parameters come from ORNL teardown and dynamometer reports
([OSTI 885676](https://www.osti.gov/biblio/885676),
[OSTI 890029](https://www.osti.gov/biblio/890029)). It is not a validation
at high current: the constant-inductance model predicts about 905 N·m stall
torque against about 340 N·m measured on the ORNL dynamometer at 250 A
(+166 %), because L_q saturates. The test suite pins that prediction as a
regression value and checks the one-sided bound against the measurement;
the MTPA angle is checked against a plausible band, not a measured value.
See [docs/validation-report.md](docs/validation-report.md) for details.

### Solver verification (Zhu & Howe)

The NGSolve FEM solver is verified against the Zhu & Howe (2002) closed-form
air-gap field solution across reference inrunner and outrunner configurations.
Acceptance bounds and current results are documented in the
[validation report](docs/validation-report.md).

Radial and axial FEM results are also cross-checked against
[ElmerFEM](https://www.elmerfem.org/). Pinned comparison records in
`scripts/elmer_radial/` and `scripts/elmer_axial/` are exercised by the core
test suite; regenerating them requires a local ElmerSolver.

### Back-EMF and torque checks (CREATOR benchmark)

The [CREATOR](https://doi.org/10.3217/sns1d-77m43) open-benchmark PMSM
(sintered-ferrite inrunner; Muetze et al., COMPEL 2025) supplies measured
back-EMF data and published circuit parameters. Computed values, references,
tolerances, and errors are reported in the
[validation report](docs/validation-report.md).

![CREATOR back-EMF — measured waveform and circuit-model fundamental](docs/images/backemf_waveform.png)

CREATOR is specified by its published circuit parameters, so its flux
linkage ψ_f is taken directly from the benchmark. The comparison therefore
validates the back-EMF and torque pipeline — electrical-speed conversion,
`E = ω_e·ψ_f`, and the MTPA torque solve — against an independent
measured-plus-published reference; they do **not** test the field solver's
own prediction of ψ_f. The nonlinear FEM field prediction and its remaining
bias are tracked in the validation report.

The trapezoidal measured waveform reflects the near-rectangular ferrite
magnets, consistent with the square-wave magnetization convention used by
both field solvers. The computed back-EMF comparison is fundamental-only
because the model currently reports a scalar fundamental rather than a
predicted back-EMF waveform.

![Verification and validation summary](docs/images/multimotor_validation.png)

## Reference data

The `data/prius_2004/` directory contains the
[Pyleecan](https://github.com/Eomys/pyleecan) machine-readable IPM
geometry (Apache-2.0) and ORNL dynamometer reference values for the Toyota
Prius 2004 traction motor. See
[`data/prius_2004/README.md`](data/prius_2004/README.md) for sources and
declared assumptions.

The `data/creator_case_pmsm/` directory contains reference data from the
CREATOR open-benchmark PMSM (Muetze et al., 2025), licensed under
CC BY-NC-ND 4.0. See
[`data/creator_case_pmsm/README.md`](data/creator_case_pmsm/README.md) for
attribution and citation information.

## Dependencies

| Package | Purpose | License |
|---------|---------|---------|
| [NumPy](https://numpy.org/) / [SciPy](https://scipy.org/) | Numerical computation | BSD-3-Clause |
| [motulator](https://github.com/Aalto-Electric-Drives/motulator) (optional) | Synchronous machine drive simulation | MIT |
| [NGSolve](https://ngsolve.org/) (optional) | 2D finite element field solver | LGPL-2.1 |
| [Matplotlib](https://matplotlib.org/) (optional) | Plotting | PSF (BSD-compatible) |
| [FastAPI](https://fastapi.tiangolo.com/) + uvicorn (optional) | Local server + dashboard | MIT / BSD-3-Clause |
| [structlog](https://www.structlog.org/) + [tomli-w](https://github.com/hukkin/tomli-w) (optional) | Server logging, TOML export | MIT or Apache-2.0 / MIT |
| [CuPy](https://cupy.dev/) (optional) | GPU-accelerated harmonic decomposition | MIT |

## Package layout

```
phasesweep/
├── machines/           # motor definitions and TOML loading
├── solvers/            # analytical and FEM field solvers
├── models/             # circuit-parameter models
├── simulation/         # motulator drive simulation
├── validation/         # measured-data import, comparison, and calibration
├── vis/                # matplotlib plotting
├── server/             # FastAPI server and job queue
├── dashboard/          # browser dashboard assets
├── registry.py         # model metadata and dispatch
├── sweep_types.py      # run configuration, results, and identifiers
└── result_store.py     # append-only JSONL result store
scripts/                # validation and maintenance tools
tests/                  # pytest suite, with dedicated FEM and slow markers
data/                   # published, measured, and derived validation data
motors/                 # motor parameter files
```

## Contributing

Bug reports, documentation fixes, and new validation anchors (datasheet
or bench data for a motor not yet in the test suite) are welcome.
See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and process.

Questions and discrepancy reports:
[GitHub Issues](https://github.com/jonelay/phase-sweep/issues).

## Citing

If you use phasesweep in a report or publication:

> J. Lay, *phasesweep: PMSM and induction-machine simulation for stator/rotor design
> exploration*, 2026. https://github.com/jonelay/phase-sweep

## License

Code is licensed under the [Apache-2.0](LICENSE).

Data files contain parameter values transcribed from cited publications
or measured by the project authors and, except as noted below, are covered
by the same Apache-2.0 license. [NOTICE](NOTICE) attributes the ORNL reports,
manufacturer datasheets, and academic publications those values come
from; if you redistribute `phasesweep`, Apache-2.0 §4(d) requires you
to carry NOTICE with it.

The file `data/prius_2004/pyleecan_geometry.json` is from the
[Pyleecan](https://github.com/Eomys/pyleecan) project (Apache-2.0).
Files in `data/creator_case_pmsm/` derived from the CREATOR dataset are
licensed under
[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) —
see [`data/creator_case_pmsm/README.md`](data/creator_case_pmsm/README.md).
Files that `scripts/fetch_creator_im_dataset.py` extracts into
`data/creator_case_im/` from the CREATOR IM dataset are licensed under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) and are not
shipped; the transcribed `creator_case_im.toml` is Apache-2.0 — see
[`data/creator_case_im/README.md`](data/creator_case_im/README.md).
The BMW i3 supplementary files `data/bmw_i3/BMWi3_GeometryAutoCAD.dxf` and
`FEMResults_1000rpm.xlsx` (Gierczynski & Grzesiak, 2021), and
`flux_maps_7x7.json` derived from the xlsx, are licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — see NOTICE.
