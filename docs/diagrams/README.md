# Architecture Diagrams

Sources (`.drawio`) live here; PNG exports go to `docs/images/`. Each
factual diagram element is mapped to a source in the claim tables below.

## D1 — `pipeline-spine.drawio` — data-flow spine

| Element | Exact claim | Source |
|---------|-------------|--------|
| `motors/*.toml` block | TOML sections `[motor] [circuit] [geometry] [winding] [materials] [drive]`; only `[circuit] n_p` required | `docs/motor-toml.md` |
| `load_motor` edge | `load_motor(path) -> Motor`; ValueError on missing/invalid fields, validation at the boundary | `phasesweep/machines/motor.py` |
| `Motor` block | frozen dataclass, "what the machine is"; composes Geometry; all electrical/winding/thermal fields optional | `phasesweep/machines/motor.py` |
| `Geometry` sub-block | Geometry as Motor sub-object | `phasesweep/machines/geometry.py` |
| `RunConfig` block | Motor + model key + solver params; "how to run the analysis" | `phasesweep/sweep_types.py` |
| `prepare_*()` edge | factories validate completeness and derive psi_f <-> B_rem at the solver boundary; "if you hold a params object, the Motor is ready for that solver" | `phasesweep/solver_params.py` |
| `*Params` block | frozen validated dataclasses AnalyticalParams … DemagScreenParams (9) | `phasesweep/solver_params.py` |
| `MODEL_REGISTRY` hub | plain dict of ModelInfo; no Protocol/ABC/plugin system | `phasesweep/registry.py` |
| computed model row | analytical, fem, drive_sim, rated_torque, stall_torque, thermal_duty, torque_speed, iron_loss, demag_screen | `phasesweep/registry.py` |
| measured-source row | backemf_capture, inductance_test, resistance_test, torque_test, airgap_flux_test, iron_loss_test | `phasesweep/registry.py` |
| measured import edge | CLI `phasesweep-import` and `POST /api/measured/{motor_name}`; source = measured/published | `phasesweep/validation/` |
| `metrics dict` edge | `fn: RunConfig -> dict` of the model's `produces` quantities | `phasesweep/registry.py` |
| `ResultStore` block | every record carries motor_config_id, model, source | `phasesweep/result_store.py` |
| crossval loop edge | comparison surface derived from registry `produces` intersections, not hardcoded; deltas computed on request, never stored | `phasesweep/validation/calibration.py` |
| server consumer | FastAPI REST (jobs/results/configs/measured) + WebSocket | `phasesweep/dashboard/` |
| dashboard consumer | no-build-step JS; panels auto-discover applicable models via `produces` | `phasesweep/dashboard/app.js` |
| CLI consumers | phasesweep-crossval, phasesweep-import, calibrate loop | `phasesweep/validation/` |

## D2 — `provenance-tiers.drawio` — three-tier hashing

| Element | Exact claim | Source |
|---------|-------------|--------|
| tier 1 box | `Geometry.config_id` hashes all geometry fields | `phasesweep/machines/geometry.py` |
| tier 2 box | `Motor.config_id` = geometry.config_id + n_p + populated optionals + materials; excludes name and drive | `phasesweep/machines/motor.py` |
| tier 3 box | `compute_run_id(rc)` = motor.config_id + model + solver params filtered by `ModelInfo.hash_fields` | `phasesweep/sweep_types.py` |
| version side-input | `model_v=N` part appended only when version > 1 | `phasesweep/sweep_types.py` compute_run_id |
| dataset_id side-input | unconditional when set — repeat captures of one test must not collide | `phasesweep/sweep_types.py` compute_run_id |
| drive-params note | drive excluded from Motor.config_id but hashed into run_id when the model's hash_fields include them | `phasesweep/sweep_types.py` |
| None-fields note | only populated optionals contribute; two Motors differing in None-vs-None hash identically | `phasesweep/machines/motor.py` |
| store edge | run_id keys the ResultStore record + index | `phasesweep/result_store.py` |
| staleness row: purge | version stamp vs registry — `populate_cache.py --purge-only` | `populate_cache.py` |
| staleness row: verify | recompute-and-compare — `populate_cache.py --verify` | `populate_cache.py` |
| staleness row: artifacts | reference_br `.npz` + mesh-cache prefix live outside the hash; manual recapture / prefix bump | `phasesweep/solvers/analytical.py` |

## Exports

- D1 → `docs/images/architecture_pipeline.png` — **full-width docs
  figure** (3.2:1 landscape); not meant for inline README embedding,
  where its labels fall below legibility
- D2 → `docs/images/architecture_provenance.png`

Both are embedded in `docs/architecture.md`, alongside the generated
`model_coverage_matrix.png`; the README links to that page rather than
carrying the figures inline.

Re-export by re-running the build scripts against the `.drawio` sources
(a local `drawio_utils` virtualenv), or from the draw.io app.
