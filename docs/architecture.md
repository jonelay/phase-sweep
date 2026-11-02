# Architecture

These figures show how a motor definition becomes a stored result, how result
IDs are computed, and which models have run for the bundled motors. Diagram
sources live in [`diagrams/`](diagrams/); [`diagrams/README.md`](diagrams/README.md)
records the source for each factual element and the repository revision checked.
The validation, simulation, and server subsystems do not yet have diagrams;
the same README documents the claim-table format used when adding one.

## Package layout

The package lives directly under `phasesweep/`. Its top-level Python modules
and packages divide configuration, computation, evidence, persistence, and
interactive consumers as follows; descriptions follow the package and module
docstrings (or, for asset-only directories, their entry-point source):

| Path | Responsibility |
| --- | --- |
| `machines/` | Machine definitions, geometry types, perturbations, and TOML loading. |
| `models/` | Rated-point and quasi-static performance models. |
| `solvers/` | Analytical and FEM field computation. |
| `simulation/` | Time-domain drive-simulation models. |
| `validation/` | Measurement comparison, cross-validation, and calibration. |
| `audit/` | Advisory induction-machine design checks against published diagnostic bands. |
| `server/` | Local FastAPI job manager, result-query API, WebSocket progress service, and MCP endpoint. |
| `dashboard/` | Static browser UI for configuration, jobs, results, sweeps, and validation. |
| `vis/` | Visualization utilities. |
| `registry.py` | Model metadata and execution-function registry. |
| `trust.py` | Read-time headline metrics and trust caveats for one result. |
| `solver_params.py` | Shared solver-parameter derivation and motor-completeness validation. |
| `sweep_types.py` | Validated run configuration and result types. |
| `geo_parallel.py` | Geometry sweep generation and parallel execution. |
| `result_store.py` | Crash-safe incremental JSONL result storage. |
| `defaults.py` | Access to solver and geometry defaults from `defaults.toml`. |
| `check_motor.py` | Staged objective checks and evidence-level reporting for one motor TOML. |

## Data-flow spine

![Pipeline spine — TOML through solver params, registry, and store to the consumers](images/architecture_pipeline.png)

A motor TOML loads into a validated, immutable `Motor`; see
[`motor-toml.md`](motor-toml.md). `RunConfig` pairs the motor with a model key
and solver settings. Each model's `prepare_*` function checks that its inputs
are complete and, when possible, derives `psi_f` from `B_rem` or vice versa.

`MODEL_REGISTRY` maps model keys to their metadata and run functions. Results
are written to `ResultStore` with their motor ID, model, and source.
Cross-validation compares only quantities that both records declare, and it
computes differences when requested rather than storing them.

## Model inventory

`MODEL_REGISTRY` in [`phasesweep/registry.py`](../phasesweep/registry.py) is
the model inventory and the source of truth for each key's source class, cost,
outputs, required motor and runtime fields, hash fields, version, validator,
and runner. Discover the current inventory and metadata programmatically with
`uv run python -c 'from phasesweep import MODEL_REGISTRY; print(*MODEL_REGISTRY.items(), sep="\n")'`;
the server and coverage generator consume this mapping directly, so this
overview does not duplicate its keys.

## Validation subsystem

`phasesweep/validation/` represents measured and published anchors as validated
`RunResult` records and imports them into the same `ResultStore` used by
computed models. Three stages form the evidence pipeline:

- **Import** — measured or published data enters as validated `RunResult`
  records, keyed by motor name so they survive TOML configuration changes.
- **Cross-validation** — resolves direct quantities, curves, bounds, and
  explicit key mappings between stored evidence and computed runs.
- **Calibration** — bounded perturbation of selected motor parameters against
  cross-validation residuals, with guards for circular, underdetermined, and
  insensitive fits.

`cli_import.py`, `cli_crossval.py`, and `cli_calibrate.py` expose those stages
without creating a second compute or persistence path.

## Simulation subsystem

`phasesweep/simulation/` is the registered time-domain branch of the main
pipeline: a motor supplies electrical, mechanical, and drive parameters, while
the `RunConfig` supplies a simulation plan and, for the two-mass model, load
mechanics. `prepare_drive_sim` validates and derives the plant parameters;
`sim_runner.py` builds and runs the motulator drive, extracts scalar response
metrics and downsampled waveforms, and returns the metrics dictionary through
the registry so the ordinary run identity and `ResultStore` path still apply.

## Local server and dashboard

`phasesweep/server/` layers phase-sweep motor loading, a `ResultStore`, and a
queued parallel job manager onto the local FastAPI/WebSocket service; its REST
routes submit registry models and sweeps, query current stored records, import
validation evidence, and manage editable user configurations. The static
`phasesweep/dashboard/` application consumes those APIs and stored-result
payloads to present job progress, run history, field and simulation waveforms,
sweep tables, model comparisons, validation summaries, and the config editor.

`server/mcp.py` mounts an MCP endpoint at `/mcp` on the same app. Its tools
read the shared job manager at call time, so agent jobs use the same worker
pool, job list, cache, and store as the dashboard; it adds no compute or
persistence path. Results reach the agent with headline metrics and caveats
from `trust.py`, which derives them from each model's `ModelInfo.headline`,
`units`, and `limits` and from provenance outputs the result already holds,
so nothing extra is stored. The server records each MCP request in memory for
the dashboard's Agent tab. Tool reference and usage: [mcp.md](mcp.md).

## Provenance and run identity

![Three-tier hashing — Geometry, Motor, RunConfig](images/architecture_provenance.png)

Result identity has three levels. `Geometry.config_id` hashes the geometry.
`Motor.config_id` adds `n_p`, materials, and populated optional fields, but
excludes the motor name and drive settings. `compute_run_id` then adds the
model key and that model's `hash_fields`. A model version greater than one
adds `model_v=N`; a supplied `dataset_id` is always included so repeated test
captures cannot collide.

Two staleness mechanisms sit outside the hash and need explicit
handling: version-stale records are purged by comparing the stamp
against the registry (`scripts/populate_cache.py --purge-only`), and records
written by changed code that never got a version bump are only caught
by recomputing (`scripts/populate_cache.py --verify`). Reference `.npz`
captures and the mesh-cache prefix are outside the hash entirely and
are recaptured or prefix-bumped by hand. See
[result-store-contract.md](result-store-contract.md) for the
portable specification of these rules, including test vectors for
downstream consumers.

## Model coverage

![Model coverage matrix — motors against computed models](images/model_coverage_matrix.png)

Rows are `motors/*.toml`; columns are computed models in the registry. A cell
shows whether the store contains a current result, the model can run but has
no stored result, or the motor lacks required fields. The ◆ counts are
measured and published records. They attach by motor *name* rather than
`motor_config_id`, so they remain associated with the physical motor when its
TOML configuration changes.

**This is a snapshot of one local store, not a project invariant.**
`output/` is gitignored, so the committed image reflects whichever
machine last rendered it. It is generated, never hand-drawn or
hand-edited:

```bash
uv run python scripts/populate_cache.py     # fill any gaps first
uv run python scripts/generate_readme_figures.py --fig 4
```

A grey cell is a statement about the motor file, not about the model:
`iron_loss` needs an `[iron]` section, `thermal_duty` and
`torque_speed` need circuit and thermal fields that the published
benchmark motors do not all carry. Light cells mean the model runs but
nothing was stored when the image was generated. The generation script is
the authority for inclusion policy.
