# Agent use (MCP)

`phasesweep-server` exposes a [Model Context Protocol](https://modelcontextprotocol.io/)
endpoint at `/mcp`. AI agents share the dashboard's job manager, worker pool,
and result store. Model inputs and outputs are documented in the
[README model table](../README.md#available-models) and
[motor-toml.md](motor-toml.md).

## 1. Start the server and connect

```bash
pip install -e ".[server]"
phasesweep-server                 # http://127.0.0.1:8000, MCP at /mcp
```

The endpoint is stateless streamable HTTP. For Claude Code:

```bash
claude mcp add --transport http phase-sweep http://127.0.0.1:8000/mcp
```

Other clients take the same URL as an HTTP (streamable) MCP server.

`[server]` covers the analytical and circuit models. `drive_sim` and
`drive_sim_two_mass` also need the `sim` extra, and the FEM models need `fem`;
`pip install -e ".[server,sim,fem]"` installs all three.

The server has **no authentication**. The Host/Origin check guards a local
browser against DNS rebinding; it is not access control. Binding a
non-loopback address (`--host 0.0.0.0`) or naming a proxy hostname
(`--allow-host`) lets every client that can reach the port submit jobs, read
results, and occupy the worker pool. Put a hosted server behind something that
authenticates.

## 2. A typical session

```
list_motors                     which configs are loaded
list_models(motor="prius_2004") which models run on it, and why the others don't
model_defaults(motor, model)    per-run params the server derives (param-gated models)
submit_job(motor, model, dry_run=True)   sub-task count, cached count, time estimate
submit_job(motor, model, params)         queue it
get_job(job_id, wait_s=60)      wait until completed / failed / cancelled
get_result(result_id)           headline, caveats, metrics
```

For a sweep, submit `model="sweep"` with `params.axes` and
`params.model_keys`, then read it with `sweep_table(job_id)` while it runs.
Fast models finish in about a second. `slow` models (FEM) can take minutes per
sub-task. Identical runs are served from the result store without recomputing.
Units are SI unless a field name says otherwise.

A captured run of the `bmw_i3_field_fundamental` agent case
(`tests/agent_cases/`, Claude Code's default model, `fem` v8), asked "What
is the air-gap flux density fundamental of the BMW i3 motor?":

```
list_motors()
list_models(motor="bmw_i3")              analytical blocked: inset rotor; fem runnable
submit_job(motor="bmw_i3", model="fem")  pending
get_job(job_id, wait_s=60)               held 33.5 s, returned completed
get_result(result_id)                    fundamental 1.115 T, thd_pct 26.1 %,
                                         b_iron_max 2.29 T; caveat limits (info)
```

Excerpt of the answer:

> The air-gap flux density fundamental for the BMW i3 (2016 IPM) motor is
> about **1.12 T**, or 1.115 T from the 2D nonlinear FEM run on the
> open-circuit magnets (`j_s` = 0, no stator current). [...] **Caveat:** the
> solution is two-dimensional, so 3D end effects are not included.

The 1.115 T is a computed value. The BMW i3 anchor has no measured or
published air-gap field to compare it with (its published FEM data are torque,
flux linkage and voltage; see `data/bmw_i3/README.md`).

## 3. Tools

| Tool | Purpose |
|------|---------|
| `list_motors` | Loaded motor configs: name, pole pairs, topology, whether geometry is present |
| `get_motor(name, raw_toml=False)` | One config as the resolved `Motor`, or its TOML section layout |
| `list_models(motor=None)` | Submittable models with `cost`, `needs_params`, `optional_params`; with `motor`, also `applicable`, `missing_params`, `reason` |
| `model_defaults(motor, model)` | Per-run params the server derives for that model on that motor (duty profile, fault current, sim plan) |
| `submit_job(motor, model, params=None, dry_run=False)` | Queue a registry model, `sweep`, or `validate` job; `dry_run=True` returns the plan and queues nothing |
| `get_job(job_id, include_subtasks=False, wait_s=0)` | Status, progress, result ids; a finished job adds per-result headlines and caveat codes. `wait_s` (0–60) holds the call until the job finishes or the time runs out |
| `list_jobs(status=None)` | Jobs this server process has seen |
| `cancel_job(job_id)` | Stop a job; in-flight sub-tasks complete |
| `sweep_table(job_id, sort_by, descending, offset, limit)` | A sweep as rows of swept dimensions × headline metrics, readable while it runs |
| `query_results(motor, model, source, limit)` | Current stored results; `source` is `computed`, `measured`, or `published` |
| `get_result(result_id, fields=None, full=False)` | One result: headline and caveats first, then status, config, and metrics |
| `validation_summary(motor)` | Computed vs measured/published comparison for a motor |

Errors come back as tool errors prefixed with an HTTP-style status
(`404: unknown motor config 'x'`, `400: ...`). `501` means the server lacks
an extra the model needs; the message names it.

The MCP surface does not edit motor configs; use the dashboard config editor
or the TOML files.

## 4. Choosing a model

Call `list_models(motor=...)` to check each model's input requirements against
the motor:

```json
{"key": "rated_torque", "cost": "fast", "kind": "single",
 "applicable": false, "missing_params": [], "reason": "rated_torque needs I_rated"}
```

- `applicable: false`: the motor lacks something the model needs, or the
  server lacks the `sim` extra a drive-sim model needs. `reason` names it;
  fix the motor file, install the extra, or pick another model. A missing
  `fem` extra is not listed here; the FEM job fails when it runs, with the
  install hint in its error.
- `missing_params`: required per-run params the server cannot derive for this
  motor (for example `load_mech` for `drive_sim_two_mass`). Pass them in
  `params`.
- `needs_params` without `missing_params`: the server derives them at submit;
  `model_defaults` shows the values it will use.
- `optional_params` have defaults (`fem` `rotation`, `cogging_torque`
  `cogging_points`).
- `fem` runs nonlinear unless `params.nonlinear` is `false`. Inset and bridged
  rotors saturate by design, so a linear solve on one is rejected at submit;
  the other FEM models default linear and list `nonlinear` in
  `missing_params` for those rotors.
- The `validate` entry lists `default_models`: the models a `validate` job runs
  when `params.models` is omitted.

## 5. Reading results

Every OK computed result carries:

- **headline**: the 2–5 outputs that summarize the model, each as
  `{key, value, unit}` in the unit stored. `max:<key>` entries are the maximum
  of a list output.
- **caveats**: `{code, severity, text, basis}`, derived when the result is
  read from the provenance outputs the model already emits. `basis` names the
  output that triggered the caveat. Records stored before a caveat existed get
  it without recomputing.

`severity` is `info` or `warn`. Report a **warn** caveat with the headline
number: it marks the number as likely optimistic. For example,
`torque_speed` on `prius_2004` reports about 905 N·m peak torque with
`constant_l_envelope` (warn): the constant-inductance envelope ignores
saturation, and the ORNL dynamometer measured about 340 N·m
([validation report](validation-report.md)).

| Code | Models | Severity | Meaning |
|------|--------|----------|---------|
| `constant_l_envelope` | `torque_speed` | warn | Envelope ignores saturation; a `[flux_map]` gives the saturated envelope |
| `stall_saturation` | `stall_torque` | warn | Stall current far past rated; constant-L torque is likely optimistic |
| `linear_iron` | `fem` | info → warn | Linear iron; warn when `b_iron_max` exceeds 1.8 T, rerun with `nonlinear=true` |
| `constant_l_mtpa` | `rated_torque`, `stall_torque` | info | MTPA point without a `[flux_map]` covering the current |
| `constant_l_operating_point` | `operating_point`, `motor_thermal` | info | Operating point without a `[flux_map]` |
| `constant_l_machine` | `drive_sim`, `drive_sim_two_mass` | info | Constant-L machine model |
| `backemf_from_psi_f` | `analytical`, `analytical_axial`, `fem_axial` | info | Back-EMF comes from the stated `psi_f`, not the field solution |
| `k_end_heuristic` | `analytical` | info | End-effect factor is a heuristic, not a 3D value |
| `newton_fell_back` | `fem` | info | Nonlinear Newton solve fell back to Picard iteration |
| `im_linear_iron`, `im_ideal_ring`, `im_no_skew` | `im_fem_*` | info | Induction FEM idealizations |
| `budget_from_rated_current` | `thermal_duty` | info | S1 loss budget inferred from rated current |
| `limits` | every model | info | Omissions inherent to the model (e.g. 2D field solution) |
| `measured_data_available` | any | info | The store holds measured/published records for this motor; `validation_summary(motor)` gives the motor-level comparison against them |

The rules live in `phasesweep/trust.py`; the table above is a summary of them.
Measured and failed results carry no caveats. `measured_data_available`
appears only on results for a loaded config itself, not on sweep points,
which are different designs that keep the motor's name.

`validation_summary` reports one-sided bound checks (computed ≥ measured)
with the anchor's note on each row. A summary backed only by such bounds
reads "within bounds only", not "validated": an over-predicting model passes
a bound, as the Prius stall torque does.

`get_result` summarizes lists longer than 32 entries (waveforms, field grids)
as `{len, min, max, first, last}` unless `full=True`. `fields=[...]` returns
only the named metrics, unsummarized, and lists absent ones in
`missing_fields`.

## 6. Planning and cost

- **Dry run**: `submit_job(..., dry_run=True)` builds the job exactly as a
  real submit would (validation, sub-tasks, cache check) and returns per-model
  sub-task and cached counts and `estimate_s`, from the median stored run time
  of each model. `estimate_complete: false` means a model has no stored runs
  and the estimate is a lower bound. Dry-run any sweep that includes a slow
  model.
- **Sweep cap**: a sweep grid larger than `--max-sweep-points` (default 500,
  the product of axis steps) is rejected at submit.
- **Shared pool**: agent jobs queue behind dashboard jobs on the same
  `--workers` processes, in submission order.

## 7. Watching an agent

The dashboard's **Agent** tab shows:

- a design-space plot of every design the agent's jobs produced, on two chosen
  quantities (a swept dimension or a headline metric), joined in exploration
  order, with warn-caveat designs outlined in amber;
- each MCP tool call live, with arguments, duration, and outcome;
- each agent job as it finishes, with the first result's headline and its
  caveats.

![Agent tab during the §2 BMW i3 run: tool calls newest first, the fem job's
headline, and the get_job call held 33.5 s by wait_s](images/mcp_agent_tab.png)

## 8. Limits

- The job list, `sweep_table`, the activity log, and the design-space plot are
  in memory and reset when the server restarts. Stored results persist and stay
  readable through `query_results` / `get_result`.
- `fem` runs nonlinear in single, validate and sweep jobs unless
  `params.nonlinear` is `false`; the `linear_iron` caveat marks a linear run.
- Measured and published data attach to a motor by name. A config that reuses
  another motor's `name` inherits its measured rows in `validation_summary`.
- Stale model versions are hidden from `query_results` after a physics change;
  the records stay in the store file.

## 9. Agent cases

`tests/agent_cases/*.toml` define the expected tool responses for a question:
runnable or blocked models, headline values, and caveats.
`tests/test_agent_cases.py` replays the tool sequences over MCP without an LLM
on every CI run. It also checks that docs quoting the case (including the
Prius example in §5) state the same numbers. Add a case when a tool or caveat
changes what an agent sees.
