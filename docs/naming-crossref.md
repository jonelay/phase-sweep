# Concept Analysis: Naming Across Domains

> **Historical reference.** The collisions below are resolved; identifiers
> shown as "current code" may have since been renamed. For current
> conventions, see [glossary.md](glossary.md).

How phase-sweep names things vs its three upstream references:
**Zhu & Howe 2002** (physics), **motulator** (drive simulation), **NGSolve** (FEM).

The goal is to identify where the same name means different things,
where the same thing has different names, and where names can be aligned.

---

## 1. Radii

The highest-risk naming collision in the codebase. Three domains use
overlapping symbols for different physical boundaries.

### Who calls what what

| Physical quantity | Zhu & Howe | motulator | phase-sweep code (pre-migration) | status |
|---|---|---|---|---|
| Stator outer radius | — | — | `_R_S` | removed → `Geometry.r_outer` |
| Stator bore (airgap boundary) | `R_s` | — | `_R_SI` | removed → `Geometry.r_stator` |
| Magnet outer radius | `R_m` | — | `_R_RO` | removed → `Geometry.r_magnet` |
| Rotor iron radius | `R_r` | — | `_R_RI` | removed → `Geometry.r_rotor` |
| Air-gap center | (derived) | — | `_R_AG` | removed → `Geometry.r_ag` (derived) |
| Stator bore (in tests) | `R_s` | — | `Rs` | `test_analytical.py` passim |
| Stator resistance (ohms) | — | `R_s` | `R_s` | current — `machines/configs.py` [circuit] |

Key collisions:

- **`R_s`** = stator bore radius (Zhu) vs stator winding resistance (motulator).
  phase-sweep uses it in the motulator sense in configs, but the tests use `Rs`
  in the Zhu sense.
- **`_R_S` ≠ `Rs`** in phase-sweep itself: `_R_S` = 1.00 (stator outer),
  `Rs` in tests = 0.70 (stator bore = `_R_SI`). They look like the same thing
  but refer to different boundaries.

### Resolution path

The spec's Geometry dataclass uses descriptive names that sidestep the collision:

| Geometry field | Zhu equivalent | FEM constant to replace |
|---|---|---|
| `r_stator` | `R_s` (bore) | `_R_SI` |
| `r_magnet` | `R_m` | `_R_RO` |
| `r_rotor` | `R_r` | `_R_RI` |
| `r_inner` | — (shaft) | `_R_RI` (outrunner: hollow) |
| `r_outer` | — | `_R_S` |

motulator's `R_s` (resistance) stays in the electrical domain and never
touches geometry code.

---

## 2. Pole pairs

| Concept | Zhu & Howe | motulator | phase-sweep prod | phase-sweep tests |
|---|---|---|---|---|
| Pole pair count | `p` | `n_p` | `n_p` | `npp` |
| Harmonic order (n × p) | `np` | — | `n` (local alias) | `npp` (same var) |

- `n_p` follows motulator. Correct for the data model.
- `npp` in tests avoids collision with `import numpy as np`. Reasonable.
- `n = n_p` inside `zhu_howe_Br` follows the paper's math notation for the
  harmonic index in `cos(nθ)`.

No changes needed. Three domains, three valid reasons for the variation.

---

## 3. Magnetization and remanence

| Concept | Zhu & Howe | motulator | phase-sweep prod | phase-sweep tests |
|---|---|---|---|---|
| PM remanence (T) | `B_r` | — | `B_rem` | `B_rem` (was `B_REM`; renamed Phase 1) |
| Magnetization (A/m) | `M` = `B_r / μ₀` | — | — | `M_n` = `B_rem / MU0` |
| Radial flux density waveform | `B_r(r,θ)` | — | `B_r` (array) | `Br` (local) |
| PM flux linkage (Vs) | — | `psi_f` | `psi_f` | — |

- Zhu uses `B_r` for both remanence and radial field component (disambiguated
  by context in the paper).
- phase-sweep's `B_rem` is an improvement — unambiguous in code. Keep it.
- `B_REM` in `TestPaperFig5` was renamed `B_rem` for consistency (Phase 1).
- `M_n` in tests follows the paper's convention for reference implementations.
  The bridge `M_n = B_rem / MU0` is the explicit conversion.
- motulator never sees remanence — it works in the `psi_f` (flux linkage) domain.
  `_derive_B_rem` in `solvers/analytical.py` bridges from `psi_f` to `B_rem`.

---

## 4. Slot count

| Concept | Zhu & Howe | motulator | phase-sweep data model | phase-sweep local vars |
|---|---|---|---|---|
| Number of stator slots | `Q_s` | — | `n_slots` | `Q` |

- `n_slots` follows the motulator `n_` prefix convention for counts (`n_p`).
- `Q` is the standard motor design shorthand (from Zhu's `Q_s`).

Current rule: use `n_slots` (via `geometry.n_slots`) on data structures
and in implementation code throughout.

---

## 5. Configuration and parameter types

| Concept | motulator pattern | phase-sweep (current) | Notes |
|---|---|---|---|
| Machine physics params | `*Pars` dataclass | `Motor` frozen dataclass | motulator: `SynchronousMachinePars` |
| Control/run config | `*Cfg` dataclass | `RunConfig` frozen dataclass | motulator: `CurrentVectorControllerCfg` |
| Drive operating point | — | `DriveParams` frozen dataclass | U_DC, MAX_I_S, W_REF, I_LIMIT |
| Validated solver params | — | `AnalyticalParams`, `FemParams`, `DriveSimParams` | Factories: `prepare_analytical()` etc. |

The three-tier split (Geometry → Motor → RunConfig) replaces the earlier
`MotorConfig` TypedDict, `MotorSweepConfig` dataclass, and
`FullMotorConfig`, aligning with motulator's Pars/Cfg separation. Solver
param factories (`solver_params.py`) validate Motor fields at the solver
boundary.

---

## 6. Mechanical and electrical quantities

| Concept | Zhu & Howe | motulator | phase-sweep (current) |
|---|---|---|---|
| Mechanical rotor speed | — | `w_M` | `w_M` |
| Electrical rotor speed | — | `w_m` | `w_m` |
| Electromagnetic torque | — | `tau_M` | `tau_M` |
| Stator current magnitude | — | `i_s_ab` (complex) | `i_s = \|i_s_ab\|` |
| Moment of inertia | — | `J` | `J` |
| DC bus voltage | — | `u_dc` | `U_DC` |

motulator convention: uppercase `_M` = mechanical, lowercase `_m` = electrical.

Code uses `w_M` for mechanical and `w_m` for electrical, matching
motulator exactly; earlier code inverted `w_m` and `w_e`.

---

## 7. Abstract radii in the A-formulation

The A-formulation solver uses topology-agnostic boundary labels
(works for both inrunner and outrunner without changing the math).

| Abstract variable | Role | Inrunner mapping | Outrunner mapping |
|---|---|---|---|
| `s` | Iron boundary (evaluation side) | `R_s` (stator bore) | `R_s` (inner iron) |
| `q` | PM / air-gap interface | `R_m` | `R_m` |
| `p` | Far iron boundary | `R_r` (rotor iron) | `R_r` (outer iron) |

In `zhu_howe_Br` (`phasesweep/solvers/analytical.py`): `p, q, s = r_rotor, r_magnet, r_stator`.

In `aform_Br` (`tests/solvers/test_analytical.py`): parameters are `s, q, p` with
docstring "s=iron BC, q=PM/airgap, p=far iron BC".

These should stay abstract — renaming to `R_s, R_m, R_r` would break the
topology-agnostic design.

---

## 8. Reluctivity / permeability

| Concept | Zhu & Howe | NGSolve convention | phase-sweep |
|---|---|---|---|
| Permeability of free space | `μ₀` | — | `_MU0` |
| Relative perm (iron) | `μ_r` (∞ assumed) | — | `Motor.mu_r_fe` (was `_MU_R_FE`) |
| Relative perm (magnet) | `μ_r` ≈ 1.05 | — | `Motor.mu_r_pm` (was `_MU_R_PM`) |
| Reluctivity function | — | `nu` or `mur` | `_bh_nu(B)`, `nu`, `nu_cf`, `gfu_nu`, `nu_pm` |

Zhu assumes infinite iron permeability (no saturation). The FEM code relaxes
this via the B-H curve and Picard iteration.

### Forms of `nu` in `solve_field_fem`

| Name | Type | What it is |
|---|---|---|
| `nu_cf` | `dict[str, float]` | Material name → reluctivity value |
| `nu` | `CoefficientFunction` | Piecewise reluctivity on mesh (from `mesh.MaterialCF`) |
| `nu_pm` | `float` | Scalar reluctivity of PM region |
| `gfu_nu` | `GridFunction` | Element-wise reluctivity (Picard iteration) |
| `_bh_nu` | function | `ν(|B|)` from B-H curve lookup |

NGSolve convention is minimal: `nu` for the CoefficientFunction, `gfu_*`
prefix for GridFunctions. Our naming follows this.

---

## 9. Result types

| Name | Type | Contains | Location |
|---|---|---|---|
| `RunResult` | dataclass (not frozen) | Config + status + metrics from any run | `sweep_types.py` |
| `ResultStore` | class | JSONL persistence layer | `result_store.py` |
| `MeasuredResult` | frozen dataclass | Imported lab/published data with comparison metadata | `validation/measured.py` |

`RunResult` (originally `SweepResult`) is the single result type for all
run types. Its `model` field (registry key)
identifies the runner; legacy `run_type` values are mapped via
`_resolve_model()` during deserialization. Two *new* types later reused the
freed-up names for different purposes: `SlimResult` (index entry NamedTuple,
`result_store.py`) and `SweepResult` (geometry-sweep container,
`geo_parallel.py`).

---

## 10. NGSolve object naming

Our code follows standard NGSolve conventions:

| Object type | NGSolve convention | Our usage | Status |
|---|---|---|---|
| FE space | `fes` | `fes`, `fes_nu`, `fes_L2` | Aligned |
| GridFunction | `gfu` | `gfu`, `gfu_nu`, `gfu_Bmag`, `gfu_Br` | Aligned |
| BilinearForm | `a` | `a` | Aligned |
| LinearForm | `f` or `lf` | `lf` | Aligned |
| Trial / test | `u`, `v` | `u`, `v` via `fes.TnT()` | Aligned |
| Material CF dict | (no convention) | `*_cf` suffix | Our invention, clear |
| Mesh regions | lowercase strings | `"stator"`, `"airgap"`, `"pm"`, `"shaft"` | Aligned |

No changes needed in this domain.

---

## 11. `cfg_name` — resolved

`build_sim` signature — neither `cfg_name` nor `MotorConfig` appears:
`build_sim(params: DriveSimParams, plan: SimPlan | None = None, *, torque_ref=None)`.

---

## Summary: naming collisions

| Collision | Resolution |
|---|---|
| `R_s` resistance vs radius | Geometry dataclass with `r_stator` etc. separates domains |
| `_R_S` ≠ `Rs` (different radii, similar names) | Descriptive names replace abbreviations |
| `w_m` mechanical vs electrical | Follows motulator: `w_M` mechanical, `w_m` electrical |
| `config`/`cfg` for 3+ types | Three-tier split (Geometry/Motor/RunConfig) |
| `Q` / `n_slots` dual naming | `Q` locals removed; `n_slots` throughout |
| `B_r` waveform vs `B_rem` remanence | Resolved by the `B_rem` choice |
| `n_p` / `npp` / `n` | Three domains, three valid reasons |
| `SweepResult` for non-sweeps | Renamed to `RunResult` |
| `p, q, s` abstract radii | Intentionally abstract, should stay |
| NGSolve naming | Already aligned |

