# Elmer independent-code verification — radial FEM

Replicates the radial `fem` model's 2D magnetostatic problem in Elmer
as an independent-code check of `phasesweep/solvers/fem_field.py` —
different mesh (structured polar vs unstructured OCC), different basis
(nodal Lagrange vs the NGSolve space), and an independently configured
nonlinear solve. **Not a shipped dependency** — nothing in
`phasesweep/` imports this; the deliverable is the pinned comparison
JSON in `pinned/`, re-checked by
`tests/crossval/test_elmer_radial_pinned.py` (no Elmer, no NGSolve at
test time). Pattern: `scripts/elmer_axial/`.

Shared parameters/tolerances/references:
`tests/crossval/radial_fem_contract.py`.

## Problem

Full 2π cross-section of the normalized Zhu baseline defined by
`tests/crossval/radial_fem_contract.py`. `MagnetoDynamics2D` (steady state) +
`MagnetoDynamicsCalcFields` + `SaveData/SaveLine`; homogeneous
Dirichlet A = 0 at r_outer; B_r sampled on the node ring at r_ag and
compared as the n_p-th harmonic (projection on a uniform resample).

Cases: smooth-bore linear, discrete arcs (α_p 0.8 / 0.95), nonlinear
BH (the fem solver's 19-point generic table), slotted open-circuit,
slotted + j_s armature current sheet.

**Rotation/cogging is deliberately excluded.** Exploratory
sliding-interface runs showed that the ripple was at the discretization
noise floor. A cogging cross-check would compare noise, not physics, until
a deliberate convergence study is done on both sides.

## Magnetization convention

Same as the axial study, verified there against the 1D circuit limit:
Elmer's constitutive relation is B = μ0·μr·(H + M), so
M = B_rem/(μ0·μr_pm) [A/m] — identical to the fem solver's coercive
field |H_c| = ν_PM·B_rem. `Magnetization i` lives in the **Material**
section (MATC on coordinates for the per-pole square-wave sign).

## SIF traps (all found the hard way)

1. `Magnetization` in a `Body Force` is **silently ignored** — solve
   completes trivially zero, exit 0 (axial study).
2. Wrong M convention (`B_rem/μ0`) produces the expected permeability-factor
   bias (axial study).
3. `H-B Curve` needs the `Variable "dummy" / Real Monotone Cubic /
   ... / End` block form; the sized-array form `H-B Curve(19,2)` fails
   to parse.
4. `Nonlinear System Max Iterations` **defaults to 1** — omit it and a
   nonlinear case silently runs one linear iteration.
5. (Rotation, excluded here but recorded:) `Mortar BC = Integer N`
   takes the master's **Boundary Condition section index**, not its
   `Target Boundaries` tag; wrong index → disconnected halves →
   exactly zero field, exit 0.

Traps 1 and 5 share a signature — exit 0, no warning, trivially zero
field — so the runner enforces the nontrivial-field floor defined by the
shared contract after every solve.

## Structured-mesh quantization

Magnet arcs and slots are θ-sector body assignment on a tensor-product
(r, θ) grid — no boolean geometry, so none of the OCC sliver failure
modes (`_MIN_ARC_GAP_M` has no analogue here). The cost: arc and slot
edges snap to element boundaries, so α_p and slot_width_ratio are
quantized by n_theta. The pinned artifact records the resulting difference;
the contract chooses cases and resolution so it remains within the gate.

## Files

- `mesh_writer.py` — polar structured mesh in ElmerSolver's native
  mesh-directory format (no ElmerGrid, no Gmsh). Bodies 1–6 = shaft,
  pm, airgap, stator, pm_gap, slot; boundary 1 = outer Dirichlet,
  boundary 2 = the r_ag sampling ring.
- `radial.sif.template` — SIF filled by the runner (`str.format`);
  iron block, nonlinear block, and body-force section are swapped in
  per case.
- `parse_saveline.py` — SaveLine ASCII → (θ, B_r), columns located by
  name from `.dat.names`; fundamental by projection.
- `run_elmer_radial.py` — orchestrates mesh → SIF → native
  `ElmerSolver` → parse → NGSolve comparison run → gates →
  `pinned/elmer_radial_cv.json`.
- `pinned/elmer_radial_cv.json` — pinned per-case B1/peak for both
  codes, analytical references, decimated waveforms.
- `cases/` — generated working dirs (gitignored).

## Toolchain pin

Results are workstation-pinned, like the axial ones. They were generated
against the **native** ElmerSolver from the CSC Ubuntu PPA; the exact
build identity is recorded in the pinned JSON. No Docker is involved;
any ElmerSolver carrying
`MagnetoDynamics2D`, `MagnetoDynamics` (CalcFields) and `SaveData`
should reproduce the numbers (the axial study's
`scripts/elmer_axial/Dockerfile` builds a suitable containerized one).
NGSolve side: the repo venv's `fem` extra at `maxh_fraction = 0.05`.

## Running

```sh
# from the repo root, end-to-end (needs ElmerSolver on PATH + fem extra):
.venv/bin/python scripts/elmer_radial/run_elmer_radial.py

# re-validate the pinned JSON against the contract (no Elmer, no NGSolve):
.venv/bin/python scripts/elmer_radial/run_elmer_radial.py --check
```

## When to regenerate

The pinned JSON carries both the Elmer and the NGSolve numbers, so it
goes stale on either side (the run IDs do not version-bust it, same as
`output/reference_br/`). Regenerate after any `fem` ModelInfo.version
bump, any change to `radial_fem_contract.py` parameters, or an
Elmer/NGSolve toolchain change — then re-run the pinned tests and
review the pinned artifact diff.

## Results

Exact per-case outputs, toolchain identity, waveforms, and analytical
references live in `pinned/elmer_radial_cv.json`. Acceptance gates live in
`tests/crossval/radial_fem_contract.py` and are enforced by
`tests/crossval/test_elmer_radial_pinned.py`. This README intentionally does
not duplicate that generated state.
