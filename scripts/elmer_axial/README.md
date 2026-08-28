# Elmer independent-code verification (Track C)

Replicates Track A's axial-flux r_mean plane problem in Elmer 2D
magnetostatics as an independent-code check of `fem_axial`. **Not a
shipped dependency** — nothing in `phasesweep/` imports this; the
deliverable is the pinned comparison JSON in `pinned/`, re-checked by
`tests/crossval/test_elmer_axial_pinned.py` (no Docker at test time).

Shared tolerances/references: `tests/crossval/axial_fem_contract.py`.

## Problem

One-pole rectangle stack (unrolled at r_mean, SI metres): x =
circumferential over one pole pitch, y = axial —
rotor_yoke / (pm_n + interpole air + pm_s) / airgap / stator_yoke.
Slotless only (v1 of this study). Dirichlet A_z = 0 on the whole outer
boundary: the two magnet-center vertical edges (mirror symmetry) plus
the outer horizontal yoke backs — the same BCs as the contract.
Machine axial B_z is the plane's B_y = -dA_z/dx, sampled on the
mid-airgap line.

## Solver choice

`MagnetoDynamics2D` (steady state) + `MagnetoDynamicsCalcFields` +
`SaveData/SaveLine`. Chosen over the legacy `StatMagSolver` because it
shares the magnetization conventions of the 3D `MagnetoDynamics` family
and works with `MagnetoDynamicsCalcFields` for nodal B output.

### Magnetization convention (the known trap)

Two findings, both verified against the CV-2 1D circuit limit before
anything else was trusted:

1. `Magnetization i` must live in the **Material** section. Placed in a
   Body Force it is silently ignored — the solve completes "trivially
   zero" with no warning.
2. Elmer's constitutive relation is B = mu0·mu_r·(H + M), so the M that
   reproduces B_rem is `M = B_rem / (mu0 * mu_r_pm)` [A/m]. The
   alternative convention (`M = B_rem/mu0`) produces the expected
   permeability-factor bias; the circuit-limit gate distinguishes them.

Trap 1 is one of a *class*: the radial harness later found a second
silent-zero mode (`Mortar BC` given a section index instead of its
`Target Boundaries` tag — disconnected halves, exactly zero field,
exit 0), and NGSolve's 3D OCC path has an analogous face-naming trap.
Signature is always exit 0 + trivially zero field, which is why both
harnesses gate every pinned field on a nontrivial-magnitude floor
(`NONTRIVIAL_FIELD_T` / `NONTRIVIAL_PEAK_T` in the contracts). Full
list: `scripts/elmer_radial/README.md` "SIF traps".

## Files

- `mesh_writer.py` — writes the structured quad mesh directly in
  ElmerSolver's native mesh-directory format (no ElmerGrid). Body ids
  1–6 = rotor_yoke, pm_n, air_interpole, pm_s, airgap, stator_yoke;
  boundary 1 = outer Dirichlet.
- `axial_pole.sif.template` — SIF filled by the runner (str.format).
- `parse_saveline.py` — SaveLine ASCII → (x, B_z), fundamental B1 by
  mirror-extension + projection (same convention as the contract),
  CV-2 plateau extraction.
- `run_elmer_axial.py` — orchestrates mesh → SIF → `docker run` →
  parse → gate comparison → `pinned/elmer_axial_cv.json`.
- `pinned/elmer_axial_cv.json` — pinned CV numbers + decimated
  mid-airgap waveforms ({x, B_z, B1} per case).
- `Dockerfile` — builds `elmer-axial:local` from the official CSC
  Ubuntu PPA (no pullable official image; see below).
- `cases/` — generated working dirs (gitignored).

## Docker image (pinned)

The committed `Dockerfile` builds a local image from the official CSC Ubuntu
PPA binaries. The pinned result records the exact solver identity used for
generation.

Any image whose `ElmerSolver` carries `MagnetoDynamics2D`,
`MagnetoDynamics` (CalcFields) and `SaveData` will reproduce the
numbers; pass it via `--image`.

## Running

```sh
# from the repo root, end-to-end (needs Docker):
.venv/bin/python scripts/elmer_axial/run_elmer_axial.py

# re-validate the pinned JSON against the contract (no Docker):
.venv/bin/python scripts/elmer_axial/run_elmer_axial.py --check
```

## When to regenerate

The pinned JSON is not version-busted by the run-ID hashing (same
policy as `output/reference_br/`). Regenerate after any `fem_axial`
ModelInfo.version bump, any change to `axial_fem_contract.py`
parameters, or an Elmer toolchain change — then re-run the pinned
tests and review the pinned artifact diff.

## Results

Exact outputs, waveforms, diagnostics, and toolchain identity live in
`pinned/elmer_axial_cv.json`. Acceptance gates live in
`tests/crossval/axial_fem_contract.py` and are enforced by
`tests/crossval/test_elmer_axial_pinned.py`. This README intentionally does
not duplicate that generated state.

CV-2 geometry note: the circuit reference assumes effectively infinite
iron permeability, so the contract chooses diagnostic yokes and permeability
that isolate the magnetization convention rather than the yoke model.
