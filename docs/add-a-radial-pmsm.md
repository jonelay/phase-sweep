# Add a radial PMSM

A procedural path from a new motor TOML to an honest statement of what has
been checked. Field-by-field schema reference:
[motor-toml.md](motor-toml.md). This page covers the geometry-led path; it
does not teach motor design.

## 1. Choose the evidence path

- **Geometry-led** (this page): supply geometry, winding, and magnet
  properties such as `B_rem`; omit `psi_f` so the circuit tier derives it.
- **Measurement-led**: supply measured `psi_f` with provenance; an optional
  `B_rem` then participates only in field calculations, and a disagreement
  between the two is reported as a warning, not an error.

Supplying both `psi_f` and `B_rem` independently is legal but couples two
sources of truth. If they disagree by more than 50% the tools warn on every
run that resolves them.

## 2. Write the TOML

Copy the intentionally hypothetical starter:

```bash
cp motors/examples/starter_radial_spm.toml motors/my_motor.toml
```

Every value in it is a labeled assumption, and its header states what the
staged check is expected to report (including a saturation warning that
coexists with ~1% solver agreement - the false-confidence trap this
workflow exists to avoid). Replace assumptions with sourced values and
keep unknown optional fields out rather than guessing; the checker
reports what each omission disables. The bundled [`motors/`](../motors/)
files are real measured motors with provenance and make better reading
than templates once your file loads.

## 3. Run the staged check

```bash
uv run python scripts/check_motor.py motors/my_motor.toml
```

This reports, in order: load and structural checks, `psi_f`/`B_rem`
authority and derived values, model applicability (with the reason each
inapplicable model is skipped), fast-model results with finite-output
checks, provenance presence, and a five-dimension status summary:

```text
CONFIG:      valid
EXECUTION:   runnable (4/4 fast models)
CONSISTENCY: ok (single source)
CROSS-CHECK: not run
MEASUREMENT: not assessed
```

A `CONFIG: valid` result establishes schema and objective invariants only.
Warnings (voltage headroom, saturation, missing provenance) are engineering
concerns to resolve by judgment; they do not fail the check. The exit code
is nonzero only for objective failures: schema errors, solver exceptions,
non-finite outputs.

## 4. Cross-check the field solution

```bash
uv run python scripts/check_motor.py motors/my_motor.toml --fem quick
```

This runs a deliberately coarse no-load FEM solve (settings are printed)
and compares the air-gap fundamental against the analytical model. Close
agreement is a numerical cross-check between two implementations of the
same idealization. It is not physical validation, and it can coexist with
severe stator saturation; the checker reports peak stator flux density
separately for exactly that reason.

## 5. Run the repository gates

```bash
uv run pytest tests/machines/test_configs.py -q
uv run pytest -m "not fem and not slow" --tb=short -q
uv run ruff check phasesweep/ scripts/ tests/
uv run mypy
```

## 6. State the evidence level

A contribution should say the highest level actually reached and what is
missing, using this vocabulary:

| Level | Established by |
|---|---|
| `RUNNABLE` | the config loads and applicable models produce finite output |
| `INTERNALLY CONSISTENT` | redundant inputs agree, or the conflict is resolved by a declared evidence path |
| `NUMERICALLY CROSS-CHECKED` | independent solvers agree at declared settings |
| `MEASUREMENT-VALIDATED` | a registered dataset with provenance and committed tolerance passes `phasesweep-crossval` |

Each level includes the ones above it and claims nothing more. In
particular: passing every check on this page means the file is internally
consistent and the software can solve it. It does not establish that the
motor is manufacturable or that the models predict measured performance.
