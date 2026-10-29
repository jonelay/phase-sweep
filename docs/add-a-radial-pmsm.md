# Add a radial PMSM

A step-by-step path from a new motor TOML to a clear record of the checks you
ran. Field-by-field schema reference:
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

Every value in it is labeled as an assumption. The starter intentionally
produces a saturation warning even though the analytical and FEM results agree
within about 1%: solver agreement does not mean the design is physically
sound. Replace assumptions with sourced values, and omit optional fields you
do not know instead of guessing. The checker reports which models each
omission disables. Once your file loads, the sourced motor files in
[`motors/`](../motors/) are better examples than the starter template; each
header records whether its values are measured, published, or estimated.

## 3. Run the staged check

```bash
phasesweep-check-motor motors/my_motor.toml
```

The checker reports structural errors, the relationship between `psi_f` and
`B_rem`, applicable models, fast-model results, and available provenance or
measurement data. It ends with this summary:

```text
CONFIG:      valid
EXECUTION:   runnable (4/4 fast models)
CONSISTENCY: ok (single source)
CROSS-CHECK: not run
MEASUREMENT: not assessed
```

`CONFIG: valid` means the file is well formed; it does not mean the motor is a
sound design. Voltage-headroom, saturation, and provenance warnings require
engineering judgment but do not make the command fail. Schema errors, solver
exceptions, and non-finite outputs produce a nonzero exit code.

## 4. Cross-check the field solution

```bash
phasesweep-check-motor motors/my_motor.toml --fem quick
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

When contributing a motor, state the highest level reached and what remains
unchecked:

| Level | Established by |
|---|---|
| `RUNNABLE` | the config loads and applicable models produce finite output |
| `INTERNALLY CONSISTENT` | redundant inputs agree, or the conflict is resolved by a declared evidence path |
| `NUMERICALLY CROSS-CHECKED` | independent solvers agree at declared settings |
| `MEASUREMENT-VALIDATED` | a registered dataset with provenance and committed tolerance passes `phasesweep-crossval` |

Each level includes the ones above it. Passing every check on this page means
the file is internally consistent and the software can solve it. It does not
show that the motor is manufacturable or that the models match measured
performance.
