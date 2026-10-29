# Contributing to phase-sweep

Thanks for your interest in contributing. This guide covers the basics.

## Reporting bugs and requesting features

Open an issue on [GitHub Issues](https://github.com/jonelay/phase-sweep/issues).
Include enough detail to reproduce the problem — motor TOML, solver
settings, and the full traceback if there is one. Please also paste
your environment info:

```bash
uv run python -c "import phasesweep; phasesweep.debug_info()"
```

## Development setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/jonelay/phase-sweep.git
cd phase-sweep
uv sync --extra all --extra test --extra dev
```

Some validation tests use a dataset that is not distributed with the repo.
To fetch it (requires network access):

```bash
uv run python scripts/fetch_creator_dataset.py
```

Tests that depend on this dataset are skipped automatically when it is
absent. FEM tests require NGSolve, which is installed via the `all`
extra above. They are skipped automatically if NGSolve is not available.

The FEM solvers are also cross-checked against ElmerFEM. Those tests
(`tests/crossval/test_elmer_*_pinned.py`) validate committed comparison
JSONs and need no Elmer install — they always run. Only regenerating
the JSONs requires ElmerSolver; see `scripts/elmer_radial/README.md`
and `scripts/elmer_axial/README.md`.

## Running tests

Start with the core lane, the same selector the pull-request CI lane
uses:

```bash
uv run pytest -m "not fem and not slow" --tb=short -q
```

The full suite is the FEM/release gate:

```bash
uv run pytest
```

Expect it to run an order of magnitude longer than the core lane. The
`fem`-marked tests under `tests/solvers/` and `tests/crossval/` dominate
the runtime and can be silent for minutes per test while a solve
converges; a quiet interval is not necessarily a hang.

## Linting and type checking

The project uses [Ruff](https://docs.astral.sh/ruff/) for linting and
import sorting, and [mypy](https://mypy-lang.org/) for type checking.
CI gates on both — check before submitting:

```bash
uv run ruff check phasesweep/ scripts/ tests/
uv run mypy
```

Auto-fix safe lint violations:

```bash
uv run ruff check --fix phasesweep/ scripts/ tests/
```

## Submitting changes

1. Open an issue first for anything beyond a small bug fix — this avoids
   wasted effort on changes that don't fit the project's direction.
2. Fork the repo and create a branch from `main`.
3. Make your changes, add or update tests as needed.
4. Ensure `uv run ruff check phasesweep/ scripts/ tests/`,
   `uv run mypy`, and `uv run pytest` all pass.
5. Open a pull request against `main`.

Keep PRs focused — one logical change per PR.

## AI-assisted development

Contributors may use AI tools provided they review, understand, and accept
responsibility for the resulting contribution. Material assistance should be
disclosed in the pull-request description or with a non-authorship commit
trailer, which records tool use without assigning authorship:

```text
Assisted-by: <tool>
```

## Scope

phase-sweep is a PMSM and induction-machine simulation toolkit for stator/rotor
design exploration — field solvers (analytical and 2D FEM), induction
equivalent-circuit and FEM models, drive simulation,
thermal-duty screening, torque and iron-loss models, sensitivity sweeps,
and a cross-validation CLI. Contributions that extend these capabilities
— bug fixes, new validation anchors, documentation improvements, and
performance work — are welcome.

Large-scope additions (new solver backends, integration with other
tools) should start as an issue discussion before any code is written.

## License

This project is licensed under the [Apache License 2.0](LICENSE).
By contributing you agree that your contributions are licensed under
the same terms.
