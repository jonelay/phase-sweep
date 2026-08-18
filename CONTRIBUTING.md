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

## Running tests

```bash
uv run pytest
```

The full suite includes FEM solves and takes ~20 minutes. For a faster
feedback loop during development, run the non-slow tier (this is what
CI runs):

```bash
uv run pytest -m "not slow" --tb=short -q
```

## Linting and type checking

The project uses [Ruff](https://docs.astral.sh/ruff/) for linting and
import sorting, and [mypy](https://mypy-lang.org/) for type checking.
CI gates on both — check before submitting:

```bash
uv run ruff check .
uv run mypy
```

Auto-fix safe lint violations:

```bash
uv run ruff check --fix .
```

## Submitting changes

1. Open an issue first for anything beyond a small bug fix — this avoids
   wasted effort on changes that don't fit the project's direction.
2. Fork the repo and create a branch from `main`.
3. Make your changes, add or update tests as needed.
4. Ensure `uv run ruff check .`, `uv run mypy`, and `uv run pytest`
   all pass.
5. Open a pull request against `main`.

Keep PRs focused — one logical change per PR.

## Scope

phase-sweep is a PMSM simulation toolkit for stator/rotor design
exploration — field solvers (analytical and 2D FEM), drive simulation,
thermal-duty screening, torque and iron-loss models, sensitivity sweeps,
and a cross-validation CLI. Contributions that extend these capabilities
— bug fixes, new validation anchors, documentation improvements, and
performance work — are welcome.

Large-scope additions (new solver backends, integration with other
tools) should start as an issue discussion before any code is written.

## License

By contributing you agree that your contributions are licensed under the
[GNU Lesser General Public License v2.1](LICENSE).
