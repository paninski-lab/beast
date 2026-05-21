# Contributing

We welcome community contributions to the BEAST repo! 
If you have found a bug or would like to request a minor change, please 
[open an issue](https://github.com/paninski-lab/beast/issues).

In order to contribute code to the repo, please follow the steps below.

We strive to maintain a fun and inclusive environment for our users and contributors.
See our [code of conduct](CODE_OF_CONDUCT.md) for more information.

## Development setup

[Fork](https://guides.github.com/activities/forking/#fork) the repo, then clone your fork and
install in editable mode with dev dependencies:

```bash
pip install -e ".[dev]"
pre-commit install
```

`pre-commit install` registers the ruff linting hook so it runs automatically on each commit.
You only need to do this once per clone.

## Running the tests

```bash
pytest
```

## Linting

We use [ruff](https://docs.astral.sh/ruff/) for linting and import sorting.
The pre-commit hook runs it automatically, but you can also run it manually:

```bash
ruff check beast tests          # check for violations
ruff check --fix beast tests    # auto-fix where possible
```

Configuration lives in `[tool.ruff]` in `pyproject.toml`.

## Type checking

We use [pyright](https://github.com/microsoft/pyright) for static type checking:

```bash
pyright beast
```

Configuration lives in `[tool.pyright]` in `pyproject.toml`.
Pyright runs in CI on every pull request but is not in the pre-commit hook (it
checks the whole project on every run, adding ~15s per commit).

## Pull requests

- Keep PRs focused — one feature or fix per PR
- Include tests for new functionality
- Ensure `pytest`, `pyright` and `ruff check` both pass before opening the PR
- Open against the `main` branch