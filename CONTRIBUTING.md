# Contributing to Core AI Optimization

The Core AI Optimization source code is open source under the [BSD-3-Clause](LICENSE) license. Contributions are welcome within a defined scope — please read this document before opening a pull request or issue.

## How you can help

The API surface is intentionally limited. This keeps the library reliable, well-tested, and maintainable.

- Reporting bugs with clear, reproducible steps
- Improving documentation
- Adding or improving tests for existing functionality
- Bug fixes for existing behavior
- Minor enhancements to existing extension points

**Not in scope at this time:**

- Major new features or algorithms — the maintainers make deliberate decisions about what enters the library. If you want to propose a new algorithm or major capability, please open a [Feature request](../../issues/new) first. Maintainers will be transparent about whether it fits the roadmap.
- Changes to the core API surface

## Build

Set up the environment as described in [README.md](README.md#getting-started). Then, from the activated venv:

```shell
# Build the package.
make build

# Build the documentation.
make docs

# Build the documentation and open it in a browser.
make docs-open
```

All make targets and their flags are listed in the [Makefile](Makefile).

## Submitting issues

Before opening an issue:

- Search [existing issues](../../issues) to avoid duplicates
- Provide a clear, concise description with code examples where applicable
- For security issues, do **not** open a public GitHub issue. Follow the [Apple Open Source security disclosure process](https://github.com/apple/.github/blob/main/SECURITY.md).

## Testing

All contributions require tests.

- **Before submitting:** Ensure all existing tests pass locally. See [README.md](README.md) for instructions.
- **New functionality:** New features and bug fixes require corresponding automated tests.
- **Numerical accuracy:** For changes that affect model output, include a test that validates the result is correct.

## Style guide

The project follows the conventions in the [Code Style Guide](docs/contributing/code_style_guide.md). Highlights:

- Python 3.11+
- PEP 8 style, enforced by `ruff`
- 100-character line limit
- Google-style docstrings

Commit messages should follow the [Conventional Commits](https://www.conventionalcommits.org/) style.

## Running Tests

Before pushing your changes, run these locally:

- `pre-commit run --all-files` — formatters, linters, and license-header checks (also run automatically on `git commit`)
- `make check` — full lint and `mypy` type-check pass
- `make test` — full test suite (parallelized with `pytest-xdist`)
- `make test-fast` — excludes tests marked `@pytest.mark.slow` for quicker iteration
- `make test-smoke` — builds the package, installs it into a clean environment, and verifies that imports plus basic quantization and palettization work end to end

A clean `make check` and `make test` are required before a pull request will be reviewed.

## Response time

Issues and pull requests are addressed on a best-effort basis. Response times vary with team bandwidth and the scope of the contribution.

## Code of conduct

This project follows the [Apple Open Source Code of Conduct](https://github.com/apple/.github/blob/main/CODE_OF_CONDUCT.md). All community members are expected to adhere to these guidelines.
