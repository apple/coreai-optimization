# CLAUDE.md

## First Principles

@agents/llm-first-principles.md

## Project Overview

@README.md

## Development Commands

- `make env` — set up the development environment (one-time; reuses an existing venv if present)
- `make test` — run all tests
- `make test-fast` — run tests excluding `@pytest.mark.slow`
- `make check` — run linting and code-quality checks
- `pytest -n auto path/to/test.py` — run a test file
- `pytest path/to/test.py::test_name` — run a single test

## uv usage

Always pass `--no-sync` to `uv run`: `uv run --no-sync --active …`.

`uv run` implicitly syncs the active project to its default-groups before running, which re-resolves dependencies and can clobber a venv's group-pinned packages — e.g. the torch pin in `.venv-lowest-torch`/`.venv-highest-torch` gets re-anchored back to the default torch. Our Make targets always prepare the environment first via `use_env`/`setup_env.sh`, so by the time `uv run` executes the deps are already correct. A `uv run` invocation should be a read-only run of a command in that prepared env, never a dependency mutation — `--no-sync` enforces that.

## Editing Guidelines

- Use `@path` to reference small files (loaded into every session automatically).
- For large documents, describe what they contain and where to find them. Claude reads them on demand when needed.
- Subdirectory `CLAUDE.md` files load only when Claude touches files in that directory — use them for area-specific rules that don't belong in the root file.
- When you discover a useful pattern, preference, or insight during a session, ask the user if they want to save it to `CLAUDE.local.md` for future sessions.

## Code Style

@docs/contributing/code_style_guide.md

## Language Specific Rules

### Markdown (.md files)

@agents/markdown_rules.md

## Additional Documents

- See `CONTRIBUTING.md` for the contribution workflow.
- See `docs/contributing/pre-commit-hook-notes.md` for the rules around individual pre-commit hooks (e.g. extending the license-header `files:` regex when adding a source file with a new extension).
- See `pyproject.toml` for Python/PyTorch version constraints, dependencies, build settings, and tool configurations (mypy, ruff, pytest).
- See `Makefile` for available make targets and their implementations.
