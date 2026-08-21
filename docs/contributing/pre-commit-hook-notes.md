# Pre-Commit Hook Notes

Notes on pre-commit hooks whose addition or modification has follow-on rules. **For the exact configuration of any hook (entry, args, file regex), read `.pre-commit-config.yaml` (root) or `external/.pre-commit-config.yaml` (its OSS-export mirror) directly — they are the single source of truth, and the two are intentional mirrors.**

## `add-license-header`

Inserts or refreshes the BSD-3-Clause license header on every file the hook's `files:` regex matches.

### Adding a source file with a new extension

When you add a source file whose extension or name isn't already covered by the hook's `files:` regex, extend the configuration so the file gets a license header. Either:

- Widen the `files:` regex in **both** `.pre-commit-config.yaml` mirrors, and add comment-style support to the script if the new extension needs a non-`#` style.
- Or, if the file genuinely shouldn't carry a header (binary, data, vendored, generated), leave the regex alone and add a one-line rationale to the PR description.

### Why an inclusion list, not an exclusion list

New file types stay silently ignored until someone deliberately adds them, rather than getting `# header` prepended (which would break parsers like JSON).

## `check-api-doc-coverage`

Fails the commit when the committed `docs/src/api/index.md` no longer matches what `docs/scripts/generate_api_index.py` produces from the package tree. The point is that a new or renamed public symbol can't ship undocumented, and that an API-surface change shows up in the diff a reviewer reads.

The hook only reports the drift — run `make render-api-index` and stage the result.

### `docs/src/api/index.md` is generated; don't edit it

The file is committed but not hand-written. Two things write it: `make render-api-index` (the cheap path, base venv) and `make docs`, whose `setup()` in `docs/src/conf.py` regenerates it before Sphinx reads sources. Any manual edit is silently overwritten by the next build, and the hook rejects it in the meantime.

`conf.py` writes it through `_write_if_changed` rather than `write_text`, so a build that changes nothing leaves the file's mtime alone. Keep it that way: an unconditional write would make every docs build touch a tracked file, so the tree looks dirty to git's stat cache even when the generated content is identical.

### Adding a public symbol

Nothing extra to do — add the symbol to its module's `__all__`, run `make render-api-index`, and commit the regenerated index alongside the code. The hook's `files:` regex already covers `src/coreai_opt/**.py`, so it runs whenever the API surface can have moved.

### Why the hook runs one test, not the module

The hook entry names `docs/tests/test_api_doc_coverage.py::test_api_doc_coverage` specifically. The other test in that module, `test_autodoc_skip_filters_external_methods`, imports `conf.py` and therefore needs Sphinx, which lives only in `.venv-docs` — it runs under `make test-docs` in CI instead. Widening the hook to the whole module would break commits for anyone who hasn't built the docs environment.
