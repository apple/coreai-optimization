# Package Release Guide

The OSS release process for Core AI Optimization is being defined. This page will document the workflow for publishing to PyPI once the public release infrastructure is finalized.

Available locally:

```bash
make build      # build the canonical, publishable wheel + sdist (uv build --no-sources)
make build-dev  # build a timestamped dev wheel (e.g. 0.2.2.dev202607231430+abc1234)
make version    # show the development version carried on the tree (e.g. 0.2.2.dev0)
make clean      # remove build artifacts
```

## Version scheme

`main` always carries the version planned for the *next* release, so ongoing
development is never mistaken for an already-published version and a release
can be stabilized, tested, and published on its own branch independently of
later changes on `main`. (The release-branch workflow itself — branch naming,
tagging, backporting fixes to `main` — will be documented separately in the
release schedule doc; this section covers only the version string mechanics.)

Concretely, `main` carries the next planned release with a `.dev0` suffix in
`src/coreai_opt/_about.py` (e.g. `0.2.2.dev0`). `.dev0` is only ever an on-tree
marker — it never appears in a built wheel; both `make build` and
`make build-dev` remove it before building.

- `make build` strips the `.dev` suffix, e.g. `0.2.2.dev0` -> `0.2.2`. A
  release is cut on a release branch by dropping the `.dev0` suffix in
  `_about.py` and tagging `v0.2.2`; the release workflow verifies the tag
  matches `_about.py`.
- `make build-dev` replaces the suffix with a unique
  `.dev<UTC-timestamp>+<short-sha>` — used by contributors, smoke tests, and
  the nightly pipeline. `DEV_VERSION=<version>` uses that version exactly
  instead.

Sorting is preserved: `0.2.2.dev0 < 0.2.2.dev202607231430+abc1234 < 0.2.2`.

### Extending the scheme downstream

A repo that vendors this one under `external/` and includes this `Makefile` —
building a single combined wheel from both trees — can insert one extra
release segment by setting `COREAI_OPT_VERSION_EXTENSION=<segment>` (e.g.
`"1"`) before calling `make build` / `make build-dev` / `make version`, all
unchanged:

- on tree: `0.2.2.dev0` + extension `1` -> `0.2.2.1.dev0`
- `make build` -> `0.2.2.1`
- `make build-dev` -> `0.2.2.1.dev<UTC-timestamp>+<short-sha>`

There is still exactly one `_about.py` (this package's own, found at
`src/coreai_opt/_about.py` or `external/src/coreai_opt/_about.py`); the
extension is a plain string composed entirely in
`scripts/release/release_utils.apply_version_extension` — no downstream file,
package, or version-composition logic is involved.

<!-- TODO: Document the chosen OSS release workflow (PyPI trusted publishing, twine upload, or uv publish). -->
