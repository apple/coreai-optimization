# Package Release Guide

The OSS release process for Core AI Optimization is still being defined. This page will document the workflow for publishing to PyPI once the public release infrastructure is finalized.

The following commands are available locally:

```bash
make build      # build the canonical, publishable wheel + sdist (uv build --no-sources)
make build-dev  # build a timestamped dev wheel (e.g. 0.2.2.dev202607231430+abc1234)
make version    # show the development version carried on the tree (e.g. 0.2.2.dev0)
make clean      # remove build artifacts
```

## Version scheme

`main` always carries the version planned for the _next_ release. This ensures that ongoing development is never mistaken for an already-published version, and that a release can be stabilized, tested, and published on its own branch, independently of later changes on `main`. (The release-branch workflow itself — branch naming, tagging, and backporting fixes to `main` — will be documented separately in the release schedule doc; this section covers only the version-string mechanics.)

`src/coreai_opt/_about.py` stores `latest_released_version` (the last tagged release, e.g. `"0.2.1"`) and computes `__version__` from it by incrementing its last number by one and adding a `.dev0` suffix (e.g. `"0.2.2.dev0"`). A pre-commit hook (`check-about-version`) verifies that `__version__` always follows this rule and that `latest_released_version` matches the repo's latest release tag. As a result, `__version__` can never look as though a release has shipped when it hasn't. The `.dev0` suffix is only a marker on the tree; it never appears in a built wheel.

- `make build` builds the release that `__version__` implies, e.g. `0.2.2`. A release is cut by tagging it (`v0.2.2`); `latest_released_version` is then hard-coded to `"0.2.2"`, which bumps `__version__` to the next candidate (`0.2.3.dev0`).
- `make build-dev` builds that same release but with a unique `.dev<UTC-timestamp>+<short-sha>` suffix instead. It is used by contributors, smoke tests, and the nightly pipeline. `DEV_VERSION=<version>` uses that version exactly instead.

Sorting is preserved: `0.2.2.dev0 < 0.2.2.dev202607231430+abc1234 < 0.2.2`.

### Extending the scheme downstream

A repo that uses this one as a submodule and includes this `Makefile` — building one combined wheel from both trees — can add its own 4th number. Set `COREAI_OPT_VERSION_EXTENSION` to the number it's about to release next (e.g. `"1"` for its first release off a given OSS release, then `"2"` for the one after that). Then call `make build`, `make build-dev`, or `make version` unchanged:

- `latest_released_version` `"0.2.1"` + extension `"1"` -> candidate: `0.2.1.1.dev0`
- `make build` -> `0.2.1.1`
- `make build-dev` -> `0.2.1.1.dev<UTC-timestamp>+<short-sha>`

The extra number is used exactly as given (`scripts/release/release_utils.apply_version_extension`); `latest_released_version`'s own last number is only bumped for OSS's own `main`, when no extension is set.

There is still only one `_about.py` (this package's own); the extra number is a plain string handled entirely in `scripts/release/release_utils.next_release_base` — no other file or package is involved.

<!-- TODO: Document the chosen OSS release workflow (PyPI trusted publishing, twine upload, or uv publish). -->
