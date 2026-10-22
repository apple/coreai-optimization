# Package Release Guide

Releases are cut and published by the [release managers team](https://github.com/orgs/apple/teams/coreai-optimization-release-managers).

The following commands are available locally:

```bash
make build      # build the artifacts for the current release to be published
make build-dev  # build a local development wheel (e.g. 1.1.0.dev202607231430+abc1234)
make build-nightly # build the nightly wheel published to PyPI as coreai-opt-nightly (e.g. 1.1.0.dev202607231430)
make version    # show the version a release would publish (e.g. 1.1.0)
make version-dev # show the development version carried on the tree (e.g. 1.1.0.dev0)
make clean      # remove build artifacts
```

## Version scheme

`main` always carries the version of the *next* release, never the one that already shipped.

There are four version formats:

| Format                                 | Example                         | What it is                                           |
| -------------------------------------- | ------------------------------- | ---------------------------------------------------- |
| `X.Y.Z`                                | `1.0.0`                         | a published release                                  |
| `X.Y.Z.dev0`                           | `1.1.0.dev0`                    | the version `main` carries in the repo               |
| `X.Y.Z.dev<UTC-timestamp>`             | `1.1.0.dev202607231430`         | a nightly, published to PyPI as `coreai-opt-nightly` |
| `X.Y.Z.dev<UTC-timestamp>+<short-sha>` | `1.1.0.dev202607231430+abc1234` | a local dev artifact, built from a specific commit   |

Say `1.0.0` has just been released. `main` then carries `1.0.1.dev0`. That reads as "working toward a release after `1.0.0`, which has not shipped": the `.dev0` suffix marks the tree as unreleased and never appears in a built wheel. Nothing on `main` can be mistaken for a published version.

`main`'s `.dev0` always defaults to the last digit plus one, so after `1.0.0` it is `1.0.1.dev0`. Once the version of the next release is known — usually a minor — a PR sets `__version__` to it before the release branch is cut.

The flow below traces one cycle. At the cut, `main` and the release branch diverge and never rejoin: the branch keeps the version it was cut with, and only `main` moves on.

```mermaid
---
title: Version change flow
---
flowchart TB
    prev["previous release schedule"] --> m1["main: 1.0.1.dev0<br/>placeholder, last digit + 1"]
    prev --> r0("1.0.0 released")
    m1 -->|"ready for release"| ask{"is the placeholder the<br/>version we want?"}
    ask -->|"No, usually a minor"| pr["PR sets __version__"]
    ask -->|Yes| fin
    pr --> fin["main: 1.1.0.dev0<br/>assume we release 1.1.0;<br/>it could also stay 1.0.1"]
    fin --> nxt["main: 1.1.1.dev0<br/>placeholder, last digit + 1"]
    fin -->|cut| rb["release/1.1.0<br/>set latest_released_version<br/>and __version__ = 1.1.0"]
    rb --> stab["stabilize"]
    stab -->|"tag v1.1.0"| rel("1.1.0 released")
    nxt --> nextsched["next release schedule"]
    rel --> nextsched
    nextsched -.->|"the process repeats"| prev
```

`src/coreai_opt/_about.py` holds the last released version and `__version__`. Both are set together in one PR when `main` moves forward: the last released version becomes the release just branched, and `__version__` becomes the one after it. The `check-about-version` pre-commit hook enforces the relation between them: `__version__` must be the last released version with exactly one of its numbers raised by one, every number after that reset to zero, and `.dev0` on the end. From `1.0.0` it accepts `1.0.1.dev0`, `1.1.0.dev0`, or `2.0.0.dev0`, and nothing else. Any other value fails the commit, and the message prints the accepted ones, so there is nothing to work out by hand.

- `make build` builds the artifacts for the current release to be published.
- `make build-nightly` builds the nightly wheel published to PyPI as `coreai-opt-nightly`, carrying a unique `.dev<UTC-timestamp>` suffix.
- `make build-dev` builds local wheels for development and testing, each carrying a unique `.dev<UTC-timestamp>+<short-sha>` suffix.

Therefore, we have the following order:

```text
1.1.0.dev0 < 1.1.0.dev202607231430 < 1.1.0.dev202607241430+abc1234 < 1.1.0
```

This is the order we want. `1.1.0.dev0` is the bare marker `main` carries, so it sorts below every wheel actually built for `1.1.0`. Each nightly and dev wheel sorts above it, and above the ones built before it, because the timestamp only grows. The published `1.1.0` sorts highest of all, so installers pick it over any dev wheel.

A release branch is the one place where the two match: it sets both `latest_released_version` and `__version__` to the release it produces, with no `.dev0` — both read `1.1.0` on `release/1.1.0`. On `main` they always differ, which is what tells a release branch apart — and what lets a repo that vendors this one pin to a release branch and still resolve the right baseline. When a commit touching `_about.py` is made on `release/<version>`, `check-about-version` requires both fields to equal `<version>`.

`__version__` must always be a literal string, never an expression.

### Release branches

1. `release/<version>` is created from `main`, once the version of the next release has been decided — that is, which digit gets one added to it. Its first commit sets `latest_released_version` and `__version__` to that version, with no `.dev0`, so the branch names its own release.
2. The tag is created on the `release/<version>` branch, never on `main`.
3. After the cut, `main` continues on to the next release's `.dev0`.
4. The `check-about-version` pre-commit hook enforces the version rules on every commit that touches `_about.py`.
5. After the cut, the release branch takes only the commits the procedure calls for, plus any must-fix issue. Those commits are later cherry-picked back to `main`.

Cut the branch before moving `main` to the next dev release.

### Nightly builds

`.github/workflows/nightly.yml` publishes a nightly wheel to PyPI as its own project, [`coreai-opt-nightly`](https://pypi.org/project/coreai-opt-nightly/), from `main` every day at 08:23 UTC (00:23 PST, 01:23 PDT). The `coreai-opt` release history therefore lists only releases. It runs `make build-nightly` and publishes the wheel via Trusted Publishing. A nightly is a snapshot of `main`, published every night; the smoke tests run on releases (`release.yml`). A scheduled run builds only when `main` has moved since the last successful nightly. The commit a nightly was built from is the head SHA of its workflow run.

A manual run (Actions → Nightly → Run workflow, or `gh workflow run nightly.yml --ref <branch>`) always builds, and publishes only when started on `main`. On any other branch it is a dry run.

Publishing relies on two settings outside the repo, each set up once:

1. On GitHub, under Settings → Environments (repository admin access required), create the `pypi-nightly` environment. Leave Required reviewers and Wait timer unchecked so the nightly publishes unattended, and under Deployment branches and tags choose Selected branches and tags with a single Branch rule for `main`.
2. On PyPI, under Your account → Publishing, add a GitHub Actions pending publisher with PyPI project name `coreai-opt-nightly`, owner `apple`, repository `coreai-optimization`, workflow `nightly.yml` (the filename, not the workflow's display name), and environment `pypi-nightly`. The first nightly upload creates the project; it then lists this publisher under `coreai-opt-nightly` → Manage → Publishing.

To verify the setup, start a manual run with `main` selected as the branch. The environment checks the branch the run was started on, so a run started from any other branch can't publish.

`coreai-opt-nightly` carries only dev versions, so installers pick them without a pre-release flag:

```bash
uv pip install --upgrade coreai-opt-nightly               # the newest nightly
uv pip install coreai-opt-nightly==1.1.0.dev202607231430  # one specific nightly
```

`coreai-opt` and `coreai-opt-nightly` install the same `coreai_opt` package, so an environment uses one or the other.

### Extending the scheme downstream

A repo that uses this one as a submodule and includes this `Makefile` — building one combined wheel from both trees — can add its own 4th number. Set `COREAI_OPT_VERSION_EXTENSION` to the number it's about to release next, then call `make build`, `make build-dev`, `make build-nightly`, or `make version` unchanged.

The extension anchors the release to the last *published* release instead of the one `__version__` is working toward:

| `latest_released_version` | `__version__` | `COREAI_OPT_VERSION_EXTENSION` | Release built |
| ------------------------- | ------------- | ------------------------------ | ------------- |
| `1.0.0`                   | `1.1.0.dev0`  | unset                          | `1.1.0`       |
| `1.0.0`                   | `1.1.0.dev0`  | `1`                            | `1.0.0.1`     |
| `1.0.0`                   | `1.1.0.dev0`  | `2`                            | `1.0.0.2`     |

The extension is used exactly as given, and starts at `1`, not `0`. `make build-nightly` and `make build-dev` add their usual `.dev<UTC-timestamp>` and `.dev<UTC-timestamp>+<short-sha>` suffixes on top.
