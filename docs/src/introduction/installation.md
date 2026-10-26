# Installation

Install the latest release from PyPI:

```bash
# pip
pip install coreai-opt

# uv
uv pip install coreai-opt
```

To build and install from source, see the [Getting Started](https://github.com/apple/coreai-optimization/#getting-started) section in the README.

Import `coreai-opt` with:

```python
import coreai_opt as opt
```

## Nightly builds

A wheel built from the latest `main` is published to PyPI every night as [`coreai-opt-nightly`](https://pypi.org/project/coreai-opt-nightly/). It provides the same `coreai_opt` package, so install it in place of `coreai-opt`:

```bash
# pip
pip install --upgrade coreai-opt-nightly

# uv
uv pip install --upgrade coreai-opt-nightly
```

Nightlies are snapshots of `main`; use a release for stable work.
