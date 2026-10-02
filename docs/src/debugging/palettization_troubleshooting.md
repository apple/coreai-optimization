# Palettization Troubleshooting

This guide helps debug issues when using [palettization](../palettization/index.md) in `coreai-opt`.

## Palettization hangs with no progress

### Symptoms

- `KMeansPalettizer.prepare()` or `coreai_opt.coreai_utils.palettize_weights()` when `cluster_dim=1` never returns and raises no error.
- The `Palettizing layers` progress bar never appears, or stays at 0%.
- The original Python process (when `num_workers >= 0`), and its worker processes when `num_workers > 1`, sit near 0% CPU.
- If you interrupt the run with Ctrl+C, the traceback ends in `torch/utils/file_baton.py`, in `wait`.

### Cause

The k-means step runs a small C++ extension that PyTorch compiles on first use (with `torch.utils.cpp_extension.load`) and caches on disk. Before PyTorch 2.14, each process that loads the extension creates a lock file in that cache and deletes it once the extension is ready. If the process is killed before it deletes the file — for example by `kill -9`, by the system running out of memory, or by a cancelled job — the lock file stays behind, and every later run waits on it forever. This is a [known PyTorch bug](https://github.com/pytorch/pytorch/issues/189245).

### Fix

First, stop the hung run, and make sure none of its worker processes are still running (check Activity Monitor on macOS, or `ps` on Linux). A worker still waiting on the lock would fail once the cache is gone.

Then delete PyTorch's extension cache. On macOS:

```bash
rm -rf ~/Library/Caches/torch_extensions
```

On Linux:

```bash
rm -rf "${XDG_CACHE_HOME:-$HOME/.cache}/torch_extensions"
```

The cache holds only compiled build artifacts, which PyTorch rebuilds on demand, so deleting it is safe. If you set `TORCH_EXTENSIONS_DIR`, PyTorch builds extensions there instead; delete the `_core` directory inside it.

Finally, rerun. PyTorch recompiles the extension on first use.

### Prevent it

Upgrade to PyTorch 2.14 or later. PyTorch 2.14 [replaced the lock file with an OS-level lock](https://github.com/pytorch/pytorch/pull/190543) that the operating system releases when the process holding it exits, even if it is killed, so a killed run no longer blocks later runs. Upgrading also gets past a stale lock file left by an older PyTorch version.
