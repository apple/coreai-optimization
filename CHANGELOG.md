# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- towncrier release notes start -->

## [0.4.0] - 2026-11-05

### Added

- Added support for Python 3.14.
- Added the public `ExportBackend.Torch` backend for fast torch inference, e.g. running evals. `finalize(backend=ExportBackend.Torch)` folds weight compression (palettization, weight quantization, pruning) into dense weights and keeps activation fake quantization.
- Added a `state_dict` argument to `prepare()` for `Quantizer` (eager and graph mode) and `MagnitudePruner`, matching `KMeansPalettizer`. It reloads a previously prepared model's compression state (calibrated qparams, pruning masks and step count, centroids), validates it against the config, and leaves the model in the same state as a regular `prepare()` call.
- Added MXINT8 quantization (`int8` with `float8_e8m0fnu` scales) in `QuantizationSpec`, supported in eager and graph mode. It is only supported for `ExportBackend.Torch` for now.
- Added the `torch.int6` and `torch.uint6` quantization dtypes. They are only supported for `ExportBackend.Torch` for now: `finalize` raises `NotImplementedError` for the Core AI and Core ML backends.
- Added `VectorKMeansConfig` (`vector_kmeans_config` on `ModuleKMeansPalettizerConfig`) to set the k-means++ candidates per seeding step (`n_local_trials`), restarts, iteration cap and convergence tolerance of vector palettization (`cluster_dim > 1`). Vector k-means now clusters on a CUDA GPU, with flash-kmeans when it is installed and a batched PyTorch loop otherwise or for sensitivity-weighted clustering, and on CPU with scikit-learn, falling back to the PyTorch loop with a warning when scikit-learn is not installed.
- K-means palettization detects weights that are already palettized for their spec (at most `2 ** n_bits` distinct values or vectors per block, e.g. a fake-palettized checkpoint) and uses those unique values as the centroids instead of running k-means. Other weights fall back to k-means after one small probe.
- Added the `sparsity_detection_threshold` option to `PalettizationSpec`. A weight with at least that fraction of exact zeros is treated as already sparse and palettized jointly with its sparsity: only its nonzero values are palettized, the zeros stay exactly zero, and the exported model stores the weight in a sparse representation together with the palettized ops for the nonzero values. Supports per-tensor and per-group-channel granularity, per-channel scales and, on Core AI, quantized LUTs and vector palettization.

### Changed

- Graph-mode `Quantizer` now skips activations and states with at most one element (e.g. the 0-d output of `x.sum()`), matching eager mode. This also fixes `prepare` failing on 0-d parameters under a per-channel weight spec.
- Vector palettization (`cluster_dim > 1`) now seeds with greedy k-means++ instead of farthest-point initialization, stops each block once its clustering error stops improving instead of running a fixed number of iterations, and is reproducible without manual seeding, including with `num_workers > 1`. It is much faster on GPU. With `num_workers > 1`, tensors that cluster on the GPU run in the main process while the worker processes handle the others.
- Pruning modules now store their step count in a `step_count` buffer, so a `MagnitudePruner` state dict saved before this change cannot be reloaded with `prepare(state_dict=...)`.

### Removed

- Removed the deprecated `ExportBackend._TORCH` alias. Use `ExportBackend.Torch` instead.
- Removed the `prev_sparsity` argument from `SparsityScheduleBase.compute_sparsity` and its subclasses. Schedules are now pure functions of the step count; `PolynomialDecaySchedule` returns the last update's sparsity for steps between updates instead of passing through `prev_sparsity` or raising `ValueError` when it is omitted.

### Fixed

- Fixed `KMeansPalettizer.prepare(state_dict=...)` failing for configs with a quantized LUT, and `MagnitudePruner` restarting its sparsity schedule from step 0 after a model saved mid-schedule was reloaded.
- Preserve original floating-point casts that lose precision or range when cleaning up cast chains.

## [0.3.0] - 2026-09-25

### Added

- Support ``ignored_ops`` in ``cast_fp32_to_fp16`` and ``cast_to_16_bit_precision`` to allow users to exclude sensitive operations (e.g. operations causing activation-level overflow like ``exp`` or ``softplus``) from FP16 casting, keeping them in FP32 with appropriate boundary casts.
- Add training-aware palettization to `KMeansPalettizer` via a `training_mode()` context manager and `step()` method, with a `PATSchedule` (`pat_schedule` on the module config) controlling when palettization activates during a training loop. `PalettizationSpec` gains a `training_strategy_spec` field for selecting pluggable training-time strategies.
- Added support for torchao 0.18.0.
- Added support for PyTorch 2.12 and 2.13.
- Graph mode quantization support with Core AI composite ops. Added tests and documentation for usage details.
- Add utility to compute analytical bits per weight (BPW) of a prepared eager-mode quantized or palettized model
- Support for memory-efficient finalize in graph mode via `finalize(..., mmap_dir=...)`, previously available only in eager mode. Supported with the Core AI backend only.
- Support new coreai-torch and coreai-core releases

### Changed

- Replace the coremltools-based 1D k-means used by palettization with a vendored C++ core that is JIT-compiled at runtime via `torch.utils.cpp_extension`. `coremltools` is no longer a runtime dependency (it is now an optional dependency, installable via the `coreml` extra). This requires a C++ compiler to be available on the host at runtime.
- Carry the next planned release with a `.dev0` suffix on `main` (e.g. `0.2.2.dev0`). `make build` strips the suffix for a clean release and `make build-dev` builds a unique, timestamped dev wheel (`0.2.2.dev<timestamp>+<shortsha>`). A repo that vendors this one can insert one extra release segment via the `COREAI_OPT_VERSION_EXTENSION` environment variable to extend the version scheme.
- When a weight is incompatible with the configured quantization block size or palettization granularity, the warning now reports the weight's fully qualified name and shape along with the module name to use in `module_name_configs`, instead of only the compression target.
- Re-cluster palettization centroids from the warmed-up weights when a `PATSchedule` enables fake palettization mid-training (`enable_fake_palettize > 0`).
- Remove upper bound on torch, torchao. Add untested message for untested versions on coreai-opt

### Fixed

- Fix setting of qscheme and float_range for fixed output range ops.
- Fix graph-mode `Quantizer` resolving a tied weight's QAT schedule from a different module than its dtype. A shared fake-quantize module's owning module is now picked by the same config priority that decides its `dtype`, instead of by graph order, so the schedule and dtype for a shared weight always come from one consistent module.
- Reject per-channel activation quantization on CoreML export
- Fix per-channel activation quantization crashing with a shape-mismatch `RuntimeError` on MaxPool/AvgPool/AdaptiveAvgPool layers whose shared observer spans an axis the pool shrinks (e.g. a spatial axis under a stride>1 pool). Axes pooling never touches (batch, channel) keep working as per-channel; only the specific unsafe axis falls back to per-tensor, with a warning explaining why and how to pick a safe axis instead.
- Fix eager Quantizer.prepare failing on models that call a supported op whose inputs are all scalars or single-element tensors
- Ensure `KMeansPalettizer.calibration_mode` cleans up temporary model checkpoints on failure and preserves caller exceptions without masking. Prevent fake palettization from remaining disabled after calibration abort, and fix sensitivity parameter name resolution for root modules.
- Finalized models in graph mode no longer contain the unused original full precision weights similar to eager mode

## [0.2.1] - 2026-07-02

### Added

- Support palettization of `ConvTranspose1d`/`ConvTranspose2d`/`ConvTranspose3d` layers via `KMeansPalettizer`
- Support for `EAGER` execution mode in model inspection utility

### Fixed

- Fixed pruning mask `dtype` to match that of the weight being pruned
- Fixes to allow better support for `bfloat16` `dtype` in palettization and quantization

## [0.2.0] - 2026-06-08

### Added

- Initial release of `coreai-opt`. See the [GitHub Releases](https://github.com/apple/coreai-optimization/releases/) page for release notes.

[0.2.0]: https://github.com/apple/coreai-optimization/commits/v0.2.0/
[0.2.1]: https://github.com/apple/coreai-optimization/compare/v0.2.0...v0.2.1/
[0.3.0]: https://github.com/apple/coreai-optimization/compare/v0.2.1...v0.3.0/
[0.4.0]: https://github.com/apple/coreai-optimization/compare/v0.3.0...v0.4.0/
