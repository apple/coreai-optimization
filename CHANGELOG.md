# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- towncrier release notes start -->

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

[0.2.1]: https://github.com/apple/coreai-optimization/releases/tag/v0.2.1
[0.2.0]: https://github.com/apple/coreai-optimization/releases/tag/v0.2.0
