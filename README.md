This is a test to check branch protection rules.
Another test
Another test
Another test
Another test

# Core AI Optimization

`coreai-opt` provides implementations of popular model optimizations such as quantization, palettization (codebook-based compression), and pruning, for PyTorch models, customized for deployment on Apple Silicon via Core AI.

Jump to: [Getting Started](#getting-started) · [Documentation](#documentation) · [Contributing](#contributing) · [Support](#support) · [License](#license) · [Related projects](#related-projects)

## Getting started

### Installation

Install the latest release from PyPI:

```bash
pip install coreai-opt
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv pip install coreai-opt
```

#### From source

This project uses [uv](https://docs.astral.sh/uv/) for environment management. Install `uv` by following the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer).

To set up the environment from a checkout:

```bash
make env
```

This creates a project-specific virtual environment `.venv` and installs all dependencies. Activate it in a new terminal session with:

```bash
source .venv/bin/activate
```

### Usage

```python
import torch
from coreai_opt.quantization import Quantizer, QuantizerConfig
from torch import nn

# A simple model and example input.
model = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10)).eval()
example_inputs = (torch.randn(1, 128),)

# Apply INT8 weight-only quantization using a built-in preset.
config = QuantizerConfig.presets.w8()
quantizer = Quantizer(model, config)
prepared_model = quantizer.prepare(example_inputs)

# Finalize for Core AI export.
finalized_model = quantizer.finalize()
```

## Documentation

For APIs, options, and detailed workflows, see the hosted documentation at [apple.github.io/coreai-optimization](https://apple.github.io/coreai-optimization/).

## Contributing

Contributions are welcome within a defined scope. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request or issue, particularly the section on contribution scope.

## Support

- [GitHub Issues](../../issues) — Bug reports, feature requests, and questions

## License

This project is licensed under the [BSD-3-Clause](LICENSE) license.

## Related projects

- [Core AI](https://developer.apple.com/documentation/coreai) — Apple's on-device AI inference stack
- [Core AI Torch](https://github.com/apple/coreai-torch) — converts PyTorch models to Core AI format; the upstream step before optimization
- [Core AI Models](https://github.com/apple/coreai-models) — ready-to-run optimized models, Python reproduction scripts, and Swift utilities for on-device integration
