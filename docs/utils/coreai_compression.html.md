# Compressing Core AI Models

The Core AI model compression path accepts a Core AI AIProgram as input and produces a compressed AIProgram (≤8 bits). Compression passes operate directly on the Core AI graph and have no dependency on PyTorch.

## Core AI Model weight compression flow

The example below shows the typical flow: load an uncompressed `.aimodel` from disk, compress its weights via one of the Core AI model compression passes, and save the compressed `.aimodel` back to disk.

To convert a PyTorch model into an `.aimodel`, see the [`coreai-torch` documentation](https://github.com/apple/coreai-torch). To ensure the model uses 16-bit precision before compressing, see [16-bit PyTorch Model Casting](casting.md).

```python
from pathlib import Path

from coreai.authoring import AIModelAsset
from coreai_opt.coreai_utils import DType, quantize_weights

# Load an uncompressed aimodel from disk
ai_asset = AIModelAsset.load(Path("model.aimodel"))
ai_program = ai_asset.program

# Compress weights to INT8
compressed_program = quantize_weights(
    coreai_program=ai_program,
    dtype=DType.INT8,
    in_place=False,
)

# Save the compressed aimodel to disk
compressed_program.optimize()
compressed_program.save_asset(Path("model_compressed.aimodel"))
```

## Core AI Model weight palettization

`palettize_weights` compresses weights in a Core AI MLIR program using K-means
palettization. It walks the IR and palettizes each eligible `coreai.constant` op.

```python
from coreai_opt.coreai_utils import CompressionGranularity, palettize_weights

# --- palettize weights ---
compressed_program = palettize_weights(
    coreai_program=coreai_program,
    n_bits=4,  # LUT has 2**n_bits entries
    lut_dtype=None,  # keep LUT entries in fp16
    granularity=CompressionGranularity.PER_CHANNEL,
    in_place=False,
)
```

For finer control, the parameters can be specified explicitly:

```python
from coreai_opt.coreai_utils import CompressionGranularity, DType, palettize_weights

# --- palettize weights with advanced options ---
compressed_program = palettize_weights(
    coreai_program=coreai_program,
    n_bits=8,  # 256-entry LUT (2**8)
    lut_dtype=DType.INT8,  # quantize LUT entries to int8
    granularity=CompressionGranularity.PER_GROUPED_CHANNEL,  # one LUT per group of channels
    group_size=16,  # group size for PER_GROUPED_CHANNEL
    enable_per_channel_scale=True,  # normalize weights by per-channel scale before clustering
    weight_num_threshold=2048,  # skip tensors with <= 2048 elements
    num_kmeans_workers=2,  # parallel workers for k-means
    enable_fast_kmeans_mode=True,  # round weights before clustering to speed up k-means
    rounding_precision=3,  # decimal places for fast k-means rounding
    in_place=True,  # modify coreai_program in-place
)
```

## Core AI Model weight quantization

`quantize_weights` compresses weights in a Core AI MLIR program by quantizing them to a
lower-precision integer or floating-point dtype. It walks the IR and quantizes each
eligible `coreai.constant` op.

```python
from coreai_opt.coreai_utils import DType, quantize_weights

# --- quantize weights ---
compressed_program = quantize_weights(
    coreai_program=coreai_program,
    dtype=DType.INT8,  # quantize weights to int8
    in_place=False,
)
```

For finer control, the parameters can be specified explicitly:

```python
from coreai_opt.coreai_utils import (
    CompressionGranularity,
    DType,
    QScheme,
    quantize_weights,
)

# --- quantize weights with advanced options ---
compressed_program = quantize_weights(
    coreai_program=coreai_program,
    dtype=DType.FP8_E4M3FN,  # quantize weights to FP8 E4M3FN
    qscheme=QScheme.SYMMETRIC,  # only symmetric is supported for FP8 dtypes
    granularity=CompressionGranularity.PER_BLOCK,  # one scale per block of axes
    block_size=32,  # block size for PER_BLOCK
    weight_num_threshold=2048,  # skip tensors with <= 2048 elements
    scale_dtype=DType.FP8_E8M0FNU,  # store scales in FP8 E8M0FNU format
    in_place=True,  # modify coreai_program in-place
)
```

## Core AI Model weight sparsification

`sparsify_weights` compresses weights in a Core AI MLIR program by pruning them to a
target sparsity level. It walks the IR and sparsifies each eligible `coreai.constant` op.

```python
from coreai_opt.coreai_utils import sparsify_weights

# --- sparsify weights ---
compressed_program = sparsify_weights(
    coreai_program=coreai_program,
    target_sparsity=0.5,  # set 50% of weights (lowest magnitude) to zero
    in_place=False,
)
```

For finer control, the parameters can be specified explicitly:

```python
from coreai_opt.coreai_utils import DType, sparsify_weights

# --- sparsify weights with advanced options ---

# Magnitude-based sparsification with joint quantization of non-zero values
compressed_program = sparsify_weights(
    coreai_program=coreai_program,
    target_sparsity=0.5,  # 50% of weights set to zero
    block_size=4,  # block sparsity: prune in blocks of 4 along the output channel axis
    quantize_dtype=DType.INT8,  # quantize non-zero values to int8
    weight_num_threshold=2048,  # skip tensors with <= 2048 elements
    in_place=True,  # modify coreai_program in-place
)

# Sparsification with joint palettization of non-zero values
compressed_program = sparsify_weights(
    coreai_program=coreai_program,
    target_sparsity=0.5,
    palettize_nbits=4,  # palettize non-zero values to 4-bit LUT
    weight_num_threshold=2048,
    in_place=False,
)
```
