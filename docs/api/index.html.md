# API Reference

## coreai_opt

coreai_opt - A library for PyTorch model compression and optimizations.

| [`coreai_opt.CoreMLExportError`](generated/coreai_opt.CoreMLExportError.md#coreai_opt.CoreMLExportError)(dtype, context)   | Raised when a model cannot be exported to the CoreML backend.   |
|----------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------|
| [`coreai_opt.ExportBackend`](generated/coreai_opt.ExportBackend.md#coreai_opt.ExportBackend)(value, \*args, \*\*kwargs)    | Enum representing supported model export backends.              |

## coreai_opt.casting

Casting related utilities including FP32 -> FP16 and INT32 -> INT16 passes.

| [`coreai_opt.casting.cast_fp32_to_fp16`](generated/coreai_opt.casting.cast_fp32_to_fp16.md#coreai_opt.casting.cast_fp32_to_fp16)(...)                      | Convert a torch exported program from FP32 to FP16 where applicable.                |
|------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| [`coreai_opt.casting.cast_int32_to_int16`](generated/coreai_opt.casting.cast_int32_to_int16.md#coreai_opt.casting.cast_int32_to_int16)(...)                | Convert INT32/INT64 tensors to INT16 in a torch exported program.                   |
| [`coreai_opt.casting.cast_to_16_bit_precision`](generated/coreai_opt.casting.cast_to_16_bit_precision.md#coreai_opt.casting.cast_to_16_bit_precision)(...) | Convert a torch exported program to 16-bit precision: FP32→FP16 and INT32/64→INT16. |

## coreai_opt.config

Configuration and specification modules for coreai_opt.

| [`coreai_opt.config.CompressionConfig`](generated/coreai_opt.config.CompressionConfig.md#coreai_opt.config.CompressionConfig)                                             | Top level configuration class for model compression.                          |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| [`coreai_opt.config.CompressionSpec`](generated/coreai_opt.config.CompressionSpec.md#coreai_opt.config.CompressionSpec)                                                   | Base class for compression specifications.                                    |
| [`coreai_opt.config.CompressionType`](generated/coreai_opt.config.CompressionType.md#coreai_opt.config.CompressionType)(value[, ...])                                     | Enum representing compression techniques applied to the model.                |
| [`coreai_opt.config.ModuleCompressionConfig`](generated/coreai_opt.config.ModuleCompressionConfig.md#coreai_opt.config.ModuleCompressionConfig)                           | Abstract base configuration class for module-level compression settings.      |
| [`coreai_opt.config.OpCompressionConfig`](generated/coreai_opt.config.OpCompressionConfig.md#coreai_opt.config.OpCompressionConfig)                                       | Abstract base configuration class for op-level compression settings.          |
| [`coreai_opt.config.WeightOnlyModuleValidationMixin`](generated/coreai_opt.config.WeightOnlyModuleValidationMixin.md#coreai_opt.config.WeightOnlyModuleValidationMixin)() | Mixin that adds weight-only validation to ModuleCompressionConfig subclasses. |
| [`coreai_opt.config.WeightOnlyOpValidationMixin`](generated/coreai_opt.config.WeightOnlyOpValidationMixin.md#coreai_opt.config.WeightOnlyOpValidationMixin)()             | Mixin that adds weight-only validation to OpCompressionConfig subclasses.     |

### coreai_opt.config.spec

Base abstractions for compression specs, simulators, and component factories.

| [`coreai_opt.config.spec.CompressionComponentFactoryBase`](generated/coreai_opt.config.spec.CompressionComponentFactoryBase.md#coreai_opt.config.spec.CompressionComponentFactoryBase)()   | Abstract base class for compression component factories.   |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------|
| [`coreai_opt.config.spec.CompressionSimulatorBase`](generated/coreai_opt.config.spec.CompressionSimulatorBase.md#coreai_opt.config.spec.CompressionSimulatorBase)(...)                     | Abstract base class for compression simulators.            |
| [`coreai_opt.config.spec.CompressionTargetTensor`](generated/coreai_opt.config.spec.CompressionTargetTensor.md#coreai_opt.config.spec.CompressionTargetTensor)(value)                      | Enum to specify the target tensor for compression.         |

## coreai_opt.coreai_utils

Core AI MLIR-level compression transforms.

| [`coreai_opt.coreai_utils.CompressionGranularity`](generated/coreai_opt.coreai_utils.CompressionGranularity.md#coreai_opt.coreai_utils.CompressionGranularity)(value)   | Enum representing the granularity of quantization for Core AI weight compression.   |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| [`coreai_opt.coreai_utils.DType`](generated/coreai_opt.coreai_utils.DType.md#coreai_opt.coreai_utils.DType)(value[, ...])                                               | Enum representing data types for Core AI weight compression.                        |
| [`coreai_opt.coreai_utils.palettize_weights`](generated/coreai_opt.coreai_utils.palettize_weights.md#coreai_opt.coreai_utils.palettize_weights)(...)                    | Palettize weights in a Core AI AIProgram (MLIR<CoreAI> IR) by using Core AI ops.    |
| [`coreai_opt.coreai_utils.quantize_weights`](generated/coreai_opt.coreai_utils.quantize_weights.md#coreai_opt.coreai_utils.quantize_weights)(...)                       | Quantize weights in a Core AI AIProgram (MLIR<CoreAI> IR) by using Core AI ops.     |
| [`coreai_opt.coreai_utils.sparsify_weights`](generated/coreai_opt.coreai_utils.sparsify_weights.md#coreai_opt.coreai_utils.sparsify_weights)(...)                       | Sparsify weights in a Core AI AIProgram (MLIR<CoreAI> IR) by using Core AI ops.     |

### coreai_opt.coreai_utils.common

Common enums and constants for coreai_opt.coreai_utils.

| [`coreai_opt.coreai_utils.common.QScheme`](generated/coreai_opt.coreai_utils.common.QScheme.md#coreai_opt.coreai_utils.common.QScheme)(value)   | Enum representing the quantization scheme.   |
|-------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------|

## coreai_opt.inspection

Utilities for inspecting model operations and compression configuration.

| [`coreai_opt.inspection.ModelInspector`](generated/coreai_opt.inspection.ModelInspector.md#coreai_opt.inspection.ModelInspector)(model, ...)   | Inspect operations in a PyTorch model for compression configuration.   |
|------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------|
| [`coreai_opt.inspection.ModelSummary`](generated/coreai_opt.inspection.ModelSummary.md#coreai_opt.inspection.ModelSummary)(model, mode)        | Complete listing of operations discovered in a model.                  |
| [`coreai_opt.inspection.ModuleContext`](generated/coreai_opt.inspection.ModuleContext.md#coreai_opt.inspection.ModuleContext)(...)             | One level of the `nn.Module` nesting hierarchy.                        |
| [`coreai_opt.inspection.ModuleInfo`](generated/coreai_opt.inspection.ModuleInfo.md#coreai_opt.inspection.ModuleInfo)(...)                      | A node in the `nn.Module` hierarchy with its directly-owned ops.       |
| [`coreai_opt.inspection.OpInfo`](generated/coreai_opt.inspection.OpInfo.md#coreai_opt.inspection.OpInfo)(op_name, ...)                         | Information about a single operation discovered in a model.            |
| [`coreai_opt.inspection.SourceFrame`](generated/coreai_opt.inspection.SourceFrame.md#coreai_opt.inspection.SourceFrame)(filename, ...)         | A single frame in the source call stack leading to an operation.       |

## coreai_opt.palettization

Palettization specification and utilities for weight compression via lookup tables.

| [`coreai_opt.palettization.KMeansPalettizer`](generated/coreai_opt.palettization.KMeansPalettizer.md#coreai_opt.palettization.KMeansPalettizer)(model)                              | K-means palettizer with integrated supported operations strategy.         |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| [`coreai_opt.palettization.KMeansPalettizerConfig`](generated/coreai_opt.palettization.KMeansPalettizerConfig.md#coreai_opt.palettization.KMeansPalettizerConfig)                   | Top-level configuration class for kmeans palettization.                   |
| [`coreai_opt.palettization.ModuleKMeansPalettizerConfig`](generated/coreai_opt.palettization.ModuleKMeansPalettizerConfig.md#coreai_opt.palettization.ModuleKMeansPalettizerConfig) | Configuration for palettizing a specific module using K-means clustering. |
| [`coreai_opt.palettization.PalettizationSpec`](generated/coreai_opt.palettization.PalettizationSpec.md#coreai_opt.palettization.PalettizationSpec)                                  | Specification for palettization compression of neural network weights.    |

### coreai_opt.palettization.config

Palettization configuration classes.

| [`coreai_opt.palettization.config.OpKMeansPalettizerConfig`](generated/coreai_opt.palettization.config.OpKMeansPalettizerConfig.md#coreai_opt.palettization.config.OpKMeansPalettizerConfig)   | Configuration class for palettization at the operation level.   |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------|

### coreai_opt.palettization.spec

Palettization specs, granularity classes, and factory functions.

| [`coreai_opt.palettization.spec.PalettizationGranularity`](generated/coreai_opt.palettization.spec.PalettizationGranularity.md#coreai_opt.palettization.spec.PalettizationGranularity)                              | Base class for palettization granularity specifications.   |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------|
| [`coreai_opt.palettization.spec.PerGroupedChannelGranularity`](generated/coreai_opt.palettization.spec.PerGroupedChannelGranularity.md#coreai_opt.palettization.spec.PerGroupedChannelGranularity)                  | Per-grouped-channel palettization granularity.             |
| [`coreai_opt.palettization.spec.PerTensorGranularity`](generated/coreai_opt.palettization.spec.PerTensorGranularity.md#coreai_opt.palettization.spec.PerTensorGranularity)                                          | Per-tensor palettization granularity.                      |
| [`coreai_opt.palettization.spec.default_weight_palettization_spec`](generated/coreai_opt.palettization.spec.default_weight_palettization_spec.md#coreai_opt.palettization.spec.default_weight_palettization_spec)() |                                                            |

## coreai_opt.pruning

Pruning infrastructure for coreai_opt.

| [`coreai_opt.pruning.MagnitudePruner`](generated/coreai_opt.pruning.MagnitudePruner.md#coreai_opt.pruning.MagnitudePruner)(model[, ...])                       | Apply magnitude-based pruning to a model.      |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------|
| [`coreai_opt.pruning.MagnitudePrunerConfig`](generated/coreai_opt.pruning.MagnitudePrunerConfig.md#coreai_opt.pruning.MagnitudePrunerConfig)                   | Top-level configuration for magnitude pruning. |
| [`coreai_opt.pruning.ModuleMagnitudePrunerConfig`](generated/coreai_opt.pruning.ModuleMagnitudePrunerConfig.md#coreai_opt.pruning.ModuleMagnitudePrunerConfig) | Module-level pruning configuration.            |
| [`coreai_opt.pruning.PruningSpec`](generated/coreai_opt.pruning.PruningSpec.md#coreai_opt.pruning.PruningSpec)                                                 | Specification for pruning tensors.             |

### coreai_opt.pruning.config

Pruning configuration exports.

| [`coreai_opt.pruning.config.ConstantSparsitySchedule`](generated/coreai_opt.pruning.config.ConstantSparsitySchedule.md#coreai_opt.pruning.config.ConstantSparsitySchedule)   | Step function: zero before `begin_step`, `target_sparsity` at and after.   |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------|
| [`coreai_opt.pruning.config.OpMagnitudePrunerConfig`](generated/coreai_opt.pruning.config.OpMagnitudePrunerConfig.md#coreai_opt.pruning.config.OpMagnitudePrunerConfig)      | Operation-level pruning configuration.                                     |
| [`coreai_opt.pruning.config.PolynomialDecaySchedule`](generated/coreai_opt.pruning.config.PolynomialDecaySchedule.md#coreai_opt.pruning.config.PolynomialDecaySchedule)      | Polynomial schedule from `initial_sparsity` to `target_sparsity`.          |
| [`coreai_opt.pruning.config.SparsityScheduleBase`](generated/coreai_opt.pruning.config.SparsityScheduleBase.md#coreai_opt.pruning.config.SparsityScheduleBase)               | Abstract base for sparsity schedules used by `MagnitudePruner`.            |

### coreai_opt.pruning.spec

Pruning spec components: specs, schemes, and parametrizations.

| [`coreai_opt.pruning.spec.ChannelStructured`](generated/coreai_opt.pruning.spec.ChannelStructured.md#coreai_opt.pruning.spec.ChannelStructured)                                 | Channel-structured pruning scheme.                                     |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------|
| [`coreai_opt.pruning.spec.PruneImplBase`](generated/coreai_opt.pruning.spec.PruneImplBase.md#coreai_opt.pruning.spec.PruneImplBase)(...)                                        | Abstract base for pruning parametrizations that mask a layer's weight. |
| [`coreai_opt.pruning.spec.PruningScheme`](generated/coreai_opt.pruning.spec.PruningScheme.md#coreai_opt.pruning.spec.PruningScheme)                                             | Base class for pruning scheme specifications.                          |
| [`coreai_opt.pruning.spec.Unstructured`](generated/coreai_opt.pruning.spec.Unstructured.md#coreai_opt.pruning.spec.Unstructured)                                                | Unstructured pruning scheme.                                           |
| [`coreai_opt.pruning.spec.default_weight_pruning_spec`](generated/coreai_opt.pruning.spec.default_weight_pruning_spec.md#coreai_opt.pruning.spec.default_weight_pruning_spec)() | Return the default pruning spec for weight tensors.                    |

## coreai_opt.quantization

Quantization compressor, configuration, specs, and granularity classes.

| [`coreai_opt.quantization.ExecutionMode`](generated/coreai_opt.quantization.ExecutionMode.md#coreai_opt.quantization.ExecutionMode)(value, ...)             | Enum representing quantization execution modes.                                                         |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| [`coreai_opt.quantization.ModuleQuantizerConfig`](generated/coreai_opt.quantization.ModuleQuantizerConfig.md#coreai_opt.quantization.ModuleQuantizerConfig) | Configuration class for quantization at the module level.                                               |
| [`coreai_opt.quantization.QuantizationSpec`](generated/coreai_opt.quantization.QuantizationSpec.md#coreai_opt.quantization.QuantizationSpec)                | Specification for quantizing tensors in neural networks.                                                |
| [`coreai_opt.quantization.Quantizer`](generated/coreai_opt.quantization.Quantizer.md#coreai_opt.quantization.Quantizer)(model[, ...])                       | Unified quantizer API that provides a single entry point for various quantization workflows, including: |
| [`coreai_opt.quantization.QuantizerConfig`](generated/coreai_opt.quantization.QuantizerConfig.md#coreai_opt.quantization.QuantizerConfig)                   | Top-level configuration class for quantization.                                                         |

### coreai_opt.quantization.config

Quantization configuration classes and execution mode.

| [`coreai_opt.quantization.config.OpQuantizerConfig`](generated/coreai_opt.quantization.config.OpQuantizerConfig.md#coreai_opt.quantization.config.OpQuantizerConfig)   | Configuration class for quantization at the operation level.          |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------|
| [`coreai_opt.quantization.config.QATSchedule`](generated/coreai_opt.quantization.config.QATSchedule.md#coreai_opt.quantization.config.QATSchedule)                     | Schedule for controlling observer and fake quantization state in QAT. |

### coreai_opt.quantization.spec

Quantization specs, schemes, granularity classes, and parameter calculators.

| [`coreai_opt.quantization.spec.GlobalMinMaxQParamsCalculator`](generated/coreai_opt.quantization.spec.GlobalMinMaxQParamsCalculator.md#coreai_opt.quantization.spec.GlobalMinMaxQParamsCalculator)(...)                   | Computes scale and zero point by tracking the running min/max.                                                           |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| [`coreai_opt.quantization.spec.MinMaxRangeCalculator`](generated/coreai_opt.quantization.spec.MinMaxRangeCalculator.md#coreai_opt.quantization.spec.MinMaxRangeCalculator)(...)                                           | Range calculator that computes the range of a given tensor as the min and max values of the tensor.                      |
| [`coreai_opt.quantization.spec.MovingAverageQParamsCalculator`](generated/coreai_opt.quantization.spec.MovingAverageQParamsCalculator.md#coreai_opt.quantization.spec.MovingAverageQParamsCalculator)(...)                | Computes the scale and zero point using a moving average of the range.                                                   |
| [`coreai_opt.quantization.spec.PerBlockGranularity`](generated/coreai_opt.quantization.spec.PerBlockGranularity.md#coreai_opt.quantization.spec.PerBlockGranularity)                                                      | Per-block quantization granularity.                                                                                      |
| [`coreai_opt.quantization.spec.PerChannelGranularity`](generated/coreai_opt.quantization.spec.PerChannelGranularity.md#coreai_opt.quantization.spec.PerChannelGranularity)                                                | Per-channel quantization granularity.                                                                                    |
| [`coreai_opt.quantization.spec.PerTensorGranularity`](generated/coreai_opt.quantization.spec.PerTensorGranularity.md#coreai_opt.quantization.spec.PerTensorGranularity)                                                   | Per-tensor quantization granularity.                                                                                     |
| [`coreai_opt.quantization.spec.QParamsCalculatorBase`](generated/coreai_opt.quantization.spec.QParamsCalculatorBase.md#coreai_opt.quantization.spec.QParamsCalculatorBase)(...)                                           | Base class for implementing logic to calculate quantization parameters (scale, zero_point, minval) given min/max values. |
| [`coreai_opt.quantization.spec.QuantizationComponentFactory`](generated/coreai_opt.quantization.spec.QuantizationComponentFactory.md#coreai_opt.quantization.spec.QuantizationComponentFactory)()                         | Factory class for creating quantization components from QuantizationSpec.                                                |
| [`coreai_opt.quantization.spec.QuantizationFormulation`](generated/coreai_opt.quantization.spec.QuantizationFormulation.md#coreai_opt.quantization.spec.QuantizationFormulation)(value)                                   | Formula used to map between quantized integers and dequantized values.                                                   |
| [`coreai_opt.quantization.spec.QuantizationGranularity`](generated/coreai_opt.quantization.spec.QuantizationGranularity.md#coreai_opt.quantization.spec.QuantizationGranularity)                                          | Base class for quantization granularity specifications.                                                                  |
| [`coreai_opt.quantization.spec.QuantizationScheme`](generated/coreai_opt.quantization.spec.QuantizationScheme.md#coreai_opt.quantization.spec.QuantizationScheme)(value)                                                  |                                                                                                                          |
| [`coreai_opt.quantization.spec.RangeCalculatorBase`](generated/coreai_opt.quantization.spec.RangeCalculatorBase.md#coreai_opt.quantization.spec.RangeCalculatorBase)(...)                                                 | Base class and registry for classes used to compute the range of a given tensor.                                         |
| [`coreai_opt.quantization.spec.RunningRangeMixin`](generated/coreai_opt.quantization.spec.RunningRangeMixin.md#coreai_opt.quantization.spec.RunningRangeMixin)(...)                                                       | Mixin for calculators that maintain running min/max range buffers.                                                       |
| [`coreai_opt.quantization.spec.StaticQParamsCalculator`](generated/coreai_opt.quantization.spec.StaticQParamsCalculator.md#coreai_opt.quantization.spec.StaticQParamsCalculator)(...)                                     | Computes scale and zero point using min/max values from the current tensor.                                              |
| [`coreai_opt.quantization.spec.default_activation_quantization_spec`](generated/coreai_opt.quantization.spec.default_activation_quantization_spec.md#coreai_opt.quantization.spec.default_activation_quantization_spec)() |                                                                                                                          |
| [`coreai_opt.quantization.spec.default_weight_quantization_spec`](generated/coreai_opt.quantization.spec.default_weight_quantization_spec.md#coreai_opt.quantization.spec.default_weight_quantization_spec)()             |                                                                                                                          |

### coreai_opt.quantization.spec.fake_quantize

Fake quantization implementation base class and default implementation.

| [`coreai_opt.quantization.spec.fake_quantize.FakeQuantizeImplBase`](generated/coreai_opt.quantization.spec.fake_quantize.FakeQuantizeImplBase.md#coreai_opt.quantization.spec.fake_quantize.FakeQuantizeImplBase)(...)   | Base class for implementing fake quantization   |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------|
