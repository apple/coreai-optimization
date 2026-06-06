# Model Examples

Check out the [coreai-models](https://github.com/apple/coreai-models/blob/main/models/README.md) project for model examples with Core AI. Several of the LLM models there utilize `coreai-opt` for compression, via various compression configs (4 bit configs, [mixed precision](../utils/mixed_precision.md) 4/8 bit configs etc).

The examples in this section use simpler models (like ResNet50) to illustrate various compression options, hyper-parameters choices and processes that you may use for determining which methods to apply to your specific model.

```{toctree}
:maxdepth: 1

resnet50
mixed_precision_palettization
edsr
```
