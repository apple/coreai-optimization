# presets

Convenient factories for common compression recipes. Each preset
returns a ready-to-use config that can be further refined by chaining
[`set_module_type()`](coreai_opt.config.CompressionConfig.md#coreai_opt.config.CompressionConfig.set_module_type),
[`set_module_name()`](coreai_opt.config.CompressionConfig.md#coreai_opt.config.CompressionConfig.set_module_name),
[`only_for()`](coreai_opt.config.CompressionConfig.md#coreai_opt.config.CompressionConfig.only_for), or
[`without()`](coreai_opt.config.CompressionConfig.md#coreai_opt.config.CompressionConfig.without).

| Preset                                                                                  | Description                                                                    |
|-----------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| [w4()](coreai_opt.quantization.ModuleQuantizerConfig.presets.w4.md)                     | int4 weight-only quantization, per-channel symmetric.                          |
| [w4_per_block()](coreai_opt.quantization.ModuleQuantizerConfig.presets.w4_per_block.md) | int4 weight-only quantization, per-block symmetric, block_size defaults to 32. |
| [w8()](coreai_opt.quantization.ModuleQuantizerConfig.presets.w8.md)                     | int8 weight-only quantization, per-channel symmetric.                          |
