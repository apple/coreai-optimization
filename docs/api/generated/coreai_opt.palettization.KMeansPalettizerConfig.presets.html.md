# presets

Convenient factories for common compression recipes. Each preset
returns a ready-to-use config that can be further refined by chaining
[`set_module_type()`](coreai_opt.config.CompressionConfig.md#coreai_opt.config.CompressionConfig.set_module_type),
[`set_module_name()`](coreai_opt.config.CompressionConfig.md#coreai_opt.config.CompressionConfig.set_module_name),
[`only_for()`](coreai_opt.config.CompressionConfig.md#coreai_opt.config.CompressionConfig.only_for), or
[`without()`](coreai_opt.config.CompressionConfig.md#coreai_opt.config.CompressionConfig.without).

| Preset                                                                | Description                                                          |
|-----------------------------------------------------------------------|----------------------------------------------------------------------|
| [w4()](coreai_opt.palettization.KMeansPalettizerConfig.presets.w4.md) | 4-bit palettization, per-grouped-channel, group_size defaults to 16. |
| [w6()](coreai_opt.palettization.KMeansPalettizerConfig.presets.w6.md) | 6-bit palettization, per-grouped-channel, group_size defaults to 16. |
| [w8()](coreai_opt.palettization.KMeansPalettizerConfig.presets.w8.md) | 8-bit palettization, per-tensor.                                     |
