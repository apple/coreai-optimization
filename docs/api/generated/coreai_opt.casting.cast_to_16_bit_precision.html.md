# coreai_opt.casting.cast_to_16_bit_precision

### coreai_opt.casting.cast_to_16_bit_precision(exported_program)

Convert a torch exported program to 16-bit precision: FP32→FP16 and INT32/64→INT16.

Runs both cast passes sequentially:
1. cast_fp32_to_fp16: FP32→FP16
2. cast_int32_to_int16: INT32/INT64→INT16

* **Parameters:**
  **exported_program** (*ExportedProgram*)
* **Return type:**
  *ExportedProgram*
