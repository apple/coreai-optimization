# coreai_opt.casting.cast_fp32_to_fp16

### coreai_opt.casting.cast_fp32_to_fp16(exported_program)

Convert a torch exported program from FP32 to FP16 where applicable.

Converts parameters, user inputs, and compute ops to FP16, inserting
casts only where values would overflow FP16 range.

* **Parameters:**
  **exported_program** (*ExportedProgram*)
* **Return type:**
  *ExportedProgram*
