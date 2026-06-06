# coreai_opt.casting.cast_int32_to_int16

### coreai_opt.casting.cast_int32_to_int16(exported_program)

Convert INT32/INT64 tensors to INT16 in a torch exported program.

Only designated ops (data operations) are converted. Positional values
(indices, strides, dimensions) are left as int32/int64.

* **Parameters:**
  **exported_program** (*ExportedProgram*)
* **Return type:**
  *ExportedProgram*
