# coreai_opt.casting.cast_to_16_bit_precision

### coreai_opt.casting.cast_to_16_bit_precision(exported_program, ignored_ops=None)

Convert a torch exported program to 16-bit precision: FP32→FP16 and INT32/64→INT16.

Runs both cast passes sequentially:
1. cast_fp32_to_fp16: FP32→FP16
2. cast_int32_to_int16: INT32/INT64→INT16

* **Parameters:**
  * **exported_program** (*ExportedProgram*) – Exported program to convert.
  * **ignored_ops** (*Collection* *[**OpOverload* *|* *OpOverloadPacket* *]*  *|* *OpOverload* *|* *OpOverloadPacket* *|* *Callable* *[* *[**Node* *]* *,* *bool* *]*  *|* *None*) – Optional operations or predicate to exclude from FP16 casting.
    Forwarded to [`cast_fp32_to_fp16()`](coreai_opt.casting.cast_fp32_to_fp16.md#coreai_opt.casting.cast_fp32_to_fp16).
* **Returns:**
  The modified exported program.
* **Return type:**
  *ExportedProgram*
