# coreai_opt.casting.cast_fp32_to_fp16

### coreai_opt.casting.cast_fp32_to_fp16(exported_program, ignored_ops=None)

Convert a torch exported program from FP32 to FP16 where applicable.

Converts parameters, user inputs, and compute ops to FP16, inserting
casts only where values would overflow FP16 range.

* **Parameters:**
  * **exported_program** (*ExportedProgram*) – Exported program to convert.
  * **ignored_ops** (*Collection* *[**OpOverload* *|* *OpOverloadPacket* *]*  *|* *OpOverload* *|* *OpOverloadPacket* *|* *Callable* *[* *[**Node* *]* *,* *bool* *]*  *|* *None*) – Optional operations or predicate to exclude from FP16 casting.
    Can be a collection/set of `OpOverload` or `OpOverloadPacket`
    instances (e.g. `{torch.ops.aten.exp, torch.ops.aten.exp.default}`),
    a single op instance, or a predicate callable taking a `torch.fx.Node`
    and returning a boolean (e.g. `lambda node: node.name == "exp_1"`).
    Ignored ops are kept in FP32 with boundary casts inserted.
* **Returns:**
  The modified exported program.
* **Return type:**
  *ExportedProgram*
