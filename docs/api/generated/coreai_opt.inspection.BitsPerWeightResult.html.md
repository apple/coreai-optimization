# coreai_opt.inspection.BitsPerWeightResult

### *class* coreai_opt.inspection.BitsPerWeightResult(bpw, per_module_map, total_bits, total_weights)

Bases: `object`

Result of a bits-per-weight computation.

* **Parameters:**
  * **bpw** (*float*)
  * **per_module_map** (*dict* *[**str* *,* *float* *]*)
  * **total_bits** (*int*)
  * **total_weights** (*int*)

#### bpw

Overall average bits per weight across all parameters
(`total_bits / total_weights`).

* **Type:**
  float

#### per_module_map

Map from module name to that module’s own
average bits per weight. Modules with no tensors are omitted.

* **Type:**
  dict[str, float]

#### total_bits

Total storage cost in bits, including amortized
compression overhead.

* **Type:**
  int

#### total_weights

Total number of parameter elements.

* **Type:**
  int

#### \_\_init_\_(bpw, per_module_map, total_bits, total_weights)

* **Parameters:**
  * **bpw** (*float*)
  * **per_module_map** (*dict* *[**str* *,* *float* *]*)
  * **total_bits** (*int*)
  * **total_weights** (*int*)
* **Return type:**
  None

#### bpw *: float*

#### per_module_map *: dict[str, float]*

#### total_bits *: int*

#### total_weights *: int*
