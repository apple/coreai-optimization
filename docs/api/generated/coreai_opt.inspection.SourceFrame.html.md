# coreai_opt.inspection.SourceFrame

### *class* coreai_opt.inspection.SourceFrame(filename, lineno, function_name, code_context)

Bases: `object`

A single frame in the source call stack leading to an operation.

Represents one level of the call hierarchy, typically a `forward()`
method in the user’s model code.

* **Parameters:**
  * **filename** (*str*)
  * **lineno** (*int*)
  * **function_name** (*str*)
  * **code_context** (*str*)

#### filename

Absolute or relative path to the source file.

* **Type:**
  str

#### lineno

Line number in the source file.

* **Type:**
  int

#### function_name

Name of the function (e.g., `"forward"`).

* **Type:**
  str

#### code_context

The source code text on that line, stripped of
leading/trailing whitespace.

* **Type:**
  str

#### \_\_init_\_(filename, lineno, function_name, code_context)

* **Parameters:**
  * **filename** (*str*)
  * **lineno** (*int*)
  * **function_name** (*str*)
  * **code_context** (*str*)
* **Return type:**
  None

#### code_context *: str*

#### filename *: str*

#### function_name *: str*

#### lineno *: int*
