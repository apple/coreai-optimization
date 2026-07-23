# Code Style Guide

This guide documents coding conventions and best practices for the Core AI Optimization project. Following these recommendations ensures consistent, maintainable code and helps both developers and AI code assistants understand project-specific patterns.

## Table of Contents

- [Table of Contents](#table-of-contents)
- [1. Package & Module Structure](#1-package--module-structure)
  - [1.1 Package Organization](#11-package-organization)
  - [1.2 Naming Conventions](#12-naming-conventions)
    - [Module Path](#module-path)
    - [Symbol Names](#symbol-names)
- [2. Private vs Public API](#2-private-vs-public-api)
  - [2.1 Understanding Visibility Levels](#21-understanding-visibility-levels)
  - [2.2 Organizing Public API](#22-organizing-public-api)
  - [2.3 `__all__` Usage in Internal vs Public Modules](#23-__all__-usage-in-internal-vs-public-modules)
- [3. Import Conventions](#3-import-conventions)
  - [3.1 Prefer Direct Imports](#31-prefer-direct-imports)
  - [3.2 Absolute vs Relative Imports](#32-absolute-vs-relative-imports)
  - [3.3 Aliasing Internal Imports in Public Modules](#33-aliasing-internal-imports-in-public-modules)
    - [Why this matters](#why-this-matters)
    - [Best practice](#best-practice)
    - [When to use this pattern](#when-to-use-this-pattern)
    - [Enforcement](#enforcement)
  - [3.4 Import Examples](#34-import-examples)
- [4. Type Annotations](#4-type-annotations)
  - [4.1 When to Annotate](#41-when-to-annotate)
  - [4.2 Examples](#42-examples)
- [5. Object Attributes](#5-object-attributes)
  - [5.1 Instance Attributes](#51-instance-attributes)
  - [5.2 Class Attributes](#52-class-attributes)
  - [5.3 Examples](#53-examples)
- [6. Method and Function Design](#6-method-and-function-design)
  - [6.1 Static Methods](#61-static-methods)
- [7. Docstrings](#7-docstrings)
  - [7.1 General Rules](#71-general-rules)
  - [7.2 One-Line Docstrings](#72-one-line-docstrings)
  - [7.3 Multiline Docstrings](#73-multiline-docstrings)
    - [7.3.1 Args Section](#731-args-section)
    - [7.3.2 Returns Section](#732-returns-section)
    - [7.3.3 Raises Section](#733-raises-section)
    - [7.3.4 Type Information in Docstrings](#734-type-information-in-docstrings)
  - [7.4 Classes](#74-classes)
  - [7.5 Modules and Packages](#75-modules-and-packages)
    - [Public Modules](#public-modules)
    - [Test Modules](#test-modules)
  - [7.6 Sphinx-Compatible Formatting (Napoleon)](#76-sphinx-compatible-formatting-napoleon)
    - [7.6.1 Napoleon Section Headers](#761-napoleon-section-headers)
    - [7.6.2 Code Examples in Docstrings](#762-code-examples-in-docstrings)
    - [7.6.3 Lists and Indentation](#763-lists-and-indentation)
    - [7.6.4 Section Ordering](#764-section-ordering)
    - [7.6.5 Links in Docstrings](#765-links-in-docstrings)

## 1. Package & Module Structure

### 1.1 Package Organization

Create `__init__.py` to declare a directory as a package when you want to use it as a collection of modules.

### 1.2 Naming Conventions

#### Module Path

Path that contains `_` is project-internal, otherwise public.

Examples:

```python
# proj_pkg/sub_pkg/helpers.py:  Public, should be in __all__ of sub_pkg/__init__.py
# proj_pkg/_utils/helpers.py:   Project-internal (has _ in path)
# proj_pkg/_helpers.py:         Project-internal (has _ in path)
```

#### Symbol Names

- Classes: `PascalCase` (e.g., `MyClass`, `SomeHelper`)
- Functions/methods: `snake_case` (e.g., `my_function()`, `some_method()`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `SOME_CONSTANT`, `MAX_VALUE`)
- Variables: `snake_case` (e.g., `some_variable`, `my_value`)
- Symbols without `_` prefix: see [Module Path](#module-path)
- Symbols with `_` prefix: file-internal only (should not be imported by other modules)

## 2. Private vs Public API

### 2.1 Understanding Visibility Levels

**1. Public API** - For external users:

- No `_` in module path
- No `_` in symbol name
- Must be listed in `__all__` of `__init__.py`

**Note:** Symbols in public modules (no `_` in path) without `_` in their name are technically public even if not in `__all__`, but this violates the style guide. Always add public symbols to `__all__`.

```python
# proj_pkg/common.py
class PublicClass:
    pass


class Helper:
    pass


# proj_pkg/__init__.py
from proj_pkg.common import PublicClass

__all__ = ["PublicClass"]  # Public API
# class Helper is also Public but not in __all__ - should be added
```

**2. Project-Internal** - Usable across the project:

- Has `_` in module path (e.g., `proj_pkg/_internal/`, `proj_pkg/_utils.py`)
- No `_` in symbol name

```python
# proj_pkg/_internal/helpers.py (has _ in path)
class InternalHelper:  # Project-internal, usable across project
    pass
```

**3. File-Internal** - Private to one file:

- Has `_` in symbol name (e.g., `_helper()`, `_PrivateClass`)
- Should not be imported by other modules

```python
# proj_pkg/api.py
def process_request() -> None:  # OK to import
    _prepare_data()


def _prepare_data() -> None:  # File-internal: don't import this
    pass
```

### 2.2 Organizing Public API

Public symbols should be exported through the `__init__.py` of the most appropriate package level for user consumption:

```python
# Example 1: General symbols applicable across all optimization techniques
# Export through /coreai_opt/__init__.py
from .common import SomeGeneralClass, SomeGeneralEnum

__all__ = ["SomeGeneralClass", "SomeGeneralEnum"]

# Import in client code
from coreai_opt import SomeGeneralClass, SomeGeneralEnum

# Example 2: Quantization-specific symbols
# Export through /coreai_opt/quantization/__init__.py
from .core import QuantizationSpecificClass

__all__ = ["QuantizationSpecificClass"]

# Import in client code
from coreai_opt.quantization import QuantizationSpecificClass

# Example 3: Palettization-specific symbols
# Export through /coreai_opt/palettization/__init__.py
from .core import PalettizationSpecificClass

__all__ = ["PalettizationSpecificClass"]

# Import in client code
from coreai_opt.palettization import PalettizationSpecificClass
```

**Guidelines:**

- Choose the export level that makes semantic sense for the symbol's scope
- General project-wide symbols → `/coreai_opt/__init__.py`
- Domain-specific symbols (quantization, palettization, etc.) → `/coreai_opt/<domains>/__init__.py`
- Public symbols can remain in their own `.py` files but should be exported through the appropriate `__init__.py` so clients don't need to know the internal structure
- Optionally use `/coreai_opt/common.py` for commonly used constants, enums, and general project-wide objects that don't belong to any specific subpackage

### 2.3 `__all__` Usage in Internal vs Public Modules

`__all__` controls wildcard import behavior (`from module import *`) and documents a module's public interface. It determines API visibility following the underscore prefix convention.

**Recommendations:**

- **Public modules (no underscore in the path):** Define `__all__` listing all public API symbols
- **Internal modules (with `_` prefix in the path):** Can use `__all__` to list project-internal symbols for convenience
  - The module remains project-internal because the package path is underscored, regardless of `__all__`

```python
# proj_pkg/utils/__init__.py (public module)
from proj_pkg.utils.helpers import public_function

__all__ = ["public_function"]  # Lists public API symbols

# External users can import as public API
from proj_pkg.utils import public_function

# proj_pkg/_utils/__init__.py (internal module)
from proj_pkg._utils.helpers import helper_function

__all__ = ["helper_function"]  # Lists project-internal symbols

# Internal users can import as project-wide helper function
from proj_pkg._utils import helper_function

# External users cannot import as public because _utils is private
```

**Note:** Using `__all__` in internal modules (with `_` prefix) provides convenience for organizing project-internal utilities. It allows developers to import from the package level (`from proj_pkg._utils import helper_function`) rather than needing to know the specific module structure (`from proj_pkg._utils.helpers import helper_function`).

## 3. Import Conventions

### 3.1 Prefer Direct Imports

**Within the project:**

- Import project-internal symbols from their defining modules
- Or use the public API if that's what you're depending on

**External users:**

- Must import only from the documented public API entrypoints

**Avoid:**

- "Chained" imports (getting a symbol via a random module that re-exported it)

```python
# proj_pkg/utils.py
class Helper:
    pass


# proj_pkg/__init__.py
from proj_pkg.utils import Helper

__all__ = ["Helper"]


# proj_pkg/myclass.py
from proj_pkg.utils import Helper

# Good: Import from original source or public API
from proj_pkg.utils import Helper  # Within project: direct from defining module
from proj_pkg import Helper  # External users or internal code depending on public API

# Avoid: Importing from re-exporting module
from proj_pkg.myclass import Helper  # Chained import - unclear origin
```

This ensures clarity about the symbol's origin and avoids confusion.

### 3.2 Absolute vs Relative Imports

**Prefer absolute imports:**

Use absolute imports from the project root as the default:

```python
# proj_pkg/subpkg_a/module.py
from proj_pkg.utils import SomeClass
from proj_pkg.subpkg_a.config import Config
```

**Relative imports are acceptable when:**

- Importing within the same subpackage
- The import is clearly internal (not part of public API)
- The import is a short hop (`.` or `..`)
- Breaking dependency cycles between modules

```python
# proj_pkg/subpkg_a/module.py
from .config import Config  # Same subpackage, short hop
from ..utils import SomeClass  # Parent package, short hop
```

**Prefer Consistency:**

- Do not mix absolute and relative import styles for the same target across different files
- Public APIs re-exported in `__init__.py` should always be imported using absolute imports, not relative

```python
# proj_pkg/subpkg_a/__init__.py exports Config
# When importing from another file:

# Good: Use absolute imports for public API
from proj_pkg.subpkg_a import Config

# Avoid: Relative imports for public API
from .subpkg_a import Config  # Don't use this style
```

### 3.3 Aliasing Internal Imports in Public Modules

When a public module needs to use project-internal symbols, always alias the imported symbol with a `_` prefix. This prevents clients from accidentally re-importing internal symbols from your public modules.

#### Why this matters

Without aliasing, project-internal symbols become accessible through public module namespaces, creating unintended API surface:

```python
# proj_pkg/api.py (public module)
from proj_pkg._internal import private_helper  # Without alias

# Client code can now accidentally access private_helper
from proj_pkg.api import private_helper  # Unintended access to internal symbol
```

#### Best practice

```python
# proj_pkg/api.py (public module)
from proj_pkg._internal import private_helper as _private_helper


class PublicApi:
    def process(self) -> None:
        result = _private_helper()  # Use the aliased name internally
```

Now clients cannot accidentally import `private_helper` from the public module:

```python
# Client code
from proj_pkg.api import PublicApi  # ✓ Works
from proj_pkg.api import private_helper  # ✗ AttributeError - not accessible
```

#### When to use this pattern

- When importing project-internal symbols into public modules to prevent unintended re-export

#### Enforcement

The pre-commit hook `check-internal-import-aliases` (defined in `.pre-commit-config.yaml`, implemented at `scripts/pre_commit/check_internal_import_aliases.py`) enforces this rule. On commit it **fails** when it finds a violation and prints each one with a suggested fix, but it does not edit files — so the commit is blocked until the imports are corrected. Fix them by hand, or run the script with `--fix` to apply the fixable ones automatically:

- Add the missing `_` alias in public modules.
- Remove the unnecessary `_` alias in private modules.
- Rename every reference to the bound name in the same file so the rewrite leaves the module compiling.
- Report (without fixing) when the bound name is shadowed inside a function scope (rare) — resolve those by hand.

Fixing is a separate, opt-in step rather than an automatic commit-time rewrite: auto-editing on commit would interleave this hook's changes with those of the formatting/linting hooks, making them hard to review or revert. Run `python scripts/pre_commit/check_internal_import_aliases.py --fix` (optionally with specific paths) to fix everything in one go, then review and commit the changes.

### 3.4 Import Examples

**Importing from Public API:**

```python
# proj_pkg/__init__.py (public module)
from proj_pkg.common import PublicClass

__all__ = ["PublicClass"]

# Client code - import from public API
from proj_pkg import PublicClass  # ✓ Public API import
```

**Importing Project-Internal to Project-Internal Module:**

```python
# proj_pkg/_internal/helpers.py (project-internal)
class InternalHelper:
    pass


# proj_pkg/_utils/processor.py (project-internal)
from proj_pkg._internal.helpers import InternalHelper  # ✓ No prefix needed
```

**Importing Project-Internal to Public Module:**

```python
# proj_pkg/_internal/helpers.py (project-internal)
def internal_helper() -> None:
    pass


# proj_pkg/api.py (public module)
from proj_pkg._internal.helpers import (
    internal_helper as _internal_helper,
)  # ✓ Aliased with _


class PublicApi:
    def process(self) -> None:
        _internal_helper()  # Use aliased name
```

**Avoid Importing File-Internal Symbols:**

```python
# proj_pkg/module.py
def _file_helper() -> None:  # File-internal
    pass


# proj_pkg/other_module.py
from proj_pkg.module import _file_helper  # ✗ Don't import _ symbols
```

## 4. Type Annotations

### 4.1 When to Annotate

**Annotate when:**

- Class properties
- Empty collections where the content type isn't obvious from context
- Function parameters and return types

**Don't annotate when:**

- Type is obvious from the assignment, e.g., `count = 0`, `msg = "error"`
- Variable receives a return value from a function call
- Simple variables with clear names
- Local variables in short scopes

### 4.2 Examples

```python
# Annotate these
config_map: dict[str, Config] = {}
metadata: dict[str, dict[str, Any]] = {}
file_set: set[Path] = set()


# Class properties - always annotate
class Model:
    name: str
    layers: list[Layer]
    config: dict[str, Any]

    def __init__(self, name: str) -> None:
        self.name = name
        self.layers = []
        self.config = {}


# Don't annotate these
msg = "Expected positive value"
count = 0
x, y = parse_coordinates(data)
result = process_data(input_value)  # Return value from function
```

## 5. Object Attributes

Object attributes include both class attributes (shared across all instances) and instance attributes (unique to each instance).

### 5.1 Instance Attributes

Instance attributes should be declared at the class level with type annotations but without assignment. This documents what attributes instances are expected to have without making them shared across instances.

```python
class Shape:
    # Declare instance attributes at class level (no default value)
    dimensions: list[int]
    name: str

    def __init__(self, dimensions: list[int], name: str):
        # Assign values inside __init__ - each instance gets its own copy
        self.dimensions = dimensions
        self.name = name
```

Rules for instance attributes:

- Declare at class level with type annotation only (no assignment)
- Never assign mutable defaults (like `[]`, `{}`) at class level - they become shared
- Assign default values inside `__init__` to ensure each instance gets its own copy
- Do not repeat type annotations inside `__init__` if already declared at class level

### 5.2 Class Attributes

Class attributes are truly shared across all instances and must be marked with `ClassVar[...]`. This makes the intention explicit and prevents confusion.

```python
from typing import ClassVar


class Shape:
    # Class attribute - shared across all instances
    max_dimensions: ClassVar[int] = 4

    # Instance attributes - each instance has its own
    dimensions: list[int]
    name: str
```

Rules for class attributes:

- Always use `ClassVar[...]` annotation for shared class-level data
- Can assign values at class level since they are intentionally shared
- Makes intention clear for both readers and type checkers

### 5.3 Examples

```python
from typing import ClassVar


class Shape:
    # Class attribute - shared across all instances
    # Use ClassVar to make it explicit this is class-level data
    registry: ClassVar[dict[str, type]] = {}
    max_dimensions: ClassVar[int] = 4

    # Instance attributes - declare at class level with type only (no default)
    # This documents what attributes instances will have
    dimensions: list[int]
    name: str
    metadata: dict[str, str]

    def __init__(self, dimensions: list[int], name: str):
        # Assign values for instance attributes inside __init__
        # Each instance gets its own copy
        self.dimensions = dimensions
        self.name = name
        self.metadata = {}
        # Do not repeat type annotations here - already declared at class level
```

Bad examples - what NOT to do:

```python
class BadShape:
    # ✗ BAD: Mutable default at class level - becomes shared across instances!
    dimensions: list[int] = []  # All instances share the same list

    # ✗ BAD: Assigning instance attribute at class level
    name: str = "default"  # Shared across all instances

    # ✗ BAD: Class attribute without ClassVar annotation
    max_dimensions: int = 4  # Ambiguous intent

    def __init__(self, dimensions: list[int]):
        # ✗ BAD: Repeating type annotation when already declared at class level
        self.dimensions: list[int] = dimensions
```

Correct version of above:

```python
from typing import ClassVar


class GoodShape:
    # ✓ GOOD: Class attribute with ClassVar
    max_dimensions: ClassVar[int] = 4

    # ✓ GOOD: Instance attributes declared at class level (no default)
    dimensions: list[int]
    name: str

    def __init__(self, dimensions: list[int], name: str = "default"):
        # ✓ GOOD: Assign values in __init__ - each instance gets its own
        self.dimensions = dimensions
        self.name = name
        # ✓ GOOD: No repeated type annotations
```

## 6. Method and Function Design

### 6.1 Static Methods

Use `@staticmethod` for any method that does not access `self` or `cls`. Do not default to instance methods — if a method only operates on its explicit arguments, make it static.

```python
class Processor:
    # ✓ GOOD: Does not use self, declared as @staticmethod
    @staticmethod
    def _compute_output_size(width: int, height: int) -> tuple[int, int]:
        return width * 2, height * 2

    # ✗ BAD: Does not use self, but declared as instance method
    def _compute_output_size(self, width: int, height: int) -> tuple[int, int]:
        return width * 2, height * 2
```

Calling a static method via `self` (e.g., `self._compute_output_size(w, h)`) is valid Python and does not need to change when converting an instance method to a static method.

## 7. Docstrings

Follow [PEP 257](https://peps.python.org/pep-0257/) and [Google Python Style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for all docstrings in the project. Sphinx generates the project's HTML documentation, and the [Napoleon](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/) extension (`sphinx.ext.napoleon`) lets it read Google-style docstrings directly — no reStructuredText (RST) knowledge needed. Just follow the formatting rules below.

### 7.1 General Rules

- Use triple **double** quotes: `"""..."""`
- Start the text immediately after the opening `"""` — no newline between the opening quotes and first line of text
- Place docstring as the first statement in a module, class, function, or method
- Write docstrings for functions/methods that meet one or more of these criteria:
  - Part of the public API
  - Project-internal utilities used across multiple modules
  - Nontrivial size or complexity
  - Non-obvious logic or behavior
- First line: short summary (one physical line, max 120 characters) ending with a period
- Preferred style: write the summary as a command — "Return X." rather than "Returns X." or "This function returns X."
  - Must be consistent within a module — don't mix imperative and descriptive mood in the same file
- For `@property` methods: use noun-phrase style ("The file path.") not verb style ("Return the file path.")
- Don't repeat the function signature in the docstring (Python introspection already provides this)
- Include type information in multiline docstrings (see [7.3.4](#734-type-information-in-docstrings))
- For multiline docstrings, place the closing `"""` on its own line

### 7.2 One-Line Docstrings

Use for simple, self-explanatory functions. Keep everything on a single line:

```python
def add(a: int, b: int) -> int:
    """Return the sum of two integers."""
    return a + b
```

Avoid:

```python
def add(a: int, b: int) -> int:
    """add(a, b) -> int"""  # ✗ Don't repeat signature
    return a + b
```

### 7.3 Multiline Docstrings

Use when describing parameters, return values, side effects, or usage details.

Format:

1. Summary line (same rules as [7.1](#71-general-rules))
2. One blank line
3. Details using section headers, plus any additional context
4. Closing `"""` on its own line

Use these section headers in multiline docstrings. Plural forms (`Examples:`, `Notes:`, `Warnings:`) also work:

| Header        | Purpose                                                                                           |
| ------------- | ------------------------------------------------------------------------------------------------- |
| `Attributes:` | Public attributes of a class                                                                      |
| `Args:`       | Function/method parameters                                                                        |
| `Returns:`    | Return value description (mutually exclusive with `Yields:`)                                      |
| `Yields:`     | Generator yield value description (mutually exclusive with `Returns:`)                            |
| `Raises:`     | Exceptions the function may raise                                                                 |
| `Note:`       | Important information the caller should be aware of                                               |
| `Warning:`    | Potential pitfalls or dangerous behavior                                                          |
| `See Also:`   | Cross-references to related functions or classes                                                  |
| `Todo:`       | Planned improvements or known limitations                                                         |
| `Example:`    | Usage examples in doctest format (`>>>` lines simulate an interactive Python session; place last) |

Use only these headers — unrecognized headers are passed through as plain RST text, which can cause "Unexpected indentation" errors when followed by indented content (see [7.6.1](#761-napoleon-section-headers)).

#### 7.3.1 Args Section

List parameters under an `Args:` header. Each entry uses the format `name (type): description`:

```python
def process_data(input_path: str, max_size: int = 1024) -> list[str]:
    """Process data from the input file.

    Args:
        input_path (str): Path to the input file to process.
        max_size (int): Maximum size in bytes to process (default 1024).

    Returns:
        list[str]: List of processed data strings.
    """
    pass
```

For longer descriptions, use continuation lines with 4-space indentation:

```python
def process_data(input_path: str, max_size: int = 1024) -> list[str]:
    """Process data from the input file.

    Args:
        input_path (str): Path to the input file to process. This can be
            a relative or absolute path.
        max_size (int): Maximum size in bytes to process (default 1024).

    Returns:
        list[str]: List of processed data strings.
    """
    pass
```

For `*args` and `**kwargs`, list them by name with their expected types:

```python
def log_messages(level: str, *args: str, **kwargs: Any) -> None:
    """Log one or more messages at the given level.

    Args:
        level (str): Log level name (e.g., "INFO", "DEBUG").
        *args (str): Message strings to log.
        **kwargs (Any): Additional key-value pairs included in the log record.
    """
    pass
```

#### 7.3.2 Returns Section

Document return values for functions that:

- Are part of the public API
- Return non-obvious values
- Have multiple possible return types or special return conditions

Omit `Returns:` for:

- Functions with obvious returns (e.g., simple getters)
- Functions returning `None` (unless the None has special meaning)
- Very simple one-line docstrings

```python
def calculate_metrics(data: list[float]) -> dict[str, float]:
    """Calculate statistical metrics from data.

    Args:
        data (list[float]): Input values for analysis.

    Returns:
        dict[str, float]: Dictionary with keys 'mean', 'median', 'stddev'.
    """
    pass


def get_status() -> str:
    """Return the current connection status."""  # Simple — no Returns: needed
    return self._status
```

#### 7.3.3 Raises Section

Document exceptions that a function may raise, especially for public APIs. Place `Raises:` after `Returns:` (see [7.6.4](#764-section-ordering) for full ordering).

```python
def divide(numerator: float, denominator: float) -> float:
    """Divide two numbers.

    Args:
        numerator (float): Value to be divided.
        denominator (float): Value to divide by.

    Returns:
        float: Result of the division.

    Raises:
        ZeroDivisionError: If denominator is zero.
    """
    if denominator == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return numerator / denominator
```

For multiple exception types:

```python
def load_config(path: str) -> dict[str, Any]:
    """Load configuration from a file.

    Args:
        path (str): Path to the configuration file.

    Returns:
        dict[str, Any]: Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file contains invalid configuration format.
        PermissionError: If the file cannot be read due to permissions.
    """
    pass
```

Guidelines:

- Document exceptions that callers should handle or be aware of
- Include the exception type and a brief explanation of when it occurs
- Don't document exceptions caused by violating the function's stated preconditions (e.g., `TypeError` from passing a wrong argument type). Document exceptions that can arise during normal, correct usage (e.g., `FileNotFoundError`, `ConnectionError`)
- Document exceptions raised by called functions if they propagate to the caller

#### 7.3.4 Type Information in Docstrings

Always include type information in multiline docstrings, even when type hints are present in the signature. Single-line docstrings don't need type information. This duplicates the hints but makes docstrings self-contained — readers of the generated API reference see types without consulting the source code.

Format: use parentheses notation `name (type): description`

Function examples:

```python
# Single-line — no type info needed
def get_timeout() -> int:
    """Return the connection timeout in seconds."""
    return self._timeout


# Multiline — include type info with semantic meaning
def parse_config(data: str) -> dict[str, Any]:
    """Parse JSON configuration string into a settings dictionary.

    Args:
        data (str): JSON-formatted configuration string.

    Returns:
        dict[str, Any]: Parsed configuration as a dictionary.

    Raises:
        ValueError: If data is not valid JSON.
    """
    ...
```

Class attributes example:

```python
class Shape:
    """Geometric shape with area calculation.

    Attributes:
        name (str): Human-readable shape name.
        sides (int): Number of sides in the shape.
        dimensions (list[float]): List of dimension measurements in meters.
    """
```

### 7.4 Classes

- Place class docstring directly under the class definition
- Leave one blank line after the docstring before the first method or attribute
- Document public attributes in an `Attributes:` section using the same format as `Args:`
- Include type annotations for class attributes on the same line as the attribute name
- Document constructor parameters in the class docstring's `Args:` section, not in a separate `__init__` docstring. The `__init__` docstring should be a brief one-liner or omitted entirely
- Properties: document in the `@property` docstring only, not in `Attributes:`. Sphinx generates property documentation from the `@property` docstring, so duplicating it in `Attributes:` produces redundant entries

```python
class Foo:
    """Short description of the class.

    Attributes:
        x (int): Description of x.
        y (str): Description of y.

    Args:
        x (int): Initial value for x.
        y (str): Initial value for y.
    """

    x: int
    y: str

    def __init__(self, x: int, y: str) -> None:
        self.x = x
        self.y = y

    @property
    def prop(self) -> float:
        """The computed property."""
        return float(self.x)
```

### 7.5 Modules and Packages

#### Public Modules

- Each public module should have a top-level docstring explaining its purpose
- Package `__init__.py` docstrings should summarize the package and list main exports

```python
"""A one-line summary of the module or program, terminated by a period.

Leave one blank line. The rest of this docstring should contain an
overall description of the module or program. Optionally, it may also
contain a brief description of exported classes and functions and/or usage
examples when the API is not self-evident.

Typical usage example:

    >>> foo = ClassFoo()
    >>> bar = foo.function_bar()
"""
```

#### Test Modules

- Test module docstrings are optional
- Only include a docstring when providing additional context beyond what the filename conveys
- Examples of useful test docstrings: special setup requirements, instructions for running tests with specific options, dependencies on external resources

Avoid docstrings that merely restate what is obvious from the filename or code structure.

```python
# ✗ Don't do this — provides no value
"""Tests for widget.py."""

# ✓ Do this — provides useful information
"""Tests for Widget class.

These tests require mock authentication tokens. Run with:
    pytest test_widget.py --mock-auth
"""
```

### 7.6 Sphinx-Compatible Formatting (Napoleon)

This project uses [Napoleon](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/) (`sphinx.ext.napoleon`) to parse Google-style docstrings into reStructuredText (RST) for the Sphinx documentation build. Content within Napoleon sections is still parsed as RST, which means certain formatting patterns that look fine in Python will cause Sphinx build errors. This subsection documents the correct patterns.

#### 7.6.1 Napoleon Section Headers

Only the headers listed in [7.3](#73-multiline-docstrings) are recognized by Napoleon. Unrecognized headers (e.g., `Usage:`, `IMPORTANT:`, `Constraints:`) cause RST "Unexpected indentation" errors. Always use a recognized header instead:

```python
def some_function():
    """Do something.

    Example:
        >>> result = some_function()
    """
```

#### 7.6.2 Code Examples in Docstrings

Napoleon's `Example:` section treats its content as free-form RST, not as a code block. Use doctest format (`>>>` and `...`) so Sphinx renders it as code:

```python
class MyClass:
    """A class.

    Example:
        >>> obj = MyClass()
        >>> obj.process()
    """
```

For multiline code, use `...` continuation markers:

```python
class MyMixin:
    """A mixin class.

    Example:
        >>> class MySubclass(MyMixin, BaseClass):
        ...     @classmethod
        ...     def get_default(cls):
        ...         return {"key": "value"}
    """
```

#### 7.6.3 Lists and Indentation

Insert a blank line between a paragraph and a following numbered or bulleted list:

```python
"""Summary.

This class provides:

1. Feature A
2. Feature B
"""
```

When nesting bullets inside a list item, add blank lines before and after the nested list:

```python
"""
- "default": Context-aware default:

  * For weights: StaticCalculator
  * For activations: MovingAverageCalculator

- "static": Direct calculation
"""
```

Continuation lines in `Args:`/`Raises:` must be indented past the description start:

```python
"""
Raises:
    ValueError: If the input is invalid and
        cannot be processed.
"""
```

#### 7.6.4 Section Ordering

Use the following order for docstring sections. Omit any sections that don't apply:

1. Summary line
2. Extended description (free-form prose)
3. `Attributes:` (classes only)
4. `Args:`
5. `Returns:` or `Yields:`
6. `Raises:`
7. `Note:` / `Warning:` / `See Also:` / `Todo:`
8. `Example:` (recommended last)

`Example:` must be the last section. While Napoleon can technically parse sections after it, content following `Example:` often causes RST parsing errors. If a note must appear alongside an example, place it before:

```python
"""Summary.

Note:
    Important note about this class.

Example:
    >>> obj = MyClass()
"""
```

#### 7.6.5 Links in Docstrings

Docstrings are parsed as RST, not Markdown. Markdown link syntax (`[text](url)`) renders as literal text in the API reference. Use one of these formats instead:

- Named link: `` `link text <https://url>`_ ``
- Bare URL (auto-linked): `https://url`

```python
"""Quantize weights to FP8 format.

For details on FP8 formats, see `the OCP spec <https://arxiv.org/pdf/2209.05433>`_.
Bare URLs are also auto-linked: https://arxiv.org/pdf/2209.05433
"""
```
