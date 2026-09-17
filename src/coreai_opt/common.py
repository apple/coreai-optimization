# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Common enums and constants for coreai_opt."""

from __future__ import annotations

import warnings
from enum import EnumMeta, StrEnum as _StrEnum, auto
from typing import TYPE_CHECKING, Any, ClassVar


class _DeprecatedMemberEnumMeta(EnumMeta):
    """Enum metaclass that emits DeprecationWarning for renamed members.

    Define ``__deprecated_aliases__`` as a dict mapping old member names to current
    member names. The metaclass validates it at class-creation time and intercepts both
    attribute access and value lookup.

    Example:
        >>> class Color(_StrEnum, metaclass=_DeprecatedMemberEnumMeta):
        ...     RED = auto()
        ...     BLUE = auto()
        ...     __deprecated_aliases__: ClassVar[dict[str, str]] = {"CRIMSON": "RED"}
        >>> Color.CRIMSON is Color.RED        # warns
        True
        >>> Color("crimson") is Color.RED     # warns, case-insensitive
        True

    """

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwds: Any,
    ) -> _DeprecatedMemberEnumMeta:
        cls = super().__new__(mcs, name, bases, namespace, **kwds)
        aliases = namespace.get("__deprecated_aliases__")
        if not aliases:
            msg = (
                f"{name} uses {mcs.__name__} but does not define a non-empty "
                f"'__deprecated_aliases__'. If there are no deprecations to "
                f"track, do not use this metaclass."
            )
            raise TypeError(msg)
        for old_name, new_name in aliases.items():
            if new_name not in cls._member_map_:
                msg = (
                    f"{name}.__deprecated_aliases__: alias {old_name!r} -> "
                    f"{new_name!r} references unknown member {new_name!r}"
                )
                raise ValueError(msg)
            if old_name in cls._member_map_:
                msg = f"{name}.__deprecated_aliases__: alias {old_name!r} shadows a real member"
                raise ValueError(msg)
        return cls

    def __getattr__(cls, name: str) -> Any:
        aliases: dict[str, str] = cls.__dict__.get("__deprecated_aliases__", {})
        if name in aliases:
            new_name = aliases[name]
            warnings.warn(
                f"{cls.__name__}.{name} is deprecated, use "
                f"{cls.__name__}.{new_name} instead. "
                f"The old name will be removed in a future release.",
                DeprecationWarning,
                stacklevel=2,
            )
            return cls[new_name]
        return super().__getattr__(name)

    def __call__(cls, value: object, *args: Any, **kwargs: Any) -> Any:
        try:
            return super().__call__(value, *args, **kwargs)
        except ValueError:
            # Only intercept simple value lookups: EnumCls("old_value").
            # Calls with extra arguments construct new enum classes and
            # must pass through unchanged.
            if args or kwargs or not isinstance(value, str):
                raise
            aliases: dict[str, str] = cls.__dict__.get("__deprecated_aliases__", {})
            value_lower = value.lower()
            for old_name, new_name in aliases.items():
                if old_name.lower() == value_lower:
                    member = cls[new_name]
                    warnings.warn(
                        f"{cls.__name__}('{value}') is deprecated, use "
                        f"{cls.__name__}('{member.value}') or "
                        f"{cls.__name__}.{new_name} instead. "
                        f"The old value will be removed in a future release.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                    return member
            raise


# CoreML compression type codes (for MIL export compatibility)
_COREML_COMPRESSION_CODES: dict[str, int] = {
    "quantization": 3,
    "palettization": 2,
    "pruning": 1,
}


class CompressionType(_StrEnum):
    """Enum representing compression techniques applied to the model.

    Each member is a string value representing the compression type.
    """

    QUANTIZATION = auto()
    PALETTIZATION = auto()
    PRUNING = auto()

    def to_coreml_code(self) -> int:
        """Convert to CoreML compression type code.

        Returns:
            CoreML-specific integer code for this compression type

        Raises:
            ValueError: If no CoreML code mapping exists for this compression type

        """
        coreml_code = _COREML_COMPRESSION_CODES.get(self.value)
        if coreml_code is None:
            msg = f"No CoreML code mapping for {self.value}"
            raise ValueError(msg)
        return coreml_code


class ExportBackend(_StrEnum, metaclass=_DeprecatedMemberEnumMeta):
    """Enum representing supported model export backends.

    Each member is a string value representing the backend format.

    Attributes:
        CoreML: Core ML format with compression metadata buffers.
        CoreAI: Core AI format with custom ops.

    """

    _TORCH = auto()
    CoreML = auto()
    CoreAI = auto()

    __deprecated_aliases__: ClassVar[dict[str, str]] = {"MIL": "CoreML", "MLIR": "CoreAI"}

    if TYPE_CHECKING:
        # Surface the deprecated aliases above for static type checkers.
        MIL: ExportBackend
        """Deprecated. Use `ExportBackend.CoreML` instead."""

        MLIR: ExportBackend
        """Deprecated. Use `ExportBackend.CoreAI` instead."""


class CoreMLExportError(ValueError):
    """Raised when a model cannot be exported to the CoreML backend."""

    def __init__(self, message: str) -> None:
        super().__init__(f"{message} Use backend=ExportBackend.CoreAI instead.")

    @classmethod
    def from_dtype(cls, dtype: Any, context: str) -> CoreMLExportError:
        """Build the error for an unsupported weight/activation/LUT dtype."""
        return cls(f"CoreML export does not support dtype {dtype} on {context}.")

    @classmethod
    def from_config(cls, config: object, context: str) -> CoreMLExportError:
        """Build the error for an unsupported quantization config attribute (e.g. granularity)."""
        return cls(f"CoreML export does not support {type(config).__name__} on {context}.")
