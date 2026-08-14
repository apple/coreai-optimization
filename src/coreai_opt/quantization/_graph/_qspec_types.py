# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Types shared by every phase of the qspec reconciliation pipeline.

Data and enums only, so no phase has to import another just to name a type.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

import torch.fx as fx

# ---------------------------------------------------------------------------
# Atomic units.
# ---------------------------------------------------------------------------


class SlotKind(enum.Enum):
    """Which side of a node a slot lives on."""

    INPUT = enum.auto()
    OUTPUT = enum.auto()


@dataclass(frozen=True)
class NodeSlot:
    """One quantization decision point, and the unit reconciliation works on.

    Each node has one ``OUTPUT`` slot (``arg_index=0``) and one ``INPUT`` slot
    per entry in ``node.all_input_nodes``, indexed by position.
    """

    node: fx.Node
    kind: SlotKind
    arg_index: int


class FieldName(enum.Enum):
    """One per settable input of
    :class:`coreai_opt.quantization.spec.QuantizationSpec`, plus
    :attr:`QUANTIZATION_TARGET`. Each reconciles independently, per
    ``_qspec_constraints._FIELD_POLICY``.

    ``quant_min`` / ``quant_max`` / ``n_bits`` / ``target_dtype`` are absent
    because ``QuantizationSpec`` derives them from ``dtype`` and ``qscheme``;
    ``ch_axis`` lives inside :attr:`GRANULARITY`; ``is_dynamic`` is torchao-only.
    """

    DTYPE = enum.auto()
    QSCHEME = enum.auto()
    QFORMULATION = enum.auto()
    GRANULARITY = enum.auto()  # carries the channel/block axis
    FLOAT_RANGE = enum.auto()  # analytic [min, max]; either bound may be None
    FAKE_QUANTIZE_CLS = enum.auto()
    QPARAM_CALCULATOR_CLS = enum.auto()
    RANGE_CALCULATOR_CLS = enum.auto()
    SCALE_DTYPE = enum.auto()
    # Weight or activation, set by which config dict the spec came from. Not a
    # QuantizationSpec attribute, but an input to construct_partial alongside
    # them, so resolution needs it to rebuild the observer.
    QUANTIZATION_TARGET = enum.auto()


@dataclass(frozen=True)
class FieldValue:
    """A value proposed for one field, plus the priority it came from.

    Lower number means higher priority, following the sort order. Reconciliation
    only ever lowers it.
    """

    value: Any
    priority: int


OP_INTRINSIC_PRIORITY = -1
"""Reserved priority for proposals from op semantics rather than user config.

Sigmoid's output being confined to ``[0, 1]`` is a fact about the operator, not
a preference, so it should outrank every user proposal. User priorities are
positions in the annotation sort order and so ``>= 0``; a negative reserved value
lets op semantics win under the ordinary "highest priority wins" rule instead of
a per-field exception.
"""


@dataclass
class ProvisionalQSpec:
    """Mutable per-observer state.

    Several slots may reference one instance; that object identity is how
    ``ShareObserverInstance`` expresses sharing.

    Attributes:
        fields (dict[FieldName, FieldValue]): Reconciled per-field state.
        declined_by (int | None): Priority of the config that declined this
            slot, or ``None``. See :attr:`declined`.
    """

    fields: dict[FieldName, FieldValue] = field(default_factory=dict)
    declined_by: int | None = None

    @property
    def declined(self) -> bool:
        """Whether a config explicitly opted this slot out of observation.

        Distinct from an empty field map, which means nothing has spoken for the
        slot — absence cannot outrank an observing peer, but a decline must.
        """
        return self.declined_by is not None


ProvisionalQSpecMap = dict[NodeSlot, ProvisionalQSpec]
"""The map every phase reads and writes.

Slots pointing at the same :class:`ProvisionalQSpec` share an observer at
runtime; slots pointing at distinct ones reconcile independently.
"""


# ---------------------------------------------------------------------------
# Cross-phase exception.
# ---------------------------------------------------------------------------


class ReconciliationError(RuntimeError):
    """Raised when reconciliation hits an unresolvable state: a must-agree field
    whose proposals disagree, or a pattern shape the generation rules can't
    handle.
    """
