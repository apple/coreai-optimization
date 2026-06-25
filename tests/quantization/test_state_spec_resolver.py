# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause


"""Resolver-level unit tests for ``StateSpecResolver``.

These tests exercise the resolver's public contract directly, without standing
up the quantizer or torch-function-mode machinery. Test fixtures construct a
small ``nn.Module`` and minimal ``ModuleCompressionComponents`` shapes by hand.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from coreai_opt._utils.insertion.torch_function.state_spec_resolver import StateSpecResolver
from coreai_opt._utils.insertion.torch_function.types import ModuleCompressionComponents
from coreai_opt._utils.torch_utils import NamedModule
from coreai_opt.config.spec import CompressionSimulatorBase


class _StubSimulator(CompressionSimulatorBase):
    """Minimal stand-in for a ``CompressionSimulatorBase`` used in tests.

    Carries a ``label`` so tests can identify which constructor produced this
    instance, and records the ``op_to_optimize`` argument the resolver passes.
    """

    def __init__(self, op_to_optimize=None, label=""):
        super().__init__()
        self.label = label
        self.op_to_optimize = op_to_optimize

    def forward(self, x):
        return x


def _stub_constructor(label):
    """Return a callable mimicking a ``PartialConstructor`` that yields ``_StubSimulator``."""

    def _make(op_to_optimize):
        return _StubSimulator(op_to_optimize=op_to_optimize, label=label)

    return _make


class _LeafA(nn.Module):
    def __init__(self):
        super().__init__()
        self.my_weight = nn.Parameter(torch.randn(2, 2))

    def forward(self, x):
        return F.linear(x, self.my_weight)


class _LeafB(nn.Module):
    def __init__(self):
        super().__init__()
        self.other_weight = nn.Parameter(torch.randn(2, 2))

    def forward(self, x):
        return F.linear(x, self.other_weight)


class _SharedWeightModel(nn.Module):
    """Two leaves whose state tensors alias the same parameter.

    After construction, ``linear2.other_weight is linear1.my_weight``.
    """

    def __init__(self):
        super().__init__()
        self.linear1 = _LeafA()
        self.linear2 = _LeafB()
        self.linear2.other_weight = self.linear1.my_weight

    def forward(self, x):
        return self.linear2(self.linear1(x))


class _ModelWithBuffer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(2, 2))
        self.register_buffer("scale", torch.ones(2))

    def forward(self, x):
        return F.linear(x, self.weight) * self.scale


def _make_resolver(model, module_components_dict, module_priority_dict):
    return StateSpecResolver(
        model=model,
        module_components_dict=module_components_dict,
        module_priority_dict=module_priority_dict,
    )


# ---------------------------------------------------------------------------
# 1. is_state_tensor membership
# ---------------------------------------------------------------------------


def test_is_state_tensor_recognizes_parameter():
    model = _LeafA()
    resolver = _make_resolver(model, {}, {"": 0})
    assert resolver.is_state_tensor(model.my_weight)


def test_is_state_tensor_recognizes_buffer():
    model = _ModelWithBuffer()
    resolver = _make_resolver(model, {}, {"": 0})
    assert resolver.is_state_tensor(model.scale)


def test_is_state_tensor_rejects_non_tensor_without_raising():
    model = _LeafA()
    resolver = _make_resolver(model, {}, {"": 0})
    assert resolver.is_state_tensor(7) is False
    assert resolver.is_state_tensor(None) is False
    assert resolver.is_state_tensor([1, 2, 3]) is False


def test_is_state_tensor_rejects_unregistered_tensor():
    model = _LeafA()
    resolver = _make_resolver(model, {}, {"": 0})
    foreign = torch.randn(2, 2)
    assert resolver.is_state_tensor(foreign) is False


# ---------------------------------------------------------------------------
# 2. get_all_local_names
# ---------------------------------------------------------------------------


def test_get_all_local_names_single_owner():
    model = _LeafA()
    resolver = _make_resolver(model, {}, {"": 0})
    assert resolver.get_all_local_names(model.my_weight) == ["my_weight"]


def test_get_all_local_names_shared_state_returns_each_owners_local_name():
    model = _SharedWeightModel()
    resolver = _make_resolver(model, {}, {"linear1": 0, "linear2": 1, "": 2})
    names = resolver.get_all_local_names(model.linear1.my_weight)
    assert sorted(names) == ["my_weight", "other_weight"]


def test_get_all_local_names_returns_empty_for_unknown_tensor():
    model = _LeafA()
    resolver = _make_resolver(model, {}, {"": 0})
    foreign = torch.randn(2, 2)
    assert resolver.get_all_local_names(foreign) == []


class _IntraModuleAliasLeaf(nn.Module):
    """Single module that aliases the same parameter under two attribute names."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(2, 2))
        self.alias = self.weight

    def forward(self, x):
        return F.linear(x, self.weight)


class _ModelWithIntraModuleAlias(nn.Module):
    def __init__(self):
        super().__init__()
        self.leaf = _IntraModuleAliasLeaf()

    def forward(self, x):
        return self.leaf(x)


def test_build_inventory_records_intra_module_aliases():
    """A single module aliasing one tensor under two attribute names must record both."""
    model = _ModelWithIntraModuleAlias()
    resolver = _make_resolver(model, {}, {"leaf": 0, "": 1})

    nm_leaf = NamedModule("leaf", model.leaf)
    entry = resolver._state_inventory[model.leaf.weight]

    assert entry.owners == [nm_leaf]
    assert set(entry.local_names) == {"weight", "alias"}

    assert set(resolver.get_all_local_names(model.leaf.weight)) == {"weight", "alias"}


# ---------------------------------------------------------------------------
# 3. resolve with op_state_spec
# ---------------------------------------------------------------------------


def test_resolve_op_state_match_caches_optimizer_at_current_priority():
    model = _LeafA()
    named = NamedModule("", model)
    components_dict = {"my_weight": _stub_constructor("op-match")}
    resolver = _make_resolver(model, {}, {"": 0})

    resolver.resolve(F.linear, model.my_weight, named, components_dict)

    optimizer = resolver.get_optimizer(model.my_weight)
    assert isinstance(optimizer, _StubSimulator)
    assert optimizer.label == "op-match"
    assert optimizer.op_to_optimize is F.linear
    cached_opt, cached_priority = resolver._optimizer_cache[model.my_weight]
    assert cached_opt is optimizer
    assert cached_priority == 0


def test_resolve_op_state_no_match_caches_none_at_current_priority():
    model = _LeafA()
    named = NamedModule("", model)
    components_dict = {"some_other_name": _stub_constructor("not-match")}
    resolver = _make_resolver(model, {}, {"": 5})

    resolver.resolve(F.linear, model.my_weight, named, components_dict)

    assert resolver.get_optimizer(model.my_weight) is None
    cached_opt, cached_priority = resolver._optimizer_cache[model.my_weight]
    assert cached_opt is None
    assert cached_priority == 5


def test_resolve_op_state_explicit_none_caches_none_at_current_priority():
    model = _LeafA()
    named = NamedModule("", model)
    components_dict = {"my_weight": None}
    resolver = _make_resolver(model, {}, {"": 3})

    resolver.resolve(F.linear, model.my_weight, named, components_dict)

    assert resolver.get_optimizer(model.my_weight) is None
    cached_opt, cached_priority = resolver._optimizer_cache[model.my_weight]
    assert cached_opt is None
    assert cached_priority == 3


# ---------------------------------------------------------------------------
# 4. resolve with module_state_spec
# ---------------------------------------------------------------------------


def test_resolve_module_state_match_cached_at_module_state_priority():
    model = _LeafA()
    named = NamedModule("", model)
    module_components_dict = {
        named: ModuleCompressionComponents(
            module_state_components={"my_weight": _stub_constructor("module-match")},
        )
    }
    resolver = _make_resolver(model, module_components_dict, {"": 0})

    resolver.resolve(F.linear, model.my_weight, named, components_dict={})

    optimizer = resolver.get_optimizer(model.my_weight)
    assert isinstance(optimizer, _StubSimulator)
    assert optimizer.label == "module-match"
    cached_opt, cached_priority = resolver._optimizer_cache[model.my_weight]
    assert cached_opt is optimizer
    assert cached_priority == StateSpecResolver._MODULE_STATE_PRIORITY


def test_resolve_module_state_not_overwritten_by_higher_priority_op_state():
    """A module_state match cached at -1 must never be overwritten by op_state.

    The skip check uses strict ``>``: ``current_priority > cached_priority``.
    Since every op-state priority is >= 0 and module-state priority is -1, the
    skip fires for any op-state visit after the module-state cache is set.
    """
    model = _SharedWeightModel()
    nm_linear1 = NamedModule("linear1", model.linear1)
    nm_linear2 = NamedModule("linear2", model.linear2)
    module_components_dict = {
        nm_linear1: ModuleCompressionComponents(
            module_state_components={"my_weight": _stub_constructor("module-spec")},
        ),
    }
    resolver = _make_resolver(model, module_components_dict, {"linear1": 0, "linear2": 5, "": 10})

    resolver.resolve(F.linear, model.linear1.my_weight, nm_linear1, components_dict={})
    cached_after_first = resolver.get_optimizer(model.linear1.my_weight)
    assert isinstance(cached_after_first, _StubSimulator)
    assert cached_after_first.label == "module-spec"

    resolver.resolve(
        F.linear,
        model.linear2.other_weight,
        nm_linear2,
        components_dict={"other_weight": _stub_constructor("op-spec-from-linear2")},
    )
    cached_after_second = resolver.get_optimizer(model.linear1.my_weight)
    assert cached_after_second is cached_after_first
    _, cached_priority = resolver._optimizer_cache[model.linear1.my_weight]
    assert cached_priority == StateSpecResolver._MODULE_STATE_PRIORITY


# ---------------------------------------------------------------------------
# 5. Priority cache ordering for shared states
# ---------------------------------------------------------------------------


def test_lower_priority_first_then_higher_priority_skipped():
    """Lower numeric priority = higher precedence; once cached, a higher-numeric
    visit must be skipped (skip check fires when ``current > cached``)."""
    model = _SharedWeightModel()
    nm_linear1 = NamedModule("linear1", model.linear1)
    nm_linear2 = NamedModule("linear2", model.linear2)
    resolver = _make_resolver(model, {}, {"linear1": 0, "linear2": 5, "": 10})

    resolver.resolve(
        F.linear,
        model.linear1.my_weight,
        nm_linear1,
        components_dict={"my_weight": _stub_constructor("from-linear1")},
    )
    first = resolver.get_optimizer(model.linear1.my_weight)
    assert first.label == "from-linear1"

    resolver.resolve(
        F.linear,
        model.linear2.other_weight,
        nm_linear2,
        components_dict={"other_weight": _stub_constructor("from-linear2")},
    )
    second = resolver.get_optimizer(model.linear2.other_weight)
    assert second is first
    assert second.label == "from-linear1"
    _, cached_priority = resolver._optimizer_cache[model.linear1.my_weight]
    assert cached_priority == 0


def test_higher_priority_first_then_lower_priority_overwrites():
    """A subsequent visit with lower numeric priority (higher precedence)
    overwrites the cache because the skip check ``current > cached`` is False."""
    model = _SharedWeightModel()
    nm_linear1 = NamedModule("linear1", model.linear1)
    nm_linear2 = NamedModule("linear2", model.linear2)
    resolver = _make_resolver(model, {}, {"linear1": 0, "linear2": 5, "": 10})

    resolver.resolve(
        F.linear,
        model.linear2.other_weight,
        nm_linear2,
        components_dict={"other_weight": _stub_constructor("from-linear2")},
    )
    first = resolver.get_optimizer(model.linear2.other_weight)
    assert first.label == "from-linear2"

    resolver.resolve(
        F.linear,
        model.linear1.my_weight,
        nm_linear1,
        components_dict={"my_weight": _stub_constructor("from-linear1")},
    )
    second = resolver.get_optimizer(model.linear1.my_weight)
    assert second.label == "from-linear1"
    _, cached_priority = resolver._optimizer_cache[model.linear1.my_weight]
    assert cached_priority == 0


def test_equal_priorities_last_writer_wins():
    """Strict ``>`` skip check means equal priorities do NOT trigger skip — the
    later writer overwrites the earlier one."""
    model = _SharedWeightModel()
    nm_linear1 = NamedModule("linear1", model.linear1)
    nm_linear2 = NamedModule("linear2", model.linear2)
    resolver = _make_resolver(model, {}, {"linear1": 0, "linear2": 0, "": 10})

    resolver.resolve(
        F.linear,
        model.linear1.my_weight,
        nm_linear1,
        components_dict={"my_weight": _stub_constructor("first-writer")},
    )
    first = resolver.get_optimizer(model.linear1.my_weight)
    assert first.label == "first-writer"

    resolver.resolve(
        F.linear,
        model.linear2.other_weight,
        nm_linear2,
        components_dict={"other_weight": _stub_constructor("second-writer")},
    )
    second = resolver.get_optimizer(model.linear2.other_weight)
    assert second is not first
    assert second.label == "second-writer"
    _, cached_priority = resolver._optimizer_cache[model.linear1.my_weight]
    assert cached_priority == 0


# ---------------------------------------------------------------------------
# 6. get_optimizer
# ---------------------------------------------------------------------------


def test_get_optimizer_returns_cached_value():
    model = _LeafA()
    named = NamedModule("", model)
    resolver = _make_resolver(model, {}, {"": 0})

    resolver.resolve(
        F.linear,
        model.my_weight,
        named,
        components_dict={"my_weight": _stub_constructor("cached")},
    )

    optimizer = resolver.get_optimizer(model.my_weight)
    assert isinstance(optimizer, _StubSimulator)
    assert optimizer.label == "cached"


def test_get_optimizer_returns_none_for_unresolved_tensor():
    model = _LeafA()
    resolver = _make_resolver(model, {}, {"": 0})
    assert resolver.get_optimizer(model.my_weight) is None


# ---------------------------------------------------------------------------
# 7. Module-state walk caching (skip redundant walks after first visit)
# ---------------------------------------------------------------------------


def test_optimizer_cache_set_after_first_visit_no_match():
    """Test that the ``_optimizer_cache`` contains the state_tensor after the first call even when
    no match.
    """
    model = _SharedWeightModel()
    nm_linear1 = NamedModule("linear1", model.linear1)
    resolver = _make_resolver(model, {}, {"linear1": 0, "linear2": 5, "": 10})

    assert model.linear1.my_weight not in resolver._optimizer_cache

    resolver.resolve(F.linear, model.linear1.my_weight, nm_linear1, components_dict={})

    assert model.linear1.my_weight in resolver._optimizer_cache
