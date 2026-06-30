# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""PyTorch 2.0 Export (PT2E) quantization implementation.

This module provides quantization functionality using PyTorch 2.0's export and
quantization APIs, supporting both post-training quantization (PTQ) and
quantization-aware training (QAT) workflows.
"""

from __future__ import annotations

import copy
import itertools
import logging
import re
from collections import defaultdict
from collections.abc import Mapping
from contextlib import contextmanager
from enum import Enum, auto
from os import PathLike
from typing import Any, NamedTuple, TypeAlias

import torch
import torchao
from torch.fx.graph_module import _USER_PRESERVED_ATTRIBUTES_KEY
from torchao.quantization.pt2e import (
    allow_exported_model_train_eval,
    disable_fake_quant,
    disable_observer,
    enable_fake_quant,
    enable_observer,
)
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_qat_pt2e
from torchao.quantization.pt2e.quantizer import (
    QuantizationAnnotation,
    Quantizer as TorchPT2EQuantizer,
)
from torchao.quantization.pt2e.quantizer.quantizer import Q_ANNOTATION_KEY

from coreai_opt._utils.config_utils import (
    ALL_TENSORS as _ALL_TENSORS,
    ConfigLevel as _ConfigLevel,
)
from coreai_opt._utils.fx_utils import (
    get_node_type as _get_node_type,
    normalize_module_fqn,
)
from coreai_opt._utils.torch_utils import (
    export_model as _export_model,
    move_model_to_eval,
    move_model_to_train,
)
from coreai_opt._utils.version_utils import version_ge as _version_ge
from coreai_opt.common import ExportBackend
from coreai_opt.config.compression_config import ModuleConfigDict, _build_module_alias_map
from coreai_opt.quantization._axis_defaults import (
    apply_weight_axis_defaults_graph as _apply_weight_axis_defaults,
    validate_activation_axes as _validate_activation_axes,
)
from coreai_opt.quantization.base_quantizer import _BaseQuantizer
from coreai_opt.quantization.config import (
    KVCacheQuantConfig,
    ModuleQuantizerConfig,
    OpQuantizerConfig,
    QuantizerConfig,
)
from coreai_opt.quantization.config.quantization_config import _ACTIVATION_SPEC_DICT
from coreai_opt.quantization.spec.fake_quantize import (
    FakeQuantizeImplBase,
)

from ._annotation_config import AnnotationConfig, AnnotationContext
from ._annotation_pattern_registry import (
    AnnotatorMatchInfo as _AnnotatorMatchInfo,
    SharedObserverModulePattern as _SharedObserverModulePattern,
    _AnnotationPatternRegistry,
    _make_kv_cache_update_pattern,
)
from ._annotation_utils import (
    _get_input_qspec_map,
    adjust_output_qspec_for_qscheme_and_propagate,
    annotate_module_level_specs as _annotate_module_level_specs,
    is_node_annotated,
)
from ._conv_bn_utils import (
    fold_conv_bn_weights as _fold_conv_bn_weights,
    remove_conv_bn_zeros_like_dtype as _remove_conv_bn_zeros_like_dtype,
)
from ._prepare_for_export import (
    _move_cache_dequant_to_output,
    prepare_for_mil_export,
    prepare_for_mlir_export,
)
from ._utils import (
    force_per_tensor_for_channel_altering_ops as _force_per_tensor_for_channel_altering_ops,
    get_source_module_name as _get_source_module_name,
    remove_fake_quant_nodes as _remove_fake_quant_nodes,
    restore_kwargs as _restore_kwargs,
    strip_non_aten_metadata_kwargs as _strip_non_aten_metadata_kwargs,
)

logger = logging.getLogger(__name__)


class _OpConfigLevel(Enum):
    """
    Enum to specify the op-level config type within a module config.

    Enum entries should be defined in order of highest priority to lowest priority.

    - OP_NAME: Applied to ops matching specific name patterns
    - OP_TYPE: Applied to ops matching specific types
    - DEFAULT: Default op input/output/state specs from the module config
    """

    OP_NAME = auto()
    OP_TYPE = auto()
    DEFAULT = auto()

    @classmethod
    def priority_order(cls) -> list[_OpConfigLevel]:
        """Return op config levels in priority order (highest to lowest)."""
        return list(cls)


class _NodePriorityConfig(NamedTuple):
    """Config attached to a node, paired with its priority within a config level.

    Attributes:
        config (OpQuantizerConfig): The op-level config to apply at this node.
        priority (int): Position of the matching module in
            ``module_config_dict[level]``. Lower = higher precedence within
            the level (matches eager-mode ``module_priority_dict`` semantics:
            ``build_module_config_dict`` processes user configs in reverse, so
            the last-listed user config claims modules first and gets the
            smallest index).
    """

    config: OpQuantizerConfig
    priority: int


class _RankedAnnotation(NamedTuple):
    """One annotator match ranked for the priority sort.

    Attributes:
        node (torch.fx.Node): The node being annotated.
        config (OpQuantizerConfig): The op-level config to apply.
        match (_AnnotatorMatchInfo): The annotator match info for ``node``.
        priority (int): Within-level priority carried over from
            :class:`_NodePriorityConfig`.
    """

    node: torch.fx.Node
    config: OpQuantizerConfig
    match: _AnnotatorMatchInfo
    priority: int


NodeConfigDict: TypeAlias = dict[_OpConfigLevel, dict[torch.fx.Node, _NodePriorityConfig]]


class _AnnotationHandler(TorchPT2EQuantizer):
    """
    Handles quantization annotations for PyTorch models using PT2E framework.

    This class extends TorchAO's PT2E quantizer to provide custom annotation
    logic for different quantization patterns and module configurations.
    """

    def __init__(
        self,
        module_configs: ModuleConfigDict,
        module_name_to_state_names_map: Mapping[str, Mapping[str, list[str]]],
        canonical_to_aliases: dict[str, list[str]],
        extra_patterns: list[type] | None = None,
        kv_cache_quant_configs: dict[str, KVCacheQuantConfig] | None = None,
    ):
        """
        Initialize the annotation handler.

        Args:
            module_configs: Dictionary mapping config level scopes to dictionaries which
                map module names to their respective quantization configurations. Each
                configuration specifies how the corresponding module should be
                quantized.
            module_name_to_state_names_map: A two level dictionary mapping module names
                to another dictionary. The inner dictionary maps full state names to all
                local names in that module which points to the state object referenced
                by the full state name.
            canonical_to_aliases: Mapping from canonical module name to all
                known names (aliases) for the same module object.  Used during node
                matching so that a node whose nn_module_stack carries an alias path is
                still matched to the correct canonical config entry.
            extra_patterns: Per-instance annotation pattern classes. Used alongside
                the global ``_AnnotationPatternRegistry`` for matching during this
                handler's lifetime; not registered globally.
            kv_cache_quant_configs: When set, a post-annotation override forcibly
                applies each cache spec to every matched cache-update op, even if a
                module-scope config would otherwise have claimed the op. Makes each
                cache spec a global-only knob that wins over module-scope shadowing.
        """
        if module_configs is None:
            raise ValueError("Module configurations cannot be None")
        self._module_configs = module_configs
        self._module_name_to_state_names_map = module_name_to_state_names_map
        self._extra_patterns: list[type] = list(extra_patterns or [])
        self._kv_cache_quant_configs = kv_cache_quant_configs or {}

        # Build reverse map: alias → canonical for O(1) lookup during node matching.
        self._alias_to_canonical: dict[str, str] = {
            alias: canonical
            for canonical, aliases in canonical_to_aliases.items()
            for alias in aliases
        }

    def _all_patterns(self) -> list[type]:
        """Globally-registered patterns plus per-instance ``extra_patterns``."""
        return list(_AnnotationPatternRegistry.list_registry_values()) + self._extra_patterns

    def _get_shared_observer_nodes(self, model: torch.fx.GraphModule) -> set[torch.fx.Node]:
        """
        Return a set of all shared observer nodes in the model.
        """
        shared_observer_annotators = [
            a_class
            for a_class in self._all_patterns()
            if issubclass(a_class, _SharedObserverModulePattern)
        ]
        shared_observer_nodes = set()
        for annotator in shared_observer_annotators:
            node_to_annotator_and_match_dict = annotator._match_all_patterns(model)

            # We only care about the nodes
            for node in node_to_annotator_and_match_dict:
                shared_observer_nodes.add(node)
        return shared_observer_nodes

    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        """
        Annotate the model with quantization specifications.

        This method applies quantization annotations to modules and operations
        based on the provided configuration. It processes module-specific configurations
        first, then handles operation-specific configurations.

        Args:
            model: The FX GraphModule to annotate.

        Returns:
            The annotated GraphModule.
        """
        if not isinstance(model, torch.fx.GraphModule):
            raise TypeError("Model must be a torch.fx.GraphModule")

        # Matching phase - apply all annotators to model
        node_to_annotator_match_info_dict = self._match_all_annotators(model)

        # Sorting phase - sort all matches in the order to try annotating in
        sorted_nodes_with_annotation_match_info = self._sort_nodes_in_annotation_order(
            model, node_to_annotator_match_info_dict
        )

        # Annotation phase - go through sorted nodes with matches list to annotate
        shared_observer_nodes = self._get_shared_observer_nodes(model)
        # Build pass-invariant context once; shared by all annotator invocations.
        context = AnnotationContext(
            module_name_to_state_names_map=self._module_name_to_state_names_map,
            shared_observer_nodes=shared_observer_nodes,
        )
        for node, config, annotator_match_info in sorted_nodes_with_annotation_match_info:
            if is_node_annotated(node):
                continue
            annotation_config = AnnotationConfig.from_quantizer_config(config)
            annotator_match_info.annotator_func(
                annotator_match_info.annotator_match,
                annotation_config,
                context,
            )

        _annotate_module_level_specs(
            self._module_configs, self._module_name_to_state_names_map, model
        )

        # Post-annotation graph pass to go through the model and adjust any qspecs which
        # are following always affine or fixed range nodes.
        for node in model.graph.nodes:
            adjust_output_qspec_for_qscheme_and_propagate(node, shared_observer_nodes)

        # Force-apply each cache spec last so it wins over any module-scope
        # annotation that may have claimed the cache op via a wildcard. Cache
        # specs are global-only knobs (see KVCacheQuantConfig docstring).
        if self._kv_cache_quant_configs:
            self._override_cache_op_annotations(model, context)
        return model

    def _override_cache_op_annotations(
        self,
        model: torch.fx.GraphModule,
        context: AnnotationContext,
    ) -> None:
        """Overwrite each cache-op node's annotation with its cache spec.

        Mirrors what the standard annotator would produce for a cache-op match
        (``annotate_n_ary_act_match`` → ``_get_input_qspec_map``), but applied
        last so it wins over any prior annotation a module-scope config may
        have produced via a wildcard ``op_input_spec``/``op_output_spec``.
        """
        for node in model.graph.nodes:
            if node.op != "call_function":
                continue
            op_type = _get_node_type(node, warn_on_failure=False)
            kc = self._kv_cache_quant_configs.get(op_type)
            if kc is None:
                continue
            ann_config = AnnotationConfig.from_quantizer_config(kc.op_quantizer_config)
            input_qspec_map = _get_input_qspec_map(node.all_input_nodes, ann_config, context)
            node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=None,
                _annotated=True,
            )

    def _match_all_annotators(
        self,
        model: torch.fx.GraphModule,
    ) -> dict[torch.fx.Node, list[_AnnotatorMatchInfo]]:
        """
        Given an exported model, use all registered annotators to match nodes in the
        model. Build a dictionary mapping nodes to _AnnotatorMatchInfo objects
        containing information about matched patterns.
        """
        node_to_annotator_match_info_dict: dict[torch.fx.Node, list[_AnnotatorMatchInfo]] = {}
        for annotator_class in self._all_patterns():
            # Match patterns for this annotator across entire model
            annotator_node_match_dict = annotator_class._match_all_patterns(model)
            for node, annotator_match_info in annotator_node_match_dict.items():
                # A node may already correspond to other matches.
                # Ex. a Conv node may have been earlier matched with Conv -> BN -> ReLU,
                # and now it is being processed again for Conv only match.
                # This is ok, we keep track of all matches associated with the node in
                # a list.
                if node in node_to_annotator_match_info_dict:
                    node_to_annotator_match_info_dict[node].append(annotator_match_info)
                else:
                    node_to_annotator_match_info_dict[node] = [annotator_match_info]
        return node_to_annotator_match_info_dict

    def _sort_nodes_in_annotation_order(
        self,
        model: torch.fx.GraphModule,
        node_to_annotator_match_info_dict: dict[torch.fx.Node, list[_AnnotatorMatchInfo]],
    ) -> list[tuple[torch.fx.Node, OpQuantizerConfig, _AnnotatorMatchInfo]]:
        """
        Produce a list of tuples of (node, config to apply, annotator match info) which
        will later be iterated to annotate matches.

        The same node can appear multiple times in the list being associated with
        different annotator match info objects if it was matched with multiple patterns.

        The list is sorted with the following criteria, in decreasing priority:
        - Config type (module_name > module_type > global)
        - Pattern length (Longer pattern > shorter pattern)
        - Config index within a config level (Later config > earlier config)
        - Topological ordering in the model (Earlier in the graph > later in the graph)
        """
        config_level_node_dicts = self._get_config_level_node_dicts(
            model, node_to_annotator_match_info_dict
        )

        nodes_with_annotation_match_info: list[
            tuple[torch.fx.Node, OpQuantizerConfig, _AnnotatorMatchInfo]
        ] = []
        for config_level_node_dict in config_level_node_dicts:
            for op_config_level in _OpConfigLevel.priority_order():
                nodes_with_annotation_match_info.extend(
                    self._expand_and_sort_nodes_for_pattern_length(
                        config_level_node_dict[op_config_level], node_to_annotator_match_info_dict
                    )
                )

        return nodes_with_annotation_match_info

    def _get_config_level_node_dicts(
        self,
        model: torch.fx.GraphModule,
        node_to_annotator_match_info_dict: dict[torch.fx.Node, list[_AnnotatorMatchInfo]],
    ) -> tuple[NodeConfigDict, NodeConfigDict, NodeConfigDict]:
        """
        Create and return three dicts corresponding to config levels for mapping nodes
        to appropriate quantizer configs using information in self._module_configs.

        The only purpose of node_to_annotator_match_info_dict is to optimize the number
        of nodes for which we track configs for since we can skip nodes which we don't
        intend to annotate.

        For each node, the config is determined by the innermost module in its
        nn_module_stack at the most specific config level possible
        (module_name > module_type > global).
        """
        global_node_config_dict: NodeConfigDict = {level: {} for level in _OpConfigLevel}
        module_type_node_config_dict: NodeConfigDict = {level: {} for level in _OpConfigLevel}
        module_name_node_config_dict: NodeConfigDict = {level: {} for level in _OpConfigLevel}

        # Precompute {canonical_key: insertion_index} once per config level so that
        # _set_config_to_use_for_node can look up a key's priority in O(1)
        config_key_index: dict[_ConfigLevel, dict[object, int]] = {
            level: {key: idx for idx, key in enumerate(self._module_configs[level].keys())}
            for level in (_ConfigLevel.MODULE_NAME, _ConfigLevel.MODULE_TYPE)
        }

        # Iterating through the nodes in topological ordering guarantees that when
        # sorting nodes later, any nodes with identical config priority and pattern
        # lengths will remain ordered by topological ordering (essentially the last
        # tiebreaker).
        for node in model.graph.nodes:
            if node in node_to_annotator_match_info_dict:
                # Try to find a config to set for the node for module_name config level
                if self._set_config_to_use_for_node(
                    node,
                    module_name_node_config_dict,
                    _ConfigLevel.MODULE_NAME,
                    config_key_index[_ConfigLevel.MODULE_NAME],
                ):
                    continue

                # Try to find a config to set for the node for module_type config level
                if self._set_config_to_use_for_node(
                    node,
                    module_type_node_config_dict,
                    _ConfigLevel.MODULE_TYPE,
                    config_key_index[_ConfigLevel.MODULE_TYPE],
                ):
                    continue

                # Take a shortcut for global since the config will be the same for all
                # other modules (no need to name match).
                global_config = list(self._module_configs[_ConfigLevel.GLOBAL].values())[0]

                config, op_config_level = self._get_config_for_node(node, global_config)
                # GLOBAL level has a single config, so priority is trivially 0.
                global_node_config_dict[op_config_level][node] = _NodePriorityConfig(
                    config, priority=0
                )

        return (module_name_node_config_dict, module_type_node_config_dict, global_node_config_dict)

    def _set_config_to_use_for_node(
        self,
        node: torch.fx.Node,
        node_config_dict: NodeConfigDict,
        config_level: _ConfigLevel,
        config_key_index: dict[object, int],
    ) -> bool:
        """
        Add a node to config entry for node_config_dict for the given config_level if
        applicable. Returns True if a config was set, False otherwise.

        The stored entry pairs the matched config with its position in
        ``self._module_configs[config_level]``. That position is used as a
        within-level priority during the sort phase: lower index = higher
        precedence.
        """
        qualified_name = _get_source_module_name(node)
        if qualified_name is None:
            return False

        name = normalize_module_fqn(qualified_name)

        # Always use the canonical name to look up in self._module_configs
        canonical = self._alias_to_canonical.get(name, name)
        if canonical not in self._module_configs[config_level]:
            return False

        config_to_use = self._module_configs[config_level][canonical]
        config_idx = config_key_index[canonical]
        config_to_use, op_config_level = self._get_config_for_node(node, config_to_use)
        node_config_dict[op_config_level][node] = _NodePriorityConfig(config_to_use, config_idx)
        return True

    @staticmethod
    def _get_config_for_node(
        node: torch.fx.Node, module_level_config: ModuleQuantizerConfig
    ) -> tuple[OpQuantizerConfig, _OpConfigLevel]:
        """
        Given a node and a module_level_config, return the config to apply for the node
        specifically.
        """
        # Use reversed to identify the last matching config
        # Check for op_name match
        for op_name, op_name_config in reversed(module_level_config.op_name_config.items()):
            try:
                if re.fullmatch(op_name, node.name):
                    return op_name_config, _OpConfigLevel.OP_NAME
            except re.error as e:
                error_msg = f"Invalid regex pattern '{op_name}' in op_name_config: {op_name_config}"
                raise ValueError(error_msg) from e

        # Check for op_type match
        for op_type, op_type_config in reversed(module_level_config.op_type_config.items()):
            node_type = _get_node_type(node)
            if node_type is not None and op_type == node_type:
                return op_type_config, _OpConfigLevel.OP_TYPE

        # If no matching op_name or op_type found, use default op input/output/state
        # settings as found in ModuleQuantizerConfig.
        config = OpQuantizerConfig(
            op_input_spec=module_level_config.op_input_spec,
            op_output_spec=module_level_config.op_output_spec,
            op_state_spec=module_level_config.op_state_spec,
        )
        return config, _OpConfigLevel.DEFAULT

    def _expand_and_sort_nodes_for_pattern_length(
        self,
        node_to_config_dict: dict[torch.fx.Node, _NodePriorityConfig],
        node_to_annotator_match_info_dict: dict[torch.fx.Node, list[_AnnotatorMatchInfo]],
    ) -> list[tuple[torch.fx.Node, OpQuantizerConfig, _AnnotatorMatchInfo]]:
        """
        Return a list of lists consisting of (node, config, annotator match info).

        This method combines information from node_to_config_dict and
        node_to_annotator_match_info_dict to create a sorted list which will later be
        traversed to apply annotations to matches.

        The sorting will ensure that nodes which matched longer patterns appear
        earlier in the list to give priority. A node can potentially match with multiple
        annotations of different lengths. In such cases, multiple tuple entries for the
        same node will be created in the sorted list, but will be interleaved with other
        potentially duplicated nodes based on length of the different patterns a node
        matched with.

        For example, say there is a model with conv1 -> relu1 ->conv2 -> relu2, and
        annotation patterns exist for Conv and Conv->ReLU. We will start with

        node_to_config_dict = {'conv1': config1, 'conv2': config2}
        # conv1 appears before conv2 in node_to_config_dict due to topological sorting

        node_to_annotator_match_info_dict =
            {'conv1': [Conv->Relu annotation match info, Conv annotation match info],
             'conv2': [Conv->Relu annotation match info, Conv annotation match info]}

        The final sorted and returned list would look like
        [['conv1', config1, Conv->ReLU annotation match info],
         ['conv2', config2, Conv->ReLU annotation match info],
         ['conv1', config1, Conv annotation match info],
         ['conv2', config2, Conv annotation match info]]

        Note how conv1 and conv2 have interleaved entries in the final sorted list since
        length 2 Conv->ReLU pattern takes precedence over length 1 Conv pattern.
        Within nodes with equal pattern lengths, the topological ordering is preserved.
        Because conv1 appears earlier in the model than conv2, conv1 appears earlier in
        the list compared to conv2 whenever the pattern length is equal.

        Args:
            node_to_config_dict: Maps nodes to corresponding configs. This is an
                ordered dict in which nodes will show up in topological order as seen in
                the exported model

            node_to_annotator_match_info_dict: Maps nodes to tuples of annotator match
                info. A node can be associated with multiple annotations, for
                example a Conv node may have matched Conv, Conv->BN, and Conv->BN->ReLU
                if these nodes were present in the model.

        Returns:
            A list of lists of (node, config, annotation match info) ordered by
            priority.
        """
        nodes_with_annotation_info: list[_RankedAnnotation] = [
            _RankedAnnotation(node, entry.config, match, entry.priority)
            for node, entry in node_to_config_dict.items()
            for match in node_to_annotator_match_info_dict[node]
        ]
        # Higher pattern_length wins; within equal length, lower priority wins
        # (later-listed user configs claim modules first, so they get the
        # smaller index in module_config_dict). Stable sort preserves
        # topological order as the final tiebreaker.
        nodes_with_annotation_info.sort(key=lambda r: (-r.match.pattern_length, r.priority))

        return [(r.node, r.config, r.match) for r in nodes_with_annotation_info]

    def validate(self, model: torch.fx.GraphModule) -> None:
        """
        Validate the annotated model.

        Args:
            model: The annotated GraphModule to validate.

        Note:
            Currently a no-op. Future implementations may add validation logic
            to ensure annotations are correctly applied.
        """
        pass


class GraphQuantizer(_BaseQuantizer):
    """
    Graph-mode quantizer implementation, built on top of ``torchao``'s PT2E framework.

    This quantizer provides a complete quantization workflow using PyTorch 2.0's
    export and quantization APIs. It supports both post-training quantization (PTQ)
    and quantization-aware training (QAT) workflows with proper state management
    to ensure correct usage patterns.

    The quantizer follows a structured workflow:
    1. prepare() - Prepares the model for quantization (required first step)
       - Produces a data-free PTQ compressed model
       - For weight-only PTQ: prepare() → finalize() is sufficient
       - For activation quantization: model may have poor accuracy until calibrated
    2(a). calibration_mode() - Context manager for calibration (optional, PTQ only)
       - Required for activation quantization to achieve good accuracy
    2(b). training_mode() - Context manager for quantization-aware training (QAT)
       - Enables fine-tuning the prepared model with the quantization active
    3. finalize() - Converts to final quantized model (required last step)

    Example:
        Basic usage with default configuration:

        >>> import torch
        >>> from coreai_opt.quantization._graph import GraphQuantizer
        >>>
        >>> # Your model
        >>> model = MyModel()
        >>>
        >>> # Create quantizer with default config
        >>> quantizer = GraphQuantizer(model)
        >>>
        >>> # Prepare for quantization
        >>> example_input = torch.randn(1, 10)
        >>> prepared_model = quantizer.prepare((example_input,))
        >>>
        >>> # For activation quantization: Run calibration data
        >>> with quantizer.calibration_mode():
        ...     for batch in calibration_dataloader:
        ...         prepared_model(batch)
        >>>
        >>> # Finalize quantized model
        >>> quantized_model = quantizer.finalize()
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: QuantizerConfig,
    ):
        """
        Initialize the PT2E quantizer.

        Args:
            model: The PyTorch model to quantize. Must be traceable by torch.export.
            config: Optional quantization configuration. If None, default configuration
                   will be used.
        """
        if not isinstance(model, torch.nn.Module):
            raise TypeError("Model must be a torch.nn.Module")

        self._validate_config(config)
        super().__init__(model, config)

    @staticmethod
    def _fill_input_output_specs_to_check_for_module_config(
        specs_to_check_for_index_or_wildcard: list[_ACTIVATION_SPEC_DICT],
        specs_to_check_for_zero_or_wildcard: list[_ACTIVATION_SPEC_DICT],
        module_config: ModuleQuantizerConfig,
    ):
        """
        Helper function to populate lists of op_input_specs and op_output_specs to
        validate.
        """
        # Combine all configs: module config + op_name_config + op_type_config
        all_configs = itertools.chain(
            [module_config],
            module_config.op_name_config.values(),
            module_config.op_type_config.values(),
        )

        # Extract specs from all configs
        for config in all_configs:
            specs_to_check_for_index_or_wildcard.append(config.op_input_spec)
            specs_to_check_for_zero_or_wildcard.append(config.op_output_spec)
            if isinstance(config, ModuleQuantizerConfig):
                specs_to_check_for_index_or_wildcard.append(config.module_input_spec)
                specs_to_check_for_index_or_wildcard.append(config.module_output_spec)

    @staticmethod
    def _validate_config(config: QuantizerConfig) -> None:
        """
        Validate the current state of support for config. This function will be
        continually updated as support for different aspects of QuantizerConfig is
        implemented.
        """
        specs_to_check_for_index_or_wildcard = []
        specs_to_check_for_zero_or_wildcard = []
        # Get op input/output specs for global
        GraphQuantizer._fill_input_output_specs_to_check_for_module_config(
            specs_to_check_for_index_or_wildcard,
            specs_to_check_for_zero_or_wildcard,
            config.global_config,
        )

        # Get op input/output specs for each module type
        for module_config in config.module_type_configs.values():
            GraphQuantizer._fill_input_output_specs_to_check_for_module_config(
                specs_to_check_for_index_or_wildcard,
                specs_to_check_for_zero_or_wildcard,
                module_config,
            )

        # Get op input/output specs for each module name
        for module_config in config.module_name_configs.values():
            GraphQuantizer._fill_input_output_specs_to_check_for_module_config(
                specs_to_check_for_index_or_wildcard,
                specs_to_check_for_zero_or_wildcard,
                module_config,
            )

        # Check for currently unsupported string based input/output tensor
        # identification.
        for spec in specs_to_check_for_index_or_wildcard:
            for key in spec:
                if isinstance(key, str) and key != _ALL_TENSORS:
                    error_msg = (
                        "Only integer indices or '*' are supported for op and module "
                        f"input and output specs currently. Got {spec}"
                    )
                    raise NotImplementedError(error_msg)

        # Check for currently unsupported nonzero index for output tensors ("*" is still
        # valid)
        for spec in specs_to_check_for_zero_or_wildcard:
            for key in spec:
                if key not in [_ALL_TENSORS, 0]:
                    error_msg = (
                        "op_output_qspec currently supports setting for '*' or 0 "
                        f"tensor only. Got {spec}"
                    )
                    raise NotImplementedError(error_msg)

    @staticmethod
    def _get_module_name_to_state_names_map(
        model: torch.nn.Module,
    ) -> Mapping[str, Mapping[str, list[str]]]:
        """
        Get a two level dictionary mapping module names to another dictionary. The
        inner dictionary maps full state names to all local names in that module which
        points to the state object referenced by the full state name.
        """
        # Step 1: get dictionaries mapping states to full names
        # A state can have multiple full names associated with it if it is shared by
        # multiple modules.
        state_to_full_name_map: defaultdict[torch.nn.Parameter, list[str]] = defaultdict(list)
        for full_name, param in model.named_parameters(remove_duplicate=False):
            state_to_full_name_map[param].append(full_name)
        for full_name, buffer in model.named_buffers(remove_duplicate=False):
            state_to_full_name_map[buffer].append(full_name)

        # Step 2: go through each module in the model and associate each full name of
        # parameters defined in that model with all local names for that parameter in
        # the model.

        # For example, consider modules with states
        # inner_model_1: (a, b, c)
        # inner_model_2: (a, b)
        # and set the following shared associations:
        # inner_model_1.a = inner_model_1.b = inner_model_2.b
        # inner_model_1.c = inner_model_2.a

        # Then we would have:
        # module_name_to_state_names["inner_model_1"]["inner_model_1.a"] = ["a", "b"]
        # module_name_to_state_names["inner_model_1"]["inner_model_1.b"] = ["a", "b"]
        # module_name_to_state_names["inner_model_1"]["inner_model_2.b"] = ["a", "b"]
        # module_name_to_state_names["inner_model_1"]["inner_model_1.c"] = ["c"]
        # module_name_to_state_names["inner_model_1"]["inner_model_2.a"] = ["c"]

        # module_name_to_state_names["inner_model_2"]["inner_model_2.b"] = ["b"]
        # module_name_to_state_names["inner_model_2"]["inner_model_1.a"] = ["b"]
        # module_name_to_state_names["inner_model_2"]["inner_model_1.b"] = ["b"]
        # module_name_to_state_names["inner_model_2"]["inner_model_1.c"] = ["a"]
        # module_name_to_state_names["inner_model_2"]["inner_model_2.a"] = ["a"]

        # Observe that since "inner_model_1.a", "inner_model_1.b", and "inner_model_2.a"
        # all refer to the same parameter object, both inner_model_1 and inner_model_2
        # contain all 3 of these full names as keys. However, from the perspective of
        # inner_model_1, there are only two local names which would point to this param:
        # "a" and "b". Thus all 3 full names are associated with ["a", "b"] in
        # module_name_to_state_names["inner_model_1"].
        # From inner_model_2's perspective, the same parameter would be referenced by
        # local name "b" only, so all 3 full names map to ["b"].

        # It is necessary to have mappings for all full names of a parameter in all
        # modules for which there is some local name pointing to the parameter because
        # during torch.export, typically only one node will be created and used wherever
        # the shared param is consumed. This node will have a node.target string
        # matching one of the multiple possible full names of a parameter. Thus a module
        # which has some local reference to the parameter would need to associate that
        # particular full name to its local reference, even if that specific module's
        # name doesn't show up in the full name used by the node.

        module_name_to_state_names = {}
        for module_name, module in model.named_modules():
            module_name_to_state_names[module_name] = defaultdict(list)
            for local_name, param in module.named_parameters(remove_duplicate=False, recurse=False):
                for full_name in state_to_full_name_map[param]:
                    module_name_to_state_names[module_name][full_name].append(local_name)
            for local_name, buffer in module.named_buffers(remove_duplicate=False, recurse=False):
                for full_name in state_to_full_name_map[buffer]:
                    module_name_to_state_names[module_name][full_name].append(local_name)

        return module_name_to_state_names

    def _kv_cache_extra_patterns(self) -> list[type]:
        """Build per-run annotation patterns for the configured cache ops.

        Returns the list of patterns to pass as ``extra_patterns`` to
        ``_AnnotationHandler``. Empty when ``kv_cache_quant_configs`` is unset.

        The final cache-op annotation is written by
        ``_AnnotationHandler._override_cache_op_annotations``; these patterns
        only ensure the cache op is visited during standard annotation. Scoped
        to the handler instance rather than the process-global
        ``_AnnotationPatternRegistry``.
        """
        configs = self._config.kv_cache_quant_configs or {}
        return [_make_kv_cache_update_pattern(op) for op in configs]

    def _validate_kv_cache_quant_ops(
        self,
        exported_model: torch.fx.GraphModule,
    ) -> None:
        """Validate ``kv_cache_quant_configs`` entries against the prepared graph.

        Runs after ``torch.export``. Catches two classes of misconfiguration
        that the annotator would otherwise let through silently:

        - A ``kv_cache_quant_configs`` key matches no node (e.g. a typo). The
          pattern would simply match nothing and the finalize-side rewrite
          would have nothing to relocate.
        - A ``kc.op_quantizer_config.op_input_spec`` keys an input index that's
          out of range for the matched op. The annotator would silently skip
          the index (``_get_input_qspec_map`` only enumerates the op's actual
          inputs), so no observer is inserted on the cache op's input edge and
          the failure would surface much later in the finalize-side rewrite.
        """
        configs = self._config.kv_cache_quant_configs
        if not configs:
            return
        for op, kc in configs.items():
            matched = [
                n
                for n in exported_model.graph.nodes
                if n.op == "call_function" and _get_node_type(n, warn_on_failure=False) == op
            ]
            if not matched:
                raise ValueError(
                    f"kv_cache_quant_configs key {op!r} matches no ops in the prepared graph."
                )
            for node in matched:
                if kc.quant_input_idx >= len(node.all_input_nodes):
                    raise ValueError(
                        f"kv_cache_quant_configs[{op!r}].op_quantizer_config.op_input_spec "
                        f"key {kc.quant_input_idx} is out of range for op {node.target!r}, "
                        f"which has {len(node.all_input_nodes)} inputs."
                    )

    @classmethod
    def get_compressible_op_names(
        cls,
        model: torch.fx.GraphModule,
    ) -> set[str]:
        """Return op names in *model* that this quantizer can target.

        Args:
            model (torch.nn.Module): The exported graph module.

        Returns:
            set[str]: Op names that can be compressed via quantization.
        """
        names: set[str] = set()
        for annotator_class in _AnnotationPatternRegistry.list_registry_values():
            node_to_match = annotator_class._match_all_patterns(model)
            names.update(node.name for node in node_to_match.keys())
        return names

    def prepare(
        self,
        example_inputs: tuple[Any, ...],
        dynamic_shapes: dict[str, Any] | tuple[Any] | list[Any] | None = None,
        export_with_no_grad: bool = True,
    ) -> torch.fx.GraphModule:
        """
        Prepare the model for quantization.

        This method exports the model using torch.export, applies quantization
        annotations, and sets up fake quantization modules. The prepared model
        represents a data-free PTQ compressed model.

        **Important Notes:**
        - For weight-only PTQ: The prepared model can be directly finalized
          (prepare() → finalize() workflow)
        - For activation quantization: The prepared model may have poor accuracy
          and should be calibrated using calibration_mode() before finalization
          to achieve good accuracy

        Args:
            example_inputs: Tuple of example inputs for model tracing. These should
                        be representative of the actual inputs the model will receive.
            dynamic_shapes: Optional dynamic shapes specification for torch.export.
                          Can be a dict mapping input names to dynamic dimensions,
                          a tuple/list of dynamic shapes per input, or None for
                          static shapes. Used to specify which dimensions can vary
                          at runtime during model export.
            export_with_no_grad: Whether to call torch.export.export within a
                               torch.no_grad() context. Defaults to True.

        Returns:
            The prepared GraphModule with quantization annotations and fake quantization
            modules inserted. This is a data-free PTQ compressed model.

        Raises:
            RuntimeError: If the model has already been prepared.
            TypeError: If example_inputs is not a tuple.
            ValueError: If example_inputs is empty or contains invalid data.
        """
        if self._is_model_prepared(self._model):
            raise RuntimeError(
                "Model has already been prepared. Cannot re-prepare a prepared model. "
            )

        if not isinstance(example_inputs, tuple):
            raise TypeError("example_inputs must be a tuple")

        if len(example_inputs) == 0:
            raise ValueError("example_inputs cannot be empty")

        # Capture original model's training state before export.
        # After export, GraphModule.training is always True regardless of the
        # original model's mode, so we must record it here to restore later.
        original_train_mode = self._model.training

        # Create annotation handler
        module_config_dict = self._config.build_module_config_dict(self._model)
        module_name_to_state_names_map = self._get_module_name_to_state_names_map(self._model)
        canonical_to_aliases, _ = _build_module_alias_map(self._model)

        # Per-instance annotation patterns for the cache ops (see
        # _kv_cache_extra_patterns for details).
        extra_patterns = self._kv_cache_extra_patterns()

        quantizer = _AnnotationHandler(
            module_config_dict,
            module_name_to_state_names_map,
            canonical_to_aliases,
            extra_patterns=extra_patterns,
            kv_cache_quant_configs=self._config.kv_cache_quant_configs,
        )

        # Collect user specified attributes that should be preserved
        preserved_attrs = {}
        if self._config.preserved_attributes is not None:
            for attr in self._config.preserved_attributes:
                if not hasattr(self._model, attr):
                    logger.warning(
                        f"Attribute '{attr}' specified in preserved_attributes was "
                        f"not found on model {type(self._model).__name__} and will "
                        f"be skipped."
                    )
                    continue
                preserved_attrs[attr] = getattr(self._model, attr)

        # Export the model to FX GraphModule
        exported_model = _export_model(
            self._model, example_inputs, dynamic_shapes, export_with_no_grad
        )

        # Catch KV cache misconfiguration (missing op type, out-of-range op_input_spec int key).
        self._validate_kv_cache_quant_ops(exported_model)

        # Prepare the model for quantization-aware training.
        # torchao < 0.16.0 asserts annotated nodes have empty kwargs,
        # so we strip metadata kwargs from non-aten nodes before and restore after.

        torchao_requires_empty_kwargs = not _version_ge(torchao, "0.16.0")
        if torchao_requires_empty_kwargs:
            saved_kwargs = _strip_non_aten_metadata_kwargs(exported_model.graph)
        try:
            prepared_model = prepare_qat_pt2e(exported_model, quantizer)
        except Exception as e:
            raise type(e)(f"prepare_qat_pt2e call failed, with error: {e}") from e
        if torchao_requires_empty_kwargs:
            _restore_kwargs(prepared_model.graph, saved_kwargs)

        # Apply post-processing fixes to the prepared model
        self._postprocess_prepared_model(prepared_model)

        # Enable model.train() / model.eval() on the exported GraphModule.
        # By default, torchao's prepare_qat_pt2e blocks these calls with NotImplementedError.
        allow_exported_model_train_eval(prepared_model)

        # Ensure fake quant is disabled and observers enabled before forward pass
        prepared_model.apply(disable_fake_quant)
        prepared_model.apply(enable_observer)

        # Run a forward pass to initialize quantization parameters
        # Place model in eval mode for this always. Switch model back to train mode
        # if original model was in train mode.
        with move_model_to_eval(prepared_model, original_state=original_train_mode):
            with torch.no_grad():
                prepared_model(*example_inputs)

        # Remove FakeQuantize nodes that were disabled during the forward
        # pass due to incompatible tensor shapes (e.g., non-divisible block sizes).
        self._remove_disabled_fake_quant_nodes(prepared_model)

        # Set up initial quantization state: fake quant enabled, observers disabled
        prepared_model.apply(enable_fake_quant)
        prepared_model.apply(disable_observer)

        # Mark the model as prepared to prevent re-preparation
        self._mark_model_as_prepared(prepared_model)

        # Attach preserved attributes to prepared model
        self._attach_preserved_attrs_to_model(prepared_model, preserved_attrs)

        # Update internal model reference
        self._model = prepared_model

        return self._model

    def finalize(
        self,
        model: torch.fx.GraphModule | None = None,
        backend: ExportBackend = ExportBackend.CoreAI,
        *,
        mmap_dir: str | PathLike[str] | None = None,
    ) -> torch.fx.GraphModule:
        """Convert fake quantization to real quantization.

        This method converts the prepared model with fake quantization modules to a
        final quantized model with actual quantized operations, ready for deployment
        to the target export backend.

        Only call ``finalize`` when exporting to a target backend. For torch-based
        evaluation, use the model returned by ``prepare()`` directly rather than
        calling ``finalize``.

        Args:
            model: Optional model to finalize. If None, uses the internal prepared model.
            backend: Target export backend for the quantized model.
                Supports CoreAI (default) and CoreML.
            mmap_dir (str | None): Not supported in graph mode. Raises
                ``ValueError`` if non-None.

        Returns:
            The finalized quantized GraphModule.

        """
        if mmap_dir is not None:
            raise ValueError(
                "mmap_dir is only supported in eager execution mode, got execution_mode=graph."
            )
        if model is None:
            model = self._model
        elif not isinstance(model, torch.fx.GraphModule):
            raise TypeError("Provided model must be a torch.fx.GraphModule")

        if not self._is_model_prepared(model):
            raise RuntimeError("Model must be prepared before finalization. Call prepare() first.")

        # Retrieve preserved attributes before conversion
        preserved_attrs = model.meta.get(_USER_PRESERVED_ATTRIBUTES_KEY, {})

        # Always first call convert_pt2e API
        try:
            finalized_model = convert_pt2e(model)
        except Exception as e:
            raise RuntimeError(f"Failed to convert model with convert_pt2e, with error: {e}") from e

        # Re-attach preserved attributes to finalized model
        self._attach_preserved_attrs_to_model(finalized_model, preserved_attrs)
        # Post-conversion processing (conv+bn folding, etc.)
        finalized_model = self._post_conversion_process(finalized_model)

        # Backend-specific processing
        match backend:
            case ExportBackend._TORCH:
                # KV-cache quantization is a no-op for the torch backend: the
                # fake-quant observers on the cache op input stay in the graph.
                # Cache-buffer retyping only happens on the CoreAI backend.
                pass

            case ExportBackend.CoreML:
                if self._config.kv_cache_quant_configs:
                    raise NotImplementedError(
                        "kv_cache_quant_configs is not supported with the CoreML "
                        "backend; the finalize-side cache-buffer retyping only "
                        "applies to the CoreAI backend."
                    )
                finalized_model = prepare_for_mil_export(finalized_model)

            case ExportBackend.CoreAI:
                finalized_model = prepare_for_mlir_export(finalized_model)
                # Relocate each cache-update op's input dq to its output edge so
                # the cache state stays in the quantized dtype.
                for op, kc in (self._config.kv_cache_quant_configs or {}).items():
                    finalized_model = _move_cache_dequant_to_output(
                        finalized_model,
                        op_type=op,
                        quant_input_idx=kc.quant_input_idx,
                    )

            case _:
                msg = f"Unsupported backend: {backend}"
                raise NotImplementedError(msg)

        # Re-enable model.train() / model.eval() on the finalized model.
        # convert_pt2e() re-applies _disallow_eval_train, so we must call
        # allow_exported_model_train_eval again on the converted model.
        allow_exported_model_train_eval(finalized_model)

        return finalized_model

    @contextmanager
    def calibration_mode(self, model: torch.fx.GraphModule | None = None):
        """
        Context manager for calibration-based post-training quantization.

        When entering this context, observers are enabled to collect statistics
        from calibration data, and fake quantization is disabled to get accurate
        statistics. When exiting, observers are disabled and fake quantization
        is re-enabled for evaluation.

        **When to use:**
        - Required for activation quantization to achieve good accuracy
        - The prepared model from prepare() may have poor accuracy for activation
          quantization until calibrated with representative data
        - Not needed for weight-only PTQ (prepare() → finalize() is sufficient)

        Args:
            model: Optional model to setup for calibration.
                    If None, uses the internal prepared model.

        Example:
            >>> quantizer = GraphQuantizer(model, config)
            >>> prepared_model = quantizer.prepare(example_inputs)
            >>> # For activation quantization, calibrate to improve accuracy:
            >>> with quantizer.calibration_mode():
            ...     for batch in calibration_dataloader:
            ...         prepared_model(batch)
            >>> finalized_model = quantizer.finalize()

        Raises:
            RuntimeError: If the model has not been prepared.
        """
        if model is not None:
            if not isinstance(model, torch.fx.GraphModule):
                raise TypeError("Provided model must be a torch.fx.GraphModule")
            self._model = model

        if not self._is_model_prepared(self._model):
            raise RuntimeError(
                "Model must be prepared before entering calibration mode. Call prepare() first."
            )

        # Enable observers and disable fake quantization for calibration
        self._model.apply(enable_observer)
        self._model.apply(disable_fake_quant)

        # Move model to eval mode and save original state
        with move_model_to_eval(self._model):
            try:
                yield
            finally:
                # Restore the state: disable observers, enable fake quantization
                self._model.apply(disable_observer)
                self._model.apply(enable_fake_quant)

    @contextmanager
    def training_mode(self, model: torch.fx.GraphModule | None = None):
        """
        Context manager for quantization-aware training (QAT) workflow.

        When entering this context, the model is configured for training with both
        observers and fake quantization enabled. This allows the model to:
        1. Set the model in training mode (model.training is set to True)
        2. Enable the observers and activate the fake quantization
        3. Using the observers, simulate quantization during forward/backward passes

        When exiting the context, the model state is restored to what it was before
        entering the training mode context (whether it was in eval mode or train mode).
        Observers are disabled but fake quantization is enabled.

        **When to use:**
        - For quantization-aware training (QAT) to fine-tune a prepared model
        - The prepared model from prepare() may have poor accuracy for weight-only
          quantization. Fine-tuning the model with the quantization enabled will help
          the weights adapt to the effects of quantization.
        - Upon calibrating an activation-quantized model, there wasn't a enough
          improvement in model accuracy. Fine-tuning the weights to adapt to the effect
          of activation (and weight) quantization can help recover the lost accuracy.

        Args:
            model: Optional model to setup for training.
                   If None, uses the internal prepared model.

        Example:
            >>> quantizer = GraphQuantizer(model, config)
            >>> prepared_model = quantizer.prepare(example_inputs)
            >>> # Fine-tune with quantization-aware training:
            >>> with quantizer.training_mode():
            ...     # Model is put in training mode
            ...     for epoch in range(num_epochs):
            ...         for batch in train_dataloader:
            ...             # Perform training step
            ...             optimizer.zero_grad()
            ...             loss = loss_fn(prepared_model(batch), targets)
            ...             loss.backward()
            ...             optimizer.step()
            ...
            >>> finalized_model = quantizer.finalize()

        Raises:
            RuntimeError: If the model has not been prepared.
            TypeError: If the provided model is not a torch.fx.GraphModule.
        """
        if model is not None:
            if not isinstance(model, torch.fx.GraphModule):
                raise TypeError("Provided model must be a torch.fx.GraphModule")
            self._model = model

        if not self._is_model_prepared(self._model):
            raise RuntimeError(
                "Model must be prepared before entering training mode. Call prepare() first."
            )

        # Configure model for training: enable both observers and fake quant
        self._model.apply(enable_observer)
        self._model.apply(enable_fake_quant)

        with move_model_to_train(self._model):
            try:
                yield
            finally:
                # Disable observers
                self._model.apply(disable_observer)
                self._model.apply(enable_fake_quant)

    def _get_fake_quantize_modules(
        self,
    ) -> Mapping[str, list[FakeQuantizeImplBase]]:
        """Map original module names to their fake quantize modules.

        Walks the FX graph to find FQ ``call_module`` nodes and resolves
        each one back to the original (pre-``torch.export``) module name
        via the ``nn_module_stack`` metadata on the FQ node's consumer
        nodes.

        Returns:
            Dict mapping original module name to the list of FQ module
            instances that were inserted for that module.
        """
        model = self._model
        modules = dict(model.named_modules(remove_duplicate=False))
        mapping: dict[str, list[FakeQuantizeImplBase]] = defaultdict(list)

        for node in model.graph.nodes:
            if node.op != "call_module":
                continue
            fq_mod = modules.get(node.target)
            if not isinstance(fq_mod, FakeQuantizeImplBase):
                continue
            # Resolve source module from the FQ node's consumer (user) nodes.
            # Fall back to producer (arg) nodes if no consumer has nn_module_stack
            # (e.g. when the FQ feeds directly into the graph output node).
            source = None
            for neighbor in list(node.users) + list(node.args):
                source = _get_source_module_name(neighbor)
                if source is not None:
                    break
            if source is not None:
                mapping[source].append(fq_mod)

        return dict(mapping)

    @staticmethod
    def _postprocess_prepared_model(model: torch.fx.GraphModule) -> None:
        """Apply post-processing fixes after prepare_qat_pt2e.

        This method applies necessary corrections to the prepared model that address
        issues or limitations in TorchAO's prepare_qat_pt2e implementation.

        Args:
            model (torch.fx.GraphModule): The graph module after
                prepare_qat_pt2e().
        """
        # Fix dtype in Conv+BN decomposition for non-float32 models
        _remove_conv_bn_zeros_like_dtype(model)

        # Channel-altering ops (flatten, reshape, transpose, etc.) share observers
        # with their inputs but invalidate axis semantics, so per-channel/per-block
        # granularity is mathematically incorrect. Force per-tensor.
        _force_per_tensor_for_channel_altering_ops(model)

        # Apply weight axis defaults for per channel and per block quantization
        _apply_weight_axis_defaults(model)

        _validate_activation_axes(model)

    @staticmethod
    def _remove_disabled_fake_quant_nodes(model: torch.fx.GraphModule) -> None:
        """Remove FakeQuantize nodes that were disabled during the forward pass."""
        modules = dict(model.named_modules(remove_duplicate=False))
        disabled_fq_nodes = set()
        for node in model.graph.nodes:
            if node.op == "call_module":
                mod = modules.get(str(node.target))
                if isinstance(mod, FakeQuantizeImplBase) and mod.is_disabled():
                    disabled_fq_nodes.add(node)
        if disabled_fq_nodes:
            _remove_fake_quant_nodes(model, disabled_fq_nodes)

    @staticmethod
    def _attach_preserved_attrs_to_model(
        model: torch.fx.GraphModule,
        preserved_attrs: dict[str, Any],
    ) -> None:
        if not preserved_attrs:
            return
        # Store preserved attributes in model.meta so they survive deepcopy
        model.meta[_USER_PRESERVED_ATTRIBUTES_KEY] = copy.deepcopy(preserved_attrs)
        # Set the preserved attributes on the model so users can access them
        # as they did before calling quantizer.prepare
        for attr_name, attr in preserved_attrs.items():
            setattr(model, attr_name, attr)

    @staticmethod
    def _post_conversion_process(model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        """Apply post-conversion transformations after convert_pt2e().

        This function handles graph transformations that need to happen after
        quantization conversion but are backend-agnostic, such as:
        - Conv+BN folding

        Args:
            model: The model after convert_pt2e()

        Returns:
            The transformed model

        """
        # Apply conv+bn folding
        return _fold_conv_bn_weights(model)
