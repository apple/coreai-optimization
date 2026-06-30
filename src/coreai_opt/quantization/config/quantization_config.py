# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Quantization config class definitions."""

from __future__ import annotations

import logging
from enum import auto
from typing import TYPE_CHECKING, ClassVar, NamedTuple, Self, TypeAlias, final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from coreai_opt._utils.config_utils import ALL_TENSORS as _ALL_TENSORS
from coreai_opt.common import (
    _DeprecatedMemberEnumMeta,
    _StrEnum,
)
from coreai_opt.config import (
    CompressionConfig,
    ModuleCompressionConfig,
    OpCompressionConfig,
)
from coreai_opt.quantization.spec import (
    QuantizationSpec,
    default_activation_quantization_spec,
    default_weight_quantization_spec,
)

if TYPE_CHECKING:
    from coreai_opt.quantization.config._presets import (
        _ModuleQuantizerConfigPresets,
        _QuantizerConfigPresets,
    )


logger = logging.getLogger(__name__)


class QATSchedule(BaseModel):
    """Schedule for controlling observer and fake quantization state in QAT.

    Defines step thresholds for enabling/disabling observers and fake quantization
    during quantization-aware training. Must be used in conjunction with the
    ``quantizer.step()`` API to advance the schedule.

    The step values correspond to the cadence at which ``quantizer.step()`` is
    called. For example, if ``step()`` is called once per batch, the thresholds
    represent batch steps; if called once per epoch, they represent epochs.

    Calling ``step()`` increments the step counter and immediately applies
    the corresponding observer/fake-quantization state. Where you place
    ``step()`` in your training loop determines when the model sees the
    new state.

    Attributes:
        enable_observer: Step count at which observers are enabled. Must be >= 0.
        enable_fake_quant: Step count at which fake quantization is enabled.
            Must be >= enable_observer.
        disable_observer: Step count at which observers are disabled. Must be
            > enable_observer and >= enable_fake_quant if provided. None means
            observers are never disabled by the schedule.

    Example:
        >>> schedule = QATSchedule(
        ...     enable_observer=0,
        ...     enable_fake_quant=500,
        ...     disable_observer=1500,
        ... )

    Note:
        In graph execution mode, when consecutive modules both quantize the
        intermediate edge (one via ``op_output_spec``, the next via
        ``op_input_spec``), graph mode deduplicates them into a single
        fake-quantize node. The schedule of the consuming module is always
        applied to the deduplicated node, irrespective of the choice of
        deduplication made by the graph preparation.

    Note:
        When two modules share a weight parameter and have different
        schedules, the schedule of the first module encountered in the
        module tree is applied. A warning is emitted for the conflict if
        there is no fake-quantize node deduplication happening (in Eager
        execution mode).
    """

    model_config = ConfigDict(frozen=True)

    class _ScheduleState(NamedTuple):
        obs_on: bool
        fq_on: bool

    enable_observer: int = Field(default=0, ge=0)
    enable_fake_quant: int = Field(default=0, ge=0)
    disable_observer: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _validate_schedule(self) -> QATSchedule:
        if self.enable_fake_quant < self.enable_observer:
            raise ValueError(
                f"enable_fake_quant ({self.enable_fake_quant}) must be >= "
                f"enable_observer ({self.enable_observer})"
            )
        if self.disable_observer is not None:
            if self.disable_observer <= self.enable_observer:
                raise ValueError(
                    f"disable_observer ({self.disable_observer}) must be > "
                    f"enable_observer ({self.enable_observer})"
                )
            if self.disable_observer < self.enable_fake_quant:
                raise ValueError(
                    f"disable_observer ({self.disable_observer}) must be >= "
                    f"enable_fake_quant ({self.enable_fake_quant})"
                )
        return self

    def _compute_state(self, step_count: int) -> _ScheduleState:
        """Return the observer/fake_quant state at the given step."""
        obs_end = self.disable_observer or float("inf")
        return self._ScheduleState(
            obs_on=self.enable_observer <= step_count < obs_end,
            fq_on=step_count >= self.enable_fake_quant,
        )


_QUANTIZATION_CONFIG = "quantization_config"
_QUANTIZATION_SPEC = "quantization_spec"
_ACTIVATION_SPEC_DICT: TypeAlias = dict[str | int, QuantizationSpec | None]
_STATE_SPEC_DICT: TypeAlias = dict[str, QuantizationSpec | None]


class ExecutionMode(_StrEnum, metaclass=_DeprecatedMemberEnumMeta):
    """Enum representing quantization execution modes.

    Each member is a string value representing the execution mode used
    for quantization.

    Attributes:
        GRAPH: Graph-based quantization using ``torch.export`` to capture the model as an FX graph,
            then applying quantization on top. Built on ``torchao``'s PT2E implementation. Requires
            the model to be exportable via ``torch.export.export``. Recommended default.
        EAGER: Eager-mode quantization that works directly on ``nn.Module`` without graph capture.
            Supports dynamic control flow (if/else, loops) and is the fallback when a model is not
            exportable.

    """

    GRAPH = auto()
    EAGER = auto()

    __deprecated_aliases__: ClassVar[dict[str, str]] = {"PT2E": "GRAPH"}

    if TYPE_CHECKING:
        # Surface the deprecated alias above for static type checkers.
        PT2E: ExecutionMode
        """Deprecated. Use ``ExecutionMode.GRAPH`` instead."""


class OpQuantizerConfig(OpCompressionConfig[QuantizationSpec]):
    """
    Configuration class for quantization at the operation level.

    This class specifies quantization settings for inputs, outputs, and state
    tensors of individual operations (ops) in a neural network. Each tensor
    can have its own quantization specification.

    Quantization for operations will require the operation to be registered. Even if
    globally all ops are configured to be quantized, ops which are not recognized will
    not be quantized.

    Attributes:
        op_input_spec (dict[str | int, QuantizationSpec | None] | None): Quantization
            specifications for operation inputs. Keys can be either all indices or all
            string names, but not a mix of both. The special key "*" can be used in both
            cases to refer to all inputs.
            Example keys:

            - int: Input index (e.g., 0 for first input, 1 for second
              input)
            - str: Named input identifier (e.g., "x", "input_0")
            - "\*": Applies to all inputs for the operation. Other tensors
              can be explicitly mentioned to override this setting.

            Values are QuantizationSpec objects or None defining how to quantize each
            input. None value represents disabling quantization.
            Default: {"\*": default_activation_quantization_spec()} (int8
            quantization for all inputs)

        op_output_spec (dict[str | int, QuantizationSpec | None] | None): Quantization
            specifications for operation outputs. Keys can be either all indices or all
            string names, but not a mix of both. The special key "*" can be used in both
            cases to refer to all outputs.
            Example keys:

            - int: Output index (e.g., 0 for first output, 1 for second
              output)
            - str: Named output identifier (e.g., "y", "output_0")
            - "\*": Applies to all outputs for the operation. Other tensors
              can be explicitly mentioned to override this setting.

            Values are QuantizationSpec objects or None defining how to quantize each
            output. None value represents disabling quantization.
            Default: {"\*": default_activation_quantization_spec()} (int8
            quantization for all outputs)

        op_state_spec (dict[str, QuantizationSpec | None] | None): Quantization
            specifications for operation state tensors (parameters, buffers, constants).
            Keys can be string names (e.g. "weight", "bias") or "\*" to refer to all
            state inputs.
            Values are QuantizationSpec objects or None defining how to quantize each
            state tensor. None value represents disabling quantization.
            Default: {"weight": default_weight_quantization_spec()} (int8
            quantization for weight inputs)

    Example:
        >>> # Quantize first input, disable first output, quantize weight tensor
        >>> op_config = OpQuantizerConfig(
        ...     op_input_spec={
        ...         0: QuantizationSpec(
        ...             dtype=torch.int8,
        ...             qscheme="symmetric",
        ...             granularity={"type": "per_tensor"},
        ...             fake_quantize_cls="default",
        ...             qparam_calculator_cls="moving_average",
        ...             range_calculator_cls="minmax",
        ...         )
        ...     },
        ...     op_output_spec={
        ...         0: None
        ...     },
        ...     op_state_spec={
        ...         "weight": QuantizationSpec(
        ...             dtype=torch.int4,
        ...             qscheme="symmetric",
        ...             granularity={"type": "per_channel", "axis": 1},
        ...             fake_quantize_cls="default",
        ...             qparam_calculator_cls="default",
        ...             range_calculator_cls="minmax",
        ...         )
        ...     }
        ... )
    """

    @classmethod
    def get_default_input_spec(cls) -> dict[str | int, QuantizationSpec | None]:
        """Provide default input spec for quantization."""
        return {_ALL_TENSORS: default_activation_quantization_spec()}

    @classmethod
    def get_default_output_spec(cls) -> dict[str | int, QuantizationSpec | None]:
        """Provide default output spec for quantization."""
        return {_ALL_TENSORS: default_activation_quantization_spec()}

    @classmethod
    def get_default_state_spec(cls) -> dict[str, QuantizationSpec | None]:
        """Provide default state spec for quantization."""
        return {"weight": default_weight_quantization_spec()}


@final
class ModuleQuantizerConfig(ModuleCompressionConfig[OpQuantizerConfig, QuantizationSpec]):
    """
    Configuration class for quantization at the module level.

    This class manages quantization settings for an entire module, including:

    - Operation-level configurations (default, by type, by name)
    - Module-level input/output quantization
    - Module-level state (parameter) quantization

    The operation configurations follow a hierarchical precedence:

    1. op_name_config (most specific - applies to operations matching a name
       pattern)
    2. op_type_config (applies to operations of a specific type)
    3. op_input/output/state_spec (least specific - applies to all operations
       not otherwise configured)

    Module-level input, output, and state settings treat the module as an opaque entity,
    setting quantization settings for specified tensors and ignoring op specific
    quantization capabilities. Module-level settings also don't check whether the
    operation receiving the quantized tensor is a registered operation or not.
    Module-level settings will override any op specific settings.

    Attributes:
        op_input_spec (dict[str | int, QuantizationSpec | None] | None): Quantization
            specifications for operation inputs applied to all registered
            operations/patterns within this module that don't have a more specific
            configuration.
            Keys can be either all indices or all string names, but not a mix of both.
            The special key "\*" can be used in both cases to refer to all inputs.
            Example keys:

            - int: Input index (e.g., 0 for first input, 1 for second input)
            - str: Named input identifier (e.g., "x", "input_0")
            - "\*": Applies to all inputs for the operation. Other tensors
              can be explicitly mentioned to override this setting.

            Values are QuantizationSpec objects or None defining how to quantize each
            input. None value represents disabling quantization.
            Default: {"\*": default_activation_quantization_spec()} (int8
            quantization for all inputs)

        op_output_spec (dict[str | int, QuantizationSpec | None] | None): Quantization
            specifications for operation inputs applied to all registered
            operations/patterns within this module that don't have a more specific
            configuration.
            Keys can be either all indices or all string names, but not a mix of both.
            The special key "\*" can be used in both cases to refer to all outputs.
            Example keys:

            - int: Output index (e.g., 0 for first output, 1 for second
              output)
            - str: Named output identifier (e.g., "y", "output_0")
            - "\*": Applies to all outputs for the operation. Other tensors
              can be explicitly mentioned to override this setting.

            Values are QuantizationSpec objects or None defining how to quantize each
            output. None value represents disabling quantization.
            Default: {"\*": default_activation_quantization_spec()} (int8
            quantization for all outputs)

        op_state_spec (dict[str, QuantizationSpec | None] | None): Quantization
            specifications for operation state tensors (parameters, buffers, constants)
            applied to all registered operations/patterns within this module that don't
            have a more specific configuration.
            Keys can be string names (e.g. "weight", "bias") or "\*" to refer to all
            state inputs.
            Values are QuantizationSpec objects or None defining how to quantize each
            state tensor. None value represents disabling quantization.
            Default: {"weight": default_weight_quantization_spec()} (int8
            quantization for weight inputs)

        op_type_config (dict[str, OpQuantizerConfig | None] | None): Operation
            type-specific configurations. Keys are operation types (e.g.,
            "linear", "conv2d"). Generally speaking, operation types will match the torch
            functional name (https://docs.pytorch.org/docs/stable/nn.functional.html) or
            operation name within the torch namespace
            (https://docs.pytorch.org/docs/stable/torch.html)
            when taking the portion of the name to the right of the last period.

            For example, to refer to a Maxpool 2D operation, take the name used for the
            torch functional, torch.nn.functional.max_pool2d, and use the portion of the
            string after the last period: "max_pool2d".

            OpQuantizerConfig objects or None defining how to quantize operations of
            that type. None value represents disabling quantization.
            Default: {} (empty dict, no type-specific configs)

        op_name_config (dict[str, OpQuantizerConfig | None] | None): Operation
            name-specific configurations. Keys are operation name patterns
            (supports regex matching). Values are OpQuantizerConfig objects or None
            defining how to quantize operations matching those names. None value
            represents disabling quantization.
            Default: {} (empty dict, no name-specific configs)

        module_input_spec (dict[str | int, QuantizationSpec | None] | None):
            Quantization specifications for module inputs. Module input settings treat
            the module as an opaque entity, setting quantization settings for input
            tensors to the module without checking whether the op receiving
            the input is quantizable. Module input settings override op level
            settings for the op receiving the module input.
            Keys can be either all indices or all string names, but not a mix of both.
            The special key "\*" can be used in both cases to refer to all module inputs.
            Example keys:

            - int: Input index (e.g., 0 for first input, 1 for second
              input)
            - str: Named input identifier (e.g., "y", "input_0")
            - "\*": Applies to all inputs for the operation. Other tensors
              can be explicitly mentioned to override this setting.

            Values are QuantizationSpec objects or None. None value represents disabling
            quantization.
            Default: {} (empty dict, no specific module input settings)

        module_output_spec (dict[str | int, QuantizationSpec | None] | None):
            Quantization specifications for module outputs. Module output settings treat
            the module as an opaque entity, setting quantization settings for
            output tensors to the module without checking whether the op
            receiving the output is quantizable. Module output settings
            override op level settings for the op receiving the module output.
            Keys can be:

            - int: Output index (e.g., 0 for first output)
            - str: Named output identifier
            - "\*": Applies to all outputs for the operation. Other tensors
              can be explicitly mentioned to override this setting.

            Values are QuantizationSpec objects or None. None value represents disabling
            quantization.
            Default: {} (empty dict, no specific module output settings)

        module_state_spec (dict[str, QuantizationSpec | None] | None): Quantization
            specifications for module state tensors (parameters, buffers, and
            constants). Module state settings will override op state settings for the
            same state tensors.
            Keys can be string names (e.g. "weight", "bias") or "\*" to refer to all
            state inputs.
            Values are QuantizationSpec objects or None. None value represents disabling
            quantization.
            Default: {} (empty dict, no specific module state settings)

        qat_schedule (QATSchedule | None): Optional QAT schedule for controlling
            observer and fake quantization state transitions during training.
            When set, the ``quantizer.step()`` API must be used to advance the
            schedule. See :class:`QATSchedule` for details. When None (default),
            both observer and fake quantization are enabled from the start of
            training.

    Example:
        >>> # Configure a module with default op config and specific settings for
        >>> # linear ops
        >>> module_config = ModuleQuantizerConfig(
        ...     # Omitted op_input/output/state_specs sets default quantization for all
        ...     # ops
        ...     op_type_config={
        ...         "linear": OpQuantizerConfig(
        ...             op_input_spec={
        ...                 0: ...
        ...             },
        ...             op_output_spec={
        ...                 0: ...
        ...             },
        ...             op_state_spec={
        ...                 "weight": ...
        ...             }
        ...         )
        ...     },
        ... )

    """

    def __init_subclass__(cls, **kwargs):
        # Prohibit subclassing due to preset limitation: presets remain bound
        # to the base class. Revisit if subclass support is needed.
        super().__init_subclass__(**kwargs)
        msg = f"{cls.__name__} cannot subclass ModuleQuantizerConfig (marked final)."
        raise TypeError(msg)

    qat_schedule: QATSchedule | None = None

    # Namespace exposing built-in preset constructors for module-level configs.
    # Wired at the bottom of this module after the class is fully defined.
    presets: ClassVar[_ModuleQuantizerConfigPresets]


class KVCacheQuantConfig(BaseModel):
    """Enables KV-cache buffer storage in a quantized dtype for one cache-update op type.

    Carries the cache-update op's quantization spec inline (via ``op_quantizer_config``)
    and is the finalize-side switch that promotes the resulting fake-quant to stored
    quantization: the dequantize is relocated from the op's input to its output and
    the cache buffer is retyped to the quantized dtype.

    Used as a value in :attr:`QuantizerConfig.kv_cache_quant_configs`, which maps the
    short op-type name to a config.

    Precondition on the cache op: it must commute with quantize/dequantize —
    i.e. a pure data-movement op (slicing, narrowing, copy). Arithmetic on
    cached values would silently produce a numerically wrong model.

    During ``prepare``, the cache spec is applied as a global-only knob with
    highest priority: any prior annotation on the cache op — from any scope, by
    any mechanism (including module-scope ``op_input_spec={"*": ...}``
    wildcards) — is **hard-overridden**, not merged. A warning is emitted at
    config-construction time when explicit ``op_type_config[op]`` collisions
    are detected. See :meth:`QuantizerConfig._validate_kv_cache_quant_configs`.

    Attributes:
        op_quantizer_config: ``OpQuantizerConfig`` whose ``op_input_spec`` selects the
            input edge to quantize. The int key indexes the op's tensor-valued,
            deduplicated inputs (``node.all_input_nodes``). ``op_input_spec`` must
            contain exactly one key (no ``"*"``, no multi-key dicts) because the
            finalize-side relocation needs a single, unambiguous input edge to act on.
            ``op_output_spec`` and ``op_state_spec`` must be explicitly set to None
            (or empty); ``OpQuantizerConfig`` otherwise applies non-None defaults
            which the validator rejects.

    Example:
        >>> QuantizerConfig(
        ...     execution_mode="graph",
        ...     global_config=ModuleQuantizerConfig(...),
        ...     kv_cache_quant_configs={
        ...         "mutable_cache_update_and_fetch": KVCacheQuantConfig(
        ...             op_quantizer_config=OpQuantizerConfig(
        ...                 op_input_spec={1: default_activation_quantization_spec()},
        ...                 op_output_spec=None,
        ...                 op_state_spec=None,
        ...             ),
        ...         ),
        ...     },
        ... )
    """

    model_config = ConfigDict(frozen=True)

    op_quantizer_config: OpQuantizerConfig

    @model_validator(mode="after")
    def _validate_op_quantizer_config(self) -> Self:
        oqc = self.op_quantizer_config

        # op_input_spec: exactly one non-negative int key (the input index of
        # the new K/V) mapped to a non-None QuantizationSpec. "*" and multi-key
        # forms break the finalize pass's single-edge contract; a None value
        # would disable quantization on that edge, leaving the finalize pass
        # with no dtype to retype the cache buffer to.
        spec = oqc.op_input_spec or {}
        nonneg_int_keys = [
            k for k in spec.keys() if isinstance(k, int) and not isinstance(k, bool) and k >= 0
        ]
        if len(nonneg_int_keys) != 1 or len(spec) != 1:
            raise ValueError(
                "KVCacheQuantConfig.op_quantizer_config.op_input_spec must "
                "contain exactly one key, which must be a non-negative int "
                f"(the input index of the new K/V); got keys {list(spec.keys())!r}."
            )
        if next(iter(spec.values())) is None:
            raise ValueError(
                "KVCacheQuantConfig.op_quantizer_config.op_input_spec must map "
                "its int key to a non-None QuantizationSpec."
            )

        # op_output_spec must be empty/None: the finalize pass synthesizes the
        # output dequantize itself; a user-supplied output spec would either
        # double-quantize or conflict with the relocated dq.
        if oqc.op_output_spec:
            raise ValueError(
                "KVCacheQuantConfig.op_quantizer_config.op_output_spec must be "
                "empty or None; the finalize pass inserts the output dequantize."
            )

        # op_state_spec must be empty/None: cache-update ops have no learnable
        # state (the cache buffer is an input, not a parameter).
        if oqc.op_state_spec:
            raise ValueError(
                "KVCacheQuantConfig.op_quantizer_config.op_state_spec must be "
                "empty or None; cache-update ops have no learnable state."
            )
        return self

    @property
    def quant_input_idx(self) -> int:
        """The single int key in ``op_quantizer_config.op_input_spec``.

        Indexes the cache op's ``all_input_nodes`` (tensor-valued, deduplicated
        inputs) — see ``op_quantizer_config`` for details.
        """
        return next(iter(self.op_quantizer_config.op_input_spec))


@final
class QuantizerConfig(CompressionConfig[ModuleQuantizerConfig]):
    """Top-level configuration class for quantization.

    This class manages the complete quantization configuration for a neural
    network model, organizing module-level configurations in a hierarchical
    structure. It inherits from CompressionConfig and specializes it for
    quantization using ModuleQuantizerConfig.

    The configuration lookup follows a hierarchical precedence (most to least
    specific):

    1. module_name_configs - Applies to module instances matching a name
       pattern (supports regex)
    2. module_type_configs - Applies to all modules of a specific type (e.g.,
       torch.nn.modules.linear.Linear)
    3. global_config - Default configuration applied to all modules not
       otherwise configured

    Attributes:
        global_config (ModuleQuantizerConfig | None): Default module-level
            quantization configuration applied to all modules that don't have
            a more specific configuration. When QuantizerConfig is initialized
            with no arguments, a default global_config is automatically created with
            standard int8 quantization.
            Setting global_config to None disables quantization by default globally.
            Default: Auto-created with int8 quantization specs when no args
            provided

        module_type_configs (dict[str, ModuleQuantizerConfig | None] | None): Module
            type-specific configurations. Keys are fully-qualified module type
            names (e.g., "torch.nn.modules.linear.Linear",
            "torch.nn.modules.conv.Conv2d"). Values are ModuleQuantizerConfig objects or
            None to disable quantization for that module type.
            Default: {} (empty dict, no type-specific configs)

        module_name_configs (dict[str, ModuleQuantizerConfig | None] | None): Module
            name-specific configurations. Keys are module name patterns
            (supports regex matching, e.g., "model.layer1.*",
            "decoder.layers.0"). Values are ModuleQuantizerConfig objects or
            None to disable quantization for matching modules.
            Default: {} (empty dict, no name-specific configs)

        preserved_attributes (list[str] | None): Names of attributes of the model
            which should be preserved on the prepared and finalized models, even if they
            are not used in the model's forward pass.

        execution_mode (ExecutionMode | str): Specifies which quantization execution
            mode to use. Options are:

            - ExecutionMode.GRAPH / "graph":
                Graph-based quantization using ``torch.export`` and FX graphs, built on
                ``torchao``'s PT2E implementation. Requires the model to be exportable.
            - ExecutionMode.EAGER / "eager":
                Works directly on ``nn.Module`` without converting to a graph representation.
                Supports dynamic control flow (if/else, loops) and doesn't require ``torch.export``.

            Default: ExecutionMode.GRAPH

        kv_cache_quant_configs (dict[str, KVCacheQuantConfig] | None): Optional
            mapping from short op-type name (as returned by ``get_node_type``,
            to the cache-update op's ``KVCacheQuantConfig``. Each entry enables
            storing the corresponding KV-cache buffer in a quantized dtype: it
            carries the op's ``OpQuantizerConfig`` inline and triggers a finalize-side
            rewrite that relocates the dequantize from the op's input to its
            output. Graph mode only; rejected for eager mode by
            :meth:`_validate_kv_cache_quant_configs`. See
            :class:`KVCacheQuantConfig` for details.
            Default: None (no KV-cache buffer quantization)

    Example:
        >>> # Create default quantizer config (auto-creates int8 global
        >>> # config)
        >>> config = QuantizerConfig()
        >>> # config.global_config is automatically created with default int8 specs
        >>>
        >>> # Disable quantization globally
        >>> config = QuantizerConfig(
        ...     global_config=None
        ... )
        >>>
        >>> # Create custom quantizer config with type-specific settings for Linear
        >>> # modules.
        >>> config = QuantizerConfig(
        ...     # Omitted global_config section defaults to int8/int8 weight/activation
        ...     # quantization for all operations
        ...     module_type_configs={
        ...         "torch.nn.modules.linear.Linear": ModuleQuantizerConfig(
        ...             op_input_spec={
        ...                 0: ...
        ...             },
        ...             op_output_spec={
        ...                 0: ...
        ...             },
        ...             op_state_spec={
        ...                 'weight': ...
        ...             }
        ...         )
        ...     },
        ... )
        >>>
        >>> # Load quantizer config from YAML file
        >>> config = QuantizerConfig.from_yaml("config.yaml")

    Notes:
        - When initialized with no arguments, a default configuration is
          created with int8 symmetric quantization for activations and
          weights
        - The from_yaml class method provides an alternative way to create
          configurations from YAML files
        - Setting a config to None explicitly disables quantization for that
          scope
        - More specific configurations (name > type > global) always override
          less specific ones

    """

    def __init_subclass__(cls, **kwargs):
        # Prohibit subclassing due to preset limitation: presets remain bound
        # to the base class. Revisit if subclass support is needed.
        super().__init_subclass__(**kwargs)
        msg = f"{cls.__name__} cannot subclass QuantizerConfig (marked final)."
        raise TypeError(msg)

    # Class attributes for config key pattern used in from_dict/from_yaml
    _CONFIG_KEY: ClassVar[str] = _QUANTIZATION_CONFIG
    _SPEC_KEY: ClassVar[str] = _QUANTIZATION_SPEC

    # Namespace exposing built-in and registered preset constructors.
    # Wired at the bottom of this module after the class is fully defined.
    presets: ClassVar[_QuantizerConfigPresets]

    preserved_attributes: list[str] | None = None
    execution_mode: ExecutionMode = ExecutionMode.GRAPH
    kv_cache_quant_configs: dict[str, KVCacheQuantConfig] | None = None

    @model_validator(mode="after")
    def _validate_kv_cache_quant_configs(self) -> Self:
        """Enforce graph-mode-only and warn on duplicate ``op_type_config[op]`` entries.

        Each cache op's spec is carried inline by ``kv_cache_quant_configs[op]``. If the
        same op key also appears under any ``op_type_config`` (in ``global_config``,
        ``module_type_configs``, or ``module_name_configs``), warns that the
        ``kv_cache_quant_configs`` entry will override it — the override is applied at
        prepare time by ``_AnnotationHandler._override_cache_op_annotations`` and does
        not mutate the user's ``QuantizerConfig``.
        """
        if not self.kv_cache_quant_configs:
            return self

        if self.execution_mode != ExecutionMode.GRAPH:
            raise ValueError(
                f"kv_cache_quant_configs is only supported with "
                f"ExecutionMode.GRAPH (got {self.execution_mode!r})."
            )

        for op in self.kv_cache_quant_configs:
            warnings_list: list[str] = []
            if self.global_config is not None and op in (self.global_config.op_type_config or {}):
                warnings_list.append("global_config.op_type_config")
            for scope_name, scope_dict in (
                ("module_type_configs", self.module_type_configs),
                ("module_name_configs", self.module_name_configs),
            ):
                for module_key, module_cfg in (scope_dict or {}).items():
                    if module_cfg is not None and op in (module_cfg.op_type_config or {}):
                        warnings_list.append(f"{scope_name}[{module_key!r}].op_type_config")

            if warnings_list:
                logger.warning(
                    "kv_cache_quant_configs[%r] is also present in %s; the "
                    "kv_cache_quant_configs entry will win at prepare time and the "
                    "duplicate entries will be ignored.",
                    op,
                    ", ".join(warnings_list),
                )
        return self

    def set_execution_mode(self, mode: ExecutionMode | str) -> Self:
        """Set the quantization execution mode.

        Args:
            mode (ExecutionMode | str): Execution mode to use.
                Accepts an ``ExecutionMode`` member (e.g. ``ExecutionMode.EAGER``)
                or its string value (e.g. ``"graph"``, ``"eager"``).

        Returns:
            Self: This config, for method chaining.

        Raises:
            ValueError: If ``mode`` is a string that is not a valid
                ``ExecutionMode`` value.

        Example:
            >>> config = QuantizerConfig.presets.w4()
            >>> config.set_execution_mode(ExecutionMode.EAGER)

        """
        self.execution_mode = ExecutionMode(mode)
        return self


# Preset wiring — after all classes so _presets can import ExecutionMode.
from coreai_opt.quantization.config._presets import (  # noqa: E402, PLC0415
    _ModuleQuantizerConfigPresets,
    _QuantizerConfigPresets,
)

ModuleQuantizerConfig.presets = _ModuleQuantizerConfigPresets(owner_cls=ModuleQuantizerConfig)
QuantizerConfig.presets = _QuantizerConfigPresets(owner_cls=QuantizerConfig)
