# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, final

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveInt,
    model_validator,
)

from coreai_opt.config import (
    CompressionConfig,
    ModuleCompressionConfig,
    OpCompressionConfig,
    WeightOnlyModuleValidationMixin,
    WeightOnlyOpValidationMixin,
)
from coreai_opt.palettization.spec import (
    PalettizationSpec,
    default_weight_palettization_spec,
)

if TYPE_CHECKING:
    from coreai_opt.palettization.config._presets import (
        _KMeansPalettizerConfigPresets,
        _ModuleKMeansPalettizerConfigPresets,
    )

_KMEANS_PALETTIZATION_CONFIG = "kmeans_palettization_config"
_PALETTIZATION_SPEC = "palettization_spec"


class PATSchedule(BaseModel):
    """Schedule for enabling palettization-aware training (PAT).

    Defines the step threshold at which a module's fake palettization
    forward pass becomes active. Used with ``KMeansPalettizer.step()``.

    Attributes:
        enable_fake_palettize: Step count at which fake palettization is
            enabled. Must be >= 0.

    Example:
        >>> schedule = PATSchedule(enable_fake_palettize=500)
    """

    model_config = ConfigDict(frozen=True)

    enable_fake_palettize: int = Field(default=0, ge=0)

    def _compute_state(self, step_count: int) -> bool:
        """Return whether fake palettization should be active at the given step."""
        return step_count >= self.enable_fake_palettize


class VectorKMeansConfig(BaseModel):
    """K-means settings for vector palettization (``cluster_dim > 1``).

    Vector palettization clusters each palettization block's ``cluster_dim``-long weight
    vectors into ``2 ** n_bits`` centroids with k-means. Each run seeds the
    centroids with k-means++, then runs Lloyd iterations (assign every vector to its
    nearest centroid, move each centroid to the mean of its vectors) until the block
    converges. These settings trade clustering quality for speed.

    Where clustering runs:

    - **CUDA GPU**: all blocks of a weight tensor are clustered together in one batch.
      Unweighted clustering (post-training palettization) uses flash-kmeans for the
      Lloyd iterations when it is installed, and otherwise a batched PyTorch loop.
      Sensitivity-weighted clustering (``calibration_mode()``) always uses the PyTorch
      loop, because flash-kmeans has no per-point weights.
    - **CPU**: scikit-learn, one block at a time. If scikit-learn is not installed, the
      batched PyTorch loop runs on CPU instead and a warning is issued; it gives the
      same results but is several times slower.

    Every backend uses the same seeding, the same stopping rule, and a seed derived
    from each tensor's contents, so results are reproducible across runs.

    Attributes:
        n_local_trials (int | None): Number of candidate centroids sampled at each
            k-means++ seeding step. The candidate that most reduces the clustering
            error is kept. ``1`` is standard k-means++. Larger values ("greedy"
            k-means++) usually give lower clustering error, and seeding time grows
            roughly in proportion. ``None`` uses scikit-learn's default,
            ``2 + int(log(n_clusters))``. Default: ``None``.
        n_init (int): How many times k-means is run from scratch on each block, each
            time starting from a different set of initial centroids. Because the result
            depends on where k-means starts, more runs make it likelier to find a good
            one; the run with the lowest clustering error is kept. Time grows linearly
            with ``n_init``. Default: 5.
        max_iter (int): Maximum number of Lloyd iterations per run. Default: 300.
        tol (float): Convergence tolerance. A block stops iterating once its clustering
            error (inertia) improves by less than ``tol``, or when ``max_iter`` is
            reached. Default: ``1e-4``.

    Example:
        >>> # Defaults: greedy k-means++, 5 runs, convergence-based stopping.
        >>> default = VectorKMeansConfig()
        >>> # Fast, approximate clustering, e.g. to check that a model exports.
        >>> fast = VectorKMeansConfig(n_local_trials=1, n_init=1, max_iter=1)
        >>> # More runs for a sensitive model.
        >>> thorough = VectorKMeansConfig(n_init=12)
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_local_trials: PositiveInt | None = None
    n_init: PositiveInt = 5
    max_iter: PositiveInt = 300
    tol: NonNegativeFloat = 1e-4


class OpKMeansPalettizerConfig(WeightOnlyOpValidationMixin, OpCompressionConfig[PalettizationSpec]):
    """
    Configuration class for palettization at the operation level.

    Palettization is a weight-only compression technique that doesn't apply
    to activations (inputs/outputs). Only op_state_spec is used to configure
    which state tensors (e.g., weights, biases) should be palettized.

    Attributes:
        op_state_spec (dict[str, PalettizationSpec | None] | None): Palettization
            specifications for operation state tensors (parameters, buffers, constants).
            Keys are string names (e.g., "weight", "bias") or "*" to refer to all
            state inputs. Values are PalettizationSpec objects or None to disable
            palettization for that state tensor.
            Default: 4-bit palettization for "weight" and "in_proj_weight"
            state tensors via ``default_weight_palettization_spec()``.

    Example:
        >>> # Palettize weights with 2-bit LUT
        >>> op_config = OpKMeansPalettizerConfig(
        ...     op_state_spec={
        ...         "weight": PalettizationSpec(n_bits=2)
        ...     }
        ... )
        >>>
        >>> # Disable palettization for a specific operation
        >>> op_config = OpKMeansPalettizerConfig(
        ...     op_state_spec=None  # No state tensors will be palettized
        ... )
    """

    @classmethod
    def get_default_state_spec(cls) -> dict[str, PalettizationSpec | None]:
        """Provide default state spec for palettization."""
        spec = default_weight_palettization_spec()
        return {"weight": spec, "in_proj_weight": spec}


@final
class ModuleKMeansPalettizerConfig(
    WeightOnlyModuleValidationMixin,
    ModuleCompressionConfig[OpKMeansPalettizerConfig, PalettizationSpec],
):
    """Configuration for palettizing a specific module using K-means clustering.

    This class manages palettization settings for an entire module, including:

    - Operation-level configurations (default, by type, by name)
    - Module-level state (parameter) palettization

    The operation configurations follow a hierarchical precedence:

    1. op_name_config (most specific - applies to operations matching a name
       pattern)
    2. op_type_config (applies to operations of a specific type)
    3. op_state_spec (least specific - applies to all operations not
       otherwise configured)

    Module-level state settings treat the module as an opaque entity,
    setting palettization settings for specified tensors and ignoring op specific
    palettization capabilities. Module-level settings also don't check whether the
    operation receiving the palettized tensor is a registered operation or not.
    Module-level settings will override any op specific settings.

    Attributes:
        op_state_spec (dict[str, PalettizationSpec | None] | None): Palettization
            specifications for operation state tensors (parameters, buffers, constants)
            applied to all registered operations/patterns within this module that don't
            have a more specific configuration.
            Keys can be string names (e.g. "weight", "bias") or "*" to refer to all
            state inputs.
            Values are PalettizationSpec objects or None defining how to palettize each
            state tensor. None value represents disabling palettization.
            Default: 4-bit palettization for "weight" and "in_proj_weight"
            state tensors via ``default_weight_palettization_spec()``.

        op_type_config (dict[str, OpKMeansPalettizerConfig | None] | None): Operation
            type-specific configurations. Keys are operation type names (e.g.,
            "aten.linear.default", "aten.conv2d.default"). Values are
            OpKMeansPalettizerConfig objects or None, defining how to palettize
            operations of that type. None value represents disabling palettization.
            Default: {} (empty dict, no type-specific configs)

        op_name_config (dict[str, OpKMeansPalettizerConfig | None] | None): Operation
            name-specific configurations. Keys are operation name patterns
            (supports regex matching). Values are OpKMeansPalettizerConfig objects or
            None, defining how to palettize operations matching those names. None value
            represents disabling palettization.
            Default: {} (empty dict, no name-specific configs)

        module_state_spec (dict[str, PalettizationSpec | None] | None): Palettization
            specifications for module state tensors (parameters, buffers, and
            constants). Module state settings will override op state settings for the
            same state tensors.
            Keys can be string names (e.g. "weight", "bias") or "*" to refer to all
            state inputs.
            Values are PalettizationSpec objects or None. None value represents
            disabling palettization.
            Default: {} (empty dict, no specific module state settings)

        enable_fast_kmeans_mode (bool): When True, enables optimizations for faster
            K-means clustering by rounding the weights before clustering if data is in
            float16 range. If weight dtype is float32, weights are cast to float16 and
            then rounded. This is not supported with ``cluster_dim > 1``. Default: True.

        rounding_precision (int): Number of decimal places to round to during fast
            K-means clustering. Higher values preserve more precision but may reduce
            speed benefits. Only used when enable_fast_kmeans_mode is True. Default: 4.

        pat_schedule (PATSchedule | None): Schedule controlling when this
            module's palettization is active during a training_mode() loop.
            If None, palettization is active immediately once training_mode()
            begins. Default: None.

        vector_kmeans_config (VectorKMeansConfig | None): K-means settings for vector
            palettization (``cluster_dim > 1``). Only valid when the module configures
            at least one spec with ``cluster_dim > 1``. If None, vector palettization
            uses the ``VectorKMeansConfig`` defaults. Default: None.

    Example:
        >>> config = ModuleKMeansPalettizerConfig()  # Uses defaults
        >>> # Or with custom settings:
        >>> from coreai_opt.palettization.spec import PalettizationSpec
        >>> config = ModuleKMeansPalettizerConfig(
        ...     op_state_spec={"weight": PalettizationSpec(n_bits=2)},
        ...     enable_fast_kmeans_mode=False,  # Disable for maximum precision
        ...     rounding_precision=6  # Higher precision when fast mode is enabled
        ... )
    """

    def __init_subclass__(cls, **kwargs):
        # Prohibit subclassing due to preset limitation: presets remain bound
        # to the base class. Revisit if subclass support is needed.
        super().__init_subclass__(**kwargs)
        msg = f"{cls.__name__} cannot subclass ModuleKMeansPalettizerConfig (marked final)."
        raise TypeError(msg)

    enable_fast_kmeans_mode: bool = True
    rounding_precision: PositiveInt = 4
    pat_schedule: PATSchedule | None = None
    vector_kmeans_config: VectorKMeansConfig | None = None

    # Namespace exposing built-in preset constructors.
    presets: ClassVar[_ModuleKMeansPalettizerConfigPresets]

    @model_validator(mode="after")
    def validate_fast_kmeans_cluster_dim_constraint(
        self,
    ) -> ModuleKMeansPalettizerConfig:
        """Validate that enable_fast_kmeans_mode is not True when cluster_dim > 1."""
        if self.op_state_spec is None:
            return self

        # Check all palettization specs within op_state_spec
        for state, spec in self.op_state_spec.items():
            if self.enable_fast_kmeans_mode and spec is not None and spec.cluster_dim > 1:
                raise ValueError(
                    f"enable_fast_kmeans_mode is not supported when cluster_dim > 1. "
                    f"Got enable_fast_kmeans_mode={self.enable_fast_kmeans_mode}, "
                    f"cluster_dim={spec.cluster_dim} for state '{state}'"
                )

        return self

    @model_validator(mode="after")
    def validate_vector_kmeans_has_vector_spec(self) -> ModuleKMeansPalettizerConfig:
        """Validate that vector_kmeans_config is only set alongside a spec with cluster_dim > 1."""
        if self.vector_kmeans_config is None:
            return self

        specs = [*(self.op_state_spec or {}).values(), *(self.module_state_spec or {}).values()]
        for op_configs in (self.op_type_config, self.op_name_config):
            for op_config in (op_configs or {}).values():
                if op_config is not None:
                    specs.extend((op_config.op_state_spec or {}).values())
        if not any(spec is not None and spec.cluster_dim > 1 for spec in specs):
            raise ValueError(
                "vector_kmeans_config only applies to vector palettization, but no palettization "
                "spec in this module config has cluster_dim > 1."
            )
        return self

    def _get_fake_module_kwargs(self) -> dict:
        """Exclude palettizer-only fields from the fake-palettize constructor args."""
        base = super()._get_fake_module_kwargs()
        base.pop("pat_schedule", None)
        return base


@final
class KMeansPalettizerConfig(CompressionConfig[ModuleKMeansPalettizerConfig]):
    """
    Top-level configuration class for kmeans palettization.

    This class manages the complete palettization configuration for a neural
    network model, organizing module-level configurations in a hierarchical
    structure. It inherits from CompressionConfig and specializes it for
    palettization using ModuleKMeansPalettizerConfig.

    The configuration lookup follows a hierarchical precedence (most to least
    specific):

    1. module_name_configs - Applies to module instances matching a name
       pattern (supports regex)
    2. module_type_configs - Applies to all modules of a specific type (e.g.,
       torch.nn.modules.linear.Linear)
    3. global_config - Default configuration applied to all modules not
       otherwise configured

    Attributes:
        global_config (ModuleKMeansPalettizerConfig | None): Default module-level
            palettization configuration applied to all modules that don't have
            a more specific configuration. When KMeansPalettizerConfig is initialized
            with no arguments, a default global_config is automatically created with
            standard 4-bit palettization.
            Setting global_config to None disables palettization by default globally.
            Default: Auto-created with 4-bit palettization spec when no args provided

        module_type_configs (dict[str, ModuleKMeansPalettizerConfig | None] | None):
            Module type-specific configurations. Keys are fully-qualified module type
            names (e.g., "torch.nn.modules.linear.Linear",
            "torch.nn.modules.conv.Conv2d"). Values are ModuleKMeansPalettizerConfig
            objects or None to disable palettization for that module type.
            Default: {} (empty dict, no type-specific configs)

        module_name_configs (dict[str, ModuleKMeansPalettizerConfig | None] | None):
            Module name-specific configurations. Keys are module name patterns
            (supports regex matching, e.g., "model.layer1.*",
            "decoder.layers.0"). Values are ModuleKMeansPalettizerConfig objects or
            None to disable palettization for matching modules.
            Default: {} (empty dict, no name-specific configs)

    Example:
        >>> import torch.nn as nn
        >>> config = KMeansPalettizerConfig()  # Uses defaults
        >>> # Or with custom settings:
        >>> config = KMeansPalettizerConfig(
        ...     module_type_configs={nn.Linear: ModuleKMeansPalettizerConfig(...)},
        ...     module_name_configs={"layer1": None}  # Skip palettization
        ... )
    """

    def __init_subclass__(cls, **kwargs):
        # Prohibit subclassing due to preset limitation: presets remain bound
        # to the base class. Revisit if subclass support is needed.
        super().__init_subclass__(**kwargs)
        msg = f"{cls.__name__} cannot subclass KMeansPalettizerConfig (marked final)."
        raise TypeError(msg)

    # Class attributes for config key pattern used in from_dict/from_yaml
    _CONFIG_KEY: ClassVar[str] = _KMEANS_PALETTIZATION_CONFIG
    _SPEC_KEY: ClassVar[str] = _PALETTIZATION_SPEC

    # Namespace exposing built-in and registered preset constructors.
    presets: ClassVar[_KMeansPalettizerConfigPresets]


# Preset wiring — after the class so _presets can resolve references at runtime.
from coreai_opt.palettization.config._presets import (  # noqa: E402, PLC0415
    _KMeansPalettizerConfigPresets,
    _ModuleKMeansPalettizerConfigPresets,
)

KMeansPalettizerConfig.presets = _KMeansPalettizerConfigPresets(owner_cls=KMeansPalettizerConfig)
ModuleKMeansPalettizerConfig.presets = _ModuleKMeansPalettizerConfigPresets(
    owner_cls=ModuleKMeansPalettizerConfig
)
