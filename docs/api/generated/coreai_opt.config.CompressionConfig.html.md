# coreai_opt.config.CompressionConfig

### *class* coreai_opt.config.CompressionConfig

Bases: `BaseModel`, `Generic`[`_T`]

Top level configuration class for model compression.

This class manages compression configurations at different scopes:
- Global configuration (applies to all modules by default)
- Module type configurations (applies to all modules of specific type)
- Module name configurations (applies to module instances identified by name)

The configuration lookup follows a hierarchical precedence, where more specific
configurations override more general ones.

Generic type \_T must be a subclass of ModuleCompressionConfig.

#### build_module_config_dict(model)

Build a mapping of module names to their quantization configurations,
separating modules by config level.

The modules are associated with configs according to the following rules:

- If a module is already associated with a config of a higher priority, the
  lower priority config is ignored (module_name > module_type > global)
- A module may match with more than one config within the same config level.
  For example, a nested module named model.outer.inner could be associated
  with module name level configs for “model.outer.inner” as well as
  “model.outer.\*”.
  Whichever config is defined later in the config list is
  the one which ends up being used.
- When a module is matched with a config, the config is applied to all of
  its child modules recursively, subject to the above priority constraint.
  For example, assume a nested module model.outer.inner.module and the
  following config:
  ```default
  module_name: {"model.outer.inner": config1},
  module_type: {type(model): config2}
  ```

  Then we would have the following associations:
  - “model”: config2 (set with module_type config)
  - “model.outer”: config2 (recursively set when setting “model”)
  - “model.outer.inner”: config1 (higher priority module_name match)
  - “model.outer.inner.module”: config1 (set as part of recursively
    processing “model.outer.inner“‘s child modules)

* **Parameters:**
  **model** (*Module*) – The model with modules to get configs for
* **Returns:**
  Dictionary with nested dictionary mapping modules to configs for each config
  level
* **Return type:**
  dict[*ConfigLevel*, dict[str,  *\_T*]]

#### *classmethod* from_dict(config_dict)

Create configuration from a dictionary.

The dictionary must contain \_CONFIG_KEY key whose value
defines the hierarchical configuration structure specifying which
compression specs to apply at different scopes (global, module type,
module name) and levels (op-level or module-level).
All compression specifications should be inline dictionaries (not
references). If this dictionary was created from YAML using from_yaml(),
any YAML anchor/alias references have already been resolved by the YAML
parser and substituted with the actual spec dictionaries.

This method requires subclasses to define \_CONFIG_KEY and \_SPEC_KEY class
attributes. It will:
1. Validate that only expected keys are present
2. Extract the nested config from config_dict[_CONFIG_KEY]
3. Create the config from the extracted data

* **Parameters:**
  **config_dict** (`dict` of `str` and values) – A nested dictionary
  of strings and values. Must have the key specified by \_CONFIG_KEY
  with the actual config nested inside.
* **Returns:**
  The created configuration object, or None if
  : config_dict is None.
* **Return type:**
  [CompressionConfig](#coreai_opt.config.CompressionConfig)
* **Raises:**
  **RuntimeError** – If:
      - Subclass doesn’t define \_CONFIG_KEY or \_SPEC_KEY
      - Unexpected keys are found in config_dict
      - The required \_CONFIG_KEY is not present in config_dict

#### *classmethod* from_yaml(yml)

Create configuration from a YAML file or stream.

* **Parameters:**
  **yml** (*IO* *|* *str* *|* *Path*) – File path or IO stream containing YAML data
* **Returns:**
  A CompressionConfig instance or None if the YAML content was empty
* **Raises:**
  **ValueError** – If the YAML content is not a dictionary
* **Return type:**
  [*CompressionConfig*](#coreai_opt.config.CompressionConfig) | None

#### get_module_config(name, module)

Get the compression config for a module with priority.

1. Module name match (supports regex)
2. Module type match
3. Global config

* **Parameters:**
  * **name** (*str*) – Name of module to get config for
  * **module** (*torch.nn.Module*) – Module to get config for
* **Returns:**
  Module config for the given module.
* **Return type:**
  \_T

#### only_for(targets: list[str | type[Module]] | tuple[str | type[Module], ...], /) → Self

#### only_for(\*targets: str | type[Module]) → Self

Restrict this config to apply only to the given module types/names.

Disables `global_config` and re-applies it as a deep-copied per-module
override on each listed target. Targets may be `nn.Module` subclasses
or module-name strings, mixed in the same call and passed either as
varargs or as a single list/tuple. All targets are validated before any
mutation happens.

* **Parameters:**
  **\*targets** – One or more `nn.Module` subclasses or module name
  strings, passed as varargs or a single list/tuple.
* **Returns:**
  `self`, for chaining.
* **Return type:**
  Self
* **Raises:**
  * **ValueError** – If no targets are provided, or if `global_config` is
        already disabled. Pass all targets in one call instead of
        chaining `only_for`.
  * **TypeError** – If a target is neither an `nn.Module` subclass nor a
        string.

#### NOTE
If a target already has an explicit override (via `set_module_type`
or `set_module_name`), `only_for` overwrites it with the former
global config. To keep per-target customizations, call `only_for`
first and `set_module_type` / `set_module_name` after.

The `ValueError` raised when `only_for` is called twice (see
`Raises`) uses a private attribute that is excluded from
`model_dump` / `to_yaml`, so a round-tripped config will
accept `only_for` again (functionally a no-op).

#### set_global(config)

Set the global config.

Accepts a `ModuleCompressionConfig` (the canonical form) or `None`
to disable compression globally.

* **Parameters:**
  **config** ( *\_T* *|* *None*)
* **Return type:**
  *Self*

#### set_module_name(module_name, config)

Set the module level compression config for a given module instance.

If the module level compression config for an existing module was
already set, the new config will override the old one.

* **Parameters:**
  * **module_name** (*str*)
  * **config** ( *\_T* *|* *None*)
* **Return type:**
  *Self*

#### set_module_type(module_type, config)

Set the module level compression config for a given module type.

If the module level compression config for an existing module type was
already set, the new config will override the old one.

* **Parameters:**
  * **module_type** (*str* *|* *type* *[**Module* *]*)
  * **config** ( *\_T* *|* *None*)
* **Return type:**
  *Self*

#### to_dict()

Convert configuration to dictionary.

* **Return type:**
  dict[str, *Any*]

#### without(targets: list[str | type[Module]] | tuple[str | type[Module], ...], /) → Self

#### without(\*targets: str | type[Module]) → Self

Exclude the given module types/names from this config.

Each target gets a per-module override of `None` (disabled). The
global config and other overrides are unchanged. Targets may be
`nn.Module` subclasses or module-name strings, mixed in the same call
and passed either as varargs or as a single list/tuple. All targets are
validated before any mutation happens. Passing no targets (or an empty
list) is a no-op.

* **Parameters:**
  **\*targets** – Zero or more `nn.Module` subclasses or module name
  strings, passed as varargs or a single list/tuple.
* **Returns:**
  `self`, for chaining.
* **Return type:**
  Self
* **Raises:**
  **TypeError** – If a target is neither an `nn.Module` subclass nor a
      string.
