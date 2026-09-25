# coreai_opt.palettization.spec.TrainingStrategySpec

### *class* coreai_opt.palettization.spec.TrainingStrategySpec

Bases: `BaseModel`, `ConfigRegistryMixin`

Base class for a fake-palettize module’s training-strategy settings.

Each subclass points `_strategy_cls` at its paired `TrainingStrategy`
behavior class; `build_strategy()` constructs that strategy from this
spec’s own fields.

#### build_strategy()

Construct this spec’s paired `TrainingStrategy` behavior instance.

* **Return type:**
  [*TrainingStrategy*](coreai_opt.palettization.spec.TrainingStrategy.md#coreai_opt.palettization.spec.TrainingStrategy)
