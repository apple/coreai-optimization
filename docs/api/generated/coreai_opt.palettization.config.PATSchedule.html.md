# coreai_opt.palettization.config.PATSchedule

### *class* coreai_opt.palettization.config.PATSchedule

Bases: `BaseModel`

Schedule for enabling palettization-aware training (PAT).

Defines the step threshold at which a module’s fake palettization
forward pass becomes active. Used with `KMeansPalettizer.step()`.

#### enable_fake_palettize

Step count at which fake palettization is
enabled. Must be >= 0.
