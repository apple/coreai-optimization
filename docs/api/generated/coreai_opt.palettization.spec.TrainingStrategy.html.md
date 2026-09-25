# coreai_opt.palettization.spec.TrainingStrategy

### *class* coreai_opt.palettization.spec.TrainingStrategy

Bases: `ABC`

Contract for a fake-palettize module’s training-time forward pass.

#### \_\_init_\_()

### Methods

| [`train_forward`](#coreai_opt.palettization.spec.TrainingStrategy.train_forward)(module, weight)   | Return the training-time output for `weight`.   |
|----------------------------------------------------------------------------------------------------|-------------------------------------------------|

#### *abstract* train_forward(module, weight)

Return the training-time output for `weight`.

Defines how the palettized weight behaves during training (e.g. frozen,
straight-through, soft assignment).

* **Parameters:**
  * **module** ( *\_FakePalettizeImplBase*)
  * **weight** (*torch.Tensor*)
* **Return type:**
  torch.Tensor
