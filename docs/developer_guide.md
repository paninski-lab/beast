# BEAST Developer Guide

This guide is aimed at contributors (human or AI agent) adding a new model to the BEAST
codebase. It covers the overall architecture, the building blocks available for reuse, and a
concrete step-by-step checklist for registering a new model end-to-end.

---

## Package layout

```
beast/
  api/
    model.py           # High-level Model wrapper (from_config, from_dir, train, predict_*)
  cli/
    commands/          # One file per CLI subcommand (train, predict, extract, extract_3d)
    main.py            # Entry point; dispatches to COMMANDS registry
  config.py            # Shared Pydantic schemas (TrainingConfig, OptimizerConfig, DataConfig,
                       #   BeastConfig) + per-model dispatch via get_beast_config_class()
  data/
    datasets.py        # BaseDataset, MultiViewDataset
    datamodules.py     # BaseDataModule, MultiViewDataModule
    samplers.py        # ContrastBatchSampler (contrastive learning)
    augmentations.py   # imgaug pipeline builders
    types.py           # ExampleDict, MultiViewExampleDict (TypedDicts)
    video.py           # Video frame extraction utilities
  geometry/
    camera.py          # c2w/w2c conversions, intrinsic normalisation, SE(3) helpers
    rotations.py       # 6D rotation ↔ matrix, quaternion utilities
    positional_encoding.py  # 2-D sinusoidal PE used by ERayZer
  inference.py         # predict_images(), predict_video() — used by Model.predict_*
  io.py                # load_config(), apply_config_overrides(), calibration loaders
  logging.py           # log_step() helper
  models/
    base.py            # BaseLightningModel (abstract; all models inherit from this)
    registry.py        # MODEL_REGISTRY, TRAIN_REGISTRY, CONFIG_REGISTRY + _register_all()
    beast_resnet/      # Per-model package — see "Model package layout" below
    beast_vit/
    erayzer/
  preprocess/          # extract pipeline (extraction.py, extraction_3d.py) + SAM segmenter
  rendering/
    gaussians_renderer.py  # gsplat-based Gaussian splatting renderer
    transformer.py         # shared transformer building blocks (used by ERayZer)
    losses.py              # rendering losses (L2, perceptual)
    dino.py                # DINOv2 perceptual feature extractor
  train.py             # Shared training loop (data setup, Lightning Trainer, callbacks)
configs/
  resnet_ae.yaml
  vit.yaml
  multiview/
    erayzer.yaml        # Reference config for ERayZer / BEAST3D
tests/                 # Mirror of beast/ package tree; see Testing section
```

---

## Data flow

```
CLI / user script
     │
     ▼
beast.io.load_config(path)
     │  reads YAML → peeks at model.model_class
     │  → get_beast_config_class(model_class) → appropriate Pydantic config class
     │  → model_validate(raw) → model_dump() → plain dict
     ▼
beast.api.model.Model.from_config(config)
     │  validates dict if not already validated
     │  → MODEL_REGISTRY[model_class](config)  →  LightningModule instance
     ▼
Model.train(output_dir)
     │  → TRAIN_REGISTRY[model_class](config, model, output_dir)
     │      (currently all three models delegate to beast.train.train)
     ▼
beast.train.train(config, model, output_dir)
     │  branches on model_class to select DataModule:
     │    'erayzer' / beast3d → MultiViewDataModule
     │    'resnet' / 'vit'    → BaseDataModule (with BaseDataset + imgaug)
     │  saves config.yaml to output_dir
     │  → pl.Trainer.fit(model, datamodule)
     ▼
BaseLightningModel subclass
     training_step / validation_step
     → get_model_outputs() → compute_loss()   (abstract; implemented per-model)
```

---

## Config architecture

Each model owns its complete Pydantic schemas in its `<model>_config.py` file:

| Schema | Location | Purpose |
|---|---|---|
| `ResnetModelConfig` | `beast_resnet/beast_resnet_config.py` | model section |
| `VitModelConfig` | `beast_vit/beast_vit_config.py` | model section |
| `ERayZerModelConfig` | `erayzer/erayzer_config.py` | model section |
| `ERayZerTrainingConfig` | `erayzer/erayzer_config.py` | training section |
| `ERayZerOptimizerConfig` | `erayzer/erayzer_config.py` | optimizer section |

Models that share the standard training loop (resnet, vit) reuse `TrainingConfig` and
`OptimizerConfig` from `beast/config.py`. Models with divergent training behaviour (ERayZer,
BEAST3D) define their own training/optimizer schemas.

Top-level full-config classes live in `beast/config.py` where they assemble model + training +
optimizer + data schemas:

- `BeastConfig` — used for `resnet` and `vit`
- `ERayZerBeastConfig` — used for `erayzer` (and subclasses like BEAST3D if they share the
  same training schema)

`get_beast_config_class(model_class)` in `config.py` is the single dispatch point. It returns
`ERayZerBeastConfig` for `'erayzer'` and `BeastConfig` for everything else. Both `load_config`
and `Model.from_config` use it.

**Adding a model with a distinct training schema** requires:
1. Define `<Model>TrainingConfig` and `<Model>OptimizerConfig` in `<model>_config.py`
2. Define `<Model>BeastConfig` in `beast/config.py`
3. Add `'<model_class>': <Model>BeastConfig` to `_MODEL_CONFIG_CLASSES` in `config.py`

---

## Model package layout

Every model lives in `beast/models/<name>/` and contains exactly:

```
beast/models/<name>/
  __init__.py          # re-exports model class, all config classes, train
  <name>_config.py     # Pydantic schemas for model (+ training/optimizer if divergent)
  <name>_model.py      # LightningModule subclass
  <name>_train.py      # training entry point; either delegates to beast.train.train or
                       #   provides a custom loop
```

The model's `model_class` string identifier (e.g. `'resnet'`, `'vit'`, `'erayzer'`) is a
stable logical name set in `Literal['<name>']` inside the config class. It does **not** have
to match the directory name (just as HuggingFace's `model_type` strings don't match directory
names).

---

## Registry

`beast/models/registry.py` holds three dicts populated by `_register_all()`:

```python
MODEL_REGISTRY: dict[str, type]           # model_class → LightningModule subclass
TRAIN_REGISTRY: dict[str, Callable]       # model_class → train function
CONFIG_REGISTRY: dict[str, type[BaseModel]] # model_class → model-section config class
```

`_register_all()` runs at import time (bottom of `registry.py`). To add a new model, add
three lines to `_register_all()` — see step 6 of the checklist below.

---

## Reusable building blocks

### Data

| Class | When to use |
|---|---|
| `BaseDataset` | Single-view image folder; handles JPEG/PNG, imgaug transforms, grayscale→RGB |
| `MultiViewDataset` | Multi-view scenes from `beast extract_3d` output; loads images + optional cameras (c2w, fxfycxcy) + optional masks |
| `BaseDataModule` | Wraps `BaseDataset`; random train/val/test split by probability |
| `MultiViewDataModule` | Wraps `MultiViewDataset`; random split by `train_fraction`; used by ERayZer and BEAST3D |

`MultiViewExampleDict` (from `beast.data.types`) is the typed batch dict returned by
`MultiViewDataset.__getitem__`. Keys: `image`, `view_names`, `video_id`, `frame_id`, and
optionally `c2w`, `fxfycxcy`, `input_mask`.

Camera tensors use the following conventions throughout:
- **c2w**: camera-to-world transform, shape `(views, 4, 4)`
- **fxfycxcy**: intrinsics in absolute pixels `(fx, fy, cx, cy)`, shape `(views, 4)`
- `normalized_intrinsics=False` must be set on the model when passing absolute-pixel intrinsics

### Model base class

`BaseLightningModel` (`beast/models/base.py`) handles:
- `configure_optimizers` — AdamW or Adam + `MultiStepLR` or `OneCycleLR`
- `training_step`, `validation_step`, `test_step` — delegate to `evaluate_batch`
- `evaluate_batch` — calls `get_model_outputs` then `compute_loss`; handles logging

Subclasses **must** implement three abstract methods:

```python
def get_model_outputs(self, batch_dict: dict) -> dict: ...
def compute_loss(self, stage: str | None, **kwargs) -> tuple[torch.Tensor, list[dict]]: ...
def predict_step(self, batch_dict: dict, batch_idx: int) -> dict: ...
```

`BaseLightningModel` does **not** implement an optimizer for ERayZer-style models (ERayZer
overrides `configure_optimizers` entirely). If your model uses a non-standard optimizer,
override `configure_optimizers` in the subclass.

### Rendering and geometry

| Module | Contents |
|---|---|
| `beast.rendering.gaussians_renderer` | Gaussian splatting renderer (wraps gsplat) |
| `beast.rendering.transformer` | Transformer building blocks used by ERayZer |
| `beast.rendering.losses` | L2, perceptual (DINO), GS regularisation losses |
| `beast.geometry.camera` | `intrinsics_to_fxfycxcy`, `normalize_camera_sequence`, `scale_intrinsics`, `w2c_to_c2w`, SE(3) helpers |
| `beast.geometry.rotations` | `rotation_6d_to_matrix`, `matrix_to_rotation_6d`, quaternion ↔ matrix |
| `beast.geometry.positional_encoding` | 2-D sinusoidal PE |

---

## Step-by-step: adding a new model

This checklist uses `beast3d` as an example. Adjust names accordingly.

### 1. Create the model package directory

```
beast/models/beast3d/
  __init__.py
  beast3d_config.py
  beast3d_model.py
  beast3d_train.py
```

### 2. Write `beast3d_config.py`

Define all Pydantic schemas the model needs. If the model shares the standard training loop
(BaseDataModule + epoch-based training), import and reuse `TrainingConfig` and `OptimizerConfig`
from `beast.config`. If it needs different training fields, define `Beast3DTrainingConfig` and
`Beast3DOptimizerConfig` here.

```python
from typing import Literal
from pydantic import BaseModel

class Beast3DModelConfig(BaseModel):
    model_class: Literal['beast3d']
    seed: int = 0
    checkpoint: str | None = None
    # ... model-specific fields ...
```

Follow the ERayZer configs as a reference for a fully typed example.

### 3. Write `beast3d_model.py`

Subclass `BaseLightningModel` (or another model if the architecture is closely related):

```python
from beast.models.base import BaseLightningModel

class Beast3D(BaseLightningModel):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        # build architecture from config['model']

    def get_model_outputs(self, batch_dict: dict) -> dict: ...
    def compute_loss(self, stage, **kwargs) -> tuple[Tensor, list[dict]]: ...
    def predict_step(self, batch_dict, batch_idx) -> dict: ...
```

If BEAST3D inherits from ERayZer, override only the methods that differ (e.g.
`_resolve_cameras` to return GT cameras from `data['c2w']` and `data['fxfycxcy']` instead of
predicting them).

### 4. Write `beast3d_train.py`

If the model uses the shared training loop unchanged:

```python
from beast.train import train

__all__ = ['train']
```

If it needs a custom loop, implement `train(config, model, output_dir)` here and make sure it
has the same signature.

### 5. Write `__init__.py`

Re-export the public API of the package:

```python
from beast.models.beast3d.beast3d_config import Beast3DModelConfig
from beast.models.beast3d.beast3d_model import Beast3D
from beast.models.beast3d.beast3d_train import train

__all__ = ['Beast3D', 'Beast3DModelConfig', 'train']
```

### 6. Register the model in `beast/models/registry.py`

Add three imports and three dict assignments inside `_register_all()`:

```python
from beast.models.beast3d.beast3d_config import Beast3DModelConfig
from beast.models.beast3d.beast3d_model import Beast3D
from beast.models.beast3d.beast3d_train import train as beast3d_train

MODEL_REGISTRY['beast3d'] = Beast3D
TRAIN_REGISTRY['beast3d'] = beast3d_train
CONFIG_REGISTRY['beast3d'] = Beast3DModelConfig
```

### 7. Update `beast/config.py`

**If the model uses `BeastConfig`** (shared `TrainingConfig` + `OptimizerConfig`): add
`Beast3DModelConfig` to the `ModelConfig` union so `BeastConfig` can validate it:

```python
ModelConfig = Annotated[
    ResnetModelConfig | VitModelConfig | Beast3DModelConfig,
    Field(discriminator='model_class'),
]
```

**If the model needs its own training/optimizer schemas**: import them and assemble a new
full-config class, then add it to `_MODEL_CONFIG_CLASSES`:

```python
from beast.models.beast3d.beast3d_config import Beast3DModelConfig  # (already imported above)

class Beast3DBeastConfig(BaseModel):
    model: Beast3DModelConfig
    training: Beast3DTrainingConfig
    optimizer: Beast3DOptimizerConfig
    data: DataConfig
    inference: bool = False

_MODEL_CONFIG_CLASSES['beast3d'] = Beast3DBeastConfig
```

### 8. Add a YAML config file

Create `configs/multiview/beast3d.yaml` (or `configs/beast3d.yaml` for non-multiview models).
Keep it alongside its `model_class: beast3d` sibling configs. It will be automatically picked
up by `TestConfigFiles` in `tests/test_config.py`.

### 9. Write tests

Mirror the source tree under `tests/`:

```
tests/models/beast3d/
  __init__.py
  test_beast3d_model.py
```

Fixtures live in `tests/conftest.py` (shared) or a `conftest.py` in the same directory.
Test assets (sample tensors, etc.) go in `tests/models/beast3d/assets/`.

The test for the model config typically looks like:

```python
class TestBeast3DModelConfig:
    def test_valid_config(self) -> None:
        Beast3DModelConfig.model_validate({'model_class': 'beast3d', ...})

    def test_missing_required_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            Beast3DModelConfig.model_validate({})
```

---

## CLI

The CLI is built from `beast/cli/commands/`. Each command module exposes two functions:

- `register_parser(subparsers)` — adds the subcommand and its arguments
- `handle(args)` — executes the command

`beast/cli/commands/__init__.py` maintains a `COMMANDS` dict that the main parser iterates.
To add a CLI subcommand for a new model operation, add a new module there.

---

## Testing conventions

- Test files mirror the source tree: `beast/a/b/c.py` → `tests/a/b/test_c.py`
- Test class names: `Test<FunctionOrClassName>`
- Fixture shared across a directory: `conftest.py` in that directory
- External test data is downloaded by `tests/conftest.py` at import time via `fetch_test_data_if_needed`
- All YAML files in `configs/` are validated automatically by `TestConfigFiles` in `tests/test_config.py`
  (files listed in `_NON_BEAST_CONFIG_NAMES` are excluded, e.g. `extraction_pipeline.yaml`
  which uses `Beast3DConfig` rather than `BeastConfig`)
