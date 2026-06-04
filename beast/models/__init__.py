"""Model architectures for BEAST.

Each model lives in its own sub-package under beast/models/<name>/ containing:

  <name>_config.py   — Pydantic schemas for model (and optionally training/optimizer)
  <name>_model.py    — LightningModule subclass
  <name>_train.py    — training entry point (delegates to beast.train.train or custom loop)
  __init__.py        — re-exports model class, config classes, and train function

The central registry (beast.models.registry) maps model_class strings to their
implementation classes and training functions.  All models inherit from
BaseLightningModel (beast.models.base) and must implement get_model_outputs(),
compute_loss(), and predict_step().

See docs/developer_guide.md for a full walkthrough of adding a new model.
"""
