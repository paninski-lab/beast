"""Shared pydantic schemas reused by every model's config.

This module must not import any model package, so that model ``*_config.py`` files can subclass
these schemas without circular imports. Top-level config assembly and per-model dispatch live in
``beast.configs.models``.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel


class BaseTrainingConfig(BaseModel):
    """Training fields common to every model.

    Model families subclass this to add their own fields and to override defaults that differ
    (e.g. ``num_epochs``, ``check_val_every_n_epoch``, ``viz_num_frames``).
    """

    train_batch_size: int
    val_batch_size: int
    test_batch_size: int = 128
    num_epochs: int = 800
    seed: int = 0
    num_workers: int = 8
    num_gpus: int = 1
    num_nodes: int = 1
    log_every_n_steps: int = 10
    check_val_every_n_epoch: int = 5
    precision: str = '32-true'
    # train on the first N frames (int) or fraction (0-1) of the train split; None uses all
    train_frames: int | float | None = None
    # also save a checkpoint every n epochs, in addition to the best-val checkpoint
    ckpt_every_n_epochs: int | None = None
    # save validation visualizations every n epochs
    viz_every_n_epochs: int | None = None
    viz_num_frames: int = 108


class BaseOptimizerConfig(BaseModel):
    """Optimizer and learning rate schedule fields, shared by every model."""

    type: Literal['Adam', 'AdamW'] = 'AdamW'
    lr: float
    wd: float = 0.05
    # adam betas; 0.95 for beta2 follows the MAE recipe (more stable than 0.999 for transformers)
    beta1: float = 0.9
    beta2: float = 0.95
    accumulate_grad_batches: int = 1
    scheduler: Literal['step', 'cosine', 'constant'] = 'cosine'
    # for cosine: scale the peak lr by global batch size / 256 (including grad accumulation)
    scale_lr_by_batch: bool = True
    # for cosine
    warmup_pct: float = 0.15
    div_factor: float = 10.0
    final_div_factor: float = 100.0
    # for step
    gamma: float = 0.5
    steps: list[int] = []


class DataConfig(BaseModel):
    """Data section; extend or replace for richer data configs."""

    data_dir: str | Path
