"""Tests for training helper functions."""

import os
import random

import numpy as np
import torch
from lightning.pytorch import callbacks as pl_callbacks

from beast.train import get_callbacks, reset_seeds


class TestResetSeeds:
    """Test the reset_seeds function."""

    def test_sets_pythonhashseed_env_var(self) -> None:
        reset_seeds(42)
        assert os.environ['PYTHONHASHSEED'] == '42'

    def test_torch_produces_same_values_after_reset(self) -> None:
        reset_seeds(0)
        a = torch.randn(5)
        reset_seeds(0)
        b = torch.randn(5)
        assert torch.equal(a, b)

    def test_numpy_produces_same_values_after_reset(self) -> None:
        reset_seeds(0)
        a = np.random.rand(5)
        reset_seeds(0)
        b = np.random.rand(5)
        assert np.array_equal(a, b)

    def test_python_random_produces_same_values_after_reset(self) -> None:
        reset_seeds(0)
        a = [random.random() for _ in range(5)]
        reset_seeds(0)
        b = [random.random() for _ in range(5)]
        assert a == b

    def test_cudnn_flags_set(self) -> None:
        reset_seeds(0)
        assert torch.backends.cudnn.benchmark is False
        assert torch.backends.cudnn.deterministic is True


class TestGetCallbacks:
    """Test the get_callbacks function."""

    def test_defaults_return_lr_monitor_and_best_checkpoint(self) -> None:
        callbacks = get_callbacks()
        assert len(callbacks) == 2
        types = {type(cb) for cb in callbacks}
        assert pl_callbacks.LearningRateMonitor in types
        assert pl_callbacks.ModelCheckpoint in types

    def test_no_lr_monitor(self) -> None:
        callbacks = get_callbacks(lr_monitor=False)
        assert not any(isinstance(cb, pl_callbacks.LearningRateMonitor) for cb in callbacks)

    def test_no_checkpointing(self) -> None:
        callbacks = get_callbacks(checkpointing=False)
        assert not any(isinstance(cb, pl_callbacks.ModelCheckpoint) for cb in callbacks)

    def test_all_disabled_returns_empty_list(self) -> None:
        assert get_callbacks(checkpointing=False, lr_monitor=False) == []

    def test_ckpt_every_n_epochs_adds_extra_checkpoint(self) -> None:
        callbacks = get_callbacks(ckpt_every_n_epochs=5)
        checkpoint_cbs = [cb for cb in callbacks if isinstance(cb, pl_callbacks.ModelCheckpoint)]
        assert len(checkpoint_cbs) == 2

    def test_best_checkpoint_monitors_val_loss(self) -> None:
        callbacks = get_callbacks(lr_monitor=False)
        ckpt = callbacks[0]
        assert isinstance(ckpt, pl_callbacks.ModelCheckpoint)
        assert ckpt.monitor == 'val_loss'
        assert ckpt.mode == 'min'

    def test_periodic_checkpoint_saves_all(self) -> None:
        callbacks = get_callbacks(lr_monitor=False, ckpt_every_n_epochs=3)
        periodic = [
            cb for cb in callbacks
            if isinstance(cb, pl_callbacks.ModelCheckpoint) and cb.monitor is None
        ]
        assert len(periodic) == 1
        assert periodic[0].every_n_epochs == 3
        assert periodic[0].save_top_k == -1
