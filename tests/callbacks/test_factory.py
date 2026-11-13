"""Tests for the callback factory."""

from lightning.pytorch import callbacks as pl_callbacks

from beast.callbacks.factory import get_callbacks


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
