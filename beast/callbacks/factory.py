"""Factory for the Lightning callbacks used during training."""

from lightning.pytorch import Callback
from lightning.pytorch import callbacks as pl_callbacks


def get_callbacks(
    checkpointing: bool = True,
    lr_monitor: bool = True,
    ckpt_every_n_epochs: int | None = None,
) -> list[Callback]:
    """Build list of Lightning callbacks shared by all models.

    Model-specific callbacks (e.g. validation visualization) are provided by each model's
    ``configure_callbacks`` method; Lightning merges them with this list.

    Parameters
    ----------
    checkpointing: whether to save the best checkpoint by validation loss
    lr_monitor: whether to log learning rate each epoch
    ckpt_every_n_epochs: if set, also save a checkpoint every n epochs

    Returns
    -------
    list of configured Lightning callbacks

    """
    callbacks = []

    if lr_monitor:
        lr_monitor_cb = pl_callbacks.LearningRateMonitor(logging_interval='epoch')
        callbacks.append(lr_monitor_cb)

    # always save out best model
    if checkpointing:
        ckpt_best_callback = pl_callbacks.ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            filename='{epoch}-{step}-best',
        )
        callbacks.append(ckpt_best_callback)

    if ckpt_every_n_epochs:
        # if ckpt_every_n_epochs is not None, save separate checkpoint files
        ckpt_callback = pl_callbacks.ModelCheckpoint(
            monitor=None,
            every_n_epochs=ckpt_every_n_epochs,
            save_top_k=-1,
        )
        callbacks.append(ckpt_callback)

    return callbacks
