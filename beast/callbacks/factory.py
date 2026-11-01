"""Factory for the Lightning callbacks used during training."""

from pathlib import Path

from lightning.pytorch import Callback
from lightning.pytorch import callbacks as pl_callbacks

from beast.callbacks.val_viz import ValidationVizCallback


def get_callbacks(
    checkpointing: bool = True,
    lr_monitor: bool = True,
    ckpt_every_n_epochs: int | None = None,
    viz_every_n_epochs: int | None = None,
    viz_num_frames: int = 108,
    output_dir: str | Path | None = None,
) -> list[Callback]:
    """Build list of Lightning callbacks for training.

    Parameters
    ----------
    checkpointing: whether to save the best checkpoint by validation loss
    lr_monitor: whether to log learning rate each epoch
    ckpt_every_n_epochs: if set, also save a checkpoint every n epochs
    viz_every_n_epochs: if set, save contact sheets of validation images and model outputs
        every n epochs; requires the model to implement ``get_viz_panels``
    viz_num_frames: number of validation frames to include in the contact sheets
    output_dir: run directory where contact sheets are saved; required if
        ``viz_every_n_epochs`` is set

    Returns
    -------
    list of configured Lightning callbacks

    Raises
    ------
    ValueError: if ``viz_every_n_epochs`` is set without ``output_dir``

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

    if viz_every_n_epochs:
        if output_dir is None:
            raise ValueError('output_dir is required when viz_every_n_epochs is set')
        callbacks.append(ValidationVizCallback(
            output_dir=output_dir,
            every_n_epochs=viz_every_n_epochs,
            num_frames=viz_num_frames,
        ))

    return callbacks
