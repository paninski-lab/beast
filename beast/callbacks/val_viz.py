"""Callback that saves contact sheets of validation images and model outputs during training."""

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities import rank_zero_only
from PIL import Image
from torch.utils.data import default_collate

from beast.models.base import BaseLightningModel

_logger = logging.getLogger(__name__)

# target side length of a contact sheet, in panels
_SHEET_SIDE_PANELS = 9


def sheet_grid_shape(n_panels: int) -> tuple[int, int]:
    """Compute the (rows, cols) of frame groups on a roughly square contact sheet.

    Each frame occupies a row of ``n_panels`` side-by-side panels. Frames are tiled so that the
    sheet is ``cols * n_panels`` panels wide and ``rows`` panels tall, with rows == the width.

    Parameters
    ----------
    n_panels: number of panels per frame (e.g. 3 for input / masked input / reconstruction)

    Returns
    -------
    tuple of (rows, cols), e.g. (9, 3) for 3 panels per frame

    """
    cols = max(1, _SHEET_SIDE_PANELS // n_panels)
    return cols * n_panels, cols


def select_viz_indices(n_available: int, n_frames: int) -> list[int]:
    """Select evenly spaced, deterministic dataset indices.

    Parameters
    ----------
    n_available: number of frames in the dataset
    n_frames: desired number of frames; capped at ``n_available``

    Returns
    -------
    sorted list of unique indices

    """
    n_frames = min(n_frames, n_available)
    if n_frames <= 0:
        return []
    return np.unique(np.linspace(0, n_available - 1, n_frames).round().astype(int)).tolist()


def make_contact_sheets(
    panels: list[torch.Tensor],
    panel_size: int = 128,
    gap: int = 4,
) -> list[np.ndarray]:
    """Tile per-frame panel groups into a series of roughly square contact sheets.

    Parameters
    ----------
    panels: list of image batches, each of shape (n_frames, 3, height, width) with values in
        [0, 1]; ``panels[k][i]`` is panel k of frame i
    panel_size: side length in pixels each panel is resized to
    gap: width in pixels of the white gutter between frames

    Returns
    -------
    list of uint8 RGB arrays of shape (sheet_height, sheet_width, 3); the last sheet is padded
        with white if there are not enough frames to fill it

    """
    n_panels = len(panels)
    n_frames = panels[0].shape[0]
    rows, cols = sheet_grid_shape(n_panels)

    # (n_frames, 3, panel_size, n_panels * panel_size): panels of a frame side by side
    resized = [
        F.interpolate(p.float(), size=(panel_size, panel_size), mode='area')
        if p.shape[-1] != panel_size or p.shape[-2] != panel_size else p.float()
        for p in panels
    ]
    strips = torch.cat(resized, dim=-1)
    # white gutter on the right and bottom of each frame
    strips = F.pad(strips, (0, gap, 0, gap), value=1.0)

    per_sheet = rows * cols
    sheets = []
    for start in range(0, n_frames, per_sheet):
        chunk = strips[start:start + per_sheet]
        if chunk.shape[0] < per_sheet:
            blank = torch.ones((per_sheet - chunk.shape[0], *chunk.shape[1:]))
            chunk = torch.cat([chunk, blank], dim=0)
        # (rows * cols) -> rows of cols strips each
        grid = chunk.view(rows, cols, *chunk.shape[1:])
        # (rows, cols, 3, H, W) -> (3, rows * H, cols * W)
        grid = grid.permute(2, 0, 3, 1, 4).reshape(
            3, rows * chunk.shape[-2], cols * chunk.shape[-1],
        )
        sheets.append((grid.permute(1, 2, 0) * 255).round().byte().numpy())
    return sheets


class ValidationVizCallback(pl.Callback):
    """Save contact sheets of fixed validation frames and model outputs every N epochs.

    The model must implement ``get_viz_panels`` (see ``BaseLightningModel``). The same evenly
    spaced validation frames are used at every save so progress can be compared across epochs.
    Sheets are written to ``<output_dir>/val_images/epoch_<EEEE>_sheet_<SS>.png``.
    """

    def __init__(
        self,
        output_dir: str | Path,
        every_n_epochs: int,
        num_frames: int = 108,
        panel_size: int = 128,
        batch_size: int = 32,
    ) -> None:
        """Initialize the callback.

        Parameters
        ----------
        output_dir: run directory; sheets are saved in a ``val_images`` subdirectory
        every_n_epochs: save every this many epochs (counted from 1); must be a multiple of
            ``check_val_every_n_epoch`` since sheets are only produced when validation runs
        num_frames: number of validation frames to visualize (capped at the validation set size)
        panel_size: side length in pixels of each panel
        batch_size: number of frames passed through the model at once

        """
        super().__init__()
        if every_n_epochs < 1:
            raise ValueError(f'every_n_epochs must be >= 1, got {every_n_epochs}')
        self.output_dir = Path(output_dir)
        self.every_n_epochs = every_n_epochs
        self.num_frames = num_frames
        self.panel_size = panel_size
        self.batch_size = batch_size
        self._batch: dict | None = None  # cached viz frames, built lazily

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        """Validate that the model supports visualization and the cadence is compatible."""
        if stage != 'fit':
            return
        self._check_model(pl_module)
        check_val = trainer.check_val_every_n_epoch or 1
        if self.every_n_epochs % check_val != 0:
            _logger.warning(
                f'viz_every_n_epochs={self.every_n_epochs} is not a multiple of '
                f'check_val_every_n_epoch={check_val}; some scheduled sheets will be skipped'
            )

    def _check_model(self, pl_module: pl.LightningModule) -> None:
        """Raise if the model does not support this callback's visualization."""
        if getattr(type(pl_module), 'get_viz_panels', None) is BaseLightningModel.get_viz_panels:
            raise NotImplementedError(
                f'{type(pl_module).__name__} does not implement get_viz_panels; '
                f'set training.viz_every_n_epochs to null for this model'
            )

    def _build_batch(self, trainer: pl.Trainer) -> dict:
        """Load the fixed set of validation frames once and cache them on CPU."""
        dataset = trainer.datamodule.val_dataset  # type: ignore[attr-defined]
        idxs = select_viz_indices(len(dataset), self.num_frames)
        items = [dataset[i] for i in idxs]
        return default_collate(items)

    @torch.no_grad()
    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Generate and save contact sheets if this epoch is due."""
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return
        try:
            self._save_sheets(trainer, pl_module)
        except Exception as exc:  # noqa: BLE001 - visualization must never break training
            _logger.warning(f'validation visualization failed: {exc}')

    def _iter_batches(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> Iterator[dict]:
        """Yield the fixed visualization frames in chunks of ``batch_size`` on the model device.

        Forward passes on these happen on rank 0 only; this is safe with SyncBatchNorm because
        the module is in eval mode here (running stats are used, no cross-rank collectives).
        """
        if self._batch is None:
            self._batch = self._build_batch(trainer)
        n_frames = len(next(v for v in self._batch.values() if isinstance(v, torch.Tensor)))
        for start in range(0, n_frames, self.batch_size):
            yield {
                k: (v[start:start + self.batch_size].to(pl_module.device)
                    if isinstance(v, torch.Tensor) else v)
                for k, v in self._batch.items()
            }

    def _save_sheets(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Run the model on the fixed frames and write the contact sheets to disk."""
        model = cast(BaseLightningModel, pl_module)
        panel_chunks: list[list[torch.Tensor]] | None = None
        for batch in self._iter_batches(trainer, pl_module):
            out = model.get_viz_panels(batch)
            if panel_chunks is None:
                panel_chunks = [[] for _ in out]
            for chunks, panel in zip(panel_chunks, out, strict=True):
                chunks.append(panel.detach().float().cpu())
        assert panel_chunks is not None, 'no validation batches to visualize'
        panels = [torch.cat(chunks, dim=0) for chunks in panel_chunks]

        sheets = make_contact_sheets(panels, panel_size=self.panel_size)
        out_dir = self.output_dir / 'val_images'
        out_dir.mkdir(parents=True, exist_ok=True)
        for idx_sheet, sheet in enumerate(sheets):
            path = out_dir / f'epoch_{trainer.current_epoch:04d}_sheet_{idx_sheet:02d}.png'
            Image.fromarray(sheet).save(path)
        if rank_zero_only.rank == 0:
            _logger.info(f'saved {len(sheets)} validation contact sheets to {out_dir}')
