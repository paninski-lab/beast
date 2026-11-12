"""Callback that saves validation visuals for multi-view models (ERayZer, BEAST3D)."""

import logging
from typing import cast

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities import rank_zero_only
from PIL import Image

from beast.callbacks.val_viz import ValidationVizCallback
from beast.models.base import BaseLightningModel
from beast.models.erayzer.visualize import (
    camera_intrinsic_stats,
    export_gaussian_glb,
    make_camera_pose_image,
    make_render_grid,
    stack_rows,
)

_logger = logging.getLogger(__name__)


def _to_uint8(img: torch.Tensor) -> 'Image.Image':
    """Convert a [3, H, W] float image in [0, 1] to a PIL image."""
    arr = (img.clamp(0.0, 1.0).permute(1, 2, 0) * 255).round().byte().numpy()
    return Image.fromarray(arr)


class MultiviewValidationVizCallback(ValidationVizCallback):
    """Save render grids, camera plots, point clouds and stats for fixed validation scenes.

    Per save (every ``every_n_epochs``), the same evenly spaced validation scenes are run through
    the model and the following are written under ``<output_dir>/val_images``:

    - ``epoch_<EEEE>_renders_<SS>.png``: ``scenes_per_sheet`` scenes per sheet; each scene is a
      GT row over a render row with one column per view (green border = input, red = target)
    - ``epoch_<EEEE>_cameras.png``: 3D plot of the predicted cameras for the first scene
    - ``pointclouds/epoch_<EEEE>.glb``: predicted Gaussians of the first scene
    - scalar intrinsic / Gaussian statistics, logged as ``val/<name>`` metrics

    The model's input/target view split is seeded during visualization, so scenes are directly
    comparable across epochs even when ``random_split`` is enabled.
    """

    def __init__(
        self,
        output_dir,
        every_n_epochs: int,
        num_frames: int = 8,
        scenes_per_sheet: int = 4,
        max_sheet_width: int = 1600,
        batch_size: int = 4,
        opacity_threshold: float = 0.05,
    ) -> None:
        """Initialize the callback.

        Parameters
        ----------
        output_dir: run directory; outputs are saved in a ``val_images`` subdirectory
        every_n_epochs: save every this many epochs (counted from 1); should be a multiple of
            ``check_val_every_n_epoch``
        num_frames: number of validation scenes to visualize (capped at the validation set size)
        scenes_per_sheet: number of scenes stacked vertically on each render sheet
        max_sheet_width: render sheets wider than this many pixels are downscaled to fit
        batch_size: number of scenes passed through the model at once
        opacity_threshold: Gaussians at or below this opacity are dropped from the point cloud

        """
        super().__init__(
            output_dir=output_dir,
            every_n_epochs=every_n_epochs,
            num_frames=num_frames,
            batch_size=batch_size,
        )
        self.scenes_per_sheet = scenes_per_sheet
        self.max_sheet_width = max_sheet_width
        self.opacity_threshold = opacity_threshold

    def _check_model(self, pl_module: pl.LightningModule) -> None:
        """Multi-view models are visualized through ``get_model_outputs``; nothing to check."""

    def _save_sheets(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Run the model on the fixed scenes and write renders, cameras, and point cloud."""
        out_dir = self.output_dir / 'val_images'
        out_dir.mkdir(parents=True, exist_ok=True)
        epoch = trainer.current_epoch

        scene_grids: list[torch.Tensor] = []
        stats: list[dict[str, float]] = []
        # seed so random input/target view splits are identical at every save
        devices = [pl_module.device] if pl_module.device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(0)
            for idx_batch, batch in enumerate(self._iter_batches(trainer, pl_module)):
                out = cast(BaseLightningModel, pl_module).get_model_outputs(batch)
                n_scenes = out['target_image'].shape[0]
                for i in range(n_scenes):
                    scene_grids.append(make_render_grid(
                        out['input_image'],
                        out['target_image'],
                        out['render'],
                        render_input=out.get('render_input'),
                        sample_idx=i,
                    ))
                stats.append(self._batch_stats(out, pl_module))
                if idx_batch == 0:
                    self._save_first_scene_extras(out, out_dir, epoch)

        n_sheets = self._save_render_sheets(scene_grids, out_dir, epoch)
        self._log_stats(stats, pl_module)
        if rank_zero_only.rank == 0:
            _logger.info(f'saved {n_sheets} multi-view validation sheets to {out_dir}')

    def _save_render_sheets(self, scene_grids: list[torch.Tensor], out_dir, epoch: int) -> int:
        """Stack scenes vertically into sheets and write them; returns the number of sheets."""
        n_sheets = 0
        for start in range(0, len(scene_grids), self.scenes_per_sheet):
            sheet = stack_rows(scene_grids[start:start + self.scenes_per_sheet])
            if sheet.shape[-1] > self.max_sheet_width:
                scale = self.max_sheet_width / sheet.shape[-1]
                size = (round(sheet.shape[-2] * scale), self.max_sheet_width)
                sheet = F.interpolate(sheet[None], size=size, mode='area')[0]
            _to_uint8(sheet).save(out_dir / f'epoch_{epoch:04d}_renders_{n_sheets:02d}.png')
            n_sheets += 1
        return n_sheets

    def _save_first_scene_extras(self, out: dict, out_dir, epoch: int) -> None:
        """Save the camera-pose plot and Gaussian point cloud for the first scene."""
        cameras = make_camera_pose_image(out['c2w_input'], out['c2w_target'])
        _to_uint8(cameras).save(out_dir / f'epoch_{epoch:04d}_cameras.png')
        gaussians = out.get('gaussians')
        if gaussians:
            path = out_dir / 'pointclouds' / f'epoch_{epoch:04d}.glb'
            try:
                n_points = export_gaussian_glb(
                    gaussians[0], path, opacity_threshold=self.opacity_threshold,
                )
                _logger.info(f'saved validation point cloud ({n_points} pts): {path}')
            except Exception as exc:  # noqa: BLE001 - e.g. optional trimesh not installed
                _logger.warning(f'could not export validation point cloud: {exc}')

    @staticmethod
    def _batch_stats(out: dict, pl_module: pl.LightningModule) -> dict[str, float]:
        """Compute intrinsic and Gaussian scalar statistics for one batch of outputs."""
        image_size = cast(BaseLightningModel, pl_module).config['model']['image_tokenizer'][
            'image_size'
        ]
        stats = camera_intrinsic_stats(out['fxfycxcy_target'], image_size)
        gaussians = out.get('gaussians')
        if gaussians:
            stats['opacity_mean'] = torch.cat(
                [g.get_opacity.flatten() for g in gaussians]
            ).mean().item()
            stats['scale_mean'] = torch.cat(
                [g.get_scaling.flatten() for g in gaussians]
            ).mean().item()
        return stats

    @staticmethod
    def _log_stats(stats: list[dict[str, float]], pl_module: pl.LightningModule) -> None:
        """Log the per-batch statistics, averaged over batches, as ``val/<name>`` metrics."""
        for name in stats[0]:
            value = sum(s[name] for s in stats) / len(stats)
            pl_module.log(f'val/{name}', value, on_epoch=True, rank_zero_only=True)
