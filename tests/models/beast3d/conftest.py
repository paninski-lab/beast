"""Fixtures for BEAST3D integration tests."""

import copy
import gc
from pathlib import Path

import lightning.pytorch as pl
import pytest
import torch

from beast.api.model import Model
from beast.data.datamodules import MultiViewDataModule


def _gsplat_cuda_available() -> bool:
    """Return True if gsplat's CUDA extension compiled and loaded successfully."""
    try:
        import gsplat.cuda._backend as gsplat_backend
        return getattr(gsplat_backend, '_C', None) is not None
    except Exception:
        return False


# Applied to every integration test; skips gracefully when gsplat CUDA is absent.
requires_gsplat_cuda = pytest.mark.skipif(
    not _gsplat_cuda_available(),
    reason='gsplat CUDA extension not available (no nvcc / unsupported GPU arch)',
)


def _multiview_masks_available(data_dir: Path) -> bool:
    """Return True if the multiview fixture contains any segmentation mask file."""
    return any(Path(data_dir).rglob('mask*'))


@pytest.fixture
def run_beast3d_model_test(tmp_path, multiview_data_dir):
    """Build, train, and run inference on a BEAST3D model.

    Mirrors run_erayzer_model_test but uses GT cameras + foreground masks. Skips
    when the multiview fixture lacks masks (BEAST3D requires use_mask=True).
    DINOv3 is disabled so the test needs no network download.
    """
    def _run(config: dict) -> None:
        if not _multiview_masks_available(multiview_data_dir):
            pytest.skip('multiview test fixture has no segmentation masks for BEAST3D')
        config = copy.deepcopy(config)
        config['model']['use_dinov3'] = False
        config['data']['data_dir'] = str(multiview_data_dir)
        config['training']['train_batch_size'] = 2
        config['training']['val_batch_size'] = 2
        config['training']['num_workers'] = 0
        config['training']['log_every_n_steps'] = 1
        config['training']['check_val_every_n_epoch'] = 1
        config['training']['max_fwdbwd_passes'] = 20
        config['optimizer']['warmup'] = 5

        model = Model.from_config(config)
        try:
            model.train(tmp_path)

            dm = MultiViewDataModule(
                data_dir=multiview_data_dir,
                image_size=config['model']['image_tokenizer']['image_size'],
                train_batch_size=2,
                val_batch_size=2,
                train_fraction=0.8,
                num_workers=0,
                use_mask=True,
            )
            dm.setup()
            trainer = pl.Trainer(accelerator='gpu', devices=1, logger=False)
            preds = trainer.predict(
                model.model,
                dataloaders=dm.val_dataloader(),
                return_predictions=True,
            )
            assert preds is not None and len(preds) > 0

            assert model.model_dir is not None
            assert len(list(model.model_dir.rglob('*.ckpt'))) == 1
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()

    return _run
