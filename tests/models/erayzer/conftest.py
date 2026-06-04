"""Fixtures for ERayZer integration tests."""

import copy
import gc

import lightning.pytorch as pl
import pytest
import torch

from beast.api.model import Model
from beast.data.datamodules import MultiViewDataModule


def _gsplat_cuda_available() -> bool:
    """Return True if gsplat's CUDA extension compiled and loaded successfully."""
    try:
        from gsplat.cuda._backend import _C  # noqa: F401
        return _C is not None
    except Exception:
        return False


# Applied to every integration test; skips gracefully when gsplat CUDA is absent
# (e.g. missing CUDA toolkit, architecture not yet supported by available wheels).
requires_gsplat_cuda = pytest.mark.skipif(
    not _gsplat_cuda_available(),
    reason='gsplat CUDA extension not available (no nvcc / unsupported GPU arch)',
)


@pytest.fixture
def run_erayzer_model_test(tmp_path, multiview_data_dir):
    """Build, train, and run inference on an ERayZer model.

    Mirrors run_model_test from tests/conftest.py but uses MultiViewDataModule
    for both training and inference, since ERayZer requires multi-view input.

    The fixture sets up a minimal step count so that at least one full training
    epoch (and therefore one validation run + best-checkpoint save) completes
    within the multiview test fixtures (20 frames, batch_size=2 → 9 steps/epoch).
    """
    def _run(config: dict) -> None:
        config = copy.deepcopy(config)
        config['data']['data_dir'] = str(multiview_data_dir)
        config['training']['train_batch_size'] = 2
        config['training']['val_batch_size'] = 2
        config['training']['num_workers'] = 0
        config['training']['log_every_n_steps'] = 1
        config['training']['check_val_every_n_epoch'] = 1
        # 20 steps covers 2 full epochs (9 steps each) so val + checkpoint run twice
        config['training']['max_fwdbwd_passes'] = 20
        # warmup must be strictly < max_fwdbwd_passes
        config['optimizer']['warmup'] = 5

        model = Model.from_config(config)
        try:
            model.train(tmp_path)

            # verify predict_step works on multiview batches
            dm = MultiViewDataModule(
                data_dir=multiview_data_dir,
                image_size=config['model']['image_tokenizer']['image_size'],
                train_batch_size=2,
                val_batch_size=2,
                train_fraction=0.8,
                num_workers=0,
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
