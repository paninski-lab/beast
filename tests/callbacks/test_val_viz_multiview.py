"""Tests for the multi-view validation visualization callback."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

from beast.callbacks import val_viz_multiview
from beast.callbacks.val_viz_multiview import MultiviewValidationVizCallback

_N_VIEWS = 3
_SIZE = 16


class _FakeGaussians:
    def __init__(self) -> None:
        self.get_opacity = torch.full((10, 1), 0.5)
        self.get_scaling = torch.full((10, 3), 0.1)


class _FakeMultiviewModel:
    """Minimal stand-in exposing what the callback uses from an ERayZer-like model."""

    def __init__(self) -> None:
        self.device = torch.device('cpu')
        self.config = {'model': {'image_tokenizer': {'image_size': _SIZE}}}
        self.log = MagicMock()
        self.n_calls = 0

    def get_model_outputs(self, batch: dict) -> dict:
        self.n_calls += 1
        b = batch['image'].shape[0]
        # random draw: the callback must seed so this is identical at every save
        rand = torch.rand(b, _N_VIEWS, 3, _SIZE, _SIZE)
        eye = torch.eye(4).expand(b, _N_VIEWS, 4, 4).clone()
        eye[..., 0, 3] = torch.arange(_N_VIEWS).float()
        return {
            'input_image': rand,
            'target_image': rand,
            'render': rand,
            'render_input': rand,
            'c2w_input': eye,
            'c2w_target': eye,
            'fxfycxcy_target': torch.full((b, _N_VIEWS, 4), _SIZE / 2),
            'gaussians': [_FakeGaussians() for _ in range(b)],
        }


def _make_trainer(n_val=10, epoch=0):
    trainer = MagicMock()
    trainer.sanity_checking = False
    trainer.is_global_zero = True
    trainer.current_epoch = epoch
    trainer.check_val_every_n_epoch = 1
    trainer.datamodule.val_dataset = [
        {'image': torch.rand(_N_VIEWS, 3, _SIZE, _SIZE), 'idx': i} for i in range(n_val)
    ]
    return trainer


@pytest.fixture
def patched_glb(monkeypatch):
    """Replace the GLB export (needs real Gaussians + trimesh) with a recorder."""
    calls = []

    def fake_export(model, path, opacity_threshold=0.0):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'glb')
        calls.append((path, opacity_threshold))
        return 10

    monkeypatch.setattr(val_viz_multiview, 'export_gaussian_glb', fake_export)
    return calls


class TestMultiviewValidationVizCallback:
    """Test the class MultiviewValidationVizCallback."""

    def test_setup_accepts_model_without_get_viz_panels(self, tmp_path) -> None:
        cb = MultiviewValidationVizCallback(tmp_path, every_n_epochs=1)
        cb.setup(_make_trainer(), SimpleNamespace(), 'fit')

    def test_writes_all_outputs(self, tmp_path, patched_glb) -> None:
        cb = MultiviewValidationVizCallback(
            tmp_path, every_n_epochs=1, num_frames=6, scenes_per_sheet=4, batch_size=3,
        )
        model = _FakeMultiviewModel()
        cb.on_validation_epoch_end(_make_trainer(epoch=0), model)
        out_dir = tmp_path / 'val_images'
        names = sorted(p.name for p in out_dir.glob('*.png'))
        # 6 scenes at 4 per sheet -> 2 render sheets, plus 1 camera plot
        assert names == [
            'epoch_0000_cameras.png', 'epoch_0000_renders_00.png', 'epoch_0000_renders_01.png',
        ]
        assert (out_dir / 'pointclouds' / 'epoch_0000.glb').exists()
        assert patched_glb[0][1] == cb.opacity_threshold
        # camera plot and point cloud written once, for the first batch only
        assert len(patched_glb) == 1

    def test_logs_scalar_stats(self, tmp_path, patched_glb) -> None:
        cb = MultiviewValidationVizCallback(tmp_path, every_n_epochs=1, num_frames=4)
        model = _FakeMultiviewModel()
        cb.on_validation_epoch_end(_make_trainer(), model)
        logged = {c.args[0]: c.args[1] for c in model.log.call_args_list}
        assert set(logged) == {
            'val/focal_fx_mean', 'val/focal_fx_std', 'val/focal_fy_mean', 'val/cx_mean',
            'val/cy_mean', 'val/opacity_mean', 'val/scale_mean',
        }
        assert logged['val/focal_fx_mean'] == pytest.approx(0.5)
        assert logged['val/opacity_mean'] == pytest.approx(0.5)
        assert all(c.kwargs['rank_zero_only'] for c in model.log.call_args_list)

    def test_downscales_wide_sheets(self, tmp_path, patched_glb) -> None:
        cb = MultiviewValidationVizCallback(
            tmp_path, every_n_epochs=1, num_frames=2, max_sheet_width=40,
        )
        cb.on_validation_epoch_end(_make_trainer(), _FakeMultiviewModel())
        img = Image.open(tmp_path / 'val_images' / 'epoch_0000_renders_00.png')
        assert img.size[0] == 40

    def test_same_scenes_and_views_every_save(self, tmp_path, patched_glb) -> None:
        cb = MultiviewValidationVizCallback(tmp_path, every_n_epochs=1, num_frames=4)
        model = _FakeMultiviewModel()
        torch.manual_seed(123)
        cb.on_validation_epoch_end(_make_trainer(epoch=0), model)
        torch.manual_seed(456)  # different global RNG state must not matter
        cb.on_validation_epoch_end(_make_trainer(epoch=1), model)
        a = np.array(Image.open(tmp_path / 'val_images' / 'epoch_0000_renders_00.png'))
        b = np.array(Image.open(tmp_path / 'val_images' / 'epoch_0001_renders_00.png'))
        assert np.array_equal(a, b)

    def test_does_not_disturb_global_rng(self, tmp_path, patched_glb) -> None:
        cb = MultiviewValidationVizCallback(tmp_path, every_n_epochs=1, num_frames=2)
        trainer, model = _make_trainer(), _FakeMultiviewModel()
        torch.manual_seed(7)
        expected = torch.rand(3)
        torch.manual_seed(7)
        cb.on_validation_epoch_end(trainer, model)
        assert torch.equal(torch.rand(3), expected)

    def test_skips_when_not_due(self, tmp_path, patched_glb) -> None:
        cb = MultiviewValidationVizCallback(tmp_path, every_n_epochs=5, num_frames=2)
        model = _FakeMultiviewModel()
        cb.on_validation_epoch_end(_make_trainer(epoch=1), model)
        assert model.n_calls == 0
        assert not (tmp_path / 'val_images').exists()

    def test_point_cloud_failure_does_not_block_other_outputs(self, tmp_path, monkeypatch) -> None:
        def broken_export(*args, **kwargs):
            raise ImportError("No module named 'trimesh'")

        monkeypatch.setattr(val_viz_multiview, 'export_gaussian_glb', broken_export)
        cb = MultiviewValidationVizCallback(tmp_path, every_n_epochs=1, num_frames=2)
        model = _FakeMultiviewModel()
        cb.on_validation_epoch_end(_make_trainer(), model)
        out_dir = tmp_path / 'val_images'
        assert (out_dir / 'epoch_0000_renders_00.png').exists()
        assert (out_dir / 'epoch_0000_cameras.png').exists()
        assert model.log.called

    def test_failure_does_not_raise(self, tmp_path) -> None:
        cb = MultiviewValidationVizCallback(tmp_path, every_n_epochs=1, num_frames=2)
        model = _FakeMultiviewModel()
        model.get_model_outputs = MagicMock(side_effect=RuntimeError('boom'))
        cb.on_validation_epoch_end(_make_trainer(), model)
