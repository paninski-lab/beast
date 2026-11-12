"""Tests for the validation visualization callback."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

from beast.callbacks.val_viz import (
    ValidationVizCallback,
    make_contact_sheets,
    select_viz_indices,
    sheet_grid_shape,
)
from beast.models.base import BaseLightningModel


class TestSheetGridShape:
    """Test the function sheet_grid_shape."""

    @pytest.mark.parametrize(
        'n_panels,expected',
        [(1, (9, 9)), (2, (8, 4)), (3, (9, 3)), (4, (8, 2)), (12, (12, 1))],
    )
    def test_sheet_grid_shape(self, n_panels, expected) -> None:
        assert sheet_grid_shape(n_panels) == expected


class TestSelectVizIndices:
    """Test the function select_viz_indices."""

    def test_evenly_spaced_and_unique(self) -> None:
        idxs = select_viz_indices(1000, 100)
        assert len(idxs) == 100 and len(set(idxs)) == 100
        assert idxs[0] == 0 and idxs[-1] == 999

    def test_capped_at_available(self) -> None:
        assert select_viz_indices(7, 100) == list(range(7))

    def test_deterministic(self) -> None:
        assert select_viz_indices(500, 50) == select_viz_indices(500, 50)

    def test_empty(self) -> None:
        assert select_viz_indices(0, 10) == []


class TestMakeContactSheets:
    """Test the function make_contact_sheets."""

    def test_sheet_count_and_square(self) -> None:
        panels = [torch.rand(60, 3, 32, 32) for _ in range(3)]
        sheets = make_contact_sheets(panels, panel_size=16, gap=2)
        # 27 frames per sheet -> 3 sheets for 60 frames
        assert len(sheets) == 3
        for sheet in sheets:
            assert sheet.dtype == np.uint8
            assert sheet.shape == (9 * (16 + 2), 3 * (3 * 16 + 2), 3)

    def test_padding_is_white(self) -> None:
        panels = [torch.zeros(1, 3, 8, 8) for _ in range(3)]
        sheet = make_contact_sheets(panels, panel_size=8, gap=2)[0]
        # first frame is black (minus gutter), last slot is blank white
        assert sheet[0, 0].tolist() == [0, 0, 0]
        assert sheet[-1, -1].tolist() == [255, 255, 255]

    def test_frame_placement(self) -> None:
        # frame i filled with value i/10; frames fill rows left to right
        n = 5
        panels = [
            torch.stack([torch.full((3, 8, 8), i / 10) for i in range(n)]) for _ in range(3)
        ]
        sheet = make_contact_sheets(panels, panel_size=8, gap=2)[0]
        frame_w = 3 * 8 + 2
        frame_h = 8 + 2
        for i in range(n):
            row, col = divmod(i, 3)
            val = sheet[row * frame_h + 1, col * frame_w + 1, 0]
            assert val == round(i / 10 * 255)

    def test_resizes_panels(self) -> None:
        panels = [torch.rand(2, 3, 64, 64) for _ in range(2)]
        sheet = make_contact_sheets(panels, panel_size=16, gap=0)[0]
        # 2 panels -> 4 cols x 8 rows
        assert sheet.shape == (8 * 16, 4 * 2 * 16, 3)


class _VizModel(BaseLightningModel):
    def get_model_outputs(self, batch_dict):
        return {}

    def compute_loss(self, stage, **kwargs):
        return torch.tensor(0.0), []

    def predict_step(self, batch_dict, batch_idx):
        return {}

    def get_viz_panels(self, batch_dict):
        x = batch_dict['image']
        return [x, 1 - x]


class _NoVizModel(_VizModel):
    get_viz_panels = BaseLightningModel.get_viz_panels


def _make_trainer(n_val=40, epoch=0, sanity=False, check_val=1):
    trainer = MagicMock()
    trainer.sanity_checking = sanity
    trainer.is_global_zero = True
    trainer.current_epoch = epoch
    trainer.check_val_every_n_epoch = check_val
    val_dataset = [{'image': torch.rand(3, 16, 16), 'idx': i} for i in range(n_val)]
    trainer.datamodule.val_dataset = val_dataset
    return trainer


class TestValidationVizCallback:
    """Test the class ValidationVizCallback."""

    def test_invalid_every_n_epochs(self, tmp_path) -> None:
        with pytest.raises(ValueError):
            ValidationVizCallback(tmp_path, every_n_epochs=0)

    def test_setup_rejects_unsupported_model(self, tmp_path) -> None:
        cb = ValidationVizCallback(tmp_path, every_n_epochs=1)
        with pytest.raises(NotImplementedError):
            cb.setup(_make_trainer(), _NoVizModel({'model': {'seed': 0}}), 'fit')

    def test_setup_accepts_supported_model(self, tmp_path) -> None:
        cb = ValidationVizCallback(tmp_path, every_n_epochs=1)
        cb.setup(_make_trainer(), _VizModel({'model': {'seed': 0}}), 'fit')

    def test_saves_sheets_on_due_epoch(self, tmp_path) -> None:
        cb = ValidationVizCallback(
            tmp_path, every_n_epochs=5, num_frames=40, panel_size=64, batch_size=16,
        )
        model = _VizModel({'model': {'seed': 0}})
        # (epoch + 1) % 5 == 0 -> due
        cb.on_validation_epoch_end(_make_trainer(epoch=4), model)
        files = sorted((tmp_path / 'val_images').glob('*.png'))
        # 2 panels -> 32 frames per sheet -> 2 sheets for 40 frames
        assert [f.name for f in files] == ['epoch_0004_sheet_00.png', 'epoch_0004_sheet_01.png']
        img = Image.open(files[0])
        # roughly square
        assert abs(img.size[0] - img.size[1]) / img.size[0] < 0.05

    def test_skips_when_not_due(self, tmp_path) -> None:
        cb = ValidationVizCallback(tmp_path, every_n_epochs=5, num_frames=10, panel_size=8)
        model = _VizModel({'model': {'seed': 0}})
        cb.on_validation_epoch_end(_make_trainer(epoch=2), model)
        assert not (tmp_path / 'val_images').exists()

    def test_skips_sanity_check(self, tmp_path) -> None:
        cb = ValidationVizCallback(tmp_path, every_n_epochs=1, num_frames=10, panel_size=8)
        model = _VizModel({'model': {'seed': 0}})
        cb.on_validation_epoch_end(_make_trainer(sanity=True), model)
        assert not (tmp_path / 'val_images').exists()

    def test_same_frames_every_save(self, tmp_path) -> None:
        cb = ValidationVizCallback(tmp_path, every_n_epochs=1, num_frames=10, panel_size=8)
        model = _VizModel({'model': {'seed': 0}})
        cb.on_validation_epoch_end(_make_trainer(epoch=0), model)
        cb.on_validation_epoch_end(_make_trainer(epoch=1), model)
        a = np.array(Image.open(tmp_path / 'val_images' / 'epoch_0000_sheet_00.png'))
        b = np.array(Image.open(tmp_path / 'val_images' / 'epoch_0001_sheet_00.png'))
        assert np.array_equal(a, b)

    def test_failure_does_not_raise(self, tmp_path) -> None:
        cb = ValidationVizCallback(tmp_path, every_n_epochs=1, num_frames=10, panel_size=8)
        model = _NoVizModel({'model': {'seed': 0}})
        cb.on_validation_epoch_end(_make_trainer(), model)  # logs a warning, no exception
