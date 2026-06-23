from pathlib import Path

import pytest
import torch

from beast.data.datasets import _IMAGENET_MEAN, _IMAGENET_STD, BaseDataset, MultiViewDataset


class TestBaseDataset:

    def test_rgb(self, base_dataset):

        # check stored object properties
        assert base_dataset.data_dir.is_dir()
        assert len(base_dataset.image_list) > 0

        # check batch properties
        idx = 3
        example = base_dataset[idx]
        assert example['image'].shape == (3, 224, 224)
        assert isinstance(example['image'], torch.Tensor)
        assert example['video'] is not None
        assert example['idx'] == idx
        assert isinstance(example['image_path'], str)
        assert Path(example['image_path']).is_file()

    def test_grayscale(self, data_dir):
        dataset = BaseDataset(data_dir=data_dir, imgaug_pipeline=None, num_channels=1)
        example = dataset[0]
        assert isinstance(example, dict)
        image = example['image']
        assert image.shape == (3, 224, 224)
        # undo per-channel ImageNet normalization before comparing channels
        mean = torch.tensor(_IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(_IMAGENET_STD).view(3, 1, 1)
        denorm = image * std + mean
        assert torch.allclose(denorm[0], denorm[1], atol=1e-5)
        assert torch.allclose(denorm[1], denorm[2], atol=1e-5)

    def test_invalid_num_channels(self, data_dir) -> None:
        with pytest.raises(ValueError, match='num_channels must be 1 or 3'):
            BaseDataset(data_dir=data_dir, imgaug_pipeline=None, num_channels=2)

    def test_data_dir_not_a_directory_raises(self, tmp_path) -> None:
        # Arrange — pass a file path where a directory is expected
        f = tmp_path / 'not_a_dir.txt'
        f.touch()
        # Act / Assert
        with pytest.raises(ValueError, match='is not a directory'):
            BaseDataset(data_dir=f, imgaug_pipeline=None)

    def test_list_indexing_returns_list(self, base_dataset) -> None:
        # Arrange
        indices = [0, 1, 2]
        # Act
        result = base_dataset[indices]
        # Assert
        assert isinstance(result, list)
        assert len(result) == len(indices)
        for item in result:
            assert 'image' in item
            assert item['image'].shape == (3, 224, 224)


# ---------------------------------------------------------------------------
# MultiViewDataset tests
# ---------------------------------------------------------------------------

_IMAGE_SIZE = 64
_N_VIEWS = 3   # cameras in fixture: camera1, camera2, camera3
_N_FRAMES = 20  # 2 sessions × 10 frames each


class TestMultiViewDataset:
    """Test the MultiViewDataset class."""

    def test_length(self, multiview_dataset) -> None:
        assert len(multiview_dataset) == _N_FRAMES

    def test_item_shapes(self, multiview_dataset) -> None:
        item = multiview_dataset[0]
        assert item['image'].shape == (_N_VIEWS, 3, _IMAGE_SIZE, _IMAGE_SIZE)
        assert item['c2w'].shape == (_N_VIEWS, 4, 4)
        assert item['fxfycxcy'].shape == (_N_VIEWS, 4)
        assert len(item['view_names']) == _N_VIEWS
        assert isinstance(item['video_id'], str)
        assert isinstance(item['frame_id'], str)

    def test_image_range(self, multiview_dataset) -> None:
        # images should be float32 in [0, 1]
        item = multiview_dataset[0]
        assert item['image'].dtype == torch.float32
        assert item['image'].min() >= 0.0
        assert item['image'].max() <= 1.0

    def test_c2w_is_valid_se3(self, multiview_dataset) -> None:
        # last row must be [0, 0, 0, 1] and rotation block must be orthogonal
        c2w = multiview_dataset[0]['c2w']
        expected_bottom = torch.tensor([0.0, 0.0, 0.0, 1.0])
        bottom_rows = expected_bottom.unsqueeze(0).expand(_N_VIEWS, -1)
        assert torch.allclose(c2w[:, 3, :], bottom_rows, atol=1e-5)
        R = c2w[:, :3, :3]
        RRt = torch.bmm(R, R.transpose(1, 2))
        assert torch.allclose(RRt, torch.eye(3).unsqueeze(0).expand(_N_VIEWS, -1, -1), atol=1e-5)

    def test_camera_normalization(self, multiview_dataset) -> None:
        # fixture is mode='test' (stable order) and normalize_cameras=True (default);
        # camera-0 is sorted-first (camera1), which is re-centered to the origin
        c2w = multiview_dataset[0]['c2w']
        assert torch.allclose(c2w[0, :3, 3], torch.zeros(3), atol=1e-5)

    def test_no_camera_normalization(self, multiview_data_dir) -> None:
        # with normalize_cameras=False translations should be large (original anipose scale)
        ds = MultiViewDataset(
            data_dir=multiview_data_dir, image_size=_IMAGE_SIZE, normalize_cameras=False,
        )
        item = ds[0]
        assert 'c2w' in item
        assert item['c2w'][:, :3, 3].abs().max() > 1.0

    def test_mask_loading(self, multiview_data_dir) -> None:
        ds = MultiViewDataset(
            data_dir=multiview_data_dir, image_size=_IMAGE_SIZE, use_mask=True,
        )
        item = ds[0]
        assert 'input_mask' in item
        assert item['input_mask'].shape == (_N_VIEWS, 1, _IMAGE_SIZE, _IMAGE_SIZE)
        assert item['input_mask'].dtype == torch.float32
        assert set(item['input_mask'].unique().tolist()).issubset({0.0, 1.0})

    def test_no_mask_by_default(self, multiview_dataset) -> None:
        assert 'input_mask' not in multiview_dataset[0]

    def test_train_mode_shuffles_views(self, multiview_data_dir) -> None:
        # over many draws, view order should not always be the same
        ds = MultiViewDataset(
            data_dir=multiview_data_dir, image_size=_IMAGE_SIZE, mode='train',
        )
        orders = [tuple(ds[0]['view_names']) for _ in range(20)]
        assert len(set(orders)) > 1

    def test_test_mode_stable_view_order(self, multiview_dataset) -> None:
        orders = [tuple(multiview_dataset[0]['view_names']) for _ in range(5)]
        assert len(set(orders)) == 1

    def test_frame_ids_subset(self, multiview_dataset, multiview_data_dir) -> None:
        # passing explicit frame_ids restricts the dataset length
        subset_ids = multiview_dataset.unique_frame_ids[:5]
        ds_sub = MultiViewDataset(
            data_dir=multiview_data_dir, image_size=_IMAGE_SIZE, frame_ids=subset_ids,
        )
        assert len(ds_sub) == 5

    def test_missing_npy_raises(self, multiview_data_dir, tmp_path) -> None:
        import shutil
        shutil.copytree(multiview_data_dir, tmp_path / 'mv')
        npy = next((tmp_path / 'mv').rglob('*.npy'))
        npy.unlink()
        ds = MultiViewDataset(data_dir=tmp_path / 'mv', image_size=_IMAGE_SIZE)
        with pytest.raises(FileNotFoundError, match='camera file missing'):
            for i in range(len(ds)):
                ds[i]

    def test_invalid_data_dir_raises(self, tmp_path) -> None:
        with pytest.raises(ValueError, match='is not a directory'):
            MultiViewDataset(data_dir=tmp_path / 'nonexistent', image_size=_IMAGE_SIZE)

    def test_missing_info_json_raises(self, tmp_path) -> None:
        (tmp_path / 'empty').mkdir()
        with pytest.raises(ValueError, match='info.json not found'):
            MultiViewDataset(data_dir=tmp_path / 'empty', image_size=_IMAGE_SIZE)
