"""Tests for the shared datamodule builders."""

import copy

from beast.data.builders import build_multiview_datamodule, build_singleview_datamodule
from beast.data.datamodules import BaseDataModule, MultiViewDataModule


class TestBuildSingleviewDatamodule:
    """Test the function build_singleview_datamodule."""

    def test_build_singleview_datamodule_default_has_no_sampler(self, config_vit) -> None:
        config = copy.deepcopy(config_vit)
        config['model']['model_params']['use_infoNCE'] = False
        datamodule = build_singleview_datamodule(config)
        assert isinstance(datamodule, BaseDataModule)
        assert datamodule.sampler_kind == 'none'
        assert datamodule.use_distributed_sampler is True

    def test_build_singleview_datamodule_infonce_selects_contrastive(self, config_vit) -> None:
        config = copy.deepcopy(config_vit)
        config['model']['model_params']['use_infoNCE'] = True
        datamodule = build_singleview_datamodule(config)
        assert datamodule.sampler_kind == 'contrastive'
        assert datamodule.use_distributed_sampler is False

    def test_build_singleview_datamodule_triplet_reads_positive_window(
        self, config_msps_vae,
    ) -> None:
        config = copy.deepcopy(config_msps_vae)
        datamodule = build_singleview_datamodule(config, sampler_kind='triplet')
        assert datamodule.sampler_kind == 'triplet'
        assert datamodule.positive_window == config['model']['model_params']['positive_window']
        assert datamodule.use_distributed_sampler is False

    def test_build_singleview_datamodule_explicit_sampler_overrides_infonce(
        self, config_vit,
    ) -> None:
        config = copy.deepcopy(config_vit)
        config['model']['model_params']['use_infoNCE'] = True
        datamodule = build_singleview_datamodule(config, sampler_kind='none')
        assert datamodule.sampler_kind == 'none'

    def test_build_singleview_datamodule_uses_config_batch_sizes(self, config_vit) -> None:
        config = copy.deepcopy(config_vit)
        datamodule = build_singleview_datamodule(config)
        assert datamodule.train_batch_size == config['training']['train_batch_size']
        assert datamodule.val_batch_size == config['training']['val_batch_size']


class TestBuildMultiviewDatamodule:
    """Test the function build_multiview_datamodule."""

    def test_build_multiview_datamodule_is_set_up(
        self, config_erayzer, multiview_data_dir,
    ) -> None:
        config = copy.deepcopy(config_erayzer)
        config['data']['data_dir'] = str(multiview_data_dir)
        datamodule = build_multiview_datamodule(config)
        assert isinstance(datamodule, MultiViewDataModule)
        assert datamodule.train_dataset is not None
        assert datamodule.val_dataset is not None
        assert datamodule.use_mask is False
        assert datamodule.use_distributed_sampler is True

    def test_build_multiview_datamodule_use_mask(
        self, config_beast3d, multiview_data_dir,
    ) -> None:
        config = copy.deepcopy(config_beast3d)
        config['data']['data_dir'] = str(multiview_data_dir)
        datamodule = build_multiview_datamodule(config, use_mask=True)
        assert datamodule.use_mask is True
