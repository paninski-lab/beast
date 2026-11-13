"""Tests for the msps_vae training entry point."""

from unittest.mock import patch

from beast.models.msps_vae.msps_vae_train import build_datamodule


class TestBuildDatamodule:
    """Test the function build_datamodule for msps_vae."""

    def test_build_datamodule_delegates_to_singleview_builder(self) -> None:
        config = {'sentinel': True}
        with patch(
            'beast.models.msps_vae.msps_vae_train.build_singleview_datamodule',
        ) as mock_builder:
            result = build_datamodule(config)
        mock_builder.assert_called_once_with(config, sampler_kind='triplet')
        assert result is mock_builder.return_value
