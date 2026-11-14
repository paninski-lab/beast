"""Tests for the erayzer training entry point."""

from unittest.mock import patch

from beast.models.erayzer.erayzer_train import build_datamodule


class TestBuildDatamodule:
    """Test the function build_datamodule for erayzer."""

    def test_build_datamodule_delegates_to_multiview_builder(self) -> None:
        config = {'sentinel': True}
        with patch(
            'beast.models.erayzer.erayzer_train.build_multiview_datamodule',
        ) as mock_builder:
            result = build_datamodule(config)
        mock_builder.assert_called_once_with(config)
        assert result is mock_builder.return_value
