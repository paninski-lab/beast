"""Tests for the beast3d training entry point."""

from unittest.mock import patch

from beast.models.beast3d.beast3d_train import build_datamodule


class TestBuildDatamodule:
    """Test the function build_datamodule for beast3d."""

    def test_build_datamodule_delegates_to_multiview_builder(self) -> None:
        config = {'sentinel': True}
        with patch(
            'beast.models.beast3d.beast3d_train.build_multiview_datamodule',
        ) as mock_builder:
            result = build_datamodule(config)
        mock_builder.assert_called_once_with(config, use_mask=True)
        assert result is mock_builder.return_value
