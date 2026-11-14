"""Tests for the beast_resnet training entry point."""

from unittest.mock import patch

from beast.models.beast_resnet.beast_resnet_train import build_datamodule


class TestBuildDatamodule:
    """Test the function build_datamodule for beast_resnet."""

    def test_build_datamodule_delegates_to_singleview_builder(self) -> None:
        config = {'sentinel': True}
        with patch(
            'beast.models.beast_resnet.beast_resnet_train.build_singleview_datamodule',
        ) as mock_builder:
            result = build_datamodule(config)
        mock_builder.assert_called_once_with(config)
        assert result is mock_builder.return_value
