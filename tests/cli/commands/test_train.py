"""Tests for the train CLI command."""

import argparse
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from beast.cli.main import build_parser


class TestRegisterParser:
    """Test the train command parser registration."""

    def test_defaults(self, tmp_path: Path) -> None:
        # Arrange
        parser = build_parser()
        config = tmp_path / 'config.yaml'
        config.touch()
        # Act
        args = parser.parse_args(['train', '--config', str(config)])
        # Assert
        assert args.output is None
        assert args.data is None
        assert args.gpus is None
        assert args.nodes is None
        assert args.overrides is None

    def test_missing_config_exits(self) -> None:
        # Arrange
        parser = build_parser()
        # Act / Assert
        with pytest.raises(SystemExit):
            parser.parse_args(['train'])

    def test_invalid_config_extension_exits(self, tmp_path: Path) -> None:
        # Arrange
        parser = build_parser()
        f = tmp_path / 'config.json'
        f.touch()
        # Act / Assert
        with pytest.raises(SystemExit):
            parser.parse_args(['train', '--config', str(f)])

    def test_overrides_parsed(self, tmp_path: Path) -> None:
        # Arrange
        parser = build_parser()
        config = tmp_path / 'config.yaml'
        config.touch()
        # Act
        args = parser.parse_args([
            'train', '--config', str(config), '--overrides', 'key=val',
        ])
        # Assert
        assert args.overrides == ['key=val']

    def test_gpu_and_node_overrides_parsed(self, tmp_path: Path) -> None:
        # Arrange
        parser = build_parser()
        config = tmp_path / 'config.yaml'
        config.touch()
        # Act
        args = parser.parse_args([
            'train', '--config', str(config), '--gpus', '2', '--nodes', '3',
        ])
        # Assert
        assert args.gpus == 2
        assert args.nodes == 3


class TestHandle:
    """Test the train command handle function."""

    def _make_args(
        self, tmp_path: Path, config: Path, output: Path | None = None,
    ) -> argparse.Namespace:
        return argparse.Namespace(
            config=config,
            output=output or (tmp_path / 'output'),
            data=None,
            gpus=None,
            nodes=None,
            overrides=None,
        )

    def test_calls_model_train(self, tmp_path: Path) -> None:
        # Arrange
        from beast.cli.commands.train import handle
        config = tmp_path / 'config.yaml'
        config.touch()
        output = tmp_path / 'output'
        args = self._make_args(tmp_path, config, output)
        mock_config = {'data': {}, 'training': {}}
        mock_model = MagicMock()
        # Act
        with (
            patch('beast.io.load_config', return_value=mock_config),
            patch('beast.api.model.Model') as MockModel,
            patch('beast.cli.commands.train._setup_model_logging'),
        ):
            MockModel.from_config.return_value = mock_model
            handle(args)
        # Assert
        mock_model.train.assert_called_once_with(output_dir=output)

    def test_applies_config_overrides(self, tmp_path: Path) -> None:
        # Arrange
        from beast.cli.commands.train import handle
        config = tmp_path / 'config.yaml'
        config.touch()
        args = self._make_args(tmp_path, config)
        args.overrides = ['training.num_epochs=10']
        mock_config = {'data': {}, 'training': {}}
        mock_model = MagicMock()
        # Act
        with (
            patch('beast.io.load_config', return_value=mock_config),
            patch('beast.io.apply_config_overrides', return_value=mock_config) as mock_overrides,
            patch('beast.api.model.Model') as MockModel,
            patch('beast.cli.commands.train._setup_model_logging'),
        ):
            MockModel.from_config.return_value = mock_model
            handle(args)
        # Assert
        mock_overrides.assert_called_once_with(mock_config, ['training.num_epochs=10'])

    def test_data_override_applied_to_config(self, tmp_path: Path) -> None:
        # Arrange
        from beast.cli.commands.train import handle
        config = tmp_path / 'config.yaml'
        config.touch()
        args = self._make_args(tmp_path, config)
        args.data = tmp_path / 'data'
        mock_config = {'data': {'data_dir': '/old'}, 'training': {}}
        mock_model = MagicMock()
        # Act
        with (
            patch('beast.io.load_config', return_value=mock_config),
            patch('beast.api.model.Model') as MockModel,
            patch('beast.cli.commands.train._setup_model_logging'),
        ):
            MockModel.from_config.return_value = mock_model
            handle(args)
        # Assert
        assert mock_config['data']['data_dir'] == str(tmp_path / 'data')

    def test_gpus_override_applied_to_config(self, tmp_path: Path) -> None:
        # Arrange
        from beast.cli.commands.train import handle
        config = tmp_path / 'config.yaml'
        config.touch()
        args = self._make_args(tmp_path, config)
        args.gpus = 2
        mock_config = {'data': {}, 'training': {'num_gpus': 1}}
        mock_model = MagicMock()
        # Act
        with (
            patch('beast.io.load_config', return_value=mock_config),
            patch('beast.api.model.Model') as MockModel,
            patch('beast.cli.commands.train._setup_model_logging'),
        ):
            MockModel.from_config.return_value = mock_model
            handle(args)
        # Assert
        assert mock_config['training']['num_gpus'] == 2

    def test_nodes_override_applied_to_config(self, tmp_path: Path) -> None:
        # Arrange
        from beast.cli.commands.train import handle
        config = tmp_path / 'config.yaml'
        config.touch()
        args = self._make_args(tmp_path, config)
        args.nodes = 3
        mock_config = {'data': {}, 'training': {'num_nodes': 1}}
        mock_model = MagicMock()
        # Act
        with (
            patch('beast.io.load_config', return_value=mock_config),
            patch('beast.api.model.Model') as MockModel,
            patch('beast.cli.commands.train._setup_model_logging'),
        ):
            MockModel.from_config.return_value = mock_model
            handle(args)
        # Assert
        assert mock_config['training']['num_nodes'] == 3

    def test_auto_output_dir_created_under_runs(self, tmp_path: Path, monkeypatch) -> None:
        # Arrange
        from beast.cli.commands.train import handle
        monkeypatch.chdir(tmp_path)
        config = tmp_path / 'config.yaml'
        config.touch()
        args = argparse.Namespace(
            config=config,
            output=None,
            data=None,
            gpus=None,
            nodes=None,
            overrides=None,
        )
        mock_config = {'data': {}, 'training': {}}
        mock_model = MagicMock()
        # Act
        with (
            patch('beast.io.load_config', return_value=mock_config),
            patch('beast.api.model.Model') as MockModel,
            patch('beast.cli.commands.train._setup_model_logging'),
        ):
            MockModel.from_config.return_value = mock_model
            handle(args)
        # Assert
        assert args.output is not None
        assert 'runs' in str(args.output)
        assert args.output.exists()


class TestSetupModelLogging:
    """Test the _setup_model_logging function."""

    def test_creates_log_file(self, tmp_path: Path) -> None:
        # Arrange
        from beast.cli.commands.train import _setup_model_logging
        root_logger = logging.getLogger()
        count_before = len(root_logger.handlers)
        # Act
        handler = _setup_model_logging(tmp_path)
        # Assert
        assert (tmp_path / 'training.log').exists()
        assert len(root_logger.handlers) == count_before + 1
        # Clean up to avoid leaking the handler into other tests
        root_logger.removeHandler(handler)
        handler.close()
