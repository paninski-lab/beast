"""Tests for the CLI main entry point."""

import sys
from argparse import ArgumentParser
from pathlib import Path
from unittest.mock import patch

import pytest

from beast.cli.main import build_parser, main

VERSION_MODULE = 'beast.cli.main.importlib.metadata'


class TestBuildParser:
    """Test the build_parser function."""

    def test_returns_argument_parser(self) -> None:
        # Act
        parser = build_parser()
        # Assert
        assert isinstance(parser, ArgumentParser)

    def test_missing_command_exits(self) -> None:
        # Arrange
        parser = build_parser()
        # Act / Assert
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_invalid_command_exits(self) -> None:
        # Arrange
        parser = build_parser()
        # Act / Assert
        with pytest.raises(SystemExit):
            parser.parse_args(['notacommand'])

    def test_version_flag_exits_with_code_0(self) -> None:
        # Arrange
        parser = build_parser()
        # Act / Assert
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(['--version'])
        assert exc_info.value.code == 0

    def test_version_flag_prints_version(self, capsys) -> None:
        # Arrange
        with patch(f'{VERSION_MODULE}.version', return_value='9.9.9'):
            parser = build_parser()
        # Act
        with pytest.raises(SystemExit):
            parser.parse_args(['--version'])
        # Assert
        assert '9.9.9' in capsys.readouterr().out

    def test_version_unknown_when_package_not_found(self, capsys) -> None:
        # Arrange
        import importlib.metadata
        with patch(
            f'{VERSION_MODULE}.version',
            side_effect=importlib.metadata.PackageNotFoundError,
        ):
            parser = build_parser()
        # Act
        with pytest.raises(SystemExit):
            parser.parse_args(['--version'])
        # Assert
        assert 'unknown' in capsys.readouterr().out

    def test_registers_extract(self, tmp_path: Path) -> None:
        # Arrange
        parser = build_parser()
        # Act
        args = parser.parse_args(['extract', '--input', '/p', '--output', str(tmp_path)])
        # Assert
        assert args.command == 'extract'

    def test_registers_train(self, tmp_path: Path) -> None:
        # Arrange
        parser = build_parser()
        config = tmp_path / 'config.yaml'
        config.touch()
        # Act
        args = parser.parse_args(['train', '--config', str(config)])
        # Assert
        assert args.command == 'train'

    def test_registers_predict(self) -> None:
        # Arrange
        parser = build_parser()
        # Act
        args = parser.parse_args(['predict', '--model', '/m', '--input', '/i'])
        # Assert
        assert args.command == 'predict'


class TestMain:
    """Test the main function."""

    def test_no_args_exits_with_code_1(self, monkeypatch) -> None:
        # Arrange
        monkeypatch.setattr(sys, 'argv', ['beast'])
        # Act / Assert
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    def test_invalid_command_exits(self, monkeypatch) -> None:
        # Arrange
        monkeypatch.setattr(sys, 'argv', ['beast', 'notacommand'])
        # Act / Assert
        with pytest.raises(SystemExit):
            main()

    def test_dispatches_to_extract_handler(self, tmp_path: Path, monkeypatch) -> None:
        # Arrange
        monkeypatch.setattr(
            sys, 'argv',
            ['beast', 'extract', '--input', '/p', '--output', str(tmp_path)],
        )
        with patch('beast.cli.commands.extract.handle') as mock_handle:
            # Act
            main()
        # Assert
        mock_handle.assert_called_once()
