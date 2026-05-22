"""Tests for the extract CLI command."""

import argparse
from pathlib import Path
from unittest.mock import patch

import pytest

from beast.cli.main import build_parser


class TestRegisterParser:
    """Test the extract command parser registration."""

    def test_defaults(self, tmp_path: Path) -> None:
        # Arrange
        parser = build_parser()
        # Act
        args = parser.parse_args(['extract', '--input', '/p', '--output', str(tmp_path)])
        # Assert
        assert args.frames_per_video == 500
        assert args.method == 'pca_kmeans'
        assert args.workers == 4

    def test_custom_values(self, tmp_path: Path) -> None:
        # Arrange
        parser = build_parser()
        # Act
        args = parser.parse_args([
            'extract',
            '--input', '/p',
            '--output', str(tmp_path),
            '--frames-per-video', '100',
            '--method', 'uniform',
            '--workers', '2',
        ])
        # Assert
        assert args.frames_per_video == 100
        assert args.method == 'uniform'
        assert args.workers == 2

    def test_random_method_accepted(self, tmp_path: Path) -> None:
        # Arrange
        parser = build_parser()
        # Act
        args = parser.parse_args([
            'extract', '--input', '/p', '--output', str(tmp_path), '--method', 'random',
        ])
        # Assert
        assert args.method == 'random'

    def test_invalid_method_exits(self, tmp_path: Path) -> None:
        # Arrange
        parser = build_parser()
        # Act / Assert
        with pytest.raises(SystemExit):
            parser.parse_args([
                'extract', '--input', '/p', '--output', str(tmp_path), '--method', 'bad',
            ])

    def test_missing_input_exits(self, tmp_path: Path) -> None:
        # Arrange
        parser = build_parser()
        # Act / Assert
        with pytest.raises(SystemExit):
            parser.parse_args(['extract', '--output', str(tmp_path)])

    def test_missing_output_exits(self) -> None:
        # Arrange
        parser = build_parser()
        # Act / Assert
        with pytest.raises(SystemExit):
            parser.parse_args(['extract', '--input', '/p'])


class TestHandle:
    """Test the extract command handle function."""

    def test_calls_extract_frames_with_correct_kwargs(self, tmp_path: Path) -> None:
        # Arrange
        from beast.cli.commands.extract import handle
        args = argparse.Namespace(
            input=Path('/some/input'),
            output=tmp_path,
            frames_per_video=500,
            method='pca_kmeans',
            workers=4,
        )
        # Act
        with patch('beast.extraction.extract_frames') as mock_extract:
            mock_extract.return_value = {'total_frames': 100, 'total_videos': 2}
            handle(args)
        # Assert
        mock_extract.assert_called_once_with(
            input_path=Path('/some/input'),
            output_dir=tmp_path,
            frames_per_video=500,
            method='pca_kmeans',
            num_workers=4,
        )

    def test_calls_extract_frames_with_custom_args(self, tmp_path: Path) -> None:
        # Arrange
        from beast.cli.commands.extract import handle
        args = argparse.Namespace(
            input=Path('/videos'),
            output=tmp_path,
            frames_per_video=100,
            method='uniform',
            workers=2,
        )
        # Act
        with patch('beast.extraction.extract_frames') as mock_extract:
            mock_extract.return_value = {'total_frames': 50, 'total_videos': 1}
            handle(args)
        # Assert
        mock_extract.assert_called_once_with(
            input_path=Path('/videos'),
            output_dir=tmp_path,
            frames_per_video=100,
            method='uniform',
            num_workers=2,
        )
