"""Tests for the predict CLI command."""

import argparse
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from beast.cli.main import build_parser


class TestRegisterParser:
    """Test the predict command parser registration."""

    def test_defaults(self) -> None:
        # Arrange
        parser = build_parser()
        # Act
        args = parser.parse_args(['predict', '--model', '/m', '--input', '/i'])
        # Assert
        assert args.batch_size == 32
        assert args.save_latents is False
        assert args.save_reconstructions is False
        assert args.output is None

    def test_missing_model_exits(self) -> None:
        # Arrange
        parser = build_parser()
        # Act / Assert
        with pytest.raises(SystemExit):
            parser.parse_args(['predict', '--input', '/i'])

    def test_missing_input_exits(self) -> None:
        # Arrange
        parser = build_parser()
        # Act / Assert
        with pytest.raises(SystemExit):
            parser.parse_args(['predict', '--model', '/m'])

    def test_save_flags_set_true(self) -> None:
        # Arrange
        parser = build_parser()
        # Act
        args = parser.parse_args([
            'predict', '--model', '/m', '--input', '/i',
            '--save_latents', '--save_reconstructions',
        ])
        # Assert
        assert args.save_latents is True
        assert args.save_reconstructions is True

    def test_custom_batch_size(self) -> None:
        # Arrange
        parser = build_parser()
        # Act
        args = parser.parse_args([
            'predict', '--model', '/m', '--input', '/i', '--batch-size', '64',
        ])
        # Assert
        assert args.batch_size == 64


class TestHandle:
    """Test the predict command handle function."""

    def test_single_video_calls_predict_video(self, tmp_path: Path) -> None:
        # Arrange
        from beast.cli.commands.predict import handle
        video_file = tmp_path / 'test.mp4'
        video_file.touch()
        args = argparse.Namespace(
            model=Path('/model'),
            input=video_file,
            output=None,
            batch_size=32,
            save_latents=True,
            save_reconstructions=False,
        )
        mock_model = MagicMock()
        # Act
        with patch('beast.api.model.Model') as MockModel:
            MockModel.from_dir.return_value = mock_model
            handle(args)
        # Assert
        mock_model.predict_video.assert_called_once_with(
            video_file=video_file,
            output_dir=None,
            batch_size=32,
            save_latents=True,
            save_reconstructions=False,
        )

    def test_directory_of_videos_calls_predict_video_per_file(self, tmp_path: Path) -> None:
        # Arrange
        from beast.cli.commands.predict import handle
        video_dir = tmp_path / 'videos'
        video_dir.mkdir()
        (video_dir / 'vid1.mp4').touch()
        (video_dir / 'vid2.mp4').touch()
        args = argparse.Namespace(
            model=Path('/model'),
            input=video_dir,
            output=None,
            batch_size=32,
            save_latents=True,
            save_reconstructions=False,
        )
        mock_model = MagicMock()
        # Act
        with patch('beast.api.model.Model') as MockModel:
            MockModel.from_dir.return_value = mock_model
            handle(args)
        # Assert
        assert mock_model.predict_video.call_count == 2

    def test_directory_of_images_calls_predict_images(self, tmp_path: Path) -> None:
        # Arrange
        from beast.cli.commands.predict import handle
        img_dir = tmp_path / 'images'
        img_dir.mkdir()
        (img_dir / 'img1.png').touch()
        (img_dir / 'img2.jpg').touch()
        args = argparse.Namespace(
            model=Path('/model'),
            input=img_dir,
            output=None,
            batch_size=32,
            save_latents=True,
            save_reconstructions=False,
        )
        mock_model = MagicMock()
        # Act
        with patch('beast.api.model.Model') as MockModel:
            MockModel.from_dir.return_value = mock_model
            handle(args)
        # Assert
        mock_model.predict_images.assert_called_once_with(
            image_dir=img_dir,
            output_dir=None,
            batch_size=32,
            save_latents=True,
            save_reconstructions=False,
        )

    def test_mixed_input_neither_predict_method_called(self, tmp_path: Path) -> None:
        # Arrange
        from beast.cli.commands.predict import handle
        mixed_dir = tmp_path / 'mixed'
        mixed_dir.mkdir()
        (mixed_dir / 'vid.mp4').touch()
        (mixed_dir / 'img.png').touch()
        args = argparse.Namespace(
            model=Path('/model'),
            input=mixed_dir,
            output=None,
            batch_size=32,
            save_latents=True,
            save_reconstructions=False,
        )
        mock_model = MagicMock()
        # Act
        with patch('beast.api.model.Model') as MockModel:
            MockModel.from_dir.return_value = mock_model
            handle(args)
        # Assert
        mock_model.predict_video.assert_not_called()
        mock_model.predict_images.assert_not_called()

    def test_no_save_flags_logs_warning(self, tmp_path: Path, caplog) -> None:
        # Arrange
        from beast.cli.commands.predict import handle
        video_file = tmp_path / 'test.mp4'
        video_file.touch()
        args = argparse.Namespace(
            model=Path('/model'),
            input=video_file,
            output=None,
            batch_size=32,
            save_latents=False,
            save_reconstructions=False,
        )
        mock_model = MagicMock()
        # Act
        with patch('beast.api.model.Model') as MockModel, \
                caplog.at_level(logging.WARNING):
            MockModel.from_dir.return_value = mock_model
            handle(args)
        # Assert
        assert any('no outputs will be saved' in r.message for r in caplog.records)
