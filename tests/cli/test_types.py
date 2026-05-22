"""Tests for CLI type validator functions."""

from pathlib import Path

import pytest

from beast.cli.types import config_file, output_dir, valid_dir, valid_file


class TestValidFile:
    """Test the valid_file function."""

    def test_valid_file_exists(self, tmp_path: Path) -> None:
        # Arrange
        f = tmp_path / 'test.yaml'
        f.touch()
        # Act
        result = valid_file(f)
        # Assert
        assert result.is_file()

    def test_valid_file_accepts_str(self, tmp_path: Path) -> None:
        # Arrange
        f = tmp_path / 'test.yaml'
        f.touch()
        # Act
        result = valid_file(str(f))
        # Assert
        assert result.is_file()

    def test_valid_file_not_exists(self, tmp_path: Path) -> None:
        # Arrange
        f = tmp_path / 'nonexistent.yaml'
        # Act / Assert
        with pytest.raises(OSError):
            valid_file(f)

    def test_valid_file_is_directory(self, tmp_path: Path) -> None:
        # Arrange
        d = tmp_path / 'some_dir'
        d.mkdir()
        # Act / Assert
        with pytest.raises(OSError):
            valid_file(d)


class TestValidDir:
    """Test the valid_dir function."""

    def test_valid_dir_exists(self, tmp_path: Path) -> None:
        # Arrange
        d = tmp_path / 'some_dir'
        d.mkdir()
        # Act
        result = valid_dir(d)
        # Assert
        assert result.is_dir()

    def test_valid_dir_accepts_str(self, tmp_path: Path) -> None:
        # Arrange
        d = tmp_path / 'some_dir'
        d.mkdir()
        # Act
        result = valid_dir(str(d))
        # Assert
        assert result.is_dir()

    def test_valid_dir_not_exists(self, tmp_path: Path) -> None:
        # Arrange
        d = tmp_path / 'nonexistent_dir'
        # Act / Assert
        with pytest.raises(OSError):
            valid_dir(d)

    def test_valid_dir_is_file(self, tmp_path: Path) -> None:
        # Arrange
        f = tmp_path / 'test.txt'
        f.touch()
        # Act / Assert
        with pytest.raises(OSError):
            valid_dir(f)


class TestConfigFile:
    """Test the config_file function."""

    def test_config_file_yaml_extension(self, tmp_path: Path) -> None:
        # Arrange
        f = tmp_path / 'config.yaml'
        f.touch()
        # Act
        result = config_file(f)
        # Assert
        assert result.is_file()

    def test_config_file_yml_extension(self, tmp_path: Path) -> None:
        # Arrange
        f = tmp_path / 'config.yml'
        f.touch()
        # Act
        result = config_file(f)
        # Assert
        assert result.is_file()

    def test_config_file_invalid_extension(self, tmp_path: Path) -> None:
        # Arrange
        f = tmp_path / 'config.json'
        f.touch()
        # Act / Assert
        with pytest.raises(ValueError):
            config_file(f)

    def test_config_file_not_exists(self, tmp_path: Path) -> None:
        # Arrange
        f = tmp_path / 'nonexistent.yaml'
        # Act / Assert
        with pytest.raises(OSError):
            config_file(f)


class TestOutputDir:
    """Test the output_dir function."""

    def test_output_dir_creates_new_directory(self, tmp_path: Path) -> None:
        # Arrange
        d = tmp_path / 'new_dir'
        # Act
        result = output_dir(d)
        # Assert
        assert result.is_dir()

    def test_output_dir_accepts_str(self, tmp_path: Path) -> None:
        # Arrange
        d = tmp_path / 'str_dir'
        # Act
        result = output_dir(str(d))
        # Assert
        assert result.is_dir()

    def test_output_dir_creates_nested_directories(self, tmp_path: Path) -> None:
        # Arrange
        d = tmp_path / 'a' / 'b' / 'c'
        # Act
        result = output_dir(d)
        # Assert
        assert result.is_dir()

    def test_output_dir_existing_dir_ok(self, tmp_path: Path) -> None:
        # Act
        result = output_dir(tmp_path)
        # Assert
        assert result.is_dir()
