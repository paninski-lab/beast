"""Tests for the high-level Model API."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from beast.api.model import Model, chdir


class TestChdir:
    """Test the chdir context manager."""

    def test_changes_and_restores_directory(self, tmp_path: Path) -> None:
        original = os.getcwd()
        with chdir(tmp_path):
            assert os.getcwd() == str(tmp_path)
        assert os.getcwd() == original

    def test_restores_directory_on_exception(self, tmp_path: Path) -> None:
        original = os.getcwd()
        with pytest.raises(RuntimeError):
            with chdir(tmp_path):
                raise RuntimeError('boom')
        assert os.getcwd() == original


class TestModelInit:
    """Test the Model.__init__ method."""

    def test_stores_model_and_config(self) -> None:
        mock_model = MagicMock()
        config = {'model': {}, 'training': {}}
        m = Model(mock_model, config, model_dir=None)
        assert m.model is mock_model
        assert m.config is config
        assert m.model_dir is None

    def test_model_dir_converted_to_path(self, tmp_path: Path) -> None:
        m = Model(MagicMock(), {}, model_dir=str(tmp_path))
        assert m.model_dir == tmp_path
        assert isinstance(m.model_dir, Path)

    def test_model_dir_none_stays_none(self) -> None:
        m = Model(MagicMock(), {}, model_dir=None)
        assert m.model_dir is None


class TestModelFromConfig:
    """Test the Model.from_config class method."""

    def _make_config(self, model_class: str) -> dict:
        return {'model': {'model_class': model_class}, 'training': {}}

    def _mock_class(self) -> MagicMock:
        mock = MagicMock(return_value=MagicMock())
        mock.__name__ = 'MockModel'
        return mock

    def test_dict_config_vit(self) -> None:
        config = self._make_config('vit')
        mock_class = self._mock_class()
        with patch.dict(Model.MODEL_REGISTRY, {'vit': mock_class}):
            m = Model.from_config(config)
        assert isinstance(m, Model)
        mock_class.assert_called_once_with(config)

    def test_dict_config_resnet(self) -> None:
        config = self._make_config('resnet')
        mock_class = self._mock_class()
        with patch.dict(Model.MODEL_REGISTRY, {'resnet': mock_class}):
            m = Model.from_config(config)
        assert isinstance(m, Model)
        mock_class.assert_called_once_with(config)

    def test_file_config(self, tmp_path: Path) -> None:
        config = self._make_config('vit')
        config_path = tmp_path / 'config.yaml'
        config_path.write_text(yaml.dump(config))
        mock_class = self._mock_class()
        with patch.dict(Model.MODEL_REGISTRY, {'vit': mock_class}):
            m = Model.from_config(config_path)
        assert isinstance(m, Model)

    def test_unknown_model_type_raises(self) -> None:
        config = self._make_config('unknown_arch')
        with pytest.raises(ValueError, match='Unknown model type'):
            Model.from_config(config)

    def test_model_dir_is_none_after_from_config(self) -> None:
        config = self._make_config('vit')
        mock_class = self._mock_class()
        with patch.dict(Model.MODEL_REGISTRY, {'vit': mock_class}):
            m = Model.from_config(config)
        assert m.model_dir is None


class TestModelFromDir:
    """Test the Model.from_dir class method."""

    def _write_config(self, model_dir: Path, model_class: str) -> None:
        config = {'model': {'model_class': model_class}, 'training': {}}
        (model_dir / 'config.yaml').write_text(yaml.dump(config))

    def _write_checkpoint(self, model_dir: Path) -> Path:
        ckpt = model_dir / 'model_best.ckpt'
        ckpt.touch()
        return ckpt

    def _mock_class(self) -> MagicMock:
        mock = MagicMock(return_value=MagicMock())
        mock.__name__ = 'MockModel'
        return mock

    def test_loads_config_and_checkpoint(self, tmp_path: Path) -> None:
        self._write_config(tmp_path, 'vit')
        self._write_checkpoint(tmp_path)
        mock_class = self._mock_class()
        mock_instance = mock_class.return_value
        mock_state = {'state_dict': {'layer': 'weights'}}
        with (
            patch.dict(Model.MODEL_REGISTRY, {'vit': mock_class}),
            patch('beast.api.model.torch.load', return_value=mock_state),
        ):
            m = Model.from_dir(tmp_path)
        assert isinstance(m, Model)
        assert m.model_dir == tmp_path
        mock_instance.load_state_dict.assert_called_once_with({'layer': 'weights'})

    def test_unknown_model_type_raises(self, tmp_path: Path) -> None:
        self._write_config(tmp_path, 'unknown_arch')
        with pytest.raises(ValueError, match='Unknown model type'):
            Model.from_dir(tmp_path)

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        self._write_config(tmp_path, 'resnet')
        self._write_checkpoint(tmp_path)
        mock_class = self._mock_class()
        mock_state = {'state_dict': {}}
        with (
            patch.dict(Model.MODEL_REGISTRY, {'resnet': mock_class}),
            patch('beast.api.model.torch.load', return_value=mock_state),
        ):
            m = Model.from_dir(str(tmp_path))
        assert m.model_dir == tmp_path


class TestModelTrain:
    """Test the Model.train method."""

    def _make_model(self, model_dir: Path | None = None) -> Model:
        return Model(MagicMock(), {'model': {}, 'training': {}}, model_dir=model_dir)

    def test_sets_model_dir(self, tmp_path: Path) -> None:
        m = self._make_model()
        output = tmp_path / 'run1'
        output.mkdir()
        with patch('beast.api.model.train', return_value=MagicMock()):
            m.train(output_dir=output)
        assert m.model_dir == output

    def test_calls_train_with_correct_args(self, tmp_path: Path) -> None:
        m = self._make_model()
        original_model = m.model  # train() reassigns m.model to its return value
        output = tmp_path / 'run1'
        output.mkdir()
        with patch('beast.api.model.train', return_value=MagicMock()) as mock_train:
            m.train(output_dir=output)
        mock_train.assert_called_once_with(m.config, original_model, output_dir=output)

    def test_updates_model_after_train(self, tmp_path: Path) -> None:
        m = self._make_model()
        output = tmp_path / 'run1'
        output.mkdir()
        trained_model = MagicMock()
        with patch('beast.api.model.train', return_value=trained_model):
            m.train(output_dir=output)
        assert m.model is trained_model


class TestModelPredictImages:
    """Test the Model.predict_images method."""

    def _make_model(self, model_dir: Path | None = None) -> Model:
        return Model(MagicMock(), {}, model_dir=model_dir)

    def test_raises_when_model_dir_is_none(self, tmp_path: Path) -> None:
        m = self._make_model(model_dir=None)
        with pytest.raises(ValueError, match='model_dir is None'):
            m.predict_images(image_dir=tmp_path)

    def test_calls_predict_images_with_correct_args(self, tmp_path: Path) -> None:
        model_dir = tmp_path / 'model'
        model_dir.mkdir()
        image_dir = tmp_path / 'images'
        image_dir.mkdir()
        m = self._make_model(model_dir=model_dir)
        with patch('beast.api.model.predict_images', return_value={}) as mock_predict:
            m.predict_images(
                image_dir=image_dir,
                output_dir=tmp_path / 'out',
                batch_size=16,
                save_latents=False,
                save_reconstructions=True,
            )
        mock_predict.assert_called_once_with(
            model=m.model,
            output_dir=tmp_path / 'out',
            source_dir=image_dir,
            batch_size=16,
            save_latents=False,
            save_reconstructions=True,
        )

    def test_default_output_dir_uses_model_dir(self, tmp_path: Path) -> None:
        model_dir = tmp_path / 'model'
        model_dir.mkdir()
        image_dir = tmp_path / 'my_images'
        image_dir.mkdir()
        m = self._make_model(model_dir=model_dir)
        with patch('beast.api.model.predict_images', return_value={}) as mock_predict:
            m.predict_images(image_dir=image_dir)
        expected_output = model_dir / 'image_predictions' / 'my_images'
        mock_predict.assert_called_once()
        assert mock_predict.call_args.kwargs['output_dir'] == expected_output

    def test_returns_predict_images_output(self, tmp_path: Path) -> None:
        model_dir = tmp_path / 'model'
        model_dir.mkdir()
        m = self._make_model(model_dir=model_dir)
        sentinel = {'latents': 'data'}
        with patch('beast.api.model.predict_images', return_value=sentinel):
            result = m.predict_images(image_dir=tmp_path)
        assert result is sentinel


class TestModelPredictVideo:
    """Test the Model.predict_video method."""

    def _make_model(self, model_dir: Path | None = None) -> Model:
        return Model(MagicMock(), {}, model_dir=model_dir)

    def test_raises_when_model_dir_is_none(self, tmp_path: Path) -> None:
        m = self._make_model(model_dir=None)
        with pytest.raises(ValueError, match='model_dir is None'):
            m.predict_video(video_file=tmp_path / 'vid.mp4')

    def test_calls_predict_video_with_correct_args(self, tmp_path: Path) -> None:
        model_dir = tmp_path / 'model'
        model_dir.mkdir()
        video_file = tmp_path / 'vid.mp4'
        m = self._make_model(model_dir=model_dir)
        with patch('beast.api.model.predict_video') as mock_predict:
            m.predict_video(
                video_file=video_file,
                output_dir=tmp_path / 'out',
                batch_size=8,
                save_latents=True,
                save_reconstructions=False,
            )
        mock_predict.assert_called_once_with(
            model=m.model,
            output_dir=tmp_path / 'out',
            video_file=video_file,
            batch_size=8,
            save_latents=True,
            save_reconstructions=False,
        )

    def test_default_output_dir_uses_model_dir(self, tmp_path: Path) -> None:
        model_dir = tmp_path / 'model'
        model_dir.mkdir()
        video_file = tmp_path / 'vid.mp4'
        m = self._make_model(model_dir=model_dir)
        with patch('beast.api.model.predict_video') as mock_predict:
            m.predict_video(video_file=video_file)
        expected_output = model_dir / 'video_predictions'
        mock_predict.assert_called_once()
        assert mock_predict.call_args.kwargs['output_dir'] == expected_output

    def test_returns_predict_video_output(self, tmp_path: Path) -> None:
        model_dir = tmp_path / 'model'
        model_dir.mkdir()
        m = self._make_model(model_dir=model_dir)
        sentinel = {'frames_processed': 42}
        with patch('beast.api.model.predict_video', return_value=sentinel):
            result = m.predict_video(video_file=tmp_path / 'vid.mp4')
        assert result is sentinel
