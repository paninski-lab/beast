"""Tests for imgaug pipeline functionality."""

import numpy as np
import pytest
from PIL import Image

from beast.data.augmentations import expand_imgaug_str_to_dict, imgaug_pipeline


class TestImgaugPipeline:
    """Test the imgaug_pipeline function."""

    def test_null_pipeline(self, base_dataset) -> None:

        image = Image.open(base_dataset.image_list[0]).convert('RGB')
        params_dict = {}
        pipe = imgaug_pipeline(params_dict)
        im_0 = pipe(images=np.expand_dims(np.asarray(image), axis=0))[0]
        assert np.allclose(np.asarray(image), im_0)

    def test_zero_probability_is_noop(self, base_dataset) -> None:

        image = Image.open(base_dataset.image_list[0]).convert('RGB')
        params_dict = {
            'ShearX': {'p': 0.0, 'kwargs': {'shear': (-30, 30)}},
            'Jigsaw': {'p': 0.0, 'kwargs': {'nb_rows': (3, 10), 'nb_cols': (5, 8)}},
            'MultiplyAndAddToBrightness': {
                'p': 0.0, 'kwargs': {'mul': (0.5, 1.5), 'add': (-5, 5)},
            },
        }
        pipe = imgaug_pipeline(params_dict)
        im_0 = pipe(images=np.expand_dims(np.asarray(image), axis=0))[0]
        assert np.allclose(np.asarray(image), im_0)

    def test_resize(self, base_dataset) -> None:

        image = Image.open(base_dataset.image_list[0]).convert('RGB')
        params_dict = {
            'Resize': {'p': 1.0, 'args': ({'height': 256, 'width': 256},), 'kwargs': {}},
        }
        pipe = imgaug_pipeline(params_dict)
        im_0 = pipe(images=np.expand_dims(np.asarray(image), axis=0))[0]
        assert im_0.shape[0] == 256
        assert im_0.shape[1] == 256
        # resize should be deterministic
        im_1 = pipe(images=np.expand_dims(np.asarray(image), axis=0))[0]
        assert np.allclose(im_0, im_1)

    def test_fliplr(self, base_dataset) -> None:

        image = Image.open(base_dataset.image_list[0]).convert('RGB')
        params_dict = {'Fliplr': {'p': 1.0, 'kwargs': {'p': 1.0}}}
        pipe = imgaug_pipeline(params_dict)
        im_0 = pipe(images=np.expand_dims(np.asarray(image), axis=0))[0]
        assert np.allclose(im_0[:, ::-1, ...], np.asarray(image))

    def test_flipud(self, base_dataset) -> None:

        image = Image.open(base_dataset.image_list[0]).convert('RGB')
        params_dict = {'Flipud': {'p': 1.0, 'kwargs': {'p': 1.0}}}
        pipe = imgaug_pipeline(params_dict)
        im_0 = pipe(images=np.expand_dims(np.asarray(image), axis=0))[0]
        assert np.allclose(im_0[::-1, :, ...], np.asarray(image))

    def test_stochastic_augmentations_change_image(self, base_dataset) -> None:

        image = Image.open(base_dataset.image_list[0]).convert('RGB')
        for params_dict in [
            {'MotionBlur': {'p': 1.0, 'kwargs': {'k': 5, 'angle': (-90, 90)}}},
            {'CoarseSalt': {'p': 1.0, 'kwargs': {'p': 0.1, 'size_percent': (0.05, 1.0)}}},
            {'Affine': {'p': 1.0, 'kwargs': {'rotate': (-90, 90)}}},
        ]:
            pipe = imgaug_pipeline(params_dict)
            im_0 = pipe(images=np.expand_dims(np.asarray(image), axis=0))[0]
            assert not np.allclose(im_0, np.asarray(image))

    def test_list_kwarg_single_element_converted(self) -> None:
        # Arrange — k=[5] is a single-element list; code converts it to k=5

        params_dict = {'MotionBlur': {'p': 1.0, 'kwargs': {'k': [5], 'angle': 0}}}
        # Act / Assert — pipeline builds successfully with the converted kwarg
        pipe = imgaug_pipeline(params_dict)
        assert pipe is not None

    def test_list_kwarg_multi_element_converted_to_tuple(self) -> None:
        # Arrange — angle=[-90, 90] is a list; code converts it to tuple (-90, 90)

        params_dict = {'MotionBlur': {'p': 1.0, 'kwargs': {'k': 5, 'angle': [-90, 90]}}}
        # Act / Assert — pipeline builds successfully with the converted kwarg
        pipe = imgaug_pipeline(params_dict)
        assert pipe is not None


class TestExpandImgaugStrToDict:
    """Test the expand_imgaug_str_to_dict function."""

    def test_none_preset_returns_empty_dict(self) -> None:

        params = expand_imgaug_str_to_dict('none')
        assert len(params) == 0

    def test_default_preset(self) -> None:

        params = expand_imgaug_str_to_dict('default')
        assert len(params) == 2
        assert 'Fliplr' in params
        assert 'CropAndPad' in params

    def test_top_down_preset(self) -> None:

        params = expand_imgaug_str_to_dict('top-down')
        assert len(params) == 3
        assert 'Fliplr' in params
        assert 'Affine' in params
        assert 'CropAndPad' in params

    def test_unknown_preset_raises(self) -> None:

        with pytest.raises(NotImplementedError):
            expand_imgaug_str_to_dict('invalid-preset')
