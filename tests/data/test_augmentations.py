"""Test imgaug pipeline functionality."""

import numpy as np
from PIL import Image


def test_imgaug_pipeline(base_dataset):

    from beast.data.augmentations import imgaug_pipeline

    idx = 0
    img_name = base_dataset.image_list[idx]
    image = Image.open(img_name).convert('RGB')

    # play with several easy-to-verify transforms

    # ------------
    # NULL
    # ------------
    params_dict = {}
    pipe = imgaug_pipeline(params_dict)
    im_0 = pipe(images=np.expand_dims(image, axis=0))
    im_0 = im_0[0]
    assert np.allclose(image, im_0)

    # pipeline should not do anything if augmentation probabilities are all zero
    params_dict = {
        'ShearX': {'p': 0.0, 'kwargs': {'shear': (-30, 30)}},
        'Jigsaw': {'p': 0.0, 'kwargs': {'nb_rows': (3, 10), 'nb_cols': (5, 8)}},
        'MultiplyAndAddToBrightness': {'p': 0.0, 'kwargs': {'mul': (0.5, 1.5), 'add': (-5, 5)}},
    }
    pipe = imgaug_pipeline(params_dict)
    im_0 = pipe(images=np.expand_dims(image, axis=0))
    im_0 = im_0[0]
    assert np.allclose(image, im_0)

    # ------------
    # Resize
    # ------------
    params_dict = {'Resize': {'p': 1.0, 'args': ({'height': 256, 'width': 256},), 'kwargs': {}}}
    pipe = imgaug_pipeline(params_dict)
    im_0 = pipe(images=np.expand_dims(image, axis=0))
    im_0 = im_0[0]
    assert im_0.shape[0] == params_dict['Resize']['args'][0]['height']
    assert im_0.shape[1] == params_dict['Resize']['args'][0]['width']

    # resize should be repeatable
    im_1 = pipe(images=np.expand_dims(image, axis=0))
    im_1 = im_1[0]
    assert np.allclose(im_0, im_1)

    # ------------
    # Fliplr
    # ------------
    params_dict = {'Fliplr': {'p': 1.0, 'kwargs': {'p': 1.0}}}
    pipe = imgaug_pipeline(params_dict)
    im_0 = pipe(images=np.expand_dims(image, axis=0))
    im_0 = im_0[0]
    im_0 = im_0[:, ::-1, ...]  # lr flip
    assert np.allclose(im_0, image)

    # ------------
    # Flipud
    # ------------
    params_dict = {'Flipud': {'p': 1.0, 'kwargs': {'p': 1.0}}}
    pipe = imgaug_pipeline(params_dict)
    im_0 = pipe(images=np.expand_dims(image, axis=0))
    im_0 = im_0[0]
    im_0 = im_0[::-1, :, ...]  # ud flip
    assert np.allclose(im_0, image)

    # ------------
    # misc
    # ------------
    # make sure various augmentations are not repeatable
    params_dict = {'MotionBlur': {'p': 1.0, 'kwargs': {'k': 5, 'angle': (-90, 90)}}}
    pipe = imgaug_pipeline(params_dict)
    im_0 = pipe(images=np.expand_dims(image, axis=0))
    im_0 = im_0[0]
    assert not np.allclose(im_0, image)  # image changed

    params_dict = {'CoarseSalt': {'p': 1.0, 'kwargs': {'p': 0.1, 'size_percent': (0.05, 1.0)}}}
    pipe = imgaug_pipeline(params_dict)
    im_0 = pipe(images=np.expand_dims(image, axis=0))
    im_0 = im_0[0]
    assert not np.allclose(im_0, image)  # image changed

    params_dict = {'Affine': {'p': 1.0, 'kwargs': {'rotate': (-90, 90)}}}
    pipe = imgaug_pipeline(params_dict)
    im_0 = pipe(images=np.expand_dims(image, axis=0))
    im_0 = im_0[0]
    assert not np.allclose(im_0, image)  # image changed


def test_expand_imgaug_str_to_dict():

    from beast.data.augmentations import expand_imgaug_str_to_dict

    # 'none' pipeline: should not contain any augmentations
    params = expand_imgaug_str_to_dict('none')
    assert len(params) == 0

    # 'default' pipeline: should only contain flips and crops
    params = expand_imgaug_str_to_dict('default')
    assert len(params) == 2
    assert 'Fliplr' in params.keys()
    assert 'CropAndPad' in params.keys()

    # 'top-down' pipeline: should only contain flips, rotations, and crops
    params = expand_imgaug_str_to_dict('top-down')
    assert len(params) == 3
    assert 'Fliplr' in params.keys()
    assert 'Affine' in params.keys()
    assert 'CropAndPad' in params.keys()
