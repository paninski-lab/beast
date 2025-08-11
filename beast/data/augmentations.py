"""Functions to build augmentation pipeline."""

from typing import Any, Callable

import imgaug.augmenters as iaa
from typeguard import typechecked


@typechecked
def imgaug_pipeline(params_dict: dict) -> Callable:
    """Create simple and flexible data transform pipeline that augments images.

    Parameters
    ----------
    params_dict: each key must be the name of a transform importable from imgaug.augmenters,
        e.g. 'Affine', 'Fliplr', etc. The value must be a dict with several optional keys:
            - 'p' (float): probability of applying transform (using imgaug.augmenters.Sometimes)
            - 'args' (list): arguments for transform
            - 'kwargs' (dict): keyword args for the transformation

    Returns
    -------
    imgaug pipeline

    Examples
    --------
    Create a pipeline with
    - Affine transformation applied 50% of the time with rotation uniformly sampled from
      (-25, 25) degrees
    - MotionBlur transformation that is applied 25% of the time with a kernel size of 5 pixels
      and blur direction uniformly sampled from (-90, 90) degrees

      >>> params_dict = {
      >>>    'Affine': {'p': 0.5, 'kwargs': {'rotate': (-25, 25)}},
      >>>    'MotionBlur': {'p': 0.25, 'kwargs': {'k': 5, 'angle': (-90, 90)}},
      >>> }

    In a config file, this will look like:
      >>> training:
      >>>   imgaug:
      >>>     Affine:
      >>>       p: 0.5
      >>>       kwargs:
      >>>         rotate: [-10, 10]
      >>>     MotionBlur:
      >>>       p: 0.25
      >>>       kwargs:
      >>>         k: 5
      >>>         angle: [-90, 90]

    """

    data_transform = []

    for transform_str, args in params_dict.items():

        transform = getattr(iaa, transform_str)
        apply_prob = args.get('p', 0.5)
        transform_args = args.get('args', ())
        transform_kwargs = args.get('kwargs', {})

        # cannot load tuples from yaml files
        # make sure any lists are converted to tuples
        # unless the list contains a single item, then pass through the item (hack for Rot90)
        for kw, arg in transform_kwargs.items():
            if isinstance(arg, list):
                if len(arg) == 1:
                    transform_kwargs[kw] = arg[0]
                else:
                    transform_kwargs[kw] = tuple(arg)

        # add transform to pipeline
        if apply_prob == 0.0:
            pass
        elif apply_prob < 1.0:
            data_transform.append(
                iaa.Sometimes(
                    apply_prob,
                    transform(*transform_args, **transform_kwargs),
                )
            )
        else:
            data_transform.append(transform(*transform_args, **transform_kwargs))

    return iaa.Sequential(data_transform)


@typechecked
def expand_imgaug_str_to_dict(params: str) -> dict[str, Any]:
    params_dict = {}
    if params == 'none':
        pass  # no augmentations
    elif params == 'default':
        # flip
        params_dict['Fliplr'] = {'p': 0.5}
        # random crop
        crop_by = 0.15  # number of pix to crop on each side of img given as a fraction
        params_dict['CropAndPad'] = {
            'p': 0.4,
            'kwargs': {'percent': (-crop_by, crop_by), 'keep_size': False},
        }
    elif params == 'top-down':
        # flip
        params_dict['Fliplr'] = {'p': 0.5}
        # rotate
        rotation = 180  # rotation uniformly sampled from (-rotation, +rotation)
        params_dict['Affine'] = {'p': 1.0, 'kwargs': {'rotate': (-rotation, rotation)}}
        # random crop
        crop_by = 0.15  # number of pix to crop on each side of img given as a fraction
        params_dict['CropAndPad'] = {
            'p': 0.4,
            'kwargs': {'percent': (-crop_by, crop_by), 'keep_size': False},
        }
    else:
        raise NotImplementedError(
            f'training.imgaug string {params} must be in ["none", "default", "top-down"]'
        )

    return params_dict
