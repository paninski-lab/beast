"""Typed dictionaries for dataset item structures."""

from typing import NotRequired, TypedDict

from jaxtyping import Float
from torch import Tensor


class ExampleDict(TypedDict):
    """Return type when calling BaseDataset.__getitem()__."""
    image: Float[Tensor, 'channels image_height image_width']
    video: str | list[str]
    idx: int | list[int]
    image_path: str | list[str]


class MultiViewExampleDict(TypedDict):
    """Return type when calling MultiViewDataset.__getitem__().

    Required fields: image, view_names, video_id, frame_id.
    Optional fields: c2w, fxfycxcy (absent when calibration files are missing),
        input_mask (present only when use_mask=True).
    """
    image: Float[Tensor, 'views channels height width']
    view_names: list[str]
    video_id: str
    frame_id: str
    # the three fields below are only used by BEAST3D (GT cameras + mask loss);
    # may become a separate subclass dict later
    c2w: NotRequired[Float[Tensor, 'views 4 4']]
    fxfycxcy: NotRequired[Float[Tensor, 'views 4']]
    input_mask: NotRequired[Float[Tensor, 'views 1 height width']]
