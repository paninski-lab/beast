"""Typed dictionaries for dataset item structures."""

from typing import TypedDict

from jaxtyping import Float
from torch import Tensor


class ExampleDict(TypedDict):
    """Return type when calling BaseDataset.__getitem()__."""
    image: (
        Float[Tensor, 'channels image_height image_width']
        | Float[Tensor, 'batch channels image_height image_width']
    )
    video: str | list[str]
    idx: int | list[int]
    image_path: str | list[str]
