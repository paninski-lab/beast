from typing import TypedDict

from jaxtyping import Float
from torch import Tensor


class ExampleDict(TypedDict):
    """Return type when calling BaseDataset.__getitem()__."""
    image: Float[Tensor, 'channels image_height image_width']
    video: str
    idx: int
