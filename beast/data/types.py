from typing import TypedDict, Union

from jaxtyping import Float
from torch import Tensor


class ExampleDict(TypedDict):
    """Return type when calling BaseDataset.__getitem()__."""
    image: Union[Float[Tensor, 'channels image_height image_width'], Float[Tensor, 'batch channels image_height image_width']]
    video: Union[str, list[str]]
    idx: Union[int, list[int]]
    image_path: Union[str, list[str]]
