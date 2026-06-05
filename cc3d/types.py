import typing

import numpy as np
from numpy.typing import NDArray

VcgT = typing.TypeVar("VcgT", np.uint8, np.uint32)
IntegerT = typing.TypeVar("IntegerT", bound=np.integer)
UnsignedIntegerT = typing.TypeVar("UnsignedIntegerT", bound=np.unsignedinteger)


class StatisticsDict(typing.TypedDict):
    voxel_counts: NDArray[np.uint32]
    bounding_boxes: NDArray[np.uint16]
    centroids: NDArray[np.float64]


class StatisticsSlicesDict(typing.TypedDict):
    voxel_counts: NDArray[np.uint32]
    bounding_boxes: list[tuple[slice, slice, slice]]
    centroids: NDArray[np.float64]
