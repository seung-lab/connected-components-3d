from collections.abc import Iterator
from typing import Any, BinaryIO, Literal, Union, overload

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray

from cc3d.types import (
    IntegerT,
    StatisticsDict,
    StatisticsSlicesDict,
    UnsignedIntegerT,
    VcgT,
)

class DimensionError(Exception):
    """The array has the wrong number of dimensions."""
    ...

@overload
def color_connectivity_graph(
    vcg: NDArray[VcgT],
    connectivity: Literal[4, 6, 8, 18, 26] = 26,
    *,
    return_N: Literal[False] = False,
) -> NDArray[VcgT]: ...
@overload
def color_connectivity_graph(
    vcg: NDArray[VcgT],
    connectivity: Literal[4, 6, 8, 18, 26],
    return_N: Literal[False] = False,
) -> NDArray[VcgT]: ...
@overload
def color_connectivity_graph(
    vcg: NDArray[VcgT],
    connectivity: Literal[4, 6, 8, 18, 26] = 26,
    *,
    return_N: Literal[True],
) -> tuple[NDArray[VcgT], int]: ...
@overload
def color_connectivity_graph(
    vcg: NDArray[VcgT],
    connectivity: Literal[4, 6, 8, 18, 26],
    return_N: Literal[True],
) -> tuple[NDArray[VcgT], int]: ...
def color_connectivity_graph(  # type: ignore[misc]
    vcg: NDArray[VcgT],
    connectivity: Literal[4, 6, 8, 18, 26] = 26,
    return_N: bool = False,
) -> Union[NDArray[VcgT], tuple[NDArray[VcgT], int]]:
    """Color the connectivity graph of a voxel connectivity graph.

    Given a voxel connectivity graph following the same bit convention as
    cc3d.voxel_connectivity_graph (see docstring), assuming an undirected
    graph (the format supports directed graphs, but this is not implemented
    for the sake of efficiency), this function will return a uint32 image
    that contains connected components labeled according to the boundaries
    described in the voxel connectivity graph (vcg).
    """
    ...

@overload
def connected_components(
    data: NDArray[Any],
    max_labels: int = -1,
    connectivity: Literal[4, 6, 8, 18, 26] = 26,
    *,
    return_N: Literal[False] = False,
    delta: float = 0,
    out_dtype: DTypeLike = None,
    out_file: Union[str, BinaryIO, None] = None,
    periodic_boundary: bool = False,
    binary_image: bool = False,
) -> NDArray[Union[np.uint16, np.uint32, np.uint64]]: ...
@overload
def connected_components(
    data: NDArray[Any],
    max_labels: int,
    connectivity: Literal[4, 6, 8, 18, 26],
    return_N: Literal[False] = False,
    delta: float = 0,
    out_dtype: DTypeLike = None,
    out_file: Union[str, BinaryIO, None] = None,
    periodic_boundary: bool = False,
    binary_image: bool = False,
) -> NDArray[Union[np.uint16, np.uint32, np.uint64]]: ...
@overload
def connected_components(
    data: NDArray[Any],
    max_labels: int = -1,
    connectivity: Literal[4, 6, 8, 18, 26] = 26,
    *,
    return_N: Literal[True],
    delta: float = 0,
    out_dtype: DTypeLike = None,
    out_file: Union[str, BinaryIO, None] = None,
    periodic_boundary: bool = False,
    binary_image: bool = False,
) -> tuple[NDArray[Union[np.uint16, np.uint32, np.uint64]], int]: ...
@overload
def connected_components(
    data: NDArray[Any],
    max_labels: int,
    connectivity: Literal[4, 6, 8, 18, 26],
    return_N: Literal[True],
    delta: float = 0,
    out_dtype: DTypeLike = None,
    out_file: Union[str, BinaryIO, None] = None,
    periodic_boundary: bool = False,
    binary_image: bool = False,
) -> tuple[NDArray[Union[np.uint16, np.uint32, np.uint64]], int]: ...
def connected_components(  # type: ignore[misc]
    data: NDArray[Any],
    max_labels: int = -1,
    connectivity: Literal[4, 6, 8, 18, 26] = 26,
    return_N: bool = False,
    delta: float = 0,
    out_dtype: DTypeLike = None,
    out_file: Union[str, BinaryIO, None] = None,
    periodic_boundary: bool = False,
    binary_image: bool = False,
) -> Union[
    NDArray[Union[np.uint16, np.uint32, np.uint64]],
    tuple[NDArray[Union[np.uint16, np.uint32, np.uint64]], int],
]:
    """Connected components applied to 3D images with handling for multiple labels.

    Args:
        data: Input weights in a 2D or 3D numpy array.
        max_labels (int): Save memory by predicting the maximum
            number of possible labels that might be output.
            Defaults to number of voxels.
        connectivity (int):
            For 3D images, 6 (voxel faces), 18 (+edges), or 26 (+corners)
            If the input image is 2D, you may specify 4 (pixel faces) or
            8 (+corners).
        return_N (bool): If True, also return the number of connected components
            as the second argument of a return tuple.
        delta (same as data): >= 0. Connect together values whose
            difference in value is <= delta. Useful for rough
            segmentations of continuously valued images.
        out_dtype: If specified, must be one of np.uint16, np.uint32, np.uint64.
            If not specified, it will be automatically determined. Most of the time,
            you should leave this off so that the smallest safe dtype will be used.
            However, in some applications you can save an up-conversion in the next
            operation by outputting the appropriately sized type instead.
        out_file: If specified, the output array will be an mmapped
            file. Can be a file-name or a file-like object.
        periodic_boundary: The boundary edges wrap around.
        binary_image: If True, regardless of the input type,
            treat as a binary image (foreground > 0, background == 0).
            Certain inputs will always be treated as a binary
            image (e.g. bool dtype, delta == max int or max float etc.).

    Returns:
        Either (OUT, N), if return_N else, OUT.

        Where OUT = 1D, 2D or 3D numpy array remapped to reflect
            the connected components sequentially numbered from 1 to N.

            The data type will be automatically determined as uint16, uint32,
            or uint64 depending on the estimate of the number of provisional
            labels required.

        And N = number of connected components
    """
    ...

def contacts(
    labels: NDArray[Any],
    connectivity: Literal[4, 6, 8, 18, 26] = 26,
    surface_area: bool = True,
    anisotropy: tuple[Union[int, float], Union[int, float], Union[int, float]] = (
        1,
        1,
        1,
    ),
) -> dict[tuple[int, int], Union[int, float]]:
    """Get the N-connected region adjacancy graph of a 3D image and the contact area between two regions.

    Args:
        labels: 3D numpy array of integer segmentation labels.
        connectivity: 6, 18, or 26 (default).
        surface_area: Should the returned value be the contact
            surface area or a simple count of neighboring voxels?
            Surface area only counts face contact as edges and corners
            have zero area.
        anisotropy: Weights for x, y, and z dimensions for computing
            surface area.

    Returns:
        A dictionary resembling { (label_1, label_2): float, ... }.
    """
    ...

def each(
    labels: NDArray[UnsignedIntegerT],
    binary: bool = False,
    in_place: bool = False,
) -> Iterator[tuple[int, NDArray[UnsignedIntegerT]]]:
    """Returns an iterator that extracts each label from a dense labeling.

    Args:
        binary: Create a binary image from each component (otherwise use the
            same dtype and label value for the mask).
        in_place: Much faster but the resulting image will be read-only.

    Examples:
    >>> for label, img in cc3d.each(labels, binary=False, in_place=False):
            process(img)

    Returns:
        An iterator.
    """
    ...

def estimate_provisional_labels(
    data: NDArray[Any],
) -> tuple[int, int, int]:
    """Estimate the number of provisional labels required for connected components."""
    ...

def region_graph(
    labels: NDArray[np.integer],
    connectivity: Literal[4, 6, 8, 18, 26] = 26,
) -> set[tuple[int, int]]:
    """Get the N-connected region adjacancy graph of a 3D image.

    For backwards compatibility. "contacts" may be more useful.

    Supports 26, 18, and 6 connectivities.

    Args:
        labels: 3D numpy array of integer segmentation labels.
        connectivity: 6, 18, or 26 (default).

    Returns:
        A set of edges between labels.
    """
    ...

@overload
def statistics(
    out_labels: NDArray[Any],
    no_slice_conversion: Literal[False] = False,
) -> StatisticsDict: ...
@overload
def statistics(
    out_labels: NDArray[Any],
    no_slice_conversion: Literal[True],
) -> StatisticsSlicesDict: ...
def statistics(  # type: ignore[misc]
    out_labels: NDArray[Any],
    no_slice_conversion: bool = False,
) -> Union[StatisticsDict, StatisticsSlicesDict]:
    """Compute basic statistics on the regions in the image.

    These are the voxel counts per label, the axis-aligned
    bounding box, and the centroid of each label.

    LIMITATION: input must be >=0 and < num voxels

    Args:
        out_labels: A numpy array of labels.
        no_slice_conversion: if True, return the bounding_boxes as
            a numpy array. This can save memory and time.

    Returns:
        A dictionary with the following structure.

        ```
        N = np.max(out_labels)
        # Index into array is the CCL label.
        {
            voxel_counts: NDArray[np.uint32], # (index is label) (N+1)
            # Structure is xmin, xmax, ymin, ymax, zmin, zmax by label
            bounding_boxes: NDArray[np.uint16] | list[tuple(slice, slice, slice)],
            # Index into list is the connected component ID, the
            # tuple of slices can be directly used to extract the
            # region of interest from out_labels using slice
            # notation.
            # Structure is x,y,z
            centroids: NDArray[np.float64], # (N+1,3)
        }
        ```
    """
    ...

def voxel_connectivity_graph(
    data: NDArray[IntegerT],
    connectivity: Literal[4, 6, 8, 18, 26] = 26,
) -> NDArray[IntegerT]:
    """Extracts the voxel connectivity graph from a multi-label image.

    A voxel is considered connected if the adjacent voxel is the same
    label.

    This output is a bitfield that represents a directed graph of the
    allowed directions for transit between voxels. If a connection is allowed,
    the respective direction is set to 1 else it set to 0.

    For 2D connectivity, the output is an 8-bit unsigned integer.

    ```
    Bits 1-4: edges     (4,8 way)
         5-8: corners   (8 way only, zeroed in 4 way)
    ```

    ```
        8       7      6      5      4      3      2      1
    ------ ------ ------ ------ ------ ------ ------ ------
      -x-y    x-y    -xy     xy     -x     +y     -x     +x
    ```

    For a 3D 26 and 18 connectivity, the output requires 32-bit unsigned integers,
    for 6-way the output are 8-bit unsigned integers.

    ```
    Bits 1-6: faces     (6,18,26 way)
        7-19: edges     (18,26 way)
       18-26: corners   (26 way)
       26-32: unused (zeroed)
    ```

    6x unused, 8 corners, 12 edges, 6 faces

    ```
        32     31     30     29     28     27     26     25     24     23
    ------ ------ ------ ------ ------ ------ ------ ------ ------ ------
    unused unused unused unused unused unused -x-y-z  x-y-z -x+y-z +x+y-z
        22     21     20     19     18     17     16     15     14     13
    ------ ------ ------ ------ ------ ------ ------ ------ ------ ------
    -x-y+z +x-y+z -x+y+z    xyz   -y-z    y-z   -x-z    x-z    -yz     yz
        12     11     10      9      8      7      6      5      4      3
    ------ ------ ------ ------ ------ ------ ------ ------ ------ ------
       -xz     xz   -x-y    x-y    -xy     xy     -z     +z     -y     +y
         2      1
    ------ ------
        -x     +x
    ```

    Args:
        data: A numpy array.
        connectivity: The connectivity to use.

    Returns:
        A uint8 or uint32 numpy array the same size as the input.
    """
    ...

def runs(labels: NDArray[np.unsignedinteger]) -> dict[int, list[tuple[int, int]]]:
    """Returns a dictionary describing where each label is located.

    Use this data in conjunction with render and erase.
    """
    ...

def draw(
    label: ArrayLike,
    runs: list[tuple[int, int]],
    image: NDArray[UnsignedIntegerT],
) -> NDArray[UnsignedIntegerT]:
    """Draws label onto the provided image according to runs."""
    ...
