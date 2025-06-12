# cython: language_level=3
"""
Cython binding for connected components applied to 3D images
with 26-connectivity and handling for multiple labels.

Author: William Silversmith
Affiliation: Seung Lab, Princeton Neuroscience Institute
Date: August 2018 - June 2023

---
This program is free software: you can redistribute it and/or modify
it under the terms of the Lesser GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
Lesser GNU General Public License for more details.

You should have received a copy of the Lesser GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
---

If you received a copy of this program in binary form, you can get 
the source code for free here: 

https://github.com/seung-lab/connected-components-3d
"""
import cython
import operator
from functools import reduce
from typing import (
  Dict, Union, Tuple, Iterator, 
  Sequence, Optional, Any, BinaryIO
)

from libc.stdlib cimport calloc, free
from libc.stdint cimport (
  int8_t, int16_t, int32_t, int64_t,
  uint8_t, uint16_t, uint32_t, uint64_t,
)
from libcpp cimport bool as native_bool
from cpython cimport array 
import array
import os
import sys

from libcpp.vector cimport vector
from libcpp.map cimport map as mapcpp
from libcpp.unordered_map cimport unordered_map
from libcpp.utility cimport pair as cpp_pair
cimport numpy as cnp
cnp.import_array()

import numpy as np
import time

cdef extern from "cc3d.hpp" namespace "cc3d":
  cdef size_t estimate_provisional_label_count[T](
    T* in_labels, int64_t sx, int64_t voxels,
    int64_t &first_foreground_row, 
    int64_t &last_foreground_row
  )

cdef extern from "cc3d_continuous.hpp" namespace "cc3d":
  cdef uint32_t* connected_components3d[T,U](
    T* in_labels, 
    int64_t sx, int64_t sy, int64_t sz,
    int64_t max_labels, int64_t connectivity, T delta,
    U* out_labels, size_t &N, 
    native_bool periodic_boundary, native_bool binary_image
  ) except +

cdef extern from "cc3d_graphs.hpp" namespace "cc3d":
  cdef OUT* extract_voxel_connectivity_graph[T,OUT](
    T* in_labels, 
    int64_t sx, int64_t sy, int64_t sz,
    int64_t connectivity, OUT *graph
  ) except +
  cdef struct pair_hash:
    size_t __call__(cpp_pair[uint64_t,uint64_t] v)
  cdef unordered_map[cpp_pair[T,T], float, pair_hash] extract_region_graph[T](
    T* labels,
    int64_t sx, int64_t sy, int64_t sz,
    float wx, float wy, float wz,
    int64_t connectivity, native_bool surface_area
  ) except +
  cdef mapcpp[T, vector[cpp_pair[size_t,size_t]]] extract_runs[T](
    T* labels, size_t sx, size_t sy, size_t sz
  )
  void set_run_voxels[T](
    T key,
    vector[cpp_pair[size_t, size_t]] all_runs,
    T* labels, size_t voxels
  ) except +
  cdef OUT* color_connectivity_graph_N[T,OUT](
    T* vcg,
    int64_t sx, int64_t sy, int64_t sz,
    int connectivity,
    OUT* out_labels,
    uint64_t& N
  ) except +

ctypedef fused UINT:
  uint8_t
  uint16_t
  uint32_t
  uint64_t

ctypedef fused INTEGER:
  UINT
  int8_t
  int16_t
  int32_t
  int64_t

ctypedef fused BBOX_T:
  uint16_t
  uint32_t

class DimensionError(Exception):
  """The array has the wrong number of dimensions."""
  pass

# from https://github.com/seung-lab/fastremap/blob/master/fastremap.pyx
@cython.binding(True)
def _reshape(
  arr:np.ndarray, 
  shape:Sequence[int], 
  order:Optional[chr]=None
) -> np.ndarray:
  """
  If the array is contiguous, attempt an in place reshape
  rather than potentially making a copy.
  Required:
    arr: The input numpy array.
    shape: The desired shape (must be the same size as arr)
  Optional: 
    order: 'C', 'F', or None (determine automatically)
  Returns: reshaped array
  """
  if order is None:
    if arr.flags['F_CONTIGUOUS']:
      order = 'F'
    elif arr.flags['C_CONTIGUOUS']:
      order = 'C'
    else:
      return arr.reshape(shape)

  cdef int nbytes = np.dtype(arr.dtype).itemsize

  if order == 'C':
    strides = [ reduce(operator.mul, shape[i:]) * nbytes for i in range(1, len(shape)) ]
    strides += [ nbytes ]
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
  else:
    strides = [ reduce(operator.mul, shape[:i]) * nbytes for i in range(1, len(shape)) ]
    strides = [ nbytes ] + strides
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

cdef int64_t even_ceil(int64_t N):
  if N & 0x1:
    return N << 1
  return N

@cython.binding(True)
def estimate_provisional_labels(data:np.ndarray) -> Tuple[int,int,int]:
  cdef uint8_t[:] arr_memview8u
  cdef uint16_t[:] arr_memview16u
  cdef uint32_t[:] arr_memview32u
  cdef uint64_t[:] arr_memview64u
  cdef float[:] arr_memviewf
  cdef double[:] arr_memviewd

  cdef int64_t first_foreground_row = 0
  cdef int64_t last_foreground_row = 0

  try:
    # We aren't going to write to the array, but some 
    # non-modifying operations we'll perform will be blocked 
    # by this flag, so we'll just unset it and reset it at 
    # the end.
    writable = data.flags.writeable
    if data.flags.owndata:
      data.setflags(write=1)

    dtype = data.dtype
    if data.flags.f_contiguous:
      sx = data.shape[0]
    else:
      sx = data.shape[-1]

    linear_data = _reshape(data, (data.size,))

    if dtype in (np.uint64, np.int64):
      arr_memview64u = linear_data.view(np.uint64)
      epl = estimate_provisional_label_count[uint64_t](
        &arr_memview64u[0], sx, linear_data.size,
        first_foreground_row, last_foreground_row
      )
    elif dtype in (np.uint32, np.int32):
      arr_memview32u = linear_data.view(np.uint32)
      epl = estimate_provisional_label_count[uint32_t](
        &arr_memview32u[0], sx, linear_data.size,
        first_foreground_row, last_foreground_row
      )
    elif dtype in (np.uint16, np.int16):
      arr_memview16u = linear_data.view(np.uint16)
      epl = estimate_provisional_label_count[uint16_t](
        &arr_memview16u[0], sx, linear_data.size,
        first_foreground_row, last_foreground_row
      )
    elif dtype in (np.uint8, np.int8, bool):
      arr_memview8u = linear_data.view(np.uint8)
      epl = estimate_provisional_label_count[uint8_t](
        &arr_memview8u[0], sx, linear_data.size,
        first_foreground_row, last_foreground_row
      )
    elif dtype == np.float32:
      arr_memviewf = linear_data
      epl = estimate_provisional_label_count[float](
        &arr_memviewf[0], sx, linear_data.size,
        first_foreground_row, last_foreground_row
      )
    elif dtype == np.float64:
      arr_memviewd = linear_data
      epl = estimate_provisional_label_count[double](
        &arr_memviewd[0], sx, linear_data.size,
        first_foreground_row, last_foreground_row
      )
    else:
      raise TypeError(
        f"Type {dtype} is not currently supported. "
        f"Supported: bool, int8, int16, int32, int64, uint8, uint16, uint32, uint64, float16, float32, float64"
      )
  finally:
    if data.flags.owndata:
      data.setflags(write=writable)

  return (epl, first_foreground_row, last_foreground_row)

@cython.binding(True)
def connected_components(
  data:np.ndarray, 
  int64_t max_labels=-1, 
  int64_t connectivity=26, 
  native_bool return_N=False,
  delta:Union[int,float] = 0, 
  out_dtype:Optional[Any] = None,
  out_file:Optional[Union[str, BinaryIO]] = None,
  periodic_boundary:bool = False,
  binary_image:bool = False,
) -> np.ndarray:
  """
  Connected components applied to 3D images with 
  handling for multiple labels.

  Required:
    data: Input weights in a 2D or 3D numpy array. 
  Optional:
    max_labels (int): save memory by predicting the maximum
      number of possible labels that might be output.
      Defaults to number of voxels.
    connectivity (int): 
      For 3D images, 6 (voxel faces), 18 (+edges), or 26 (+corners)
      If the input image is 2D, you may specify 4 (pixel faces) or
        8 (+corners).
    return_N (bool): if True, also return the number of connected components
      as the second argument of a return tuple.
    delta (same as data): >= 0. Connect together values whose 
      difference in value is <= delta. Useful for rough 
      segmentations of continuously valued images.
    out_dtype: if specified, must be one of np.uint16, np.uint32, np.uint64.
      If not specified, it will be automatically determined. Most of the time,
      you should leave this off so that the smallest safe dtype will be used.
      However, in some applications you can save an up-conversion in the next 
      operation by outputting the appropriately sized type instead.
    out_file: If specified, the output array will be an mmapped
      file. Can be a file-name or a file-like object.
    periodic_boundary: the boundary edges wrap around
    binary_image: if True, regardless of the input type,
      treat as a binary image (foreground > 0, background == 0).
      Certain inputs will always be treated as a binary 
      image (e.g. bool dtype, delta == max int or max float etc.).

  let OUT = 1D, 2D or 3D numpy array remapped to reflect
    the connected components sequentially numbered from 1 to N. 

    The data type will be automatically determined as uint16, uint32, 
    or uint64 depending on the estimate of the number of provisional 
    labels required.
  
  let N = number of connected components

  Returns:
    if return_N: (OUT, N)
    else: OUT
  """
  is_torch = hasattr(data, "cpu")
  if is_torch:
    # don't need to call .detach() b/c its read-only
    data = data.cpu().numpy() 

  cdef int dims = len(data.shape)
  if dims not in (1,2,3):
    raise DimensionError("Only 1D, 2D, and 3D arrays supported. Got: " + str(dims))

  if dims == 2 and connectivity not in (4, 8, 6, 18, 26):
    raise ValueError("Only 4, 8, and 6, 18, 26 connectivities are supported for 2D images. Got: " + str(connectivity))
  elif dims != 2 and connectivity not in (6, 18, 26):
    raise ValueError("Only 6, 18, and 26 connectivities are supported for 3D images. Got: " + str(connectivity))

  if periodic_boundary and connectivity not in (4, 8, 6):
    raise ValueError(f"periodic_boundary is not yet implemented for {connectivity}-connectivity.")
  if periodic_boundary and delta != 0:
    raise ValueError(f"periodic_boundary is not yet implemented continuous data.")

  if data.size == 0:
    dtype = data.dtype
    if out_dtype is not None:
      dtype = out_dtype
    out_labels = np.zeros(shape=(0,), dtype=dtype)
    if return_N:
      return (out_labels, 0)
    return out_labels

  order = 'F' if data.flags.f_contiguous else 'C'

  while len(data.shape) < 3:
    if order == 'C':
      data = data[np.newaxis, ...]
    else: # F
      data = data[..., np.newaxis ]

  if not data.flags.c_contiguous and not data.flags.f_contiguous:
    data = np.copy(data, order=order)

  if data.dtype == np.float16:
    if delta == 0:
      data = data.view(np.uint16)
    else:
      raise TypeError("float16 is not supported for continuous images (delta != 0).")

  shape = list(data.shape)

  if order == 'C':
    shape.reverse()

  cdef int sx = shape[0]
  cdef int sy = shape[1]
  cdef int sz = shape[2]

  cdef uint8_t[:,:,:] arr_memview8u
  cdef uint16_t[:,:,:] arr_memview16u
  cdef uint32_t[:,:,:] arr_memview32u
  cdef uint64_t[:,:,:] arr_memview64u
  cdef float[:,:,:] arr_memviewf
  cdef double[:,:,:] arr_memviewd

  cdef int64_t voxels = <int64_t>sx * <int64_t>sy * <int64_t>sz
  cdef cnp.ndarray[uint16_t, ndim=1] out_labels16 = np.array([], dtype=np.uint16)
  cdef cnp.ndarray[uint32_t, ndim=1] out_labels32 = np.array([], dtype=np.uint32)
  cdef cnp.ndarray[uint64_t, ndim=1] out_labels64 = np.array([], dtype=np.uint64)

  epl, first_foreground_row, last_foreground_row = estimate_provisional_labels(data)

  if max_labels <= 0:
    max_labels = voxels
  max_labels = min(max_labels, epl, voxels)

  dtype = data.dtype
  binary_image = binary_image or (dtype == bool)

  if np.issubdtype(dtype, np.floating):
    delta = float(delta)
    binary_image = binary_image or (delta == np.finfo(dtype).max)
  else:
    delta = int(delta)
    binary_image = binary_image or (delta == np.iinfo(dtype).max)

  # OpenCV made a great point that for binary images,
  # the highest number of provisional labels is 
  # 1 0  for a 4-connected that's 1/2 the size + 1 
  # 0 1  for black.
  # For 3D six-connected data the same ratio holds
  # for a 2x2x2 block, where 1/2 the slots are filled
  # in the worst case. 
  # For 8 connected, since 2x2 bocks are always connected,
  # at most 1/4 + 1 of the pixels can be labeled. For 26
  # connected, 2x2x2 blocks are connected, so at most 1/8 + 1
  cdef int64_t union_find_voxels = even_ceil(data.shape[0]) * even_ceil(data.shape[1]) * even_ceil(data.shape[2])

  if binary_image:
    if connectivity in (4,6):
      max_labels = min(max_labels, (union_find_voxels // 2) + 1)
    elif connectivity == (8,18):
      max_labels = min(max_labels, (union_find_voxels // 4) + 1)
    else: # 26
      max_labels = min(max_labels, (union_find_voxels // 8) + 1)

  if out_dtype is not None:
    out_dtype = np.dtype(out_dtype)
    if out_dtype not in (np.uint16, np.uint32, np.uint64):
      raise ValueError(
        f"Explicitly defined out_dtype ({out_dtype}) must be one of: "
        f"np.uint16, np.uint32, np.uint64"
      )
    if np.iinfo(out_dtype).max < max_labels:
      raise ValueError(
        f"Explicitly defined out_dtype ({out_dtype}) is too small "
        f"to contain the estimated maximum number of labels ({max_labels})."
      )
  elif max_labels < np.iinfo(np.uint16).max:
    out_dtype = np.uint16
  elif max_labels < np.iinfo(np.uint32).max:
    out_dtype = np.uint32
  else:
    out_dtype = np.uint64

  def output_factory(out_file, voxels, out_dtype):
    if out_file is None:
      return np.zeros( (voxels,), dtype=out_dtype, order='F' )
    else:
      if isinstance(out_file, str):
        with open(out_file, "wb") as f:
          os.ftruncate(f.fileno(), voxels * np.dtype(out_dtype).itemsize)
      return np.memmap(out_file, order='F', dtype=out_dtype, shape=(voxels,))

  if out_dtype == np.uint16:
    out_labels16 = output_factory(out_file, voxels, out_dtype)
    out_labels = out_labels16
  elif out_dtype == np.uint32:
    out_labels32 = output_factory(out_file, voxels, out_dtype)
    out_labels = out_labels32
  elif out_dtype == np.uint64:
    out_labels64 = output_factory(out_file, voxels, out_dtype)
    out_labels = out_labels64

  cdef size_t N = 0
  
  try:
    # We aren't going to write to the array, but some 
    # non-modifying operations we'll perform will be blocked 
    # by this flag, so we'll just unset it and reset it at 
    # the end.
    writable = data.flags.writeable
    if data.flags.owndata:
      data.setflags(write=1)

    # This first condition can only happen if there 
    # is a single X axis aligned foreground row. Let's handle
    # it hyper efficiently.
    if delta == 0 and first_foreground_row == last_foreground_row and first_foreground_row >= 0:
      N = epl_special_row(first_foreground_row, sx, sy, data, out_labels)
    elif dtype in (np.uint64, np.int64):
      arr_memview64u = data.view(np.uint64)
      if out_dtype == np.uint16:
        connected_components3d[uint64_t, uint16_t](
          &arr_memview64u[0,0,0],
          sx, sy, sz, max_labels, connectivity, delta,
          <uint16_t*>&out_labels16[0], N, 
          periodic_boundary, binary_image
        )
      elif out_dtype == np.uint32:
        connected_components3d[uint64_t, uint32_t](
          &arr_memview64u[0,0,0],
          sx, sy, sz, max_labels, connectivity, delta,
          <uint32_t*>&out_labels32[0], N, 
          periodic_boundary, binary_image
        )
      elif out_dtype == np.uint64:
        connected_components3d[uint64_t, uint64_t](
          &arr_memview64u[0,0,0],
          sx, sy, sz, max_labels, connectivity, delta,
          <uint64_t*>&out_labels64[0], N, 
          periodic_boundary, binary_image
        )
    elif dtype in (np.uint32, np.int32):
      arr_memview32u = data.view(np.uint32)
      if out_dtype == np.uint16:
        connected_components3d[uint32_t, uint16_t](
          &arr_memview32u[0,0,0],
          sx, sy, sz, max_labels, connectivity, delta,
          <uint16_t*>&out_labels16[0], N, 
          periodic_boundary, binary_image
        )
      elif out_dtype == np.uint32:
        connected_components3d[uint32_t, uint32_t](
          &arr_memview32u[0,0,0],
          sx, sy, sz, max_labels, connectivity, delta,
          <uint32_t*>&out_labels32[0], N, 
          periodic_boundary, binary_image
        )
      elif out_dtype == np.uint64:
        connected_components3d[uint32_t, uint64_t](
          &arr_memview32u[0,0,0],
          sx, sy, sz, max_labels, connectivity, delta,
          <uint64_t*>&out_labels64[0], N, 
          periodic_boundary, binary_image
        )
    elif dtype in (np.uint16, np.int16):
      arr_memview16u = data.view(np.uint16)
      if out_dtype == np.uint16:
        connected_components3d[uint16_t, uint16_t](
          &arr_memview16u[0,0,0],
          sx, sy, sz, max_labels, connectivity, delta,
          <uint16_t*>&out_labels16[0], N, 
          periodic_boundary, binary_image
        )
      elif out_dtype == np.uint32:
        connected_components3d[uint16_t, uint32_t](
          &arr_memview16u[0,0,0],
          sx, sy, sz, max_labels, connectivity, delta,
          <uint32_t*>&out_labels32[0], N, 
          periodic_boundary, binary_image
        )
      elif out_dtype == np.uint64:
        connected_components3d[uint16_t, uint64_t](
          &arr_memview16u[0,0,0],
          sx, sy, sz, max_labels, connectivity, delta,
          <uint64_t*>&out_labels64[0], N, 
          periodic_boundary, binary_image
        )
    elif dtype in (np.uint8, np.int8, bool):
      arr_memview8u = data.view(np.uint8)
      if out_dtype == np.uint16:
        connected_components3d[uint8_t, uint16_t](
          &arr_memview8u[0,0,0],
          sx, sy, sz, max_labels, connectivity, delta,
          <uint16_t*>&out_labels16[0], N, 
          periodic_boundary, binary_image
        )
      elif out_dtype == np.uint32:
        connected_components3d[uint8_t, uint32_t](
          &arr_memview8u[0,0,0],
          sx, sy, sz, max_labels, connectivity, delta,
          <uint32_t*>&out_labels32[0], N, 
          periodic_boundary, binary_image
        )
      elif out_dtype == np.uint64:
        connected_components3d[uint8_t, uint64_t](
          &arr_memview8u[0,0,0],
          sx, sy, sz, max_labels, connectivity, delta,
          <uint64_t*>&out_labels64[0], N, 
          periodic_boundary, binary_image
        )
    elif dtype == np.float32:
      arr_memviewf = data
      if out_dtype == np.uint16:
        connected_components3d[float, uint16_t](
          &arr_memviewf[0,0,0],
          sx, sy, sz, max_labels, connectivity, delta,
          <uint16_t*>&out_labels16[0], N, 
          periodic_boundary, binary_image
        )
      elif out_dtype == np.uint32:
        connected_components3d[float, uint32_t](
          &arr_memviewf[0,0,0],
          sx, sy, sz, max_labels, connectivity, delta,
          <uint32_t*>&out_labels32[0], N, 
          periodic_boundary, binary_image
        )
      elif out_dtype == np.uint64:
        connected_components3d[float, uint64_t](
          &arr_memviewf[0,0,0],
          sx, sy, sz, max_labels, connectivity, delta,
          <uint64_t*>&out_labels64[0], N, 
          periodic_boundary, binary_image
        )
    elif dtype == np.float64:
      arr_memviewd = data
      if out_dtype == np.uint16:
        connected_components3d[double, uint16_t](
          &arr_memviewd[0,0,0],
          sx, sy, sz, max_labels, connectivity, delta,
          <uint16_t*>&out_labels16[0], N, 
          periodic_boundary, binary_image
        )
      elif out_dtype == np.uint32:
        connected_components3d[double, uint32_t](
          &arr_memviewd[0,0,0],
          sx, sy, sz, max_labels, connectivity, delta,
          <uint32_t*>&out_labels32[0], N, 
          periodic_boundary, binary_image
        )
      elif out_dtype == np.uint64:
        connected_components3d[double, uint64_t](
          &arr_memviewd[0,0,0],
          sx, sy, sz, max_labels, connectivity, delta,
          <uint64_t*>&out_labels64[0], N, 
          periodic_boundary, binary_image
        )
    else:
      raise TypeError(
        f"Type {dtype} is not currently supported. "
        f"Supported: bool, int8, int16, int32, int64, uint8, uint16, uint32, uint64, float16, float32, float64"
      )
  finally:
    if data.flags.owndata:
      data.setflags(write=writable)

  out_labels = _final_reshape(out_labels, sx, sy, sz, dims, order)

  if is_torch:
    import torch
    out_labels = torch.from_numpy(out_labels)

  if return_N:
    return (out_labels, N)
  return out_labels

def _final_reshape(out_labels, sx, sy, sz, dims, order):
  if dims == 3:
    if order == 'C':
      out_labels = out_labels.reshape( (sz, sy, sx), order=order)
    else:
      out_labels = out_labels.reshape( (sx, sy, sz), order=order)
  elif dims == 2:
    if order == 'C':
      out_labels = out_labels.reshape( (sy, sx), order=order)
    else:
      out_labels = out_labels.reshape( (sx, sy), order=order)
  else:
    out_labels = out_labels.reshape( (sx), order=order)

  return out_labels

cdef size_t epl_special_row(
  size_t foreground_row, size_t sx, size_t sy,
  data, out_labels, size_t N = 0
):
  cdef size_t start = foreground_row * sx
  cdef size_t rz = foreground_row // sy
  cdef size_t ry = foreground_row - rz * sy

  cdef size_t i = 0
  cdef int64_t last_label = 0
  if data.flags.f_contiguous:
    for i in range(sx):
      if data[i,ry,rz] == 0:
        last_label = 0
        continue
      elif data[i,ry,rz] == last_label:
        out_labels[start + i] = N
        continue
      else:
        N += 1
        out_labels[start + i] = N
        last_label = data[i,ry,rz]
  else:
    for i in range(sx):
      if data[rz,ry,i] == 0:
        last_label = 0
        continue
      elif data[rz,ry,i] == last_label:
        out_labels[start + i] = N
        continue
      else:
        N += 1
        out_labels[start + i] = N
        last_label = data[rz,ry,i]

  return N

@cython.binding(True)
def statistics(
  out_labels:np.ndarray, 
  no_slice_conversion:bool = False,
) -> dict:
  """
  Compute basic statistics on the regions in the image.
  These are the voxel counts per label, the axis-aligned
  bounding box, and the centroid of each label.
  
  LIMITATION: input must be >=0 and < num voxels

  no_slice_conversion: if True, return the bounding_boxes as 
    a numpy array. This can save memory and time.

  Returns:
    Let N = np.max(out_labels)
    Index into array is the CCL label.
    {
      voxel_counts: np.ndarray[uint64_t] (index is label) (N+1)

      # Structure is xmin,xmax,ymin,ymax,zmin,zmax by label
      bounding_boxes: List[ tuple(slice, slice, slice), ... ]
        Index into list is the connected component ID, the 
        tuple of slices can be directly used to extract the
        region of interest from out_labels using slice
        notation.

      # Structure is x,y,z
      centroids: np.ndarray[float64] (N+1,3)
    }
  """
  while out_labels.ndim < 2:
    out_labels = out_labels[..., np.newaxis]

  if out_labels.dtype == bool:
    out_labels = out_labels.view(np.uint8)

  cdef uint64_t voxels = out_labels.size;
  cdef uint64_t sx = out_labels.shape[0]
  cdef uint64_t sy = out_labels.shape[1]
  cdef uint64_t sz = (out_labels.shape[2] if out_labels.ndim > 2 else 1)

  if voxels == 0:
    return {
      "voxel_counts": None,
      "bounding_boxes": None,
      "centroids": None,
    }

  cdef uint64_t N = np.max(out_labels)
  cdef int64_t N_min = 0 

  if N > voxels:
    raise ValueError(
      f"Statistics can only be computed on volumes containing labels with values lower than the number of voxels. Max: {N}"
    )

  out_dtype = np.dtype(out_labels.dtype)

  if np.issubdtype(out_dtype, np.signedinteger):
    N_min = np.min(out_labels)
    if N_min < 0:
      raise ValueError(
        f"Statistics can only be computed on volumes containing labels with values >= 0. Min: {N_min}"
      )
    out_labels = out_labels.view(f'u{out_dtype.itemsize}')

  cdef cnp.ndarray[uint16_t] bounding_boxes16
  cdef cnp.ndarray[uint32_t] bounding_boxes32

  if np.any(np.array([sx,sy,sz]) > np.iinfo(np.uint16).max):
    if out_labels.ndim == 2:
      bounding_boxes32 = np.zeros(4 * (N + 1), dtype=np.uint32)
      return _statistics_helper2d(out_labels, no_slice_conversion, bounding_boxes32, N)
    else:
      bounding_boxes32 = np.zeros(6 * (N + 1), dtype=np.uint32)
      return _statistics_helper3d(out_labels, no_slice_conversion, bounding_boxes32, N)
  else:
    if out_labels.ndim == 2:
      bounding_boxes16 = np.zeros(4 * (N + 1), dtype=np.uint16)
      return _statistics_helper2d(out_labels, no_slice_conversion, bounding_boxes16, N)
    else:
      bounding_boxes16 = np.zeros(6 * (N + 1), dtype=np.uint16)
      return _statistics_helper3d(out_labels, no_slice_conversion, bounding_boxes16, N)

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def _statistics_helper2d(
  cnp.ndarray[UINT, ndim=2] out_labels, 
  native_bool no_slice_conversion,
  cnp.ndarray[BBOX_T, ndim=1] bounding_boxes,
  uint64_t N
):
  cdef uint64_t voxels = out_labels.size;
  cdef uint64_t sx = out_labels.shape[0]
  cdef uint64_t sy = out_labels.shape[1]

  cdef cnp.ndarray[uint32_t] counts = np.zeros(N + 1, dtype=np.uint32)
  cdef cnp.ndarray[double] centroids = np.zeros(2 * (N + 1), dtype=np.float64)

  cdef BBOX_T x = 0
  cdef BBOX_T y = 0

  cdef uint64_t label = 0

  bounding_boxes[::2] = np.iinfo(bounding_boxes.dtype).max

  if out_labels.flags.f_contiguous:
    for y in range(sy):
      for x in range(sx):
        label = <uint64_t>out_labels[x,y]
        counts[label] += 1
        bounding_boxes[4 * label + 0] = <BBOX_T>min(bounding_boxes[4 * label + 0], x)
        bounding_boxes[4 * label + 1] = <BBOX_T>max(bounding_boxes[4 * label + 1], x)
        bounding_boxes[4 * label + 2] = <BBOX_T>min(bounding_boxes[4 * label + 2], y)
        bounding_boxes[4 * label + 3] = <BBOX_T>max(bounding_boxes[4 * label + 3], y)
        centroids[2 * label + 0] += <double>x
        centroids[2 * label + 1] += <double>y
  else:
    for x in range(sx):
      for y in range(sy):
        label = <uint64_t>out_labels[x,y]
        counts[label] += 1
        bounding_boxes[4 * label + 0] = <BBOX_T>min(bounding_boxes[4 * label + 0], x)
        bounding_boxes[4 * label + 1] = <BBOX_T>max(bounding_boxes[4 * label + 1], x)
        bounding_boxes[4 * label + 2] = <BBOX_T>min(bounding_boxes[4 * label + 2], y)
        bounding_boxes[4 * label + 3] = <BBOX_T>max(bounding_boxes[4 * label + 3], y)
        centroids[2 * label + 0] += <double>x
        centroids[2 * label + 1] += <double>y

  for label in range(N+1):
    if <double>counts[label] == 0:
      centroids[2 * label + 0] = float('NaN')
      centroids[2 * label + 1] = float('NaN')
    else:
      centroids[2 * label + 0] /= <double>counts[label]
      centroids[2 * label + 1] /= <double>counts[label]

  bbxes = bounding_boxes.reshape((N+1,4))

  output = {
    "voxel_counts": counts,
    "bounding_boxes": bbxes,
    "centroids": centroids.reshape((N+1,2)),
  }

  if no_slice_conversion:
    return output

  cdef BBOX_T xs = 0, xe = 0, ys = 0, ye = 0

  slices = []
  for xs, xe, ys, ye in bbxes:
    if xs < voxels and ys < voxels:
      slices.append((slice(xs, int(xe+1)), slice(ys, int(ye+1))))
    else:
      slices.append(None)

  output["bounding_boxes"] = slices

  return output

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def _statistics_helper3d(
  cnp.ndarray[UINT, ndim=3] out_labels, 
  native_bool no_slice_conversion,
  cnp.ndarray[BBOX_T, ndim=1] bounding_boxes,
  uint64_t N
):
  cdef uint64_t voxels = out_labels.size;
  cdef uint64_t sx = out_labels.shape[0]
  cdef uint64_t sy = out_labels.shape[1]
  cdef uint64_t sz = out_labels.shape[2]

  cdef cnp.ndarray[uint32_t] counts = np.zeros(N + 1, dtype=np.uint32)
  cdef cnp.ndarray[double] centroids = np.zeros(3 * (N + 1), dtype=np.float64)

  cdef BBOX_T x = 0
  cdef BBOX_T y = 0
  cdef BBOX_T z = 0

  cdef uint64_t label = 0

  bounding_boxes[::2] = np.iinfo(bounding_boxes.dtype).max

  if out_labels.flags.f_contiguous:
    for z in range(sz):
      for y in range(sy):
        for x in range(sx):
          label = <uint64_t>out_labels[x,y,z]
          counts[label] += 1
          bounding_boxes[6 * label + 0] = <BBOX_T>min(bounding_boxes[6 * label + 0], x)
          bounding_boxes[6 * label + 1] = <BBOX_T>max(bounding_boxes[6 * label + 1], x)
          bounding_boxes[6 * label + 2] = <BBOX_T>min(bounding_boxes[6 * label + 2], y)
          bounding_boxes[6 * label + 3] = <BBOX_T>max(bounding_boxes[6 * label + 3], y)
          bounding_boxes[6 * label + 4] = <BBOX_T>min(bounding_boxes[6 * label + 4], z)
          bounding_boxes[6 * label + 5] = <BBOX_T>max(bounding_boxes[6 * label + 5], z)
          centroids[3 * label + 0] += <double>x
          centroids[3 * label + 1] += <double>y
          centroids[3 * label + 2] += <double>z
  else:
    for x in range(sx):
      for y in range(sy):
        for z in range(sz):    
          label = <uint64_t>out_labels[x,y,z]
          counts[label] += 1
          bounding_boxes[6 * label + 0] = <BBOX_T>min(bounding_boxes[6 * label + 0], x)
          bounding_boxes[6 * label + 1] = <BBOX_T>max(bounding_boxes[6 * label + 1], x)
          bounding_boxes[6 * label + 2] = <BBOX_T>min(bounding_boxes[6 * label + 2], y)
          bounding_boxes[6 * label + 3] = <BBOX_T>max(bounding_boxes[6 * label + 3], y)
          bounding_boxes[6 * label + 4] = <BBOX_T>min(bounding_boxes[6 * label + 4], z)
          bounding_boxes[6 * label + 5] = <BBOX_T>max(bounding_boxes[6 * label + 5], z)
          centroids[3 * label + 0] += <double>x
          centroids[3 * label + 1] += <double>y
          centroids[3 * label + 2] += <double>z

  for label in range(N+1):
    if <double>counts[label] == 0:
      centroids[3 * label + 0] = float('NaN')
      centroids[3 * label + 1] = float('NaN')
      centroids[3 * label + 2] = float('NaN')
    else:
      centroids[3 * label + 0] /= <double>counts[label]
      centroids[3 * label + 1] /= <double>counts[label]
      centroids[3 * label + 2] /= <double>counts[label]

  bbxes = bounding_boxes.reshape((N+1,6))

  output = {
    "voxel_counts": counts,
    "bounding_boxes": bbxes,
    "centroids": centroids.reshape((N+1,3)),
  }

  if no_slice_conversion:
    return output

  cdef:
    BBOX_T xs = 0, xe = 0
    BBOX_T ys = 0, ye = 0
    BBOX_T zs = 0, ze = 0

  slices = []
  for xs, xe, ys, ye, zs, ze in bbxes:
    if xs < voxels and ys < voxels and zs < voxels:
      slices.append((slice(xs, int(xe+1)), slice(ys, int(ye+1)), slice(zs, int(ze+1))))
    else:
      slices.append(None)

  output["bounding_boxes"] = slices

  return output

@cython.binding(True)
def color_connectivity_graph(
  vcg, 
  connectivity = 26, 
  return_N = False, 
) -> np.ndarray:
  """
  Given a voxel connectivity graph following the same bit convention as 
  cc3d.voxel_connectivity_graph (see docstring), assuming an undirected
  graph (the format supports directed graphs, but this is not implemented
  for the sake of efficiency), this function will return a uint32 image
  that contains connected components labeled according to the boundaries 
  described in the voxel connectivity graph (vcg).
  """
  cdef int dims = len(vcg.shape)
  if dims not in (2,3):
    raise DimensionError("Only 2D, and 3D arrays supported. Got: " + str(dims))

  if dims == 2 and connectivity not in [4,8,6,26]:
    raise ValueError(f"Only 4 and 8 connectivity is supported for 2D images. Got: {connectivity}")
  elif dims != 2 and connectivity not in [6,26]:
    raise ValueError(f"Only 6 and 26 connectivity are supported for 3D images. Got: {connectivity}")

  dtype = vcg.dtype
  if dtype not in [np.uint8, np.uint32]:
    raise ValueError(f"Only uint8 and uint32 are supported. Got: {vcg.dtype}")

  if vcg.size == 0:
    return np.zeros([0] * dims, dtype=np.uint32, order="F")

  while vcg.ndim < 3:
    vcg = vcg[..., np.newaxis]

  vcg = np.asfortranarray(vcg)

  shape = vcg.shape

  cdef int sx = shape[0]
  cdef int sy = shape[1]
  cdef int sz = shape[2]

  if connectivity in [6, 26] and sz > 1 and dtype != np.uint32:
    raise ValueError(f"Only uint32 is supported for 3d connectivites. Got: {vcg.dtype}")

  cdef uint8_t[:,:,:] arr_memview8u
  cdef uint32_t[:,:,:] arr_memview32u
  cdef uint32_t[:,:,:] out_labels32

  cdef int64_t voxels = <int64_t>sx * <int64_t>sy * <int64_t>sz
  out_labels = np.zeros( (sx,sy,sz,), dtype=np.uint32, order='F' )
  out_labels32 = out_labels

  cdef size_t N = 0

  if dtype == np.uint32:
    arr_memview32u = vcg
    color_connectivity_graph_N[uint32_t, uint32_t](
      &arr_memview32u[0,0,0],
      sx, sy, sz,
      connectivity,
      &out_labels32[0,0,0],
      N
    )
  else:
    arr_memview8u = vcg
    color_connectivity_graph_N[uint8_t, uint32_t](
      &arr_memview8u[0,0,0],
      sx, sy, sz,
      connectivity,
      &out_labels32[0,0,0],
      N
    )

  while out_labels.ndim > dims:
    out_labels = out_labels[...,0]

  if return_N:
    return (out_labels, N)
  return out_labels

@cython.binding(True)
def voxel_connectivity_graph(
  data:np.ndarray, 
  int64_t connectivity=26
) -> np.ndarray:
  """
  Extracts the voxel connectivity graph from a multi-label image.
  A voxel is considered connected if the adjacent voxel is the same
  label.

  This output is a bitfield that represents a directed graph of the 
  allowed directions for transit between voxels. If a connection is allowed, 
  the respective direction is set to 1 else it set to 0.

  For 2D connectivity, the output is an 8-bit unsigned integer.

  Bits 1-4: edges     (4,8 way)
       5-8: corners   (8 way only, zeroed in 4 way)

       8      7      6      5      4      3      2      1
  ------ ------ ------ ------ ------ ------ ------ ------
    -x-y    x-y    -xy     xy     -x     +y     -x     +x

  For a 3D 26 and 18 connectivity, the output requires 32-bit unsigned integers,
    for 6-way the output are 8-bit unsigned integers.

  Bits 1-6: faces     (6,18,26 way)
      7-19: edges     (18,26 way)
     18-26: corners   (26 way)
     26-32: unused (zeroed)

  6x unused, 8 corners, 12 edges, 6 faces

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

  Returns: uint8 or uint32 numpy array the same size as the input
  """
  cdef int dims = len(data.shape)
  if dims not in (1,2,3):
    raise DimensionError("Only 1D, 2D, and 3D arrays supported. Got: " + str(dims))

  if dims == 2 and connectivity not in (4, 8, 6, 18, 26):
    raise ValueError("Only 4, 8, and 6, 18, 26 connectivities are supported for 2D images. Got: " + str(connectivity))
  elif dims != 2 and connectivity not in (6, 18, 26):
    raise ValueError("Only 6, 18, and 26 connectivities are supported for 3D images. Got: " + str(connectivity))

  out_dtype = np.uint32
  if connectivity in (4, 8, 6):
    out_dtype = np.uint8

  if data.size == 0:
    return np.zeros(shape=(0,), dtype=out_dtype)

  data = np.asfortranarray(data)

  while len(data.shape) < 3:
    data = data[..., np.newaxis ]

  shape = list(data.shape)

  cdef int sx = shape[0]
  cdef int sy = shape[1]
  cdef int sz = shape[2]

  cdef uint8_t[:,:,:] arr_memview8u
  cdef uint16_t[:,:,:] arr_memview16u
  cdef uint32_t[:,:,:] arr_memview32u
  cdef uint64_t[:,:,:] arr_memview64u

  cdef uint64_t voxels = <uint64_t>sx * <uint64_t>sy * <uint64_t>sz
  cdef cnp.ndarray[uint8_t, ndim=1] graph8 = np.array([], dtype=np.uint8)
  cdef cnp.ndarray[uint32_t, ndim=1] graph32 = np.array([], dtype=np.uint32)

  if out_dtype == np.uint8:
    graph8 = np.zeros( (voxels,), dtype=out_dtype, order='F' )
    graph = graph8
  elif out_dtype == np.uint32:
    graph32 = np.zeros( (voxels,), dtype=out_dtype, order='F' )
    graph = graph32

  dtype = data.dtype
  
  if dtype in (np.uint64, np.int64):
    arr_memview64u = data.view(np.uint64)
    if out_dtype == np.uint8:
      extract_voxel_connectivity_graph[uint64_t, uint8_t](
        &arr_memview64u[0,0,0],
        sx, sy, sz, connectivity, 
        <uint8_t*>&graph8[0]
      )
    elif out_dtype == np.uint32:
      extract_voxel_connectivity_graph[uint64_t, uint32_t](
        &arr_memview64u[0,0,0],
        sx, sy, sz, connectivity, 
        <uint32_t*>&graph32[0]
      )
  elif dtype in (np.uint32, np.int32):
    arr_memview32u = data.view(np.uint32)
    if out_dtype == np.uint8:
      extract_voxel_connectivity_graph[uint32_t, uint8_t](
        &arr_memview32u[0,0,0],
        sx, sy, sz, connectivity, 
        <uint8_t*>&graph8[0]
      )
    elif out_dtype == np.uint32:
      extract_voxel_connectivity_graph[uint32_t, uint32_t](
        &arr_memview32u[0,0,0],
        sx, sy, sz, connectivity, 
        <uint32_t*>&graph32[0]
      )
  elif dtype in (np.uint16, np.int16):
    arr_memview16u = data.view(np.uint16)
    if out_dtype == np.uint8:
      extract_voxel_connectivity_graph[uint16_t, uint8_t](
        &arr_memview16u[0,0,0],
        sx, sy, sz, connectivity, 
        <uint8_t*>&graph8[0]
      )
    elif out_dtype == np.uint32:
      extract_voxel_connectivity_graph[uint16_t, uint32_t](
        &arr_memview16u[0,0,0],
        sx, sy, sz, connectivity, 
        <uint32_t*>&graph32[0]
      )
  elif dtype in (np.uint8, np.int8, bool):
    arr_memview8u = data.view(np.uint8)
    if out_dtype == np.uint8:
      extract_voxel_connectivity_graph[uint8_t, uint8_t](
        &arr_memview8u[0,0,0],
        sx, sy, sz, connectivity, 
        <uint8_t*>&graph8[0]
      )
    elif out_dtype == np.uint32:
      extract_voxel_connectivity_graph[uint8_t, uint32_t](
        &arr_memview8u[0,0,0],
        sx, sy, sz, connectivity, 
        <uint32_t*>&graph32[0]
      )
  else:
    raise TypeError("Type {} not currently supported.".format(dtype))

  if dims == 3:
    return graph.reshape( (sx, sy, sz), order='F')
  elif dims == 2:
    return graph.reshape( (sx, sy), order='F')
  else:
    return graph.reshape( (sx), order='F')

@cython.binding(True)
def region_graph(
  labels:np.ndarray,
  int connectivity=26,
) -> set:
  """
  Get the N-connected region adjacancy graph of a 3D image.
  For backwards compatibility. "contacts" may be more useful.

  Supports 26, 18, and 6 connectivities.

  labels: 3D numpy array of integer segmentation labels
  connectivity: 6, 18, or 26 (default)

  Returns: set of edges between labels
  """
  res = contacts(labels, connectivity=connectivity)
  return set(res.keys())

@cython.binding(True)
def contacts(
  labels:np.ndarray, 
  int connectivity=26, 
  surface_area:bool = True, 
  anisotropy:Tuple[int,int,int] = (1,1,1)
) -> Dict[Tuple[int,int], float]:
  """
  Get the N-connected region adjacancy graph of a 3D image
  and the contact area between two regions.

  Supports 26, 18, and 6 connectivities.

  labels: 3D numpy array of integer segmentation labels
  connectivity: 6, 18, or 26 (default)
  surface_area: should the returned value be the contact
    surface area or a simple count of neighboring voxels?
    Surface area only counts face contact as edges and corners
    have zero area.
  anisotropy: weights for x, y, and z dimensions for computing
    surface area.

  Returns: { (label_1, label_2): float, ... }
  """
  while len(labels.shape) < 3:
    labels = labels[..., np.newaxis ]

  return _contacts(labels, connectivity, surface_area, anisotropy)

def _contacts(
  cnp.ndarray[INTEGER, ndim=3, cast=True] labels,
  int connectivity=26,
  surface_area=True,
  anisotropy=(1,1,1), 
):
  if connectivity == 8 and labels.shape[2] == 1:
    connectivity = 26
  if connectivity == 4 and labels.shape[2] == 1:
    connectivity = 6

  if connectivity not in (6, 18, 26):
    raise ValueError("Only 6, 18, and 26 connectivities are supported. Got: " + str(connectivity))

  labels = np.asfortranarray(labels)

  cdef unordered_map[cpp_pair[INTEGER,INTEGER], float, pair_hash] res = extract_region_graph(
    <INTEGER*>&labels[0,0,0],
    labels.shape[0], labels.shape[1], labels.shape[2],
    anisotropy[0], anisotropy[1], anisotropy[2],
    connectivity, surface_area
  )

  output = {}
  for pair in res:
    output[(pair.first.first, pair.first.second)] = pair.second

  return output

## These below functions are concerned with fast rendering
## of a densely labeled image into a series of binary images.

@cython.binding(True)
def runs(labels:np.ndarray):
  """
  Returns a dictionary describing where each label is located.
  Use this data in conjunction with render and erase.
  """
  return _runs(_reshape(labels, (labels.size,)))

def _runs(
    cnp.ndarray[UINT, ndim=1, cast=True] labels
  ):
  if labels.dtype in (np.uint8, bool):
    return extract_runs[uint8_t](<uint8_t*>&labels[0], labels.size)
  elif labels.dtype == np.uint16:
    return extract_runs[uint16_t](<uint16_t*>&labels[0], labels.size)
  elif labels.dtype == np.uint32:
    return extract_runs[uint32_t](<uint32_t*>&labels[0], labels.size)
  elif labels.dtype == np.uint64:
    return extract_runs[uint64_t](<uint64_t*>&labels[0], labels.size)
  else:
    raise TypeError("Unsupported type: " + str(labels.dtype))

def draw(
  label:np.ndarray, 
  vector[cpp_pair[size_t, size_t]] runs,
  image:np.ndarray
) -> np.ndarray:
  """
  Draws label onto the provided image according to 
  runs.
  """
  return _draw(label, runs, _reshape(image, (image.size,)))

def _draw( 
  label, 
  vector[cpp_pair[size_t, size_t]] runs,
  cnp.ndarray[UINT, ndim=1, cast=True] image
):
  if image.dtype == bool:
    set_run_voxels[uint8_t](label != 0, runs, <uint8_t*>&image[0], image.size)
  elif image.dtype == np.uint8:
    set_run_voxels[uint8_t](label, runs, <uint8_t*>&image[0], image.size)
  elif image.dtype == np.uint16:
    set_run_voxels[uint16_t](label, runs, <uint16_t*>&image[0], image.size)
  elif image.dtype == np.uint32:
    set_run_voxels[uint32_t](label, runs, <uint32_t*>&image[0], image.size)
  elif image.dtype == np.uint64:  
    set_run_voxels[uint64_t](label, runs, <uint64_t*>&image[0], image.size)
  else:
    raise TypeError("Unsupported type: " + str(image.dtype))

  return image

@cython.embedsignature(True)
def _erase( 
  vector[cpp_pair[size_t, size_t]] runs, 
  image:np.ndarray
) -> np.ndarray:
  """
  Erases (sets to 0) part of the provided image according to 
  runs.
  """
  return draw(0, runs, image)

@cython.binding(True)
def each(
  labels:np.ndarray, 
  binary:bool = False, 
  in_place:bool = False
) -> Iterator[Tuple[int, np.ndarray]]:
  """
  Returns an iterator that extracts each label from a dense labeling.

  binary: create a binary image from each component (otherwise use the
    same dtype and label value for the mask)
  in_place: much faster but the resulting image will be read-only

  Example:
  for label, img in cc3d.each(labels, binary=False, in_place=False):
    process(img)

  Returns: iterator
  """
  all_runs = runs(labels)
  order = 'F' if labels.flags['F_CONTIGUOUS'] else 'C'

  dtype = labels.dtype
  if binary:
    dtype = bool

  class ImageIterator():
    def __len__(self):
      return len(all_runs) - int(0 in all_runs)
    def __iter__(self):
      for key, rns in all_runs.items():
        if key == 0:
          continue
        img = np.zeros(labels.shape, dtype=dtype, order=order)
        draw(key, rns, img)
        yield (key, img)

  class InPlaceImageIterator(ImageIterator):
    def __iter__(self):
      img = np.zeros(labels.shape, dtype=dtype, order=order)
      for key, rns in all_runs.items():
        if key == 0:
          continue
        draw(key, rns, img)
        img.setflags(write=0)
        yield (key, img)
        img.setflags(write=1)
        _erase(rns, img)

  if in_place:
    return InPlaceImageIterator()
  return ImageIterator()
