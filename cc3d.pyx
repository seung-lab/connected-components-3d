"""
Cython binding for connected components applied to 3D images
with 26-connectivity and handling for multiple labels.

Author: William Silversmith
Affiliation: Seung Lab, Princeton Neuroscience Institute
Date: August 2018 - June 2019

---
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
---

If you received a copy of this program in binary form, you can get 
the source code for free here: 

https://github.com/seung-lab/connected-components-3d
"""

from libc.stdlib cimport calloc, free
from libc.stdint cimport (
  int8_t, int16_t, int32_t, int64_t,
  uint8_t, uint16_t, uint32_t, uint64_t,
)
from libcpp cimport bool
from cpython cimport array 
import array
import sys

from libcpp.vector cimport vector
cimport numpy as cnp
import numpy as np

__VERSION__ = '1.14.0'

cdef extern from "cc3d.hpp" namespace "cc3d":
  cdef uint32_t* connected_components3d[T,U](
    T* in_labels, 
    int64_t sx, int64_t sy, int64_t sz,
    int64_t max_labels, int64_t connectivity,
    U* out_labels, bool sparse
  )

  cdef vector[T] extract_region_graph[T](
    T* labels,
    int64_t sx, int64_t sy, int64_t sz,
    int64_t connectivity,
  )

ctypedef fused INTEGER:
  uint8_t
  uint16_t
  uint32_t
  uint64_t
  int8_t
  int16_t
  int32_t
  int64_t

class DimensionError(Exception):
  """The array has the wrong number of dimensions."""
  pass

cdef int64_t even_ceil(int64_t N):
  if N & 0x1:
    return N << 1
  return N

def connected_components(
  data, int64_t max_labels=-1, 
  int64_t connectivity=26, out_dtype=np.uint32,
  sparse=False
):
  """
  ndarray connected_components(
    data, int64_t max_labels=-1, 
    int64_t connectivity=26, out_dtype=np.uint32
  )

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
    out_dtype: Sets the output data type of the output.
    sparse: (bool) if the dataset is known to be sparse
      perform some optimizations that can either reduce memory
      or increase speed.
  
  Returns: 2D or 3D numpy array remapped to reflect
    the connected components.
  """
  dims = len(data.shape)
  if dims not in (1,2,3):
    raise DimensionError("Only 1D, 2D, and 3D arrays supported. Got: " + str(dims))

  out_dtype = np.dtype(out_dtype)
  if out_dtype not in (np.uint16, np.uint32, np.uint64):
    raise TypeError(
      """Only unsigned 16, 32, and 64 bit out data types are 
      supported through the python interface (C++ can handle any integer type)."""
    )

  if dims == 2 and connectivity not in (4, 8, 6, 18, 26):
    raise ValueError("Only 4, 8, and 6, 18, 26 connectivities are supported for 2D images. Got: " + str(connectivity))
  elif dims != 2 and connectivity not in (6, 18, 26):
    raise ValueError("Only 6, 18, and 26 connectivities are supported for 3D images. Got: " + str(connectivity))

  if data.size == 0:
    return np.zeros(shape=(0,), dtype=np.uint32)

  order = 'F' if data.flags['F_CONTIGUOUS'] else 'C'

  while len(data.shape) < 3:
    if order == 'C':
      data = data[np.newaxis, ...]
    else: # F
      data = data[..., np.newaxis ]

  if not data.flags['C_CONTIGUOUS'] and not data.flags['F_CONTIGUOUS']:
    data = np.copy(data, order=order)

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

  cdef uint64_t voxels = <uint64_t>sx * <uint64_t>sy * <uint64_t>sz
  cdef cnp.ndarray[uint16_t, ndim=1] out_labels16 = np.array([], dtype=np.uint16)
  cdef cnp.ndarray[uint32_t, ndim=1] out_labels32 = np.array([], dtype=np.uint32)
  cdef cnp.ndarray[uint64_t, ndim=1] out_labels64 = np.array([], dtype=np.uint64)

  if out_dtype == np.uint16:
    out_labels16 = np.zeros( (voxels,), dtype=out_dtype, order='C' )
    out_labels = out_labels16
  elif out_dtype == np.uint32:
    out_labels32 = np.zeros( (voxels,), dtype=out_dtype, order='C' )
    out_labels = out_labels32
  elif out_dtype == np.uint64:
    out_labels64 = np.zeros( (voxels,), dtype=out_dtype, order='C' )
    out_labels = out_labels64

  if max_labels <= 0:
    max_labels = voxels

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

  if data.dtype == np.bool:
    if connectivity in (4,6):
      max_labels = min(max_labels, (union_find_voxels // 2) + 1)
    elif connectivity == (8,18):
      max_labels = min(max_labels, (union_find_voxels // 4) + 1)
    else: # 26
      max_labels = min(max_labels, (union_find_voxels // 8) + 1)

  dtype = data.dtype
  
  if dtype in (np.uint64, np.int64):
    arr_memview64u = data.view(np.uint64)
    if out_dtype == np.uint16:
      connected_components3d[uint64_t, uint16_t](
        &arr_memview64u[0,0,0],
        sx, sy, sz, max_labels, connectivity,
        <uint16_t*>&out_labels16[0], <bool>sparse
      )
    elif out_dtype == np.uint32:
      connected_components3d[uint64_t, uint32_t](
        &arr_memview64u[0,0,0],
        sx, sy, sz, max_labels, connectivity,
        <uint32_t*>&out_labels32[0], <bool>sparse
      )
    elif out_dtype == np.uint64:
      connected_components3d[uint64_t, uint64_t](
        &arr_memview64u[0,0,0],
        sx, sy, sz, max_labels, connectivity,
        <uint64_t*>&out_labels64[0], <bool>sparse
      )
  elif dtype in (np.uint32, np.int32):
    arr_memview32u = data.view(np.uint32)
    if out_dtype == np.uint16:
      connected_components3d[uint32_t, uint16_t](
        &arr_memview32u[0,0,0],
        sx, sy, sz, max_labels, connectivity,
        <uint16_t*>&out_labels16[0], <bool>sparse
      )
    elif out_dtype == np.uint32:
      connected_components3d[uint32_t, uint32_t](
        &arr_memview32u[0,0,0],
        sx, sy, sz, max_labels, connectivity,
        <uint32_t*>&out_labels32[0], <bool>sparse
      )
    elif out_dtype == np.uint64:
      connected_components3d[uint32_t, uint64_t](
        &arr_memview32u[0,0,0],
        sx, sy, sz, max_labels, connectivity,
        <uint64_t*>&out_labels64[0], <bool>sparse
      )
  elif dtype in (np.uint16, np.int16):
    arr_memview16u = data.view(np.uint16)
    if out_dtype == np.uint16:
      connected_components3d[uint16_t, uint16_t](
        &arr_memview16u[0,0,0],
        sx, sy, sz, max_labels, connectivity,
        <uint16_t*>&out_labels16[0], <bool>sparse
      )
    elif out_dtype == np.uint32:
      connected_components3d[uint16_t, uint32_t](
        &arr_memview16u[0,0,0],
        sx, sy, sz, max_labels, connectivity,
        <uint32_t*>&out_labels32[0], <bool>sparse
      )
    elif out_dtype == np.uint64:
      connected_components3d[uint16_t, uint64_t](
        &arr_memview16u[0,0,0],
        sx, sy, sz, max_labels, connectivity,
        <uint64_t*>&out_labels64[0], <bool>sparse
      )
  elif dtype in (np.uint8, np.int8, np.bool):
    arr_memview8u = data.view(np.uint8)
    if out_dtype == np.uint16:
      connected_components3d[uint8_t, uint16_t](
        &arr_memview8u[0,0,0],
        sx, sy, sz, max_labels, connectivity,
        <uint16_t*>&out_labels16[0], <bool>sparse
      )
    elif out_dtype == np.uint32:
      connected_components3d[uint8_t, uint32_t](
        &arr_memview8u[0,0,0],
        sx, sy, sz, max_labels, connectivity,
        <uint32_t*>&out_labels32[0], <bool>sparse
      )
    elif out_dtype == np.uint64:
      connected_components3d[uint8_t, uint64_t](
        &arr_memview8u[0,0,0],
        sx, sy, sz, max_labels, connectivity,
        <uint64_t*>&out_labels64[0], <bool>sparse
      )
  else:
    raise TypeError("Type {} not currently supported.".format(dtype))

  if dims == 3:
    if order == 'C':
      return out_labels.reshape( (sz, sy, sx), order=order)
    else:
      return out_labels.reshape( (sx, sy, sz), order=order)
  elif dims == 2:
    if order == 'C':
      return out_labels.reshape( (sy, sx), order=order)
    else:
      return out_labels.reshape( (sx, sy), order=order)
  else:
    return out_labels.reshape( (sx), order=order)

def region_graph(
    cnp.ndarray[INTEGER, ndim=3, cast=True] labels,
    int connectivity=26
  ):
  """
  Get the N-connected region adjacancy graph of a 3D image.

  Supports 26, 18, and 6 connectivities.

  labels: 3D numpy array of integer segmentation labels
  connectivity: 6, 16, or 26 (default)

  Returns: set of edges
  """
  if connectivity not in (6, 18, 26):
    raise ValueError("Only 6, 18, and 26 connectivities are supported. Got: " + str(connectivity))

  labels = np.asfortranarray(labels)

  cdef vector[INTEGER] res = extract_region_graph(
    <INTEGER*>&labels[0,0,0],
    labels.shape[0], labels.shape[1], labels.shape[2],
    connectivity
  )

  output = set()
  cdef size_t i = 0

  for i in range(res.size() // 2):
    output.add(
      (res[i * 2], res[i*2 + 1])
    )

  return output


