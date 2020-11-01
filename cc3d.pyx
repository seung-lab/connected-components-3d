"""
Cython binding for connected components applied to 3D images
with 26-connectivity and handling for multiple labels.

Author: William Silversmith
Affiliation: Seung Lab, Princeton Neuroscience Institute
Date: August 2018 - October 2020

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

import operator
from functools import reduce

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

__VERSION__ = '2.0.0'

cdef extern from "cc3d.hpp" namespace "cc3d":
  cdef uint32_t* connected_components3d[T,U](
    T* in_labels, 
    int64_t sx, int64_t sy, int64_t sz,
    int64_t max_labels, int64_t connectivity,
    U* out_labels, size_t &N
  )
  cdef size_t zeroth_pass[T](
    T* in_labels, int64_t sx, int64_t voxels
  )

cdef extern from "cc3d_graphs.hpp" namespace "cc3d":
  cdef OUT* extract_voxel_connectivity_graph[T,OUT](
    T* in_labels, 
    int64_t sx, int64_t sy, int64_t sz,
    int64_t connectivity, OUT *graph
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

# from https://github.com/seung-lab/fastremap/blob/master/fastremap.pyx
def reshape(arr, shape, order=None):
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

def compute_zeroth_pass(data):
  cdef uint8_t[:] arr_memview8u
  cdef uint16_t[:] arr_memview16u
  cdef uint32_t[:] arr_memview32u
  cdef uint64_t[:] arr_memview64u

  dtype = data.dtype
  sx = data.shape[0]
  data = reshape(data, (data.size,))

  if dtype in (np.uint64, np.int64):
    arr_memview64u = data.view(np.uint64)
    return zeroth_pass[uint64_t](&arr_memview64u[0], sx, data.size)
  elif dtype in (np.uint32, np.int32):
    arr_memview32u = data.view(np.uint32)
    return zeroth_pass[uint32_t](&arr_memview32u[0], sx, data.size)
  elif dtype in (np.uint16, np.int16):
    arr_memview16u = data.view(np.uint16)
    return zeroth_pass[uint16_t](&arr_memview16u[0], sx, data.size)
  elif dtype in (np.uint8, np.int8, np.bool):
    arr_memview8u = data.view(np.uint8)
    return zeroth_pass[uint8_t](&arr_memview8u[0], sx, data.size)
  else:
    raise TypeError("Type {} not currently supported.".format(dtype))

def connected_components(
  data, int64_t max_labels=-1, 
  int64_t connectivity=26, bool zeroth_pass=True,
  bool return_N=False
):
  """
  ndarray connected_components(
    data, max_labels=-1, 
    connectivity=26, zeroth_pass=True,
    return_N=False
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
    zeroth_pass (bool): if True, perform a preliminary pass to
      compute an estimate of the number of provisional labels.

      The hope is that this extra pass will reduce memory usage 
      and improve execution time by avoiding unnecesary
      memory initializations.

      For some high performance situations with known quantities, it 
      may make more sense to provide a manual estimate of max_labels 
      and switch this off, but unless you know what you're doing keep
      this enabled.

      The name "zeroth pass" is in reference to the Rosenfeld and Pfaltz
      two-pass scheme which consists of a scan for equivalences and then
      a second pass for relabeling that this CCL variant is derived from.
    return_N (bool): if True, also return the number of connected components
      as the second argument of a return tuple.

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
  cdef int dims = len(data.shape)
  if dims not in (1,2,3):
    raise DimensionError("Only 1D, 2D, and 3D arrays supported. Got: " + str(dims))

  if dims == 2 and connectivity not in (4, 8, 6, 18, 26):
    raise ValueError("Only 4, 8, and 6, 18, 26 connectivities are supported for 2D images. Got: " + str(connectivity))
  elif dims != 2 and connectivity not in (6, 18, 26):
    raise ValueError("Only 6, 18, and 26 connectivities are supported for 3D images. Got: " + str(connectivity))

  if data.size == 0:
    return np.zeros(shape=(0,), dtype=data.dtype)

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

  cdef int64_t voxels = <int64_t>sx * <int64_t>sy * <int64_t>sz
  cdef cnp.ndarray[uint16_t, ndim=1] out_labels16 = np.array([], dtype=np.uint16)
  cdef cnp.ndarray[uint32_t, ndim=1] out_labels32 = np.array([], dtype=np.uint32)
  cdef cnp.ndarray[uint64_t, ndim=1] out_labels64 = np.array([], dtype=np.uint64)

  if max_labels <= 0:
    max_labels = voxels
  max_labels = min(max_labels, voxels)

  if zeroth_pass:
    max_labels = min(max_labels, compute_zeroth_pass(data) + 1)

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

  if max_labels < np.iinfo(np.uint16).max:
    out_dtype = np.uint16
  elif max_labels < np.iinfo(np.uint32).max:
    out_dtype = np.uint32
  else:
    out_dtype = np.uint64

  if out_dtype == np.uint16:
    out_labels16 = np.zeros( (voxels,), dtype=out_dtype, order='C' )
    out_labels = out_labels16
  elif out_dtype == np.uint32:
    out_labels32 = np.zeros( (voxels,), dtype=out_dtype, order='C' )
    out_labels = out_labels32
  elif out_dtype == np.uint64:
    out_labels64 = np.zeros( (voxels,), dtype=out_dtype, order='C' )
    out_labels = out_labels64

  dtype = data.dtype

  cdef size_t N = 0
  
  if dtype in (np.uint64, np.int64):
    arr_memview64u = data.view(np.uint64)
    if out_dtype == np.uint16:
      connected_components3d[uint64_t, uint16_t](
        &arr_memview64u[0,0,0],
        sx, sy, sz, max_labels, connectivity,
        <uint16_t*>&out_labels16[0], N
      )
    elif out_dtype == np.uint32:
      connected_components3d[uint64_t, uint32_t](
        &arr_memview64u[0,0,0],
        sx, sy, sz, max_labels, connectivity,
        <uint32_t*>&out_labels32[0], N
      )
    elif out_dtype == np.uint64:
      connected_components3d[uint64_t, uint64_t](
        &arr_memview64u[0,0,0],
        sx, sy, sz, max_labels, connectivity,
        <uint64_t*>&out_labels64[0], N
      )
  elif dtype in (np.uint32, np.int32):
    arr_memview32u = data.view(np.uint32)
    if out_dtype == np.uint16:
      connected_components3d[uint32_t, uint16_t](
        &arr_memview32u[0,0,0],
        sx, sy, sz, max_labels, connectivity,
        <uint16_t*>&out_labels16[0], N
      )
    elif out_dtype == np.uint32:
      connected_components3d[uint32_t, uint32_t](
        &arr_memview32u[0,0,0],
        sx, sy, sz, max_labels, connectivity,
        <uint32_t*>&out_labels32[0], N
      )
    elif out_dtype == np.uint64:
      connected_components3d[uint32_t, uint64_t](
        &arr_memview32u[0,0,0],
        sx, sy, sz, max_labels, connectivity,
        <uint64_t*>&out_labels64[0], N
      )
  elif dtype in (np.uint16, np.int16):
    arr_memview16u = data.view(np.uint16)
    if out_dtype == np.uint16:
      connected_components3d[uint16_t, uint16_t](
        &arr_memview16u[0,0,0],
        sx, sy, sz, max_labels, connectivity,
        <uint16_t*>&out_labels16[0], N
      )
    elif out_dtype == np.uint32:
      connected_components3d[uint16_t, uint32_t](
        &arr_memview16u[0,0,0],
        sx, sy, sz, max_labels, connectivity,
        <uint32_t*>&out_labels32[0], N
      )
    elif out_dtype == np.uint64:
      connected_components3d[uint16_t, uint64_t](
        &arr_memview16u[0,0,0],
        sx, sy, sz, max_labels, connectivity,
        <uint64_t*>&out_labels64[0], N
      )
  elif dtype in (np.uint8, np.int8, np.bool):
    arr_memview8u = data.view(np.uint8)
    if out_dtype == np.uint16:
      connected_components3d[uint8_t, uint16_t](
        &arr_memview8u[0,0,0],
        sx, sy, sz, max_labels, connectivity,
        <uint16_t*>&out_labels16[0], N
      )
    elif out_dtype == np.uint32:
      connected_components3d[uint8_t, uint32_t](
        &arr_memview8u[0,0,0],
        sx, sy, sz, max_labels, connectivity,
        <uint32_t*>&out_labels32[0], N
      )
    elif out_dtype == np.uint64:
      connected_components3d[uint8_t, uint64_t](
        &arr_memview8u[0,0,0],
        sx, sy, sz, max_labels, connectivity,
        <uint64_t*>&out_labels64[0], N
      )
  else:
    raise TypeError("Type {} not currently supported.".format(dtype))

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

  if return_N:
    return (out_labels, N)
  return out_labels

def voxel_connectivity_graph(data, int64_t connectivity=26):
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
  elif dtype in (np.uint8, np.int8, np.bool):
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

def region_graph(
    cnp.ndarray[INTEGER, ndim=3, cast=True] labels,
    int connectivity=26
  ):
  """
  Get the N-connected region adjacancy graph of a 3D image.

  Supports 26, 18, and 6 connectivities.

  labels: 3D numpy array of integer segmentation labels
  connectivity: 6, 16, or 26 (default)

  Returns: set of edges between labels
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


