"""
Cython binding for connected components applied to 3D images
with 26-connectivity and handling for multiple labels.

Author: William Silversmith
Affiliation: Seung Lab, Princeton Neuroscience Institute
Date: August 2018
"""

from libc.stdlib cimport calloc, free
from libc.stdint cimport (
  int8_t, int16_t, int32_t, int64_t,
  uint8_t, uint16_t, uint32_t, uint64_t,
)
from cpython cimport array 
import array
import sys

from libcpp.vector cimport vector
cimport numpy as cnp
import numpy as np

__VERSION__ = '1.0.6'

cdef extern from "cc3d.hpp" namespace "cc3d":
  cdef uint32_t* connected_components3d[T](
    T* in_labels, 
    int sx, int sy, int sz,
    int64_t max_labels
  )

def connected_components(data, int64_t max_labels=-1):
  """
  Connected components applied to 3D images
  with 26-connectivity and handling for multiple labels.

  Required:
    data: Input weights in a 2D or 3D numpy array. 
  Optional:
    max_labels (int): save memory by predicting the maximum
      number of possible labels that might be output.
      Defaults to number of voxels.
  
  Returns: 2D or 3D numpy array remapped to reflect
    the connected components.
  """
  dims = len(data.shape)
  assert dims in (1, 2, 3)

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

  # The default C order of 4D numpy arrays is (channel, depth, row, col)
  # col is the fastest changing index in the underlying buffer. 
  # fpzip expects an XYZC orientation in the array, namely nx changes most rapidly. 
  # Since in this case, col is the most rapidly changing index, 
  # the inputs to fpzip should be X=col, Y=row, Z=depth, F=channel
  # If the order is F, the default array shape is fine.
  if order == 'C':
    shape.reverse()

  cdef int sx = shape[0]
  cdef int sy = shape[1]
  cdef int sz = shape[2]

  cdef uint8_t[:,:,:] arr_memview8u
  cdef uint16_t[:,:,:] arr_memview16u
  cdef uint32_t[:,:,:] arr_memview32u
  cdef uint64_t[:,:,:] arr_memview64u
  cdef int8_t[:,:,:] arr_memview8
  cdef int16_t[:,:,:] arr_memview16
  cdef int32_t[:,:,:] arr_memview32
  cdef int64_t[:,:,:] arr_memview64

  cdef uint32_t* labels 
  cdef uint64_t voxels = <uint64_t>sx * <uint64_t>sy * <uint64_t>sz

  if max_labels <= 0:
    max_labels = voxels

  dtype = data.dtype
  
  if dtype == np.uint64:
    arr_memview64u = data
    labels = connected_components3d[uint64_t](
      &arr_memview64u[0,0,0],
      sx, sy, sz, max_labels
    )
  elif dtype == np.uint32:
    arr_memview32u = data
    labels = connected_components3d[uint32_t](
      &arr_memview32u[0,0,0],
      sx, sy, sz, max_labels
    )
  elif dtype == np.uint16:
    arr_memview16u = data
    labels = connected_components3d[uint16_t](
      &arr_memview16u[0,0,0],
      sx, sy, sz, max_labels
    )
  elif dtype in (np.uint8, np.bool):
    arr_memview8u = data.astype(np.uint8)
    labels = connected_components3d[uint8_t](
      &arr_memview8u[0,0,0],
      sx, sy, sz, max_labels
    )
  elif dtype == np.int64:
    arr_memview64 = data
    labels = connected_components3d[int64_t](
      &arr_memview64[0,0,0],
      sx, sy, sz, max_labels
    )
  elif dtype == np.int32:
    arr_memview32 = data
    labels = connected_components3d[int32_t](
      &arr_memview32[0,0,0],
      sx, sy, sz, max_labels
    )
  elif dtype == np.int16:
    arr_memview16 = data
    labels = connected_components3d[int16_t](
      &arr_memview16[0,0,0],
      sx, sy, sz, max_labels
    )
  elif dtype == np.int8:
    arr_memview8 = data
    labels = connected_components3d[int8_t](
      &arr_memview8[0,0,0],
      sx, sy, sz, max_labels
    )
  else:
    raise TypeError("Type {} not currently supported.".format(dtype))

  cdef uint32_t[:] labels_view = <uint32_t[:voxels]>labels

  # This construct is required by python 2.
  # Python 3 can just do np.frombuffer(vec_view, ...)
  buf = bytearray(labels_view[:])
  free(labels)

  output = np.frombuffer(buf, dtype=np.uint32)

  if dims == 3:
    if order == 'C':
      return output.reshape( (sz, sy, sx), order=order)
    else:
      return output.reshape( (sx, sy, sz), order=order)
  elif dims == 2:
    if order == 'C':
      return output.reshape( (sy, sx), order=order)
    else:
      return output.reshape( (sx, sy), order=order)
  else:
    return output.reshape( (sx), order=order)