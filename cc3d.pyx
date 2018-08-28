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

__VERSION__ = '1.0.0'

cdef extern from "cc3d.hpp" namespace "cc3d":
  cdef uint16_t* connected_components3d[T](
    T* in_labels, 
    int sx, int sy, int sz
  )

def connected_components(data):
  """
  Parameters:
   Data: Input weights in a 2D or 3D numpy array. 
  
  Returns: 1D numpy array containing indices of the path from
    source to target including source and target.
  """
  dims = len(data.shape)
  assert dims in (2, 3)

  if data.size == 0:
    return np.zeros(shape=(0,), dtype=np.uint32)

  if dims == 2:
    data = data[:, :, np.newaxis]

  cdef int cols = data.shape[0]
  cdef int rows = data.shape[1]
  cdef int depth = data.shape[2]

  cdef uint8_t[:,:,:] arr_memview8u
  cdef uint16_t[:,:,:] arr_memview16u
  cdef uint32_t[:,:,:] arr_memview32u
  cdef uint64_t[:,:,:] arr_memview64u
  cdef int8_t[:,:,:] arr_memview8
  cdef int16_t[:,:,:] arr_memview16
  cdef int32_t[:,:,:] arr_memview32
  cdef int64_t[:,:,:] arr_memview64

  cdef uint16_t* labels 

  dtype = data.dtype
  
  if dtype == np.uint64:
    arr_memview64u = data
    labels = connected_components3d[uint64_t](
      &arr_memview64u[0,0,0],
      rows, cols, depth
    )
  elif dtype == np.uint32:
    arr_memview32u = data
    labels = connected_components3d[uint32_t](
      &arr_memview32u[0,0,0],
      rows, cols, depth
    )
  elif dtype == np.uint16:
    arr_memview16u = data
    labels = connected_components3d[uint16_t](
      &arr_memview16u[0,0,0],
      rows, cols, depth
    )
  elif dtype in (np.uint8, np.bool):
    arr_memview8u = data
    labels = connected_components3d[uint8_t](
      &arr_memview8u[0,0,0],
      rows, cols, depth
    )
  elif dtype == np.int64:
    arr_memview64 = data
    labels = connected_components3d[int64_t](
      &arr_memview64[0,0,0],
      rows, cols, depth
    )
  elif dtype == np.int32:
    arr_memview32 = data
    labels = connected_components3d[int32_t](
      &arr_memview32[0,0,0],
      rows, cols, depth
    )
  elif dtype == np.int16:
    arr_memview16 = data
    labels = connected_components3d[int16_t](
      &arr_memview16[0,0,0],
      rows, cols, depth
    )
  elif dtype == np.int8:
    arr_memview8 = data
    labels = connected_components3d[int8_t](
      &arr_memview8[0,0,0],
      rows, cols, depth
    )
  else:
    raise TypeError("Type {} not currently supported.".format(dtype))

  cdef int voxels = cols * rows * depth
  cdef uint16_t[:] labels_view = <uint16_t[:voxels]>labels

  # This construct is required by python 2.
  # Python 3 can just do np.frombuffer(vec_view, ...)
  buf = bytearray(labels_view[:])
  free(labels)
  order = 'F' if data.flags['F_CONTIGUOUS'] else 'C'
  return np.frombuffer(buf, dtype=np.uint16).reshape( (cols, rows, depth), order=order)