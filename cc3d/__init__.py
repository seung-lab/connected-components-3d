from typing import (
  Dict, Union, Tuple, Iterator, 
  Sequence, Optional, Any, BinaryIO
)

import fastcc3d
from fastcc3d import (
  connected_components,
  statistics,
  each,
  contacts,
  region_graph,
  voxel_connectivity_graph,
  color_connectivity_graph,
  estimate_provisional_labels,
)

import numpy as np

def dust(
  img:np.ndarray, 
  threshold:Union[int,float], 
  connectivity:int = 26,
  in_place:bool = False,
) -> np.ndarray:
  """
  Remove from the input image connected components
  smaller than threshold ("dust"). The name of the function
  can be read as a verb "to dust" the image.

  img: 2D or 3D image
  threshold: discard components smaller than this in voxels
  connectivity: cc3d connectivity to use
  in_place: whether to modify the input image or perform
    dust 

  Returns: dusted image
  """
  orig_dtype = img.dtype
  img = _view_as_unsigned(img)

  if not in_place:
    img = np.copy(img)

  cc_labels, N = connected_components(
    img, connectivity=connectivity, return_N=True
  )
  stats = statistics(cc_labels)
  mask_sizes = stats["voxel_counts"]
  del stats

  to_mask = [ 
    i for i in range(1, N+1) if mask_sizes[i] < threshold 
  ]

  if len(to_mask) == 0:
    return img

  mask = np.isin(cc_labels, to_mask)
  del cc_labels
  np.logical_not(mask, out=mask)
  np.multiply(img, mask, out=img)
  return img.view(orig_dtype)

def largest_k(
  img:np.ndarray,
  k:int,
  connectivity:int = 26,
  delta:Union[int,float] = 0,
  return_N:bool = False,
) -> np.ndarray:
  """
  Returns the k largest connected components
  in the image.
  """
  assert k >= 0

  order = "C" if img.flags.c_contiguous else "F"

  if k == 0:
    return np.zeros(img.shape, dtype=np.uint16, order=order)

  cc_labels, N = connected_components(
    img, connectivity=connectivity, 
    return_N=True, delta=delta,
  )
  if N <= k:
    if return_N:
      return cc_labels, N
    return cc_labels

  cts = statistics(cc_labels)["voxel_counts"]  
  preserve = [ (i,ct) for i,ct in enumerate(cts) if i > 0 ]
  preserve.sort(key=lambda x: x[1])
  preserve = [ x[0] for x in preserve[-k:] ]

  shape, dtype = cc_labels.shape, cc_labels.dtype
  rns = fastcc3d.runs(cc_labels)

  order = "C" if cc_labels.flags.c_contiguous else "F"
  del cc_labels
  
  cc_out = np.zeros(shape, dtype=dtype, order=order)
  for i, label in enumerate(preserve):
    fastcc3d.draw(i+1, rns[label], cc_out)
  
  if return_N:
    return cc_out, len(preserve)
  return cc_out

def _view_as_unsigned(img:np.ndarray):
  if np.issubdtype(img.dtype, np.unsignedinteger) or img.dtype == bool:
    return img
  elif img.dtype == np.int8:
    return img.view(np.uint8)
  elif img.dtype == np.int16:
    return img.view(np.uint16)
  elif img.dtype == np.int32:
    return img.view(np.uint32)
  elif img.dtype == np.int64:
    return img.view(np.uint64)

  return img
