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

def _view_as_unsigned(img:np.ndarray) -> np.ndarray:
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


class DisjointSet:
  def __init__(self):
    self.data = {} 
  def makeset(self, x):
    self.data[x] = x
    return x
  def find(self, x):
    if not x in self.data:
      return None
    i = self.data[x]
    while i != self.data[i]:
      self.data[i] = self.data[self.data[i]]
      i = self.data[i]
    return i
  def union(self, x, y):
    i = self.find(x)
    j = self.find(y)
    if i is None:
      i = self.makeset(x)
    if j is None:
      j = self.makeset(y)

    if i < j:
      self.data[j] = i
    else:
      self.data[i] = j

def connected_components_stack(
  stacked_images:Sequence[np.ndarray], 
  connectivity:int = 26,
  return_N:bool = False,
  out_dtype:Optional[Any] = None,
):
  """
  This is for performing connected component labeling
  on an array larger than RAM.

  stacked_images is a sequence of 3D images that are of equal
  width and height (x,y) and arbitrary depth (z). For example,
  you might define a generator that produces a tenth of your
  data at a time. The data must be sequenced in z order from
  z = 0 to z = depth - 1.

  Each 3D image will have CCL run on it and then compressed
  into crackle format (https://github.com/seung-lab/crackle)
  which is highly compressed but still usable and randomly
  accessible by z-slice. 

  The bottom previous slice and top current
  slice will be analyzed to produce a merged image.

  The final output will be a CrackleArray. You
  can access parts of the image using standard array
  operations, write the array data to disk using arr.binary
  or fully decompressing the array using arr.decompress()
  to obtain a numpy array (but presumably this will blow
  out your RAM since the image is so big).
  """
  try:
    import crackle
    import fastremap
  except ImportError:
    print("You need to pip install connected-components-3d[stack]")
    raise

  full_binary = None
  bottom_cc_img = None
  bottom_cc_labels = None

  if connectivity not in (6,26):
    raise ValueError(f"Connectivity must be 6 or 26. Got: {connectivity}")

  offset = 0

  for image in stacked_images:
    cc_labels, N = connected_components(
      image, connectivity=connectivity,
      return_N=True, out_dtype=np.uint64,
    )
    cc_labels[cc_labels != 0] += offset
    offset += N
    binary = crackle.compress(cc_labels)

    if full_binary is None:
      full_binary = binary
      bottom_cc_img = image[:,:,-1]
      bottom_cc_labels = cc_labels[:,:,-1]
      continue

    top_cc_labels = cc_labels[:,:,0]

    equivalences = DisjointSet()

    buniq = fastremap.unique(bottom_cc_labels)
    tuniq = fastremap.unique(top_cc_labels)

    for u in buniq:
      equivalences.makeset(u)
    for u in tuniq:
      equivalences.makeset(u)

    if connectivity == 6:
      for y in range(image.shape[1]):
        for x in range(image.shape[0]):
          if bottom_cc_labels[x,y] == 0 or top_cc_labels[x,y] == 0:
            continue
          if bottom_cc_img[x,y] == image[x,y,0]:
            equivalences.union(bottom_cc_labels[x,y], top_cc_labels[x,y])
    else:
      for y in range(image.shape[1]):
        for x in range(image.shape[0]):
          if bottom_cc_labels[x,y] == 0:
            continue

          for y0 in range(max(y - 1, 0), min(y + 1, image.shape[1] - 1) + 1):
            for x0 in range(max(x - 1, 0), min(x + 1, image.shape[0] - 1) + 1):
              if top_cc_labels[x0,y0] == 0:
                continue
              
              if bottom_cc_img[x,y] == image[x0,y0,0]:
                equivalences.union(
                  bottom_cc_labels[x,y], top_cc_labels[x0,y0]
                )
    
    relabel = {}
    for u in buniq:
      relabel[int(u)] = int(equivalences.find(u))
    for u in tuniq:
      relabel[int(u)] = int(equivalences.find(u))

    full_binary = crackle.zstack([
      full_binary,
      binary,
    ])
    full_binary = crackle.remap(full_binary, relabel, preserve_missing_labels=True)

    bottom_cc_img = image[:,:,-1]
    bottom_cc_labels = cc_labels[:,:,-1]
    bottom_cc_labels = fastremap.remap(bottom_cc_labels, relabel, preserve_missing_labels=True)

  if crackle.contains(full_binary, 0):
    start = 0
  else:
    start = 1

  full_binary, mapping = crackle.renumber(full_binary, start=start)
  arr = crackle.CrackleArray(full_binary).refit()

  if return_N:
    return arr, arr.num_labels()
  else:
    return arr



