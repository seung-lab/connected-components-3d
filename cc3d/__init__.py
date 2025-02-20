from typing import (
  Dict, Union, Tuple, List, Iterator, 
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
  threshold:Union[int,float,Tuple[int,int],Tuple[float,float],List[int],List[float]], 
  connectivity:int = 26,
  in_place:bool = False,
  binary_image:bool = False,
  precomputed_ccl:bool = False,
  invert:bool = False,
  return_N:bool = False,
) -> np.ndarray:
  """
  Remove from the input image connected components
  smaller than threshold ("dust"). The name of the function
  can be read as a verb "to dust" the image.

  img: 2D or 3D image
  threshold: 
    (int) discard components smaller than this in voxels
    (tuple/list) discard components outside this range [lower, upper)
  connectivity: cc3d connectivity to use
  in_place: whether to modify the input image or perform
    dust 
  precomputed_ccl: for performance, avoid computing a CCL
    pass since the input is already a CCL output from this
    library.
  invert: switch the threshold direction. For scalar input,
    this means less than converts to greater than or equal to,
    for ranged input, switch from between to outside of range.

  Returns: dusted image
  """
  orig_dtype = img.dtype
  img = _view_as_unsigned(img)

  if not in_place:
    img = np.copy(img)

  if precomputed_ccl:
    cc_labels = img
    N = np.max(cc_labels)
  else:
    cc_labels, N = connected_components(
      img, connectivity=connectivity, 
      return_N=True, binary_image=bool(binary_image),
    )
  
  stats = statistics(cc_labels, no_slice_conversion=True)
  mask_sizes = stats["voxel_counts"]
  del stats

  if isinstance(threshold, (tuple, list)):
    to_mask = [ 
      i for i in range(1, N+1) if not (threshold[0] <= mask_sizes[i] < threshold[1])
    ]
  else:
    to_mask = [ 
      i for i in range(1, N+1) if mask_sizes[i] < threshold 
    ]

  if invert:
    dust_N = len(to_mask)
    if dust_N and to_mask[0] == 0:
      dust_N -= 1
  else:
    dust_N = N - len(to_mask)
    if len(to_mask) and to_mask[0] == 0:
      dust_N += 1

  if len(to_mask) == 0:
    if invert:
      img = np.zeros(img.shape, dtype=img.dtype, order="F")

    if return_N:
      return (img, dust_N)
    else:
      return img

  mask = np.isin(cc_labels, to_mask, assume_unique=True, invert=invert)
  del cc_labels
  img[mask] = 0
  img = img.view(orig_dtype)

  if return_N:
    return (img, dust_N)
  return img

def largest_k(
  img:np.ndarray,
  k:int,
  connectivity:int = 26,
  delta:Union[int,float] = 0,
  return_N:bool = False,
  binary_image:bool = False,
  precomputed_ccl:bool = False,
) -> np.ndarray:
  """
  Returns the k largest connected components
  in the image.

  k: number of components to return (>= 0)
  connectivity: 
    (2d) 4 [edges], 8 [edges+corners] 
    (3d) 6 [faces], 18 [faces+edges], or 26 [faces+edges+corners]
  delta: if using a continuous image, the allowed difference
    in adjacent voxel values
  return_N: return value is (image, N)
  binary_image: treat the input image as a binary image
  precomputed_ccl: for performance, avoid computing a CCL
    pass since the input is already a CCL output from this
    library.

  NOTE: Performance may increase if you have the fastremap
    library installed. This may also change the numbering
    of the output.
  """
  assert k >= 0

  order = "C" if img.flags.c_contiguous else "F"

  if k == 0:
    return np.zeros(img.shape, dtype=np.uint16, order=order)
  
  if precomputed_ccl:
    cc_labels = np.copy(img, order="F")
    N = np.max(cc_labels)
  else:
    cc_labels, N = connected_components(
      img, connectivity=connectivity, 
      return_N=True, delta=delta,
      binary_image=bool(binary_image),
    )

  if N <= k:
    if return_N:
      return cc_labels, N
    return cc_labels

  cts = statistics(cc_labels, no_slice_conversion=True)["voxel_counts"]  
  
  if k == 1:
    cc_out = (cc_labels == (np.argmax(cts[1:]) + 1))
    if return_N:
      return cc_out, 1
    return cc_out

  preserve = np.argpartition(cts[1:], len(cts) - k - 1)[-k:]
  preserve += 1
  preserve = [ (label,cts[label]) for label in preserve ]
  preserve.sort(key=lambda x: x[1])
  preserve = [ int(x[0]) for x in preserve ]

  try:
    import fastremap
    cc_out = fastremap.mask_except(cc_labels, preserve, in_place=True)
    fastremap.renumber(cc_out, in_place=True)
  except ImportError:
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
  binary_image:bool = False,
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
      binary_image=bool(binary_image),
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
          if ((not binary_image and bottom_cc_img[x,y] == image[x,y,0]) 
            or (binary_image and bottom_cc_img[x,y] and image[x,y,0])):

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
              
              if ((not binary_image and bottom_cc_img[x,y] == image[x0,y0,0])
                  or (binary_image and bottom_cc_img[x,y] and image[x0,y0,0])):

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



