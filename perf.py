import cc3d
from tqdm import tqdm
import scipy.ndimage.measurements
import numpy as np

import fastremap

def cc3d_test(labels):
  labels, remap = fastremap.renumber(labels)
  res = cc3d.connected_components(labels)
  N = np.max(labels)
  for segid in tqdm(range(1, N+1)):
    extracted = (res == segid)

def ndimage_test(labels):
  s = [
    [[1,1,1], [1,1,1], [1,1,1]],
    [[1,1,1], [1,1,1], [1,1,1]],
    [[1,1,1], [1,1,1], [1,1,1]]
  ]

  uniques = np.unique(labels)[1:]
  for segid in tqdm(uniques):
    extracted = (labels == segid)
    res, N = scipy.ndimage.measurements.label(extracted, structure=s)
    for ccid in tqdm(range(1,N+1)):
      extracted = (res == ccid)


labels = np.random.randint(0,100, (512,512,512), dtype=np.uint8)

ndimage_test(labels)
# cc3d_test(labels)