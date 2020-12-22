import cc3d
from tqdm import tqdm
import scipy.ndimage.measurements
import numpy as np
import time
import fastremap

def cc3d_test_multilabel(labels):
  labels, remap = fastremap.renumber(labels)
  res = cc3d.connected_components(labels)
  N = np.max(labels)
  for segid in tqdm(range(1, N+1)):
    extracted = (res == segid)

def cc3d_test_multilabel_series(labels):
  res = cc3d.connected_components(labels)
  for label, img in tqdm(cc3d.series(res, in_place=True)):
    pass

def ndimage_test_multilabel(labels):
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

def ndimage_binary_image(labels, connectivity=26):
  if connectivity == 26:
    s = [
      [[1,1,1], [1,1,1], [1,1,1]],
      [[1,1,1], [1,1,1], [1,1,1]],
      [[1,1,1], [1,1,1], [1,1,1]]
    ]
  elif connectivity == 6:
    s = None
  else:
    raise ValueError("Unsupported connectivity. " + str(connectivity))

  for label in labels:
    scipy.ndimage.measurements.label(label, structure=s) # black

def cc3d_binary_image(labels, connectivity=26):
  for label in labels:
    cc3d.connected_components(label, connectivity=connectivity) # blue

# binary tests
s = time.time()
labels = [ 
  np.random.randint(0,2, size=(384, 384, 384), dtype=np.bool) 
  for _ in range(10)
]
print(time.time() - s)
s = time.time()
cc3d_binary_image(labels, 6)
# ndimage_binary_image(labels, 6)
print(time.time() - s)

# multilabel tests
# labels = np.random.randint(0,100, (512,512,512), dtype=np.uint8)
# ndimage_test(labels)
# cc3d_test(labels)