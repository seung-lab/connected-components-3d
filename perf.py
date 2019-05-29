import cc3d
from tqdm import tqdm
import scipy.ndimage.measurements
import numpy as np

labels = np.random.randint(0,100, (512,512,512), dtype=np.uint8)
uniques = np.unique(labels)[1:]

def cc3d_test():
  res = cc3d.connected_components(labels)
  for segid in tqdm(uniques):
    extracted = res[labels == segid]

def ndimage_test():
  s = [
    [[1,1,1], [1,1,1], [1,1,1]],
    [[1,1,1], [1,1,1], [1,1,1]],
    [[1,1,1], [1,1,1], [1,1,1]]
  ]

  for segid in tqdm(uniques):
    extracted = (labels == segid)
    res = scipy.ndimage.measurements.label(labels, structure=s)

# ndimage_test()
# cc3d_test()