import cc3d
from tqdm import tqdm
import scipy.ndimage.measurements
import numpy as np
import fastremap
import time

def cc3d_mutlilabel_extraction(labels):
  res, N = cc3d.connected_components(labels, return_N=True)
  for label, extracted in tqdm(cc3d.each(labels, binary=True, in_place=True), total=N):
    pass
  # for segid in tqdm(range(1, N+1)):
  #   extracted = (res == segid)

def ndimage_mutlilabel_extraction(labels):
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

def pinky40subvol():
  from cloudvolume import CloudVolume
  return CloudVolume('gs://neuroglancer/wms/skeletonization/pinky40subvol', 
    mip=2, progress=True, cache=True)[:,:,:][...,0]



labels = pinky40subvol()
cc3d_mutlilabel_extraction(labels)
# ndimage_mutlilabel_extraction(labels)

# voxels = labels.size
# times = []
# # for label, img in tqdm(cc3d.each(labels, binary=False, in_place=True)):
# #   s = time.time()
# #   cc3d.connected_components(img)
# #   # scipy.ndimage.measurements.label(labels, structure=structure)
# #   e = time.time()
# #   dt = e-s
# #   times.append(dt)
# #   prettyprint(dt, voxels)

# for i in range(10):
#   s = time.time()
#   cc3d.connected_components(labels)#, connectivity=26)
#   # scipy.ndimage.measurements.label(labels)
#   e = time.time()
#   dt = e-s
#   times.append(dt)
#   prettyprint(dt, voxels)

# times = times[1:]
# mean = sum(times) / float(len(times))

# print("mean:")
# prettyprint(mean, voxels)
# print("stddev: %.2f" % np.std(times))