import cc3d
from tqdm import tqdm
import scipy.ndimage.measurements
import numpy as np
import fastremap
import time

def prettyprint(tm, voxels):
  print("{:.2f} sec. {:.2f} MVx/sec".format(tm, voxels / tm / (1000*1000)))

def summary(times, voxels):
  times = times[1:]
  mean = sum(times) / float(len(times))

  print("mean:")
  prettyprint(mean, voxels)
  print("stddev: %.2f" % np.std(times))

structures = {
  6: [
    [[0,0,0], [0,1,0], [0,0,0]],
    [[0,1,0], [1,1,1], [0,1,0]],
    [[0,0,0], [0,1,0], [0,0,0]]
  ],
  18: [
    [[0,1,0], [1,1,1], [0,1,0]],
    [[1,1,1], [1,1,1], [1,1,1]],
    [[0,1,0], [1,1,1], [0,1,0]]
  ],
  26: [
    [[1,1,1], [1,1,1], [1,1,1]],
    [[1,1,1], [1,1,1], [1,1,1]],
    [[1,1,1], [1,1,1], [1,1,1]]
  ]
}

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

def test_binary_image_processing(labels, connectivity=26):
  voxels = labels.size
  times = []

  for label, img in tqdm(cc3d.each(labels, binary=True, in_place=True)):
    s = time.time()
    cc3d.connected_components(img)
    # scipy.ndimage.measurements.label(labels, structure=structures[connectivity])
    e = time.time()
    dt = e-s
    times.append(dt)
    prettyprint(dt, voxels)
  summary(times, voxels)

def test_multilabel_speed(labels, connectivity=26):
  voxels = labels.size
  times = []

  for i in range(10):
    s = time.time()
    cc3d.connected_components(labels, connectivity=connectivity)
    e = time.time()
    dt = e-s
    times.append(dt)
    prettyprint(dt, voxels)
  summary(times, voxels)

def test_continuous_speed(labels, connectivity=8):
  voxels = labels.size
  upper = np.max(labels)

  def _run(delta):
    print(f"Delta {delta}...")
    times = []
    for i in range(10):
      s = time.time()
      cc3d.connected_components(labels, connectivity=connectivity, delta=delta)
      e = time.time()
      dt = e-s
      times.append(dt)
    summary(times, voxels)

  for delta in range(0,int(upper),int(upper/10)):
    _run(delta)
  _run(upper)


# cc3d_mutlilabel_extraction(labels)
# ndimage_mutlilabel_extraction(labels)
# test_multilabel_speed(labels)

labels = np.load("Misc_pollen.npy").astype(np.float32)
test_continuous_speed(labels)



