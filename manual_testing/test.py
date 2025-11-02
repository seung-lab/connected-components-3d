import cc3d
import numpy as np
import scipy.ndimage

from cloudvolume.lib import save_images
from cloudvolume import CloudVolume, view
import time

from PIL import Image

def twod():
  img = Image.open('test2d.png')
  return np.array(img)[:,:,0].T

def threed():
  labels = np.zeros( (480, 480, 3), dtype=np.uint8)
  tmp = Image.open('test3d-1.png')
  labels[:,:,0] = np.array(tmp)[:,:,0].T
  tmp = Image.open('test3d-2.png')
  labels[:,:,1] = np.array(tmp)[:,:,0].T
  tmp = Image.open('test3d-3.png')
  labels[:,:,2] = np.array(tmp)[:,:,0].T
  return labels


labels = twod()
# labels = threed()

print(np.max(labels))
labels = np.asfortranarray(labels)

start = time.time()
# labels = scipy.ndimage.measurements.label(labels)[0]
labels = cc3d.connected_components(labels)
print(time.time() - start, "sec")

print(np.unique(labels).shape)

view(labels, segmentation=True)

