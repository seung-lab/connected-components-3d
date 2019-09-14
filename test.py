import cc3d
import numpy as np
from dataIO import ReadH5File

data_in = ReadH5File("/home/frtim/wiring/raw_data/segmentations/JWR/cell032_downsampled.h5")

print("data was read in; shape: " + str(data_in.shape) + "; DataType is: " + str(data_in.dtype))

# labels_in = np.ones((3,3,3), dtype=np.int32)
# labels_in[1,1,1]=3
# labels_in[2,2,2]=4

labels_in = 1 - data_in[500:700,1500:2000,1500:2000]
print("lables in created! Size is: " + str(labels_in.shape) + " maximum is: " + str(np.max(labels_in)))


# print("Labels in:")
# print(labels_in)

connectivity = 6 # only 26, 18, and 6 are allowed
labels_out = cc3d.connected_components(labels_in, connectivity=connectivity)
print(labels_out.shape)

# print("Labels out:")
# print(labels_out)

# You can extract individual components like so:
N = np.max(labels_out)
print("Conntected Regions found: " + str(N))

# determine indices, numbers and counts for the connected regions
unique, unique_inverse, counts = np.unique(labels_out, return_counts=True, return_inverse = True)
print(dict(zip(unique, counts)))

#further examine component 3 (random choice)
idx_compThree = np.argwhere(labels_out==3)
#idx_conpThree = np.unravel_index(idx_componentThree, (200,500,500))

print(idx_compThree)
print(labels_out[idx_compThree])

#idx = np.unravel_index(unique_inverse[:15], (200,500,500))

#print(labels_out[idx])

# for segid in range(1, N+1):
#   extracted_image = labels_out * (labels_out == segid)
#   print("Showing extracted label " + str(segid) +":" )
#   print(extracted_image)
