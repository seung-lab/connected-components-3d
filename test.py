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

## DEBUG:

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time
from scipy.spatial import ConvexHull, convex_hull_plot_2d

for region in range(5,6):
    #further examine component 3 (random choice)
    # find coordinates of points that belong to component three
    idx_compThree = np.argwhere(labels_out==region)
    # debug: print indices and check that all points are labeld as three
    print(idx_compThree)
    cods = np.array([idx_compThree[:,0],idx_compThree[:,1],idx_compThree[:,2]]).transpose()
    print(cods)

    # debug: plot points as 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(cods[:,0],cods[:,1],cods[:,2])
    plt.show()
    hull = ConvexHull(cods)
    print(hull)

#idx = np.unravel_index(unique_inverse[:15], (200,500,500))

#print(labels_out[idx])

# for segid in range(1, N+1):
#   extracted_image = labels_out * (labels_out == segid)
#   print("Showing extracted label " + str(segid) +":" )
#   print(extracted_image)
