import cc3d
import numpy as np
from dataIO import ReadH5File
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d

global x_start
x_start = 500
global x_end
x_end = 700
global x_size
x_size = x_end-x_start

global y_start
y_start = 1500
global y_end
y_end = 2000
global y_size
y_size = y_end-y_start

global z_start
z_start = 1500
global z_end
z_end = 2000
global z_size
z_size = z_end-z_start

def is2D(points):
    if all_same(points[:,0]) or all_same(points[:,1]) or all_same(points[:,2]):
        return True
    else:
        return False

def all_same(items):
    return all(x == items[0] for x in items)

def convHull2D(points):
    if all_same(points[:,0]):
        cods = np.array([points[:,1],points[:,2]]).transpose()
        hull = ConvexHull(cods)
        boundary_idx = np.unique(hull.simplices)
        boundary_pts_cods = points[boundary_idx,:]

    if all_same(points[:,1]):
        cods = np.array([points[:,0],points[:,2]]).transpose()
        hull = ConvexHull(cods)
        boundary_idx = np.unique(hull.simplices)
        boundary_pts_cods = points[boundary_idx,:]

    if all_same(points[:,2]):
        cods = np.array([points[:,0],points[:,1]]).transpose()
        hull = ConvexHull(cods)
        boundary_idx = np.unique(hull.simplices)
        boundary_pts_cods = points[boundary_idx,:]

    return boundary_pts_cods

def convHull3D(points):
    hull = ConvexHull(points)
    boundary_idx = np.unique(hull.simplices)
    boundary_pts_cods = points[boundary_idx,:]

    return boundary_pts_cods

def getadjcomp(p):

    # set label to -1 if outside of boundary (needed for whole detection)
    comp = np.zeros((1,6))

    # store component number that is adjacend
    if (p[0]+1 < x_size): comp[0,0] = labels_in[p[0]+1,p[1],p[2]]
    else: comp[0,0] = -1
    if (p[0]-1 > 0): comp[0,1] = labels_in[p[0]-1,p[1],p[2]]
    else: comp[0,1] = -1
    if (p[1]+1 < y_size): comp[0,2] = labels_in[p[0],p[1]+1,p[2]]
    else: comp[0,2] = -1
    if (p[1]-1 > 0): comp[0,3] = labels_in[p[0],p[1]-1,p[2]]
    else: comp[0,3] = -1
    if (p[2]+1 < z_size): comp[0,4] = labels_in[p[0],p[1],p[2]+1]
    else: comp[0,4] = -1
    if (p[2]-1 > 0): comp[0,5] = labels_in[p[0],p[1],p[2]-1]
    else: comp[0,5] = -1

    return comp

def checkifwhole(boundary_pts_cods):

    isWhole = False
    adjComp = np.zeros((6, boundary_pts_cods.shape[0]))
    counter = 0

    for p in boundary_pts_cods:
        adjComp[:,counter] = getadjcomp(p)
        counter = counter + 1

    if -1 in adjComp:
        isWhole = False
        print("Not a Whole, connected to Boundary!")

    elif len(np.unique(adjComp))==2:
        isWhole = True
        print("Whole detected!")

    elif len(np.unique(adjComp))==1:
        print("Error, this connected component was detected wrong!")

    else:
        print("This connected component is not a whole (>2 neighbors)!")



    #TODO check if only one adjacent component exists and if so, classify as whole


data_in = ReadH5File("/home/frtim/wiring/raw_data/segmentations/JWR/cell032_downsampled.h5")

print("data was read in; shape: " + str(data_in.shape) + "; DataType is: " + str(data_in.dtype))

# labels_in = np.ones((3,3,3), dtype=np.int32)
# labels_in[1,1,1]=3
# labels_in[2,2,2]=4

labels_in = 1 - data_in[x_start:x_end,y_start:y_end,z_start:z_end]
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


for region in range(2,50):
    print("Loading component " + str(region) +"...")
    #further examine component 3 (random choice)
    # find coordinates of points that belong to component three
    idx_compThree = np.argwhere(labels_out==region)
    # debug: print indices and check that all points are labeld as three
    #print(idx_compThree)
    cods = np.array([idx_compThree[:,0],idx_compThree[:,1],idx_compThree[:,2]]).transpose()
    #print(cods)

    # debug: plot points as 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(cods[:,0],cods[:,1],cods[:,2],c='b')

    if is2D(cods):
        boundary_pts_cods = convHull2D(cods)
    else:
        boundary_pts_cods = convHull3D(cods)

    checkifwhole(boundary_pts_cods)

    ax.scatter(boundary_pts_cods[:,0],boundary_pts_cods[:,1],boundary_pts_cods[:,2],c='r')

    plt.show()

#idx = np.unravel_index(unique_inverse[:15], (200,500,500))

#print(labels_out[idx])

# for segid in range(1, N+1):
#   extracted_image = labels_out * (labels_out == segid)
#   print("Showing extracted label " + str(segid) +":" )
#   print(extracted_image)
