import cc3d
import numpy as np
from dataIO import ReadH5File
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import time

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
z_start = 1800
global z_end
z_end = 2000
global z_size
z_size = z_end-z_start

# input lables (0 is background, 1 = neuron1, (2 = neuron2,...))
# defined later:
# global labels

# this function checks if an array of points contains a 2D or a 3D structure
def is2D(points):
    if all_same(points[:,0]) or all_same(points[:,1]) or all_same(points[:,2]):
        return True
    else:
        return False

# function to check if all elements in items have the same value
def all_same(items):
    return all(x == items[0] for x in items)

# fits a convex hull to a 2D point object and returns the coordinates of the points that describe the border
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

# fits a convex hull to a 3D point object and returns the coordinates of the points that describe the hull surface
def convHull3D(points):
    hull = ConvexHull(points)
    boundary_idx = np.unique(hull.simplices)
    boundary_pts_cods = points[boundary_idx,:]

    return boundary_pts_cods

# returns the 6 adjacent components for a point, adjacent component is -1 if out of boundary
def getadjcomp(p):

    # set label to -1 if outside of boundary (needed for whole detection)
    comp = np.zeros((1,6))

    # store component number that is adjacend
    if (p[0]+1 < x_size): comp[0,0] = labels[p[0]+1,p[1],p[2]]
    else: comp[0,0] = -1
    if (p[0]-1 > 0): comp[0,1] = labels[p[0]-1,p[1],p[2]]
    else: comp[0,1] = -1
    if (p[1]+1 < y_size): comp[0,2] = labels[p[0],p[1]+1,p[2]]
    else: comp[0,2] = -1
    if (p[1]-1 > 0): comp[0,3] = labels[p[0],p[1]-1,p[2]]
    else: comp[0,3] = -1
    if (p[2]+1 < z_size): comp[0,4] = labels[p[0],p[1],p[2]+1]
    else: comp[0,4] = -1
    if (p[2]-1 > 0): comp[0,5] = labels[p[0],p[1],p[2]-1]
    else: comp[0,5] = -1

    return comp

# checks the adjacent coponents of an array of boundary points and applies rules to check if whole (see code)
def checkifwhole(boundary_pts_cods):

    isWhole = False
    adjComp = np.zeros((6, boundary_pts_cods.shape[0]))
    counter = 0
    connectedNeuron = -1

    for p in boundary_pts_cods:
        adjComp[:,counter] = getadjcomp(p)
        counter = counter + 1

    if -1 in adjComp:
        isWhole = False
        print("Not a Whole, connected to Boundary!")

    elif len(np.unique(adjComp))==2:
        isWhole = True

        #find Neuron that this whole is connected to
        connectedNeuron = np.max(np.absolute(np.unique(adjComp)))
        print("Whole detected! Conntected to Neuron " + str(int(connectedNeuron)))

        #check if whole is composed of Zeros
        if (np.min(np.absolute(np.unique(adjComp)))!= 0):
            isWhole = False
            print("Error! Whole is not composed of 0 and hence is not a valid Whole!")

    elif len(np.unique(adjComp))==1:
        print("Error, this connected component was detected wrong!")

    else:
        print("This connected component is not a whole (>2 neighbors)!")

    return isWhole, connectedNeuron

#read data from HD5, given the file path
def readData(filename):
    # read in data block
    data_in = ReadH5File(filename)

    global labels
    labels = data_in[x_start:x_end,y_start:y_end,z_start:z_end]

    print("data was read in; shape: " + str(labels.shape) + "; DataType is: " + str(data_in.dtype))

#compute the connected Com ponent labels
def computeConnectedComp():
    lables_inverse = 1 - labels
    connectivity = 6 # only 26, 18, and 6 are allowed
    labels_out = cc3d.connected_components(lables_inverse, connectivity=connectivity)

    # You can extract individual components like so:
    n_comp = np.max(labels_out)
    print("Conntected Regions found: " + str(n_comp))

    # determine indices, numbers and counts for the connected regions
    unique, counts = np.unique(labels_out, return_counts=True)
    print("Conntected regions and associated points: ")
    print(dict(zip(unique, counts)))

    return labels_out, n_comp

# fill a whole by changing the labels to the neuron it belongs to
def fillWhole(cods,connectedNeuron):

    labels[cods[:,0],cods[:,1],cods[:,2]] = np.ones((cods.shape[0],))*connectedNeuron
    print("Whole has been filled!!")

def main():

    # turn Visualization on and off
    Viz = False

    # read in data (written to global variable labels")
    readData("/home/frtim/wiring/raw_data/segmentations/JWR/cell032_downsampled.h5")

    #compute the labels of the conencted connected components
    labels_out, n_comp = computeConnectedComp()

    # check if connected component is a whole)
    n_start = 2 if Viz else 0
    for region in range(n_start,n_comp):
        print("Loading component " + str(region) +"...")
        # find coordinates of points that belong to component
        idx_compThree = np.argwhere(labels_out==region)
        # find coordinates of connected component
        cods = np.array([idx_compThree[:,0],idx_compThree[:,1],idx_compThree[:,2]]).transpose()
        # check if selected points are in a plane (2D object) and compute points that define hull surface
        if is2D(cods):
            boundary_pts_cods = convHull2D(cods)
        else:
            boundary_pts_cods = convHull3D(cods)

        # check if connected component is a whole
        isWhole, connectedNeuron = checkifwhole(boundary_pts_cods)

        # fill whole if detected
        if isWhole:
            fillWhole(cods, connectedNeuron)

        #fill wholes
        if (Viz):
            # debug: plot points as 3D scatter plot, extreme points in red
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(cods[:,0],cods[:,1],cods[:,2],c='b')
            ax.scatter(boundary_pts_cods[:,0],boundary_pts_cods[:,1],boundary_pts_cods[:,2],c='r')
            plt.show()

if __name__== "__main__":
  main()
