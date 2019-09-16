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
        coods = np.array([points[:,1],points[:,2]]).transpose()
        hull = ConvexHull(coods)
        boundary_idx = np.unique(hull.simplices)
        boundary_pts_coods = points[boundary_idx,:]

    if all_same(points[:,1]):
        coods = np.array([points[:,0],points[:,2]]).transpose()
        hull = ConvexHull(coods)
        boundary_idx = np.unique(hull.simplices)
        boundary_pts_coods = points[boundary_idx,:]

    if all_same(points[:,2]):
        coods = np.array([points[:,0],points[:,1]]).transpose()
        hull = ConvexHull(coods)
        boundary_idx = np.unique(hull.simplices)
        boundary_pts_coods = points[boundary_idx,:]

    return boundary_pts_coods

# fits a convex hull to a 3D point object and returns the coordinates of the points that describe the hull surface
def convHull3D(points):
    hull = ConvexHull(points)
    boundary_idx = np.unique(hull.simplices)
    boundary_pts_coods = points[boundary_idx,:]

    return boundary_pts_coods

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
def checkifwhole(boundary_pts_coods):

    isWhole = False
    adjComp = np.zeros((6, boundary_pts_coods.shape[0]))
    counter = 0
    connectedNeuron = -1

    for p in boundary_pts_coods:
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
def fillWhole(coods,connectedNeuron):

    labels[coods[:,0],coods[:,1],coods[:,2]] = np.ones((coods.shape[0],))*connectedNeuron
    print("Whole has been filled!!")

# find the coordinates of the points that belong to a selected connected component
def findCoodsOfComp(compQuery, compLabels):
    print("Loading component " + str(compQuery) +"...")
    # find coordinates of points that belong to component
    idx_comp = np.argwhere(compLabels==compQuery)
    # find coordinates of connected component
    coods = np.array([idx_comp[:,0],idx_comp[:,1],idx_comp[:,2]]).transpose()
    return coods

# find the points that describe the hull space of a given set of points
def findHullPoints(points):
    # check if selected points are in a plane (2D object) and compute points that define hull surface
    if is2D(points):
        HullPts= convHull2D(points)
    else:
        HullPts = convHull3D(points)

    return HullPts

# show points of a cloud in blue and mark the hull points in red
def runViz(coods, hull_coods):
    # debug: plot points as 3D scatter plot, extreme points in red
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(coods[:,0],coods[:,1],coods[:,2],c='b')
    ax.scatter(hull_coods[:,0],hull_coods[:,1],hull_coods[:,2],c='r')
    plt.show()

def main():

    # turn Visualization on and off
    Viz = True

    # read in data (written to global variable labels")
    readData("/home/frtim/wiring/raw_data/segmentations/JWR/cell032_downsampled.h5")

    #compute the labels of the conencted connected components
    labels_out, n_comp = computeConnectedComp()

    # check if connected component is a whole)
    # start at 1 as component 0 is always the neuron itself, which has label 1
    n_start = 2 if Viz else 1
    for region in range(n_start,n_comp):

        # find coordinates of points that belong to the selected component
        coods = findCoodsOfComp(region, labels_out)

        # find coordinates that describe the hull space
        hull_coods = findHullPoints(coods)

        # check if connected component is a whole
        isWhole, connectedNeuron = checkifwhole(hull_coods)

        # fill whole if detected
        if isWhole: fillWhole(coods, connectedNeuron)

        # run visualization
        if Viz: runViz(coods,hull_coods)


if __name__== "__main__":
  main()
