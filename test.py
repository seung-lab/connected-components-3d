import cc3d
import numpy as np
from dataIO import ReadH5File
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import time
from scipy.spatial import distance
import h5py
from numba import njit

global x_start
x_start = 500
global x_end
x_end = 773
global x_size
x_size = x_end-x_start

global y_start
y_start = 1000
global y_end
y_end = 2000
global y_size
y_size = y_end-y_start

global z_start
z_start = 1000
global z_end
z_end = 2000
global z_size
z_size = z_end-z_start

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

    labels = data_in[x_start:x_end,y_start:y_end,z_start:z_end]

    print("data read in; shape: " + str(data_in.shape) + "; DataType: " + str(data_in.dtype) + "; cut to: " + str(labels.shape))
    return labels

# write data to H5 file
def writeData(filename,labels):

    filename_comp = filename + "_" + str(time.time())[:10] +".h5"

    with h5py.File(filename_comp, 'w') as hf:
        # should cover all cases of affinities/images
        hf.create_dataset("main", data=labels, compression='gzip')

#compute the connected Com ponent labels
def computeConnectedComp(labels):
    lables_inverse = 1 - labels
    connectivity = 6 # only 26, 18, and 6 are allowed
    labels_out = cc3d.connected_components(lables_inverse, connectivity=connectivity)

    # You can extract individual components like so:
    n_comp = np.max(labels_out) + 1
    print("Conntected Regions found: " + str(n_comp))

    # determine indices, numbers and counts for the connected regions
    # unique, counts = np.unique(labels_out, return_counts=True)
    # print("Conntected regions and associated points: ")
    # print(dict(zip(unique, counts)))

    return labels_out, n_comp

# show points of a cloud in blue and mark the hull points in red
def runViz(coods, hull_coods):
    # debug: plot points as 3D scatter plot, extreme points in red
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(coods[:,0],coods[:,1],coods[:,2],c='b')
    ax.scatter(hull_coods[:,0],hull_coods[:,1],hull_coods[:,2],c='r')
    plt.show()

# compute statistics for the connected conmponents that have been found
def doStatistics(isWhole, coods, hull_coods, connectedNeuron, statTable, cnt):

    # account for components starting with 1
    cnt = cnt - 1

    # First column: Check if connected component is a whole
    if isWhole: statTable[cnt,0] = 1
    else: statTable[cnt,1] = 0

    # check if this is a 3D whole (1 if is 3D, otherwise 0)
    if is2D(coods): statTable[cnt,1] = 0
    else: statTable[cnt,2] = 1

    # number of points
    statTable[cnt,3] = int(coods.shape[0])

    # number of hull points
    n_hull_points = int(hull_coods.shape[0])
    statTable[cnt,4] = n_hull_points

    # intermediate step: compute pairwise distances between all hull points
    d_table = distance.cdist(hull_coods, hull_coods, 'euclidean')

    # average distance between hull points
    avg_hull_dist = np.sum(d_table)/(n_hull_points*n_hull_points-n_hull_points)
    statTable[cnt,5] = avg_hull_dist

    # maximum distance between hull points
    max_hull_dist = np.max(d_table)
    statTable[cnt,6] = max_hull_dist

    # intermediate step: find mid point (as the average over all points)
    mid_point = np.mean(coods, axis=0, keepdims=True)

    # intermediate step: find hull mid point (as the average over the hull points)
    hull_mid_point = np.mean(hull_coods, axis=0, keepdims=True)

    # compute mean, median and std for distance for all points from mid point
    d_allPoints_to_mid_table = distance.cdist(mid_point, coods, 'euclidean')
    d_allPoints_to_mid_mean = np.mean(d_allPoints_to_mid_table)
    d_allPoints_to_mid_median = np.median(d_allPoints_to_mid_table)
    d_allPoints_to_mid_std = np.std(d_allPoints_to_mid_table)
    statTable[cnt,7] = d_allPoints_to_mid_mean
    statTable[cnt,8] = d_allPoints_to_mid_median
    statTable[cnt,9] = d_allPoints_to_mid_std

    # compute mean, median and std for distance for all points from hull mid point
    d_allPoints_to_hullmid_table = distance.cdist(hull_mid_point, coods, 'euclidean')
    d_allPoints_to_hullmid_mean = np.mean(d_allPoints_to_hullmid_table)
    d_allPoints_to_hullmid_median = np.median(d_allPoints_to_hullmid_table)
    d_allPoints_to_hullmid_std = np.std(d_allPoints_to_hullmid_table)
    statTable[cnt,10] = d_allPoints_to_hullmid_mean
    statTable[cnt,11] = d_allPoints_to_hullmid_median
    statTable[cnt,12] = d_allPoints_to_hullmid_std

    # compute mean, median and std for distance for hull points from mid point
    d_hullPoints_to_mid_table = distance.cdist(mid_point, hull_coods, 'euclidean')
    d_hullPoints_to_mid_mean = np.mean(d_hullPoints_to_mid_table)
    d_hullPoints_to_mid_median = np.median(d_hullPoints_to_mid_table)
    d_hullPoints_to_mid_std = np.std(d_hullPoints_to_mid_table)
    statTable[cnt,13] = d_hullPoints_to_mid_mean
    statTable[cnt,14] = d_hullPoints_to_mid_median
    statTable[cnt,15] = d_hullPoints_to_mid_std

    # compute mean, median and std for distance for hull points from hull mid point
    d_hullPoints_to_hullmid_table = distance.cdist(hull_mid_point, hull_coods, 'euclidean')
    d_hullPoints_to_hullmid_mean = np.mean(d_hullPoints_to_hullmid_table)
    d_hullPoints_to_hullmid_median = np.median(d_hullPoints_to_hullmid_table)
    d_hullPoints_to_hullmid_std = np.std(d_hullPoints_to_hullmid_table)
    statTable[cnt,16] = d_hullPoints_to_mid_mean
    statTable[cnt,17] = d_hullPoints_to_mid_median
    statTable[cnt,18] = d_hullPoints_to_mid_std

    #distance between hull mid point and all points mid point
    d_mid_to_hullmid = np.linalg.norm(hull_mid_point-mid_point)
    statTable[cnt,19] = d_mid_to_hullmid

    return statTable

# write statistics to a .txt filename
def writeStatistics(statTable, statistics_path, sample_name):
    filename = statistics_path + sample_name.replace("/","_").replace(".","_") + "_statistics_" + str(time.time())[:10] + ".txt"

    header_a = "number,isWhole,is3D,nPoints,nHullPoints,avgHullDist,maxHullDist,"
    header_b = "d_allPoints_to_mid_mean,d_allPoints_to_mid_median,d_allPoints_to_mid_std,"
    header_c = "d_allPoints_to_hullmid_mean,d_allPoints_to_hullmid_median,d_allPoints_to_hullmid_std,"
    header_d = "d_hullPoints_to_mid_mean,d_hullPoints_to_mid_median,d_hullPoints_to_mid_std,"
    header_e = "d_hullPoints_to_hullmid_mean,d_hullPoints_to_hullmid_median,d_hullPoints_to_hullmid_std,"
    header_f = "d_mid_to_hullmid"

    header =  header_a + header_b + header_c + header_d + header_e + header_f

    if(header.count(',')!=(statTable.shape[1]-1)):
        print("Error! Header variables are not equal to number of columns in the statistics!")
    np.savetxt(filename, statTable, delimiter=',', header=header)

@njit
def findAdjCompSets(labels_out, n_comp):

    #adj_comp = [[] for _ in range(n_comp)]
    neighbor_sets = set()
    for ix in range(0, x_size-1):
        for iy in range(0, y_size-1):
            for iz in range(0, z_size-1):

                curr_comp = labels_out[ix,iy,iz]

                if curr_comp != labels_out[ix+1,iy,iz]:
                    neighbor_sets.add((curr_comp, labels_out[ix+1,iy,iz]))
                    neighbor_sets.add((labels_out[ix+1,iy,iz], curr_comp))
                if curr_comp != labels_out[ix,iy+1,iz]:
                    neighbor_sets.add((curr_comp, labels_out[ix,iy+1,iz]))
                    neighbor_sets.add((labels_out[ix,iy+1,iz], curr_comp))
                if curr_comp != labels_out[ix,iy,iz+1]:
                    neighbor_sets.add((curr_comp, labels_out[ix,iy,iz+1]))
                    neighbor_sets.add((labels_out[ix,iy,iz+1], curr_comp))

    for ix in [0, x_size-1]:
        for iy in range(0, y_size):
            for iz in range(0, z_size):
                curr_comp = labels_out[ix,iy,iz]
                neighbor_sets.add((curr_comp, -1))

    for ix in range(0, x_size):
        for iy in [0, y_size-1]:
            for iz in range(0, z_size):
                curr_comp = labels_out[ix,iy,iz]
                neighbor_sets.add((curr_comp, -1))

    for ix in range(0, x_size):
        for iy in range(0, y_size):
            for iz in [0, z_size-1]:
                curr_comp = labels_out[ix,iy,iz]
                neighbor_sets.add((curr_comp, -1))

    return neighbor_sets

# create string of connected components that are a whole
def findWholesList(adjComp_sets, n_comp):

    # find the components that each connected component is connected to
    adj_comp = [[] for _ in range(n_comp)]
    for s in range(len(adjComp_sets)):
        temp = adjComp_sets.pop()
        if temp[1] not in adj_comp[temp[0]]:
            adj_comp[temp[0]].append(temp[1])

    #find connected components that are a whole
    wholes = []
    for c in range(n_comp):
        # check that only connected to one component and that this component is not border (which is numbered as -1)
        if (len(adj_comp[c]) is 1):
            if(adj_comp[c][0] is not -1):
                wholes.append(c)

    print("found " + str(len(wholes)) + " wholes")
    return wholes

@njit
def fillWholes(labels, labels_out, wholes_set):

    for ix in range(0, x_size):
        for iy in range(0, y_size):
            for iz in range(0, z_size):

                curr_comp = labels_out[ix,iy,iz]

                if curr_comp in wholes_set:
                    # assign all wholes ID 2 to be able to visualize them
                    labels[ix,iy,iz] = 2

    return labels

def main():

    # turn Visualization on and off
    Viz = False
    saveStatistics = True
    n_features = 20
    statistics_path = "/home/frtim/wiring/statistics/"
    data_path = "/home/frtim/wiring/raw_data/segmentations/"
    sample_name = "JWR/cell032_downsampled.h5"
    output_name = "JWR/cell032_downsampled_filled_viz"

    # needed to time the code (n_functions as the number of subfunctions considered for timing)

    # read in data (written to global variable labels")
    labels = readData(data_path+sample_name)

    #compute the labels of the conencted connected components
    labels_out, n_comp = computeConnectedComp(labels)

    adjComp_sets = findAdjCompSets(labels_out, n_comp)
    print(adjComp_sets)

    wholes_array = findWholesList(adjComp_sets, n_comp)
    print(wholes_array)

    wholes_set = set(wholes_array)

    labels = fillWholes(labels, labels_out, wholes_set)

    # if saveStatistics: statTable = np.ones((n_comp-1, n_features))*-1
    # for region in range(n_start,n_comp):
    #
    #     print("Loading component " + str(region) +"...")
    #     # find coordinates of points that belong to the selected component
    #     coods = findCoodsOfComp(region, labels_out)
    #
    #     print("finding points that describe the hull...")
    #     # find coordinates that describe the hull space
    #     hull_coods = findHullPoints(coods)
    #
    #     print("Checking if this is a whole...")
    #     # check if connected component is a whole
    #     isWhole, connectedNeuron = checkifwhole(hull_coods)
    #
    #     print("Felling Whole...")
    #     # fill whole if detected
    #     if isWhole: fillWhole(coods, connectedNeuron)
    #
    #     print("Computing statistics...")
    #     # compute statistics and save to numpy array
    #     if saveStatistics: statTable = doStatistics(isWhole, coods, hull_coods, connectedNeuron, statTable, region)
    #
    #     print("Running Visualization...")
    #     # run visualization
    #     if Viz: runViz(coods,hull_coods)
    #
    # # save the statistics file to a .txt file
    # if saveStatistics: writeStatistics(statTable, statistics_path, sample_name)
    #
    # write filled data to H5
    writeData(data_path+output_name, labels)

if __name__== "__main__":
  main()
