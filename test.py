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
from numba.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
import scipy.ndimage.interpolation
import math

# set will be deprecated soon on numba, but until now an alternative has not been implemented
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

# [z_start,z_end,y_start,y_end,x_start,x_end]
# box = [600,728,1024,2048,1024,2048]
# box = [0,773,1200,2600,0,3328]

#read data from HD5, given the file path
def readData(box, filename):
    # read in data block
    data_in = ReadH5File(filename)

    labels = data_in[box[0]:box[1],box[2]:box[3],box[4]:box[5]]

    print("data read in; shape: " + str(data_in.shape) + "; DataType: " + str(data_in.dtype) + "; cut to: " + str(labels.shape))

    return labels

# downsample data by facter downsample
def downsampleDataClassic(box, downsample, labels):

    if downsample%2!=0:
        print("Error, downsampling only possible for even integers")


    labels = labels[::downsample,::downsample,::downsample]
    print("downsampled to: " + str(labels.shape))

    #update box size according to samplint factor
    box = [int(b*(1/downsample))for b in box]

    return box, labels

# downsample with max
@njit
def downsampleDataMax(box, downsample, labels):

    if downsample%2!=0:
        print("Error, downsampling only possible for even integers")
    box_down = [int(b*(1/downsample))for b in box]
    labels_down = np.zeros((box_down[1],box_down[3],box_down[5]),dtype=np.uint8)

    # dsf = downsamplefactor
    dsf = downsample

    for ix in range(0, box_down[1]-box_down[0]):
        for iy in range(0, box_down[3]-box_down[2]):
            for iz in range(0, box_down[5]-box_down[4]):
                labels_down[ix,iy,iz]=np.max(labels[ix*dsf:(ix+1)*dsf,iy*dsf:(iy+1)*dsf,iz*dsf:(iz+1)*dsf])

    return box_down, labels_down

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

# find sets of adjacent components
@njit
def findAdjCompSets(box, labels_out, n_comp):

    neighbor_sets = set()

    for ix in range(0, box[1]-box[0]-1):
        for iy in range(0, box[3]-box[2]-1):
            for iz in range(0, box[5]-box[4]-1):

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

    for ix in [0, box[1]-box[0]-1]:
        for iy in range(0, box[3]-box[2]):
            for iz in range(0, box[5]-box[4]):
                curr_comp = labels_out[ix,iy,iz]
                neighbor_sets.add((curr_comp, -1))

    for ix in range(0, box[1]-box[0]):
        for iy in [0, box[3]-box[2]-1]:
            for iz in range(0, box[5]-box[4]):
                curr_comp = labels_out[ix,iy,iz]
                neighbor_sets.add((curr_comp, -1))

    for ix in range(0, box[1]-box[0]):
        for iy in range(0, box[3]-box[2]):
            for iz in [0, box[5]-box[4]-1]:
                curr_comp = labels_out[ix,iy,iz]
                neighbor_sets.add((curr_comp, -1))

    return neighbor_sets

# for statistics: additinallz count occurence of each component
@njit
def findAdjCompSetsWithCount(box, labels_out, n_comp):

    counts = np.zeros((n_comp),dtype=np.uint64)

    neighbor_sets = set()

    for ix in range(0, box[1]-box[0]-1):
        for iy in range(0, box[3]-box[2]-1):
            for iz in range(0, box[5]-box[4]-1):

                curr_comp = labels_out[ix,iy,iz]

                counts[curr_comp] += 1

                if curr_comp != labels_out[ix+1,iy,iz]:
                    neighbor_sets.add((curr_comp, labels_out[ix+1,iy,iz]))
                    neighbor_sets.add((labels_out[ix+1,iy,iz], curr_comp))
                if curr_comp != labels_out[ix,iy+1,iz]:
                    neighbor_sets.add((curr_comp, labels_out[ix,iy+1,iz]))
                    neighbor_sets.add((labels_out[ix,iy+1,iz], curr_comp))
                if curr_comp != labels_out[ix,iy,iz+1]:
                    neighbor_sets.add((curr_comp, labels_out[ix,iy,iz+1]))
                    neighbor_sets.add((labels_out[ix,iy,iz+1], curr_comp))

    for ix in [0, box[1]-box[0]-1]:
        for iy in range(0, box[3]-box[2]):
            for iz in range(0, box[5]-box[4]):
                curr_comp = labels_out[ix,iy,iz]
                neighbor_sets.add((curr_comp, -1))

    for ix in range(0, box[1]-box[0]):
        for iy in [0, box[3]-box[2]-1]:
            for iz in range(0, box[5]-box[4]):
                curr_comp = labels_out[ix,iy,iz]
                neighbor_sets.add((curr_comp, -1))

    for ix in range(0, box[1]-box[0]):
        for iy in range(0, box[3]-box[2]):
            for iz in [0, box[5]-box[4]-1]:
                curr_comp = labels_out[ix,iy,iz]
                neighbor_sets.add((curr_comp, -1))

    return neighbor_sets, counts

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
    non_wholes = []

    for c in range(n_comp):
        # check that only connected to one component and that this component is not border (which is numbered as -1)
        if (len(adj_comp[c]) is 1 and adj_comp[c][0] is not -1):
            wholes.append(c)
        else:
            non_wholes.append(c)

    # convert to sets
    wholes_set = set(wholes)
    non_wholes_set = set(non_wholes)

    # print("Wholes (total of " + str(len(wholes_set)) + "):")
    # print(wholes_set)
    # print("Non-Wholes (total of " + str(len(non_wholes_set)) + "):")
    # print(non_wholes_set)

    return wholes_set, non_wholes_set

# fill detedted wholes and give non_wholes their ID (for visualization)
@njit
def fillWholes(box_dyn_down, labels_cut, labels_cut_out_down, wholes_set, non_wholes_set, downsample):

    dsf = downsample
    box = box_dyn_down

    for ix in range(0, box[1]-box[0]):
        for iy in range(0, box[3]-box[2]):
            for iz in range(0, box[5]-box[4]):

                curr_comp = labels_cut_out_down[ix,iy,iz]

                # assign all wholes ID 2 to be able to visualize them
                if curr_comp in wholes_set:
                    labels_cut[ix*dsf:(ix+1)*dsf,iy*dsf:(iy+1)*dsf,iz*dsf:(iz+1)*dsf] = 2

                # assign non_wholes their componene ID to be able to visualize them (except background and neuron)
                if curr_comp in non_wholes_set and curr_comp != 0 and curr_comp != 1:
                    labels_cut[ix*dsf:(ix+1)*dsf,iy*dsf:(iy+1)*dsf,iz*dsf:(iz+1)*dsf] = curr_comp

    return labels_cut

def main():

    saveStatistics = False
    n_features = 20
    statistics_path = "/home/frtim/wiring/statistics/"
    data_path = "/home/frtim/wiring/raw_data/segmentations/JWR/test_volume/"
    sample_name = "cell032_downsampled_test.h5"
    output_name = "cell032_downsampled_test_filled"

    # bos size
    box = [0,200,0,1000,0,1000]

    print("box is: " + str(box))

    # read in data
    labels = readData(box, data_path+sample_name)

    # factor by which to downsample input
    downsample = 2

    # downsample data
    box_down, labels_down = downsampleDataClassic(box, downsample, labels)

    print("box_down is: " + str(box_down))

    #specify block overlap
    overlapInPx = False
    overlap = 0 #(one direction, total is twice as much)
    rel_b_size = 0.5

    #compute number of blocks and block size
    bs_z = int(rel_b_size*(box_down[1]-box_down[0]))
    n_blocks_z = math.floor((box_down[1]-box_down[0])/bs_z)
    bs_y = int(rel_b_size*(box_down[3]-box_down[2]))
    n_blocks_y = math.floor((box_down[3]-box_down[2])/bs_y)
    bs_x = int(rel_b_size*(box_down[5]-box_down[4]))
    n_blocks_x = math.floor((box_down[5]-box_down[4])/bs_x)

    print("nblocks: " + str(n_blocks_z) + ", " + str(n_blocks_y) + ", " + str(n_blocks_x))

    total_wholes_found = 0
    total_non_wholes_found = 0
    cell_counter = 0

    #box explanation
    # box = box in original volume, that is being processed

    #process blocks
    for bz in range(n_blocks_z):
        for by in range(n_blocks_y):
            for bx in range(n_blocks_x):

                if not overlapInPx:
                    z_min = math.floor(bz*bs_z*(1-overlap))
                    z_max = math.ceil((bz+1)*bs_z*(1+overlap))
                    y_min = math.floor(by*bs_y*(1-overlap))
                    y_max = math.ceil((by+1)*bs_y*(1+overlap))
                    x_min = math.floor(bx*bs_x*(1-overlap))
                    x_max = math.ceil((bx+1)*bs_x*(1+overlap))

                # TODO check that this doesnt become negative
                if overlapInPx:
                    z_min = bz*bs_z-overlap
                    z_max = (bz+1)*bs_z+overlap
                    y_min = by*bs_y-overlap
                    y_max = (by+1)*bs_y+overlap
                    x_min = bx*bs_x-overlap
                    x_max = (bx+1)*bs_x+overlap

                # extend last block to the border (to avoid small blocks where wholes cannot be detected)
                if bz == n_blocks_z-1: z_max = box_down[1]
                if by == n_blocks_y-1: y_max = box_down[3]
                if bx == n_blocks_x-1: x_max = box_down[5]

                box_down_dyn = [bz*bs_z,z_max,by*bs_y,y_max,bx*bs_x,x_max]
                box_dyn = [int(b*downsample)for b in box_down_dyn]


                print("box down_dyn: " + str(box_down_dyn))
                print("box _dyn: " + str(box_dyn))
                # box_down_dyn_ext = [z_min,z_max,y_min,y_max,x_min,x_max]
                # check if this is the last box (has to be enlarged)

                #take only part of block
                labels_cut_down = labels_down[box_down_dyn[0]:box_down_dyn[1],box_down_dyn[2]:box_down_dyn[3],box_down_dyn[4]:box_down_dyn[5]]
                labels_cut = labels[box_dyn[0]:box_dyn[1],box_dyn[2]:box_dyn[3],box_dyn[4]:box_dyn[5]]
                #compute the labels of the conencted connected components
                labels_cut_out_down, n_comp = computeConnectedComp(labels_cut_down)

                # compute the sets of connected components (also including boundary)
                adjComp_sets = findAdjCompSets(box_down_dyn, labels_cut_out_down, n_comp)

                # compute lists of wholes and non_wholes (then saved as set for compability with njit)
                wholes_set, non_wholes_set = findWholesList(adjComp_sets, n_comp)

                # fill detected wholes and visualize non_wholes
                if len(wholes_set)!=0:
                    labels_cut_filled = fillWholes(box_down_dyn, labels_cut, labels_cut_out_down, wholes_set, non_wholes_set, downsample)
                    labels[box_dyn[0]:box_dyn[1],box_dyn[2]:box_dyn[3],box_dyn[4]:box_dyn[5]] = labels_cut_filled
                    
                    total_wholes_found += len(wholes_set)
                    total_non_wholes_found += len(non_wholes_set)

                cell_counter+=1

    # print out total of found wholes
    print("Cells processed: " + str(cell_counter))
    print("Wholes filled (total): " + str(total_wholes_found))
    print("Non wholes detected (total): " + str(total_non_wholes_found))

    # write filled data to H5
    writeData(data_path+output_name, labels)

if __name__== "__main__":
  main()
