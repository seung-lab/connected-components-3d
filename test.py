import cc3d
import numpy as np
from dataIO import ReadH5File
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import time
from scipy.spatial import distance
import h5py
from numba import njit, types
from numba.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
import scipy.ndimage.interpolation
import math
from numba.typed import Dict
import os
import psutil

# set will be deprecated soon on numba, but until now an alternative has not been implemented
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

# [z_start,z_end,y_start,y_end,x_start,x_end]
# box = [600,728,1024,2048,1024,2048]
# box = [0,773,1200,2600,0,3328]

#read data from HD5, given the file path
def readData(box, filename):
    # read in data block
    data_in = ReadH5File(filename, box)

    labels = data_in

    print("data read in; shape: " + str(data_in.shape) + "; DataType: " + str(data_in.dtype))

    return labels

# downsample data by facter downsample
def downsampleDataClassic(box, downsample, labels):

    if downsample%2!=0 and downsample!=1:
        print("Error, downsampling only possible for even integers")


    labels = labels[::downsample,::downsample,::downsample]
    print("downsampled to: " + str(labels.shape))

    #update box size according to samplint factor
    box = [int(b*(1/downsample))for b in box]

    return box, labels

# downsample with max
@njit
def downsampleDataMax(box, downsample, labels):

    if downsample%2!=0 and downsample!=1:
        print("Error, downsampling only possible for even integers")
    box_down = [int(b*(1/downsample))for b in box]
    labels_down = np.zeros((box_down[1],box_down[3],box_down[5]),dtype=np.uint16)

    # dsf = downsamplefactor
    dsf = downsample

    for ix in range(0, box_down[1]-box_down[0]):
        for iy in range(0, box_down[3]-box_down[2]):
            for iz in range(0, box_down[5]-box_down[4]):
                labels_down[ix,iy,iz]=np.max(labels[ix*dsf:(ix+1)*dsf,iy*dsf:(iy+1)*dsf,iz*dsf:(iz+1)*dsf])

    return box_down, labels_down

@njit
def downsampleDataMin(box, downsample, labels):

    if downsample%2!=0 and downsample!=1:
        print("Error, downsampling only possible for even integers")
    box_down = [int(b*(1/downsample))for b in box]
    labels_down = np.zeros((box_down[1],box_down[3],box_down[5]),dtype=np.uint16)
    # dsf = downsamplefactor
    dsf = downsample

    for ix in range(0, box_down[1]-box_down[0]):
        for iy in range(0, box_down[3]-box_down[2]):
            for iz in range(0, box_down[5]-box_down[4]):
                labels_down[ix,iy,iz]=np.min(labels[ix*dsf:(ix+1)*dsf,iy*dsf:(iy+1)*dsf,iz*dsf:(iz+1)*dsf])

    return box_down, labels_down

# write data to H5 file
def writeData(filename,labels):

    filename_comp = filename + "_" + str(time.time())[:10] +".h5"

    with h5py.File(filename_comp, 'w') as hf:
        # should cover all cases of affinities/images
        hf.create_dataset("main", data=labels, compression='gzip')

#compute the connected Com ponent labels
def computeConnectedComp(labels, printOn):
    connectivity = 6 # only 26, 18, and 6 are allowed
    labels_out = cc3d.connected_components(labels, connectivity=connectivity, max_labels=100000000)

    # You can extract individual components like so:
    n_comp = np.max(labels_out) + 1

    if printOn:
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
def findAdjCompSets(box, labels, labels_out, n_comp):

    neighbor_set_comp = set()
    neighbor_label = Dict.empty(key_type=types.uint16,value_type=types.uint16)

    for ix in range(0, box[1]-box[0]-1):
        for iy in range(0, box[3]-box[2]-1):
            for iz in range(0, box[5]-box[4]-1):

                curr_comp = labels_out[ix,iy,iz]

                if curr_comp != labels_out[ix+1,iy,iz]:
                    neighbor_set_comp.add((curr_comp, labels_out[ix+1,iy,iz]))
                    neighbor_set_comp.add((labels_out[ix+1,iy,iz], curr_comp))

                    neighbor_label[labels_out[ix,iy,iz]]=labels[ix+1,iy,iz]
                    neighbor_label[labels_out[ix+1,iy,iz]]=labels[ix,iy,iz]

                if curr_comp != labels_out[ix,iy+1,iz]:
                    neighbor_set_comp.add((curr_comp, labels_out[ix,iy+1,iz]))
                    neighbor_set_comp.add((labels_out[ix,iy+1,iz], curr_comp))

                    neighbor_label[labels_out[ix,iy,iz]]=labels[ix,iy+1,iz]
                    neighbor_label[labels_out[ix,iy+1,iz]]=labels[ix,iy,iz]

                if curr_comp != labels_out[ix,iy,iz+1]:
                    neighbor_set_comp.add((curr_comp, labels_out[ix,iy,iz+1]))
                    neighbor_set_comp.add((labels_out[ix,iy,iz+1], curr_comp))

                    neighbor_label[labels_out[ix,iy,iz]]=labels[ix,iy,iz+1]
                    neighbor_label[labels_out[ix,iy,iz+1]]=labels[ix,iy,iz]

    for ix in [0, box[1]-box[0]-1]:
        for iy in range(0, box[3]-box[2]):
            for iz in range(0, box[5]-box[4]):
                curr_comp = labels_out[ix,iy,iz]
                neighbor_set_comp.add((curr_comp, -1))

    for ix in range(0, box[1]-box[0]):
        for iy in [0, box[3]-box[2]-1]:
            for iz in range(0, box[5]-box[4]):
                curr_comp = labels_out[ix,iy,iz]
                neighbor_set_comp.add((curr_comp, -1))

    for ix in range(0, box[1]-box[0]):
        for iy in range(0, box[3]-box[2]):
            for iz in [0, box[5]-box[4]-1]:
                curr_comp = labels_out[ix,iy,iz]
                neighbor_set_comp.add((curr_comp, -1))

    return neighbor_set_comp, neighbor_label

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
def findAssociatedComp(adjComp_sets, neighbor_label, n_comp):

    # find the components that each connected component is connected to
    adj_comp = [[] for _ in range(n_comp)]
    for s in range(len(adjComp_sets)):
        temp = adjComp_sets.pop()
        if temp[1] not in adj_comp[temp[0]]:
            adj_comp[temp[0]].append(temp[1])

    #find connected components that are a whole
    associated_comp = Dict.empty(key_type=types.uint16,value_type=types.uint16)

    for c in range(n_comp):
        # check that only connected to one component and that this component is not border (which is numbered as -1)
        if (len(adj_comp[c]) is 1 and adj_comp[c][0] is not -1):
            associated_comp[c] = neighbor_label[c]
            # associated_comp[c] = 65000
        else:
            associated_comp[c] = 0

    return associated_comp

# fill detedted wholes and give non_wholes their ID (for visualization)
@njit
def fillWholes(box_down_dyn_ext, labels_cut_ext, labels_cut_out_down_ext, associated_comp, downsample):

    dsf = downsample
    box = box_down_dyn_ext

    for ix in range(0, box[1]-box[0]):
        for iy in range(0, box[3]-box[2]):
            for iz in range(0, box[5]-box[4]):

                if np.min(labels_cut_ext[ix*dsf:(ix+1)*dsf,iy*dsf:(iy+1)*dsf,iz*dsf:(iz+1)*dsf]) == 0:
                # assign all wholes ID 2 to be able to visualize them
                # labels_cut_ext[ix*dsf:(ix+1)*dsf,iy*dsf:(iy+1)*dsf,iz*dsf:(iz+1)*dsf] = associated_comp[labels_cut_out_down_ext[ix,iy,iz]]
                    labels_cut_ext[ix*dsf:(ix+1)*dsf,iy*dsf:(iy+1)*dsf,iz*dsf:(iz+1)*dsf] = associated_comp[labels_cut_out_down_ext[ix,iy,iz]]
                # # assign non_wholes their componene ID to be able to visualize them (except background and neuron)
                # if curr_comp in non_wholes and curr_comp != 0 and curr_comp != 1:
                #     labels_cut_ext[ix*dsf:(ix+1)*dsf,iy*dsf:(iy+1)*dsf,iz*dsf:(iz+1)*dsf] = curr_comp

    return labels_cut_ext

# compute extended boxes
@njit
def getBoxes(box_down, overlap, overlap_d, downsample, bz, bs_z, n_blocks_z, by, bs_y, n_blocks_y, bx, bs_x, n_blocks_x):

        # down refers to downsampled scale, ext to extended boxes (extended by the overlap)
        # compute the downsampled dynamic box
        z_min_down_dyn = bz*bs_z
        z_max_down_dyn = (bz+1)*bs_z if ((bz+1)*bs_z+overlap_d <= box_down[1] and bz != n_blocks_z-1) else box_down[1]
        y_min_down_dyn = by*bs_y
        y_max_down_dyn = (by+1)*bs_y if ((by+1)*bs_y+overlap_d <= box_down[3] and by != n_blocks_y-1) else box_down[3]
        x_min_down_dyn = bx*bs_x
        x_max_down_dyn = (bx+1)*bs_x if ((bx+1)*bs_x+overlap_d <= box_down[5] and bx != n_blocks_x-1) else box_down[5]

        box_down_dyn = [z_min_down_dyn,z_max_down_dyn,y_min_down_dyn,y_max_down_dyn,x_min_down_dyn,x_max_down_dyn]
        box_dyn = [int(b*downsample)for b in box_down_dyn]

        # compute the downsampled dynamic box (extended by the overlap)
        z_min_down_dyn_ext = bz*bs_z-overlap_d if (bz*bs_z-overlap_d >= 0) else 0
        z_max_down_dyn_ext = (bz+1)*bs_z+overlap_d if ((bz+1)*bs_z+overlap_d <= box_down[1] and bz != n_blocks_z-1) else box_down[1]
        y_min_down_dyn_ext = by*bs_y-overlap_d if (by*bs_y-overlap_d >= 0) else 0
        y_max_down_dyn_ext = (by+1)*bs_y+overlap_d if ((by+1)*bs_y+overlap_d <= box_down[3] and by != n_blocks_y-1) else box_down[3]
        x_min_down_dyn_ext = bx*bs_x-overlap_d if (bx*bs_x-overlap_d >= 0) else 0
        x_max_down_dyn_ext = (bx+1)*bs_x+overlap_d if ((bx+1)*bs_x+overlap_d <= box_down[5] and bx != n_blocks_x-1) else box_down[5]

        box_down_dyn_ext = [z_min_down_dyn_ext,z_max_down_dyn_ext,y_min_down_dyn_ext,y_max_down_dyn_ext,x_min_down_dyn_ext,x_max_down_dyn_ext]
        box_dyn_ext = [int(b*downsample)for b in box_down_dyn_ext]

        z_start = overlap_d*downsample if (bz*bs_z-overlap_d >= 0) else 0
        z_end = -overlap_d*downsample if ((bz+1)*bs_z+overlap_d <= box_down[1] and bz != n_blocks_z-1 and overlap!=0) else (box_dyn[1]-box_dyn[0]+overlap)
        y_start = overlap_d*downsample if (by*bs_y-overlap_d >= 0) else 0
        y_end = -overlap_d*downsample if ((by+1)*bs_y+overlap_d <= box_down[3] and by != n_blocks_y-1 and overlap!=0) else (box_dyn[3]-box_dyn[2]+overlap)
        x_start = overlap_d*downsample if (bx*bs_x-overlap_d >= 0) else 0
        x_end = -overlap_d*downsample if ((bx+1)*bs_x+overlap_d <= box_down[5] and bx != n_blocks_x-1 and overlap!=0) else (box_dyn[5]-box_dyn[4]+overlap)

        # compute the indexing to get from the extended dynamic box back to the standard size dynamic box
        box_idx = [z_start, z_end, y_start, y_end, x_start, x_end]

        # print("box_down_dyn_ext: " + str(box_down_dyn_ext))
        # print("box_dyn_ext: " + str(box_dyn_ext))
        # print("box_down_dyn: " + str(box_down_dyn))
        # print("box_dyn: " + str(box_dyn))
        # print(z_start,z_end,y_start,y_end,x_start,x_end)

        return box_down_dyn, box_dyn, box_down_dyn_ext, box_dyn_ext, box_idx

# process whole filling process for chung of data
def processData(py, labels, downsample, overlap, rel_block_size):

        # read in chunk size
        box = [0,labels.shape[0],0,labels.shape[1],0,labels.shape[2]]

        print('memory use at c:', py.memory_info()[0]/2.**30)

        # downsample data
        if downsample > 1:
            box_down, labels_down = downsampleDataMin(box, downsample, labels)
        else :
            box_down = box
            labels_down = labels

        print('memory use at d:', py.memory_info()[0]/2.**30)

        #specify block overlap in downsampled domain
        overlap_d = int(overlap/downsample)

        # compute number of blocks and block size
        bs_z = int(rel_block_size*(box_down[1]-box_down[0]))
        n_blocks_z = math.floor((box_down[1]-box_down[0])/bs_z)
        bs_y = int(rel_block_size*(box_down[3]-box_down[2]))
        n_blocks_y = math.floor((box_down[3]-box_down[2])/bs_y)
        bs_x = int(rel_block_size*(box_down[5]-box_down[4]))
        n_blocks_x = math.floor((box_down[5]-box_down[4])/bs_x)

        print("nblocks: " + str(n_blocks_z) + ", " + str(n_blocks_y) + ", " + str(n_blocks_x))
        print("block size: " + str(bs_z) + ", " + str(bs_y) + ", " + str(bs_x))

        #counters
        total_wholes_found = 0
        total_non_wholes_found = 0
        cell_counter = 0

        # print connected components only if all data processed in one
        printOn = True if n_blocks_z == 1 else False

        print('memory use at e:', py.memory_info()[0]/2.**30)

        # process blocks by iterating over all bloks
        for bz in range(n_blocks_z):
            for by in range(n_blocks_y):
                for bx in range(n_blocks_x):

                    print('Bock {} ...'.format(bz+1), end='\r')

                    print('memory use at e a:', py.memory_info()[0]/2.**30)

                    # compute boxes (description in function)
                    box_down_dyn, box_dyn, box_down_dyn_ext, box_dyn_ext, box_idx = getBoxes(
                        box_down, overlap, overlap_d, downsample, bz, bs_z, n_blocks_z, by, bs_y, n_blocks_y, bx, bs_x, n_blocks_x)
                    print('memory use at e b:', py.memory_info()[0]/2.**30)
                    # take only part of block
                    labels_cut_down_ext = labels_down[
                        box_down_dyn_ext[0]:box_down_dyn_ext[1],box_down_dyn_ext[2]:box_down_dyn_ext[3],box_down_dyn_ext[4]:box_down_dyn_ext[5]]

                    print('memory use at e c:', py.memory_info()[0]/2.**30)

                    labels_cut_ext = labels[
                        box_dyn_ext[0]:box_dyn_ext[1],box_dyn_ext[2]:box_dyn_ext[3],box_dyn_ext[4]:box_dyn_ext[5]]

                    print('memory use at f:', py.memory_info()[0]/2.**30)
                    # compute the labels of the conencted connected components
                    labels_cut_out_down_ext, n_comp = computeConnectedComp(labels_cut_down_ext, printOn)
                    print('memory use at g:', py.memory_info()[0]/2.**30)
                    # compute the sets of connected components (also including boundary)
                    adjComp_sets, neighbor_label = findAdjCompSets(box_down_dyn_ext, labels_cut_down_ext, labels_cut_out_down_ext, n_comp)

                    print("ADJCompSets, len, nbytes:" + str(len(adjComp_sets)))
                    print("Neighbor label, len, nbytes:" + str(len(neighbor_label)))
                    # compute lists of wholes and non_wholes (then saved as set for compability with njit)
                    associated_comp = findAssociatedComp(adjComp_sets, neighbor_label, n_comp)

                    print("Associated Comp, len, nbytes:" + str(len(associated_comp)))

                    print('memory use at h:', py.memory_info()[0]/2.**30)
                    # fill detected wholes and visualize non_wholes
                    if len(associated_comp)>2:
                        labels_cut_filled = fillWholes(box_down_dyn_ext, labels_cut_ext, labels_cut_out_down_ext, associated_comp, downsample)
                        labels[box_dyn[0]:box_dyn[1],box_dyn[2]:box_dyn[3],box_dyn[4]:box_dyn[5]] = labels_cut_filled[
                            box_idx[0]:box_idx[1],box_idx[2]:box_idx[3],box_idx[4]:box_idx[5]]

                        total_wholes_found += len({k: v for k, v in associated_comp.items() if v != 0})

                    cell_counter+=1
                    print('memory use at i:', py.memory_info()[0]/2.**30)

        # print out total of found wholes
        print("Cells processed: " + str(cell_counter))
        print("Wholes filled (total): " + str(total_wholes_found))



        return labels

def main():

    saveStatistics = False
    n_features = 20
    statistics_path = "/home/frtim/wiring/statistics/"
    data_path = "/home/frtim/wiring/raw_data/segmentations/Zebrafinch/test_volume/"
    sample_name = "0000"
    vizWholes = False
    # sample_name = "cell032_downsampled.h5"
    # output_name = "cell032_downsampled_filled"

    # define psutil
    pid = os.getpid()
    py = psutil.Process(pid)

    # bos size
    box = [0,128,0,2000,0,2000]
    # box = [0,773,0,3328,0,3328]

    print("-----------------------------------------------------------------")

    print('memory use at a:', py.memory_info()[0]/2.**30)

    # read in data
    labels = readData(box, data_path+sample_name+".h5")
    print(labels.nbytes)

    print('memory use at b:', py.memory_info()[0]/2.**30)

    start_time = time.time()

    print("-----------------------------------------------------------------")

    # process chunk of data
    # overlap in points in one direction (total is twice)
    labels = processData(py, labels, downsample=1, overlap=0, rel_block_size=1)

    print('memory use at j:', py.memory_info()[0]/2.**30)

    print("-----------------------------------------------------------------")
    print("Time elapsed: " + str(time.time() - start_time))

    # write filled data to H5
    output_name = "output/" + sample_name
    writeData(data_path+output_name, labels)
    # process again to check success
    print('memory use at k:', py.memory_info()[0]/2.**30)

    if vizWholes:
        labels_inp = readData(box, data_path+sample_name+".h5")
        wholes = np.subtract(labels, labels_inp)
        output_name = "output/" + sample_name + "_wholes"
        print("max_label is: " + str(np.max(wholes)))
        writeData(data_path+output_name, wholes)

    # print("-----------------------------------------------------------------")
    # print("----------------------CHECK--------------------------------------")
    # labels = processData(labels, downsample=1, overlap=0, rel_block_size=1)

    # write filled data to H5
    # writeData(data_path+output_name, labels)

if __name__== "__main__":
  main()
