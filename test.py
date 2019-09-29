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

# get shape of data saved in H5
def getBoxAll(filename):

    # return the first h5 dataset from this file
    with h5py.File(filename, 'r') as hf:
        keys = [key for key in hf.keys()]
        d = hf[keys[0]]
        box = [0,d.shape[0],0,d.shape[1],0,d.shape[2]]

    return box

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

    filename_comp = filename +".h5"

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

# write statistics to a .txt filename
def writeStatistics(n_comp, isWhole, comp_counts, comp_mean, comp_var, statistics_path, sample_name):

    numbering = np.zeros((n_comp,1))
    numbering[:,0] = np.linspace(1,n_comp,num=n_comp)

    # create table that is later written to .txt
    statTable = np.hstack((numbering, isWhole, comp_counts, comp_mean, comp_var))

    filename = statistics_path + sample_name.replace("/","_").replace(".","_") + "_statistics" + ".txt"

    header = "number,isWhole,nPoints,mean_z,mean_y,mean_x,var_z,var_y,var_x"

    if(header.count(',')!=(statTable.shape[1]-1)):
        print("Error! Header variables are not equal to number of columns in the statistics!")
    np.savetxt(filename, statTable, delimiter=',', header=header)

# find sets of adjacent components
@njit
def findAdjLabelSet(box, labels, labels_out, n_comp):

    neighbor_label_set = set()

    for ix in range(0, box[1]-box[0]-1):
        for iy in range(0, box[3]-box[2]-1):
            for iz in range(0, box[5]-box[4]-1):

                curr_comp = labels_out[ix,iy,iz]

                if curr_comp != labels_out[ix+1,iy,iz]:
                    neighbor_label_set.add((labels_out[ix,iy,iz],labels[ix+1,iy,iz]))
                    neighbor_label_set.add((labels_out[ix+1,iy,iz],labels[ix,iy,iz]))

                if curr_comp != labels_out[ix,iy+1,iz]:
                    neighbor_label_set.add((labels_out[ix,iy,iz],labels[ix,iy+1,iz]))
                    neighbor_label_set.add((labels_out[ix,iy+1,iz],labels[ix,iy,iz]))

                if curr_comp != labels_out[ix,iy,iz+1]:
                    neighbor_label_set.add((labels_out[ix,iy,iz],labels[ix,iy,iz+1]))
                    neighbor_label_set.add((labels_out[ix,iy,iz+1],labels[ix,iy,iz]))

    for ix in [0, box[1]-box[0]-1]:
        for iy in range(0, box[3]-box[2]):
            for iz in range(0, box[5]-box[4]):
                neighbor_label_set.add((labels_out[ix,iy,iz], -1))

    for ix in range(0, box[1]-box[0]):
        for iy in [0, box[3]-box[2]-1]:
            for iz in range(0, box[5]-box[4]):
                neighbor_label_set.add((labels_out[ix,iy,iz], -1))

    for ix in range(0, box[1]-box[0]):
        for iy in range(0, box[3]-box[2]):
            for iz in [0, box[5]-box[4]-1]:
                neighbor_label_set.add((labels_out[ix,iy,iz], -1))

    return neighbor_label_set

# for statistics: additinallz count occurence of each component
@njit
def getStat(box, labels_out, n_comp):

    comp_counts = np.zeros((n_comp,1),dtype=np.uint64)
    comp_mean = np.zeros((n_comp,3),dtype=np.float64)
    comp_var = np.zeros((n_comp,3),dtype=np.float64)

    # Compute the mean in each direction and count points for each component
    for iz in range(0, box[1]-box[0]):
        for iy in range(0, box[3]-box[2]):
            for ix in range(0, box[5]-box[4]):

                curr_comp = labels_out[iz,iy,ix]

                comp_counts[curr_comp] =  comp_counts[curr_comp] + 1

                comp_mean[curr_comp, 0] = comp_mean[curr_comp, 0] + iz
                comp_mean[curr_comp, 1] = comp_mean[curr_comp, 1] + iy
                comp_mean[curr_comp, 2] = comp_mean[curr_comp, 2] + ix

    comp_mean[:,0] = np.divide(comp_mean[:,0],comp_counts[:,0])
    comp_mean[:,1] = np.divide(comp_mean[:,1],comp_counts[:,0])
    comp_mean[:,2] = np.divide(comp_mean[:,2],comp_counts[:,0])

    # Compute the Variance in each direction
    for iz in range(0, box[1]-box[0]):
        for iy in range(0, box[3]-box[2]):
            for ix in range(0, box[5]-box[4]):

                curr_comp = labels_out[iz,iy,ix]

                comp_var[curr_comp, 0] = comp_var[curr_comp, 0] + (iz-comp_mean[curr_comp, 0])**2
                comp_var[curr_comp, 1] = comp_var[curr_comp, 1] + (iy-comp_mean[curr_comp, 1])**2
                comp_var[curr_comp, 2] = comp_var[curr_comp, 2] + (ix-comp_mean[curr_comp, 2])**2

    comp_var[:,0] = np.divide(comp_var[:,0],comp_counts[:,0])
    comp_var[:,1] = np.divide(comp_var[:,1],comp_counts[:,0])
    comp_var[:,2] = np.divide(comp_var[:,2],comp_counts[:,0])

    return comp_counts, comp_mean, comp_var

# create string of connected components that are a whole
def findAssociatedLabels(neighbor_label_set, n_comp):

    neighbor_labels = [[] for _ in range(n_comp)]
    for s in range(len(neighbor_label_set)):
        temp = neighbor_label_set.pop()
        if temp[1] not in neighbor_labels[temp[0]]:
            neighbor_labels[temp[0]].append(temp[1])

    #find connected components that are a whole
    associated_label = Dict.empty(key_type=types.uint16,value_type=types.uint16)
    isWhole = np.ones((n_comp,1), dtype=np.int8)*-1

    for c in range(n_comp):
        # check that only connected to one component and that this component is not border (which is numbered as -1)
        if (len(neighbor_labels[c]) is 1 and neighbor_labels[c][0] is not -1):
            associated_label[c] = neighbor_labels[c][0]
            isWhole[c] = 1
            # associated_label[c] = 65000
            # associated_label[c] = 0
        else:
            associated_label[c] = 0
            isWhole[c] = 0
            # associated_label[c] = neighbor_labels[c][0] + 5

    return associated_label, isWhole

# fill detedted wholes and give non_wholes their ID (for visualization)
@njit
def fillWholes(box_down_dyn_ext, labels_cut_ext, labels_cut_out_down_ext, associated_label, downsample):

    dsf = downsample
    box = box_down_dyn_ext

    for ix in range(0, box[1]-box[0]):
        for iy in range(0, box[3]-box[2]):
            for iz in range(0, box[5]-box[4]):

                if labels_cut_ext[ix,iy,iz] == 0:
                    labels_cut_ext[ix,iy,iz] = associated_label[labels_cut_out_down_ext[ix,iy,iz]]

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
def processData(saveStatistics, statistics_path, sample_name, labels, downsample, overlap, rel_block_size):

        # read in chunk size
        box = [0,labels.shape[0],0,labels.shape[1],0,labels.shape[2]]

        # downsample data
        if downsample > 1:
            box_down, labels_down = downsampleDataMin(box, downsample, labels)
        else:
            box_down = box
            labels_down = labels

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
        cell_counter = 0

        # print connected components only if all data processed in one
        printOn = True if n_blocks_z < 11 else False

        # process blocks by iterating over all bloks
        for bz in range(n_blocks_z):
            for by in range(n_blocks_y):
                for bx in range(n_blocks_x):

                    print('Bock {} ...'.format(bz+1), end='\r')

                    # compute boxes (description in function)
                    box_down_dyn, box_dyn, box_down_dyn_ext, box_dyn_ext, box_idx = getBoxes(
                        box_down, overlap, overlap_d, downsample, bz, bs_z, n_blocks_z, by, bs_y, n_blocks_y, bx, bs_x, n_blocks_x)

                    # take only part of block
                    labels_cut_down_ext = labels_down[
                        box_down_dyn_ext[0]:box_down_dyn_ext[1],box_down_dyn_ext[2]:box_down_dyn_ext[3],box_down_dyn_ext[4]:box_down_dyn_ext[5]]

                    # take only part of block
                    labels_cut_ext = labels[
                        box_dyn_ext[0]:box_dyn_ext[1],box_dyn_ext[2]:box_dyn_ext[3],box_dyn_ext[4]:box_dyn_ext[5]]

                    print("Compute connected Components...")
                    # compute the labels of the conencted connected components
                    labels_cut_out_down_ext, n_comp = computeConnectedComp(labels_cut_down_ext, printOn)

                    print("Find Sets of Adjacent Components...")
                    # compute the sets of connected components (also including boundary)
                    neighbor_label_set = findAdjLabelSet(box_down_dyn_ext, labels_cut_down_ext, labels_cut_out_down_ext, n_comp)

                    print("Find Associated Components...")
                    # compute lists of wholes and non_wholes (then saved as set for compability with njit)
                    associated_label, isWhole = findAssociatedLabels(neighbor_label_set, n_comp)

                    if saveStatistics:
                        print("Computing n, mean, std ...")
                        comp_counts, comp_mean, comp_var = getStat(box_down_dyn_ext, labels_cut_out_down_ext, n_comp)

                        print("Writing Statistics...")
                        writeStatistics(n_comp, isWhole, comp_counts, comp_mean, comp_var, statistics_path, sample_name)

                    print("Fill wholes...")
                    # fill detected wholes and visualize non_wholes
                    labels_cut_filled = fillWholes(box_down_dyn_ext, labels_cut_ext, labels_cut_out_down_ext, associated_label, downsample)
                    labels[box_dyn[0]:box_dyn[1],box_dyn[2]:box_dyn[3],box_dyn[4]:box_dyn[5]] = labels_cut_filled[
                        box_idx[0]:box_idx[1],box_idx[2]:box_idx[3],box_idx[4]:box_idx[5]]

                    total_wholes_found += len({k: v for k, v in associated_label.items() if v != 0})

                    cell_counter+=1

        # print out total of found wholes
        print("Cells processed: " + str(cell_counter))
        print("Wholes filled (total): " + str(total_wholes_found))



        return labels

def processFile(data_path, sample_name, saveStatistics, vizWholes):

    output_path = data_path + sample_name + "_outp_" + str(time.time())[:10] +"/"
    os.mkdir(output_path)

    # bos size
    # box = [0,128,0,1000,0,1000]
    box = getBoxAll(data_path+sample_name+".h5")

    print("-----------------------------------------------------------------")

    # read in data
    labels = readData(box, data_path+sample_name+".h5")

    start_time = time.time()

    print("-----------------------------------------------------------------")

    # process chunk of data
    # overlap in points in one direction (total is twice)
    labels = processData(saveStatistics, output_path, sample_name, labels, downsample=1, overlap=0, rel_block_size=1)

    print("-----------------------------------------------------------------")
    print("Time elapsed: " + str(time.time() - start_time))

    # write filled data to H5
    output_name = "_filled"
    writeData(output_path+sample_name+output_name, labels)

    # compute negative to visualize filled wholes
    if vizWholes:
        labels_inp = readData(box, data_path+sample_name+".h5")
        neg = np.subtract(labels, labels_inp)
        output_name = "_wholes"
        writeData(output_path+sample_name+output_name, neg)

def main():

    # saveStatistics = True
    data_path = "/home/frtim/wiring/raw_data/segmentations/Zebrafinch/"
    output_path = "/home/frtim/wiring/raw_data/segmentations/Zebrafinch/sample_volume/"
    sample_name = "0768"
    # vizWholes = True

    box = [0,128,0,2000,0,2000]

    for i in range(0,15):
        name = str(i*128).zfill(4)
        print("Processing file " + name)
        if i is 0:
            labels_concat = readData(box, data_path+sample_name+".h5")
        else:
            labels_temp = readData(box, data_path+sample_name+".h5")
            labels_concat = np.concatenate((labels_concat,labels_temp),axis=0)
            del labels_temp

        print("Curent shape is: ", labels_concat.shape)

    print(labels_concat.nbytes)
    output_name = "concat_0_to_14_1920_2000_2000"
    writeData(output_path+output_name, labels_concat)

if __name__== "__main__":
  main()
