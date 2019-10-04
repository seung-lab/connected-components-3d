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
import sys

# set will be deprecated soon on numba, but until now an alternative has not been implemented
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

#read data from HD5, given the file path
def readData(box, filename):
    # read in data block
    data_in = ReadH5File(filename, box)

    labels = data_in

    # print("data read in; shape: " + str(data_in.shape) + "; DataType: " + str(data_in.dtype))

    return labels

# get shape of data saved in H5
def getBoxAll(filename):

    # return the first h5 dataset from this file
    with h5py.File(filename, 'r') as hf:
        keys = [key for key in hf.keys()]
        d = hf[keys[0]]
        box = [0,d.shape[0],0,d.shape[1],0,d.shape[2]]

    return box

@njit
def downsampleDataMinBorderAware(box, downsample, labels):

    if downsample%2!=0 and downsample!=1:
        raise ValueError("Error, downsampling only possible for even integers")
    box_down = [int(b*(1/downsample))for b in box]
    labels_down = np.zeros((box_down[1],box_down[3],box_down[5]),dtype=np.uint16)
    # dsf = downsamplefactor
    dsf = downsample

    for ix in range(0, box_down[1]-box_down[0]):
        for iy in range(0, box_down[3]-box_down[2]):
            for iz in range(0, box_down[5]-box_down[4]):

                temp = labels[ix*dsf:(ix+1)*dsf,iy*dsf:(iy+1)*dsf,iz*dsf:(iz+1)*dsf].flat
                if temp[0]==0:
                    labels_down[ix,iy,iz]=0
                else:
                    color = temp[0]
                    cont = True
                    i = 1
                    labels_down[ix,iy,iz]=color
                    while cont and i<len(temp):
                        if temp[i]==0:
                            labels_down[ix,iy,iz]=0
                            cont = False
                        elif temp[1]!=color:
                            labels_down[ix,iy,iz]=2000
                            labels_down[ix-1,iy,iz]=2000
                            labels_down[ix,iy-1,iz]=2000
                            labels_down[ix,iy,iz-1]=2000
                            cont=False
                        i = i+1
                # labels_down[ix,iy,iz]=np.min(labels[ix*dsf:(ix+1)*dsf,iy*dsf:(iy+1)*dsf,iz*dsf:(iz+1)*dsf])

    return box_down, labels_down

@njit
def downsampleDataMin(box, downsample, labels):

    if downsample%2!=0 and downsample!=1:
        raise ValueError("Error, downsampling only possible for even integers")
    box_down = [int(b*(1/downsample))for b in box]
    labels_down = np.zeros((box_down[1],box_down[3],box_down[5]),dtype=np.uint16)
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
def computeConnectedComp26(labels):
    connectivity = 26 # only 26, 18, and 6 are allowed
    labels_out = cc3d.connected_components(labels, connectivity=connectivity, max_labels=45000000)

    # You can extract individual components like so:
    n_comp = np.max(labels_out) + 1

    del labels_out
    # print("Conntected Regions found: " + str(n_comp))

    # determine indices, numbers and counts for the connected regions
    # unique, counts = np.unique(labels_out, return_counts=True)
    # print("Conntected regions and associated points: ")
    # print(dict(zip(unique, counts)))

    return n_comp

#compute the connected Com ponent labels
def computeConnectedComp6(labels):
    connectivity = 6 # only 26, 18, and 6 are allowed
    labels_out = cc3d.connected_components(labels, connectivity=connectivity, max_labels=45000000)

    # You can extract individual components like so:
    n_comp = np.max(labels_out) + 1

    # print("Conntected Regions found: " + str(n_comp))

    # determine indices, numbers and counts for the connected regions
    # unique, counts = np.unique(labels_out, return_counts=True)
    # print("Conntected regions and associated points: ")
    # print(dict(zip(unique, counts)))

    return labels_out, n_comp

# write statistics to a .txt filename
def writeStatistics(n_comp, isWhole, comp_counts, comp_mean, comp_var, data_path, sample_name):

    numbering = np.zeros((n_comp,1))
    numbering[:,0] = np.linspace(1,n_comp,num=n_comp)

    # create table that is later written to .txt
    statTable = np.hstack((numbering, isWhole, comp_counts, comp_mean, comp_var))

    filename = data_path + sample_name.replace("/","_").replace(".","_") + "_statistics" + ".txt"

    header = "number,isWhole,nPoints,mean_z,mean_y,mean_x,var_z,var_y,var_x"

    if(header.count(',')!=(statTable.shape[1]-1)):
        raise ValueError("Error! Header variables are not equal to number of columns in the statistics!")
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
    associated_label = Dict.empty(key_type=types.uint64,value_type=types.uint64)
    isWhole = np.ones((n_comp,1), dtype=np.int8)*-1

    for c in range(n_comp):
        # check that only connected to one component and that this component is not border (which is numbered as -1)
        if (len(neighbor_labels[c]) is 1 and neighbor_labels[c][0] is not -1 and neighbor_labels[c][0] is not 2000):
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
def fillWholes(box_dyn, labels, labels_cut_out_down, associated_label, downsample):

    dsf = downsample
    box = box_dyn

    for iz in range(box[0], box[1]):
        for iy in range(box[2], box[3]):
            for ix in range(box[4], box[5]):

                ic = math.floor((iz - box[0])/dsf)
                ib = math.floor((iy - box[2])/dsf)
                ia = math.floor((ix - box[4])/dsf)

                if labels[iz,iy,ix] == 0:
                    # if np.random.rand() < 0.0001 and associated_label[labels_cut_out_down[ic,ib,ia]]!=0:
                    #     print("Printing...")
                    #     print(iz,iy,ix,ic,ib,ia)
                    #     print(labels[iz-4:iz+4,iy-4:iy+4,ix])
                    #     print(labels_cut_out_down[ic-2:ic+2,ib-2:ib+2,ia])
                    #     printOn = True
                    labels[iz,iy,ix] = associated_label[labels_cut_out_down[ic,ib,ia]]

                # if printOn:
                #     print(labels[iz-4:iz+4,iy-4:iy+4,ix])
                #     print(associated_label[labels_cut_out_down[ic,ib,ia]])
                #     printOn = False

    return labels

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

        z_start = overlap_d if (bz*bs_z-overlap_d >= 0) else 0
        z_end = -overlap_d if ((bz+1)*bs_z+overlap_d <= box_down[1] and bz != n_blocks_z-1 and overlap!=0) else (box_down_dyn[1]-box_down_dyn[0]+overlap_d)
        y_start = overlap_d if (by*bs_y-overlap_d >= 0) else 0
        y_end = -overlap_d if ((by+1)*bs_y+overlap_d <= box_down[3] and by != n_blocks_y-1 and overlap!=0) else (box_down_dyn[3]-box_down_dyn[2]+overlap_d)
        x_start = overlap_d if (bx*bs_x-overlap_d >= 0) else 0
        x_end = -overlap_d if ((bx+1)*bs_x+overlap_d <= box_down[5] and bx != n_blocks_x-1 and overlap!=0) else (box_down_dyn[5]-box_down_dyn[4]+overlap_d)

        # compute the indexing to get from the extended dynamic box back to the standard size dynamic box
        box_idx = [z_start, z_end, y_start, y_end, x_start, x_end]

        # print("box_down_dyn_ext: " + str(box_down_dyn_ext))
        # print("box_dyn_ext: " + str(box_dyn_ext))
        # print("box_down_dyn: " + str(box_down_dyn))
        # print("box_dyn: " + str(box_dyn))
        # print("box_idx: " + str(box_idx))

        return box_down_dyn, box_dyn, box_down_dyn_ext, box_dyn_ext, box_idx

# process whole filling process for chung of data
def processData(saveStatistics, output_path, sample_name, labels, downsample, overlap, rel_block_size, borderAware):

        # read in chunk size
        box = [0,labels.shape[0],0,labels.shape[1],0,labels.shape[2]]

        # downsample data
        if downsample > 1:
            if borderAware:
                box_down, labels_down = downsampleDataMinBorderAware(box, downsample, labels)
                output_name = "dsp_borderAware" + str(downsample)
                writeData(output_path+output_name, labels_down)
            else:
                box_down, labels_down = downsampleDataMin(box, downsample, labels)
                output_name = "dsp_" + str(downsample)
                writeData(output_path+output_name, labels_down)
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

        # process blocks by iterating over all bloks
        for bz in range(n_blocks_z):
            for by in range(n_blocks_y):
                for bx in range(n_blocks_x):

                    # compute boxes (description in function)
                    box_down_dyn, box_dyn, box_down_dyn_ext, box_dyn_ext, box_idx = getBoxes(
                        box_down, overlap, overlap_d, downsample, bz, bs_z, n_blocks_z, by, bs_y, n_blocks_y, bx, bs_x, n_blocks_x)

                    # take only part of block
                    labels_cut_down_ext = labels_down[
                        box_down_dyn_ext[0]:box_down_dyn_ext[1],box_down_dyn_ext[2]:box_down_dyn_ext[3],box_down_dyn_ext[4]:box_down_dyn_ext[5]]

                    # take only part of block
                    labels_cut_ext = labels[
                        box_dyn_ext[0]:box_dyn_ext[1],box_dyn_ext[2]:box_dyn_ext[3],box_dyn_ext[4]:box_dyn_ext[5]]

                    # print(str(str(bz+1)+":Compute connected Components...").format(bz+1), end='\r')
                    # compute the labels of the conencted connected components
                    labels_cut_out_down_ext, n_comp = computeConnectedComp6(labels_cut_down_ext)

                    # print(str(str(bz+1)+":Find Sets of Adjacent Components...").format(bz+1), end='\r')
                    # compute the sets of connected components (also including boundary)
                    neighbor_label_set = findAdjLabelSet(box_down_dyn_ext, labels_cut_down_ext, labels_cut_out_down_ext, n_comp)

                    # print(str(str(bz+1)+":Find Associated Components......").format(bz+1), end='\r')
                    # compute lists of wholes and non_wholes (then saved as set for compability with njit)
                    associated_label, isWhole = findAssociatedLabels(neighbor_label_set, n_comp)

                    if saveStatistics:
                        # print(str(str(bz+1)+":Computing n, mean, std .........").format(bz+1), end='\r')
                        comp_counts, comp_mean, comp_var = getStat(box_down_dyn_ext, labels_cut_out_down_ext, n_comp)

                        # print(str(str(bz+1)+":Writing Statistics..............").format(bz+1), end='\r')
                        writeStattistics(n_comp, isWhole, comp_counts, comp_mean, comp_var, output_path, sample_name)

                    # print(str(str(bz+1)+":Fill wholes..................").format(bz+1), end='\r')
                    # fill detected wholes and visualize non_wholes

                    labels_cut_out_down = labels_cut_out_down_ext[box_idx[0]:box_idx[1],box_idx[2]:box_idx[3],box_idx[4]:box_idx[5]]
                    labels = fillWholes(box_dyn, labels, labels_cut_out_down, associated_label, downsample)

                    total_wholes_found += np.count_nonzero(isWhole)
                    cell_counter+=1

        # print out total of found wholes
        print("Cells processed: " + str(cell_counter))
        print("Wholes filled (total): " + str(total_wholes_found))

        del labels_cut_down_ext, labels_cut_ext, labels_cut_out_down, labels_cut_out_down_ext, labels_down, associated_label, isWhole, neighbor_label_set

        return labels, total_wholes_found

def processFile(box, data_path, sample_name, ID, saveStatistics, vizWholes, steps, downsample, overlap, rel_block_size, borderAware):

    for i in range(len(overlap)):
        if overlap[i]%downsample[i] != 0:
            raise ValueError("Overlap must be a multiple of downsample!")


    output_path = data_path + ID + "/"
    if os.path.exists(output_path):
        raise ValueError("Folderpath " + data_path + " already exists!")
    else:
        os.mkdir(output_path)

    print("-----------------------------------------------------------------")

    # read in data
    labels = readData(box, data_path+sample_name+".h5")

    start_time = time.time()

    print("-----------------------------------------------------------------")

    # process chunk of data
    # overlap in points in one direction (total is twice)

    if steps == 3:
        labels, _ = processData(saveStatistics=saveStatistics, output_path=output_path, sample_name=ID,
                    labels=labels, downsample=downsample[1], overlap=overlap[1], rel_block_size=rel_block_size[1], borderAware=borderAware)

    if steps >= 1:
        labels, n_wholes = processData(saveStatistics=saveStatistics, output_path=output_path, sample_name=ID,
                    labels=labels, downsample=downsample[0], overlap=overlap[0], rel_block_size=rel_block_size[0], borderAware=borderAware)
    else:
        raise ValueError("Number of steps is smaller than 1")

    if steps >= 2:
        labels, _ = processData(saveStatistics=saveStatistics, output_path=output_path, sample_name=ID,
                    labels=labels, downsample=downsample[1], overlap=overlap[1], rel_block_size=rel_block_size[1], borderAware=borderAware)

    print("-----------------------------------------------------------------")
    print("Time elapsed during " + str(steps) + " steps: " + str(time.time() - start_time))

    # write filled data to H5
    output_name = "filled_" + ID
    writeData(output_path+output_name, labels)

    # compute negative to visualize filled wholes
    if vizWholes:
        labels_inp = readData(box, data_path+sample_name+".h5")
        neg = np.subtract(labels, labels_inp)
        output_name = "wholes_" + ID
        writeData(output_path+output_name, neg)

    del labels_inp, neg, labels
    return n_wholes

def concatFiles(box, slices_s, slices_e, output_path, data_path):

    for i in range(slices_s,slices_e+1):
        sample_name = str(i*128).zfill(4)
        print(str("Processing file " + sample_name).format(sample_name), end='\r')
        if i is slices_s:
            labels_concat = readData(box, data_path+sample_name+".h5")
        else:
            labels_temp = readData(box, data_path+sample_name+".h5")
            labels_old = labels_concat.copy()
            del labels_concat
            labels_concat = np.concatenate((labels_old,labels_temp),axis=0)
            del labels_temp

    print("Concat size/ shape: " + str(labels_concat.nbytes) + '/ ' + str(labels_concat.shape))
    writeData(output_path, labels_concat)

    del labels_concat

def evaluateWholes(folder_path,ID,sample_name,n_wholes):
    print("Evaluating wholes...")
    # load gt wholes
    gt_wholes_filepath = folder_path+"/gt/wholes_gt"+".h5"
    box = getBoxAll(gt_wholes_filepath)
    wholes_gt = readData(box, gt_wholes_filepath)

    # load block wholes
    inBlocks_wholes_filepath = folder_path+"/"+ID+"/"+"wholes_"+ID+".h5"
    box = getBoxAll(inBlocks_wholes_filepath)
    wholes_inBlocks = readData(box, inBlocks_wholes_filepath)

# check that both can be converted to int16
    if np.max(wholes_gt)>32767 or np.max(wholes_inBlocks)>32767:
        raise ValueError("Cannot convert wholes to int16 (max is >32767)")

    wholes_gt = wholes_gt.astype(np.int16)
    wholes_inBlocks = wholes_inBlocks.astype(np.int16)
    wholes_gt = np.subtract(wholes_gt, wholes_inBlocks)
    diff = wholes_gt
    # free some RAM
    del wholes_gt, wholes_inBlocks

    print("Freed memory")

    if np.min(diff)<0:
        FP = diff.copy()
        FP[FP>0]=0
        print(FP.shape)
        n_points_FP = np.count_nonzero(FP)
        n_comp_FP = computeConnectedComp26(FP)-1
        print("FP classifications (points/components): " + str(n_points_FP) + "/ " +str(n_comp_FP))
        del FP
    else:
        print("No FP classification")

    if np.max(diff)>0:
        FN = diff.copy()
        FN[FN<0]=0
        n_points_FN = np.count_nonzero(FN)
        n_comp_FN = computeConnectedComp26(FN)-1
        print("FN classifications (points/components): " + str(n_points_FN) + "/ " +str(n_comp_FN))
        print("Percentage (total wholes is "+str(n_wholes)+"): "+str(float(n_comp_FN)/float(n_wholes)))
        del FN
    else:
        print("No FN calssifications")

    output_name = 'diff_wholes_'+ID
    writeData(folder_path+"/"+ID+"/"+output_name, diff)

    del diff

def main():

    data_path = "/home/frtim/wiring/raw_data/segmentations/Zebrafinch/"
    output_path = "/home/frtim/wiring/raw_data/segmentations/Zebrafinch/stacked_volumes/"
    vizWholes = True
    saveStatistics = False
    box_concat = [0,128,0,400,0,400]
    slices_start = 4
    slices_end = 6

    # sample_name = "ZF_concat_0to15_2048_2048"
    # folder_path = output_path + sample_name + "_outp/"
    # n_wholes = 1472441

    sample_name = "ZF_concat_"+str(slices_start)+"to"+str(slices_end)+"_"+str(box_concat[3])+"_"+str(box_concat[5])
    folder_path = output_path + sample_name + "_outp_" + time.strftime("%Y%m%d_%H_%M_%S") + "/"
    os.mkdir(folder_path)

    # concat files
    concatFiles(box=box_concat, slices_s=slices_start, slices_e=slices_end, output_path=folder_path+sample_name, data_path=data_path)

    # compute groundtruth (in one block)
    box = getBoxAll(folder_path+sample_name+".h5")
    n_wholes = processFile(box=box, data_path=folder_path, sample_name=sample_name, ID="gt", saveStatistics=saveStatistics, vizWholes=vizWholes,
                steps=1, downsample=[1], overlap=[0], rel_block_size=[1], borderAware=False)


    # ID = "BA_"+str(borderAware)+"_DS_"+str(downsample)+"_OL_"+str(overlap)+"_BS_"+str(block_size).replace(".","_")+"_S_" +str(steps)
    #
    # timestr0 = time.strftime("%Y%m%d_%H_%M_%S")
    # f = open(folder_path + ID + timestr0 + '.txt','w')
    # sys.stdout = f
    # print(ID)
    #
    # box = getBoxAll(folder_path+sample_name+".h5")
    # processFile(box=box, data_path=folder_path, sample_name=sample_name, ID=ID, saveStatistics=saveStatistics, vizWholes=vizWholes,
    #             steps=steps, downsample=[downsample,1], overlap=[0,overlap*(downsample-1)], rel_block_size=[1,block_size], borderAware=borderAware)
    # evaluateWholes(folder_path=folder_path,ID=ID,sample_name=sample_name,n_wholes=n_wholes)



if __name__== "__main__":
  main()
