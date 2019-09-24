import os
import h5py
import struct


import numpy as np
from PIL import Image


# def GridSize(prefix):
#     # return the size of the grid for this prefix
#     return meta_data.MetaData(prefix).GridSize()
#
#
#
# def Resolution(prefix):
#     # return the resolution for this prefix
#     return meta_data.MetaData(prefix).Resolution()
#
#
#
# def ReadImage(filename):
#     # return the image corresponding to this file
#     im = np.array(Image.open(filename))
#
#     return im


def ReadH5File(filename,box):
    # return the first h5 dataset from this file
    with h5py.File(filename, 'r') as hf:
        keys = [key for key in hf.keys()]
        print("Data keys are: ", str(keys))
        d = hf[keys[0]]
        print("Data shape: ", str(d.shape))
        # load selected part of data
        data = np.array(d[box[0]:box[1],box[2]:box[3],box[4]:box[5]])
# ,dtype=np.dtype(np.bool_)
    return data



def WriteH5File(data, filename, dataset):
    with h5py.File(filename, 'w') as hf:
        # should cover all cases of affinities/images
        hf.create_dataset(dataset, data=data, compression='gzip')




# def ReadPoints(prefix, label, dataset):
#     # get the filename for the segmentation
#     point_cloud_filename = '{}/{}/{:06d}.pts'.format(dataset, prefix, label)
#
#     prefix_zres, prefix_yres, prefix_xres = GridSize(prefix)
#
#     with open(point_cloud_filename, 'rb') as fd:
#         zres, yres, xres, npoints, = struct.unpack('qqqq', fd.read(32))
#         assert (zres == prefix_zres)
#         assert (yres == prefix_yres)
#         assert (xres == prefix_xres)
#         point_cloud = struct.unpack('%sq' % npoints, fd.read(8 * npoints))
#
#     return point_cloud
#
#
#
# def ReadAllPoints(prefix, dataset):
#     labels = [int(label[:-4]) for label in sorted(os.listdir('{}/{}'.format(dataset, prefix)))]
#
#     point_clouds = {}
#
#     # read all individual point clouds
#     for label in labels:
#         point_clouds[label] = ReadPoints(prefix, label, dataset)
#
#     return point_clouds
#
#
#
# def ReadWidths(prefix, label):
#     # get the filename with all of the widths
#     width_filename = 'widths/{}/{:06d}.pts'.format(prefix, label)
#
#     prefix_zres, prefix_yres, prefix_xres = GridSize(prefix)
#
#     widths = {}
#
#     with open(width_filename, 'rb') as fd:
#         zres, yres, xres, nelements, = struct.unpack('qqqq', fd.read(32))
#         assert (zres == prefix_zres)
#         assert (yres == prefix_yres)
#         assert (xres == prefix_xres)
#
#         for _ in range(nelements):
#             index, width, = struct.unpack('qf', fd.read(12))
#             widths[index] = width
#
#     # return the dictionary of widths for each skeleton point
#     return widths
