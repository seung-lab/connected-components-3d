import neuroglancer
import numpy as np
import sys
import h5py
import sys

ip='localhost' # or public IP of the machine for sharable display
port=98092 # change to an unused port number
neuroglancer.set_server_bind_address(bind_address=ip,bind_port=port)
viewer=neuroglancer.Viewer()

#read data from HD5, given the file path
def readData(filename):
    # read in data block
    data_in = ReadH5File(filename)

    labels = data_in

    print("data read in; shape: " + str(data_in.shape) + "; DataType: " + str(data_in.dtype) + "; cut to: " + str(labels.shape))

    return labels

def ReadH5File(filename):
    # return the first h5 dataset from this file
    with h5py.File(filename, 'r') as hf:
        keys = [key for key in hf.keys()]
        print("Data keys are: ", str(keys))
        data = np.array(hf[keys[0]])
# ,dtype=np.dtype(np.bool_)
    return data

data_path = "/home/frtim/wiring/raw_data/segmentations/"
sample_name = "JWR/test_volume/output/cell032_downsampled_1569513764.h5"

res=[20,18,18]; # resolution of the data

print ('load im')
im = readData(data_path+sample_name)
im = im.astype(np.uint16)

uniq = np.expand_dims(np.unique(im[::5,::5,::5]),axis=1).transpose()
print(uniq.shape)
np.savetxt(sys.stdout.buffer, uniq, delimiter=',', fmt='%d')

print("Max label is: " + str(np.max(im)))
print("Min label is: " + str(np.min(im)))

with viewer.txn() as s:
    s.layers.append(
        name='labels',
        layer=neuroglancer.LocalVolume(
            data=im,
            voxel_size=res,
            volume_type='segmentation'
        ))

print(viewer)
