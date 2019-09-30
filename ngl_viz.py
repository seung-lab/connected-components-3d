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
    return data

def loadViz(path, caption, res):

    print("-----------------------------------------------------------------")
    print ('loading ' + caption + "...")
    gt = readData(path)
    gt = gt.astype(np.uint16)

    uniq = np.expand_dims(np.unique(gt[::2,::2,::2]),axis=1).transpose()
    np.savetxt(sys.stdout.buffer, uniq, delimiter=',', fmt='%d')

    with viewer.txn() as s:
        s.layers.append(
            name=caption,
            layer=neuroglancer.LocalVolume(
                data=gt,
                voxel_size=res,
                volume_type='segmentation'
            ))

data_path = "/home/frtim/wiring/raw_data/segmentations/Zebrafinch/sample_volume/"
sample_name= "concat_5_600"
sample_name_gt = "concat_5_600_gt"

# file_name_org = data_path + sample_name + "_outp/" + sample_name + ".h5"
# file_name_filled = data_path + sample_name + "_outp/" + sample_name + "_filled.h5"
# file_name_nonwholes = data_path + sample_name + "_outp/" + sample_name + "_nonwholes.h5"
file_name_wholes = data_path + sample_name + "_outp/" + sample_name + "_wholes.h5"
file_name_wholes_gt = data_path + sample_name_gt + "_outp/" + sample_name_gt + "_wholes.h5"


res=[20,18,18]; # resolution of the data

# loadViz(path=file_name_org, caption="original", res=res)
# loadViz(path=file_name_filled, caption="filled", res=res)
loadViz(path=file_name_wholes, caption="wholes", res=res)
loadViz(path=file_name_wholes_gt, caption="wholes_gt", res=res)
# loadViz(path=file_name_nonwholes, caption="non_wholes", res=res)

print("----------------------------DONE---------------------------------")
print(viewer)
print("-----------------------------------------------------------------")
