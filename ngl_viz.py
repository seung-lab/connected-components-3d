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

def loadViz(path, caption, res, idRes):

    print("-----------------------------------------------------------------")
    print ('loading ' + caption + "...")
    gt = readData(path)
    gt = gt.astype(np.uint16)

    uniq = np.expand_dims(np.unique(gt[::idRes,::idRes,::idRes]),axis=1).transpose()
    np.savetxt(sys.stdout.buffer, uniq, delimiter=',', fmt='%d')

    with viewer.txn() as s:
        s.layers.append(
            name=caption,
            layer=neuroglancer.LocalVolume(
                data=gt,
                voxel_size=res,
                volume_type='segmentation'
            ))

idRes = 1 #which resolution to use to search for IDs
res=[20,18,18]; # resolution of the data
res_4 = [80,72,72]
data_path = "/home/frtim/wiring/raw_data/segmentations/Zebrafinch/stacked_volumes/"
sample_name= "concat_5_500_test"

file_name_org =             data_path + sample_name + "_outp/" + sample_name + ".h5"
file_name_filled_gt =       data_path + sample_name + "_outp/" + sample_name + "_filled_gt.h5"
file_name_filled_inBlocks = data_path + sample_name + "_outp/" + sample_name + "_filled_inBlocks.h5"
file_name_wholes_gt =       data_path + sample_name + "_outp/" + sample_name + "_wholes_gt.h5"
file_name_wholes_inBlocks = data_path + sample_name + "_outp/" + sample_name + "_wholes_inBlocks.h5"
file_name_diff_wholes =     data_path + sample_name + "_outp/" + sample_name + "_diff_wholes.h5"
file_name_dsp =             data_path + sample_name + "_outp/" + sample_name + "_dsp_4.h5"

loadViz(path=file_name_org,             caption="original",         res=res, idRes=idRes)
# loadViz(path=file_name_filled_gt,       caption="filled_gt",        res=res, idRes=idRes)
# loadViz(path=file_name_filled_inBlocks, caption="filled_inBlocks",  res=res, idRes=idRes)
# loadViz(path=file_name_wholes_gt,       caption="wholes_gt",        res=res, idRes=idRes)
# loadViz(path=file_name_wholes_inBlocks, caption="wholes_inBlocks",  res=res, idRes=idRes)
loadViz(path=file_name_diff_wholes,     caption="diff_wholes",      res=res, idRes=idRes)
loadViz(path=file_name_dsp,             caption="dsp",              res=res_4, idRes=idRes)

print("----------------------------DONE---------------------------------")
print(viewer)
print("-----------------------------------------------------------------")
