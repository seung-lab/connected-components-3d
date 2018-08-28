from cloudvolume.lib import save_images

import cc3d
import numpy as np

input_labels = np.zeros( (17,17,1), dtype=np.bool )
input_labels[:] = 1
# input_labels[4:,:,:] = 2
input_labels[:,8,:] = 0
input_labels[8,:,:] = 0

input_labels[8,0,0] = 1
# input_labels[8,8,0]= 1

output_labels = cc3d.connected_components(input_labels).astype(np.uint8)

print(output_labels[:,:,0])

save_images(input_labels, directory='./save_images/input')
save_images(output_labels * 25, directory='./save_images/output')