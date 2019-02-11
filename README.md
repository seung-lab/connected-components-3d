[![Build Status](https://travis-ci.org/seung-lab/connected-components-3d.svg?branch=master)](https://travis-ci.org/seung-lab/connected-components-3d) [![PyPI version](https://badge.fury.io/py/connected-components-3d.svg)](https://badge.fury.io/py/connected-components-3d)

Connected Components 3D
=======================

Implementation of connected components in three dimensions using a 26-connected neighborhood. This package uses a 3D variant of the two pass method by Rosenfeld and Pflatz augmented with Union-Find. This implementation is compatible with images containing many different labels, not just binary images. It can be used with 2D or 3D images. 

I wrote this package because I was working on densely labeled 3D biomedical images of brain tissue (e.g. 512x512x512 voxels). Other off the shelf implementations I reviewed were limited to binary images. This rendered these other packages too slow for my use case as it required masking each label and running the connected components algorithm once each time. For reference, there are often between hundreds to thousands of labels in a given volume. The benefit of this package is that it labels all connected components in one shot, improving performance by one or more orders of magnitude.

## Python `pip` Installaction

If binaries are available for your platform:

```bash
pip install connected-components-3d
```

Otherwise:  

*Requires a C++ compiler.*  

```bash
pip install numpy
pip install connected-components-3d --no-binary :all:
```

## Python Manual Installation

*Requires a C++ compiler.*

```bash
pip install -r requirements.txt
python setup.py develop
```

## Python Use

```python
from cc3d import connected_components
import numpy as np

labels_in = np.ones((512, 512, 512), dtype=np.int32)
labels_out = connected_components(labels_in)

# You can extract individual components like so:
segids = [ x for x in np.unique(labels_out) if x != 0 ]
extracted_image = labels_out * (labels_out == segid[0])
```

If you know approximately how many labels you are going to generate, you can save substantial memory by specifying a number a bit above that range. The max label ID in your input labels must be less than `max_labels`.

```python
labels_out = connected_components(labels_in, max_labels=20000)
```

*Note: C and Fortran order arrays will be processed in row major and column major order respectively, so the numbering of labels will be "transposed". The scare quotes are there because the dimensions of the array will not change.*

## C++ Use 

```cpp
#include "cc3d.hpp"

// 3d array represented as 1d array
int* labels = new int[512*512*512](); 

int* cc_labels = cc3d::connected_components3d<int>(
  labels, /*sx=*/512, /*sy=*/512, /*sz=*/512
);
```

## Algorithm Description

The algorithm contained in this package is an elaboration into 3D images of the 2D image connected components algorithm described by Rosenfeld and Pflatz in 1968 [1] (which is well illustrated by [this youtube video](https://www.youtube.com/watch?v=ticZclUYy88)) using an equivalency list implemented as a Union-Find disjoint set with path compression [2].  

In RP's two-pass method for 2D images, you raster scan and every time you first encounter a foreground pixel, mark it with a new label if the pixels to its top and left are background. If there is a preexisting label in its neighborhood, use that label instead. Whenever you see that two labels are adjacent, record they are equivalent as this will be used in the second pass. This equivalency table can be constructed in several ways, but some popular approaches are Union-Find with path compression and Selkow's algorithm (which can avoid pipeline stalls). However, Selkow's algorithm is designed for two trees of depth two, appropriate for binary images. We would like to process multiple labels at the same time, making union-find mandatory.

In the second pass, the pixels are relabeled using the equivalency table. Union-Find (disjoint sets) establishes one label as the root label of a tree, and the root is considered the representative label. Each pixel is then labeled with the representative label.

To move to a 3D 26-connected neighborhood, we must first note that RP's method is 4-connected, in that they only examine the pixel to the top and left. In 2D, the 8-connected version would have looked at the top, left, top-left, and top-right. In a 3D 26-connected case, we have to look at the top, left, top-left, top-right, the same pattern shifted by z-1, and the voxel located at (x, y, z-1).

In the literature, modern connected components algorithms appear to do better than the simple one I selected by about 2x-5x depending on the data. 
There appear to be some modern competing approaches involving decision trees, and an approach called "Light Speed Labeling". I picked this algorithm mainly because it is easy to understand and implement.  

In order to make a reasonably fast implementation, I implemented union-find with union by rank and path compression. I conservatively used two arrays (a uint32 rank array, and the IDs as the image data type) equal to the size of the image for the union-find data structure instead of a sparse map. The union-find data structure plus the output labels means the memory consumption will be input + output + rank + equivalences. If your input labels are 32-bit, the memory usage will be 4x the input size. This becomes more problematic when 64-bit labels are used, but if you know something about your data, you can decrease the size of the union-find data structure.

## References

1. A. Rosenfeld and J. Pfaltz. "Sequential Operations in Digital Picture Processing". Journal of the ACM. Vol. 13, Issue 4, Oct. 1966, Pg. 471-494. doi: 10.1145/321356.321357 ([link](https://dl.acm.org/citation.cfm?id=321357))
2. R. E. Tarjan. "Efficiency of a good but not linear set union algorithm". Journal of the ACM, 22:215-225, 1975. ([link](https://dl.acm.org/citation.cfm?id=321884))
