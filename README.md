[![Build Status](https://travis-ci.org/seung-lab/connected-components-3d.svg?branch=master)](https://travis-ci.org/seung-lab/connected-components-3d)

Connected Components 3D
=======================

Implementation of connected components in three dimensions using a 26-connected neighborhood.  
Uses a 3D variant of the two pass algorithm by Rosenfeld and Pflatz augmented with Union-Find. This implementation is compatible with images containing many different labels, not just binary images.

Modern connected components algorithms appear to  do better by 2x-5x depending on the data, but there is
no superlinear improvement. I picked this algorithm mainly because it is easy to understand and implement. In order to make a reasonably fast implementation, I conservatively used a large array for the union-find data structure instead of a map. This makes the memory usage larger than it would otherwise be, but possibly still small enough for our purposes.

Essentially, you raster scan, and every time you first encounter a foreground pixel, mark it with a new label if the pixels to its top and left are background. If there is a preexisting label in its neighborhood, use that label instead. Whenever you see that two labels are adjacent, record that we should unify them in the next pass. This equivalency table can be constructed in several ways, but some popular approaches are Union-Find with path compression and Selkow's algorithm (which can avoid pipeline stalls). However, Selkow's algorithm is designed for two trees of depth two, appropriate for binary images. We would like to process multiple labels at the same time, making union-find mandatory.

In the next pass, the pixels are relabeled using the equivalency table. Union-Find (disjoint sets) establishes one label as the root label of a tree, and so the root is considered the representative label. Each pixel is labeled with the representative label.
 
There appear to be some modern competing approaches involving decision trees, and an approach called "Light Speed Labeling". If memory becomes a problem, these algorithms might be of use.


## C++ Use 

```cpp
#include "cc3d.hpp"

// 3d array represented as 1d array
int* labels = new int[512*512*512](); 

path = cc3d::connected_components3d<int>(
  labels, /*sx=*/512, /*sy=*/512, /*sz=*/512
);
```

## Python Installation

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
```

If you know approximately how many labels you are going to generate, you can save substantial memory by specifying a number a bit above that range. The max label ID in your input labels must be less than `max_labels`.

```python
labels_out = connected_components(labels_in, max_labels=20000)
```

## Tips for Memory Pressure Management

If the allocations are too much, you have options if you know something about the data. If you know that there will be fewer than 2<sup>16</sup> labels, you can try using a uint16_t output instead of a uint32_t output. If you know the minimum component size, you can try allocating a smaller union-find data structure.

## References

1. A. Rosenfeld and J. Pfaltz. "Sequential Operations in Digital Picture Processing". Journal of the ACM. Vol. 13, Issue 4, Oct. 1966, Pg. 471-494. doi: 10.1145/321356.321357 ([link](https://dl.acm.org/citation.cfm?id=321357))
2. R. E. Tarjan. "Efficiency of a good but not linear set union algorithm". Journal of the ACM, 22:215-225, 1975. ([link](https://dl.acm.org/citation.cfm?id=321884))
