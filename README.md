[![Build Status](https://travis-ci.org/seung-lab/connected-components-3d.svg?branch=master)](https://travis-ci.org/seung-lab/connected-components-3d) [![PyPI version](https://badge.fury.io/py/connected-components-3d.svg)](https://badge.fury.io/py/connected-components-3d)

Connected Components 3D
=======================

Implementation of connected components in three dimensions using a 26, 18, or 6 connected neighborhood in 3D or 4 and 8-connected in 2D. This package uses a 3D variant of the two pass method by Rosenfeld and Pflatz augmented with Union-Find and a decision tree based on the 2D 8-connected work of Wu, Otoo, and Suzuki. This implementation is compatible with images containing many different labels, not just binary images. It can be used with 2D or 3D images. 

I wrote this package because I was working on densely labeled 3D biomedical images of brain tissue (e.g. 512x512x512 voxels). Other off the shelf implementations I reviewed were limited to binary images. This rendered these other packages too slow for my use case as it required masking each label and running the connected components algorithm once each time. For reference, there are often between hundreds to thousands of labels in a given volume. The benefit of this package is that it labels all connected components in one shot, improving performance by one or more orders of magnitude.

Check out [benchmarks](https://github.com/seung-lab/connected-components-3d/tree/master/benchmarks) to see a comparison with SciPy on a few different tasks.  

## Python `pip` Installaction

If compatible binaries are available for your platform, installation is particularly simple. 

```bash
pip install connected-components-3d
```

If compatible binaries are not available, you can install from source as follows. 

*Requires a C++ compiler.*  

```bash
pip install numpy
pip install connected-components-3d --no-binary :all:
```

Occasionally, you may appear to successfully install cc3d, but on import you'll see an error that includes: `numpy.ufunc size changed, may indicate binary incompatibility`. cc3d was compiled against numpy 1.16+ and unfortunately, there was a backwards incompatibilty between numpy 1.15 and 1.16. You can either try upgrading numpy or compiling from source in this case.

## Python Manual Installation

*Requires a C++ compiler.*

```bash
pip install -r requirements.txt
python setup.py develop
```

## Python Use

Important limitation: Only label values less than or equal to the size of the image in voxels (pixels) are supported currently. If you want to use larger values, consider using [fastremap.renumber](https://github.com/seung-lab/fastremap/).

```python
import cc3d
import numpy as np

labels_in = np.ones((512, 512, 512), dtype=np.int32)
labels_out = cc3d.connected_components(labels_in) # 26-connected

connectivity = 6 # only 26, 18, and 6 are allowed
labels_out = cc3d.connected_components(labels_in, connectivity=connectivity)

# You can adjust the bit width of the output to accomodate
# different expected image statistics with memory usage tradeoffs.
# uint16, uint32 (default), and uint64 are supported.
labels_out = cc3d.connected_components(labels_in, out_dtype=np.uint16)

# You can extract individual components like so:
N = np.max(labels_out)
for segid in range(1, N+1):
  extracted_image = labels_out * (labels_out == segid)
  process(extracted_image)

# We also include a region adjacency graph function 
# that returns a set of undirected edges.
graph = cc3d.region_graph(labels_out, connectivity=connectivity) 
```

If you know approximately how many labels you are going to generate, you can save some memory by specifying a number a safety factor above that range. The max label ID in your input labels must be less than `max_labels`.

```python
labels_out = connected_components(labels_in, max_labels=20000)
```

*Note: C and Fortran order arrays will be processed in row major and column major order respectively, so the numbering of labels will be "transposed". The scare quotes are there because the dimensions of the array will not change.*

## C++ Use 

```cpp
#include "cc3d.hpp"

// 3d array represented as 1d array
int* labels = new int[512*512*512](); 

uint32_t* cc_labels = cc3d::connected_components3d<int>(
  labels, /*sx=*/512, /*sy=*/512, /*sz=*/512
);

// The default template parameter for output type is uint32_t
uint64_t* cc_labels = cc3d::connected_components3d<int, uint64_t>(
  labels, /*sx=*/512, /*sy=*/512, /*sz=*/512
);

uint16_t* cc_labels = cc3d::connected_components3d<int, uint16_t>(
  labels, /*sx=*/512, /*sy=*/512, /*sz=*/512, 
  /*connectivity=*/18 // default is 26 connected
);

// edges is [ e11, e12, e21, e22, ... ]
std::vector<uint64_t> edges = cc3d::extract_region_graph<uint64_t>(
  labels, /*sx=*/512, /*sy=*/512, /*sz=*/512, 
  /*connectivity=*/18 // default is 26 connected
);
```

## Algorithm Description

The algorithm contained in this package is an elaboration into 3D images of the 2D image connected components algorithm described by Rosenfeld and Pflatz (RP) in 1968 [1] (which is well illustrated by [this youtube video](https://www.youtube.com/watch?v=ticZclUYy88)) using an equivalency list implemented as Tarjan's Union-Find disjoint set with path compression and balancing [2] and augmented with a decision tree based on work by Wu, Otoo, and Suzuki (WOS). [3] The description below describes the 26-connected algorithm, but once you understand it, deriving 18 and 6 are simple.

### First Principles in 2D

In RP's 4-connected two-pass method for binary 2D images, the algorithm raster scans and every time it first encounters a foreground pixel (the pixels to its top and left are background), it marks it with a new label. If there is a preexisting label in its neighborhood, it uses that label instead. Whenever two labels are adjacent, it records they are equivalent so that they can be relabeled consistently in the second pass. This equivalency table can be constructed in several ways, but some popular approaches are Union-Find with path compression with balancing by rank and Selkow's algorithm (which can avoid pipeline stalls). [4] However, Selkow's algorithm is designed for two trees of depth two, appropriate for binary images. We would like to process multiple labels at the same time, making Union-Find preferable.

In the second pass, the pixels are relabeled using the equivalency table. Union-Find establishes one label as the root label of a tree, and the root is considered the representative label. Each pixel is then labeled with the representative label. Union-Find is therefore appropriate for representing disjoint sets. Path compression with balancing radically reduces the height of the tree, which accelerates the second pass.  

WOS approached the problem of accelerating 8-connected 2D connected components on binary images. 8-connected labeling is achieved by extending RP's forward pass mask to the top left and top right corner pixels. In Union-Find based connected components algorithms, the unify step in the first pass is the most expensive step. WOS showed how to optimize away a large fraction of these calls using a decision tree that takes advantage of local topology. For example, since the top-center neighbor of the current pixel is also adjacent to the other mask elements, all of which have already been processed by virtue of the raster scan direction, if it is present it is sufficient to copy its value and move on. If it is absent, pick one of the remaining foreground pixels, copy their value, and use unify for the mask element on the right as it is now known to be non-neighboring with the left hand side. WOS's algorithm continues in this fashion until a match is found or all mask elements are processed at which point a new label is created.  

For several years, this algorithm was the world's fastest, though it has been superceded by a newer work that exchanges the static decision tree for a dynamic one or precalculated generated one amongst other improvements. However, WOS's work is significant for both its simplicity and speed and thus serves as the inspiration for this library. For 2D 8-connected images, we provide a specialization using Wu et al's original decision tree for a slight performance boost.

We're interested in exploring the block based approaches of Grana, Borghesani, and Cucchiara ([5],[7]), however their approach appears to critically rely on binary images. We'll continue to think about ways to incorporate it. We also considered the approach of He et al [8] which is also supposed to modestly faster than than WOS. However, it substitutes the Union-Find data structure (one array) with three arrays, which imposes a memory requirement that is at odds with our goal of processing large images.

### Extending to 3D

The approach presented below is very similar to that of Sutheebanjard [6]. To move to a 3D 26-connected neighborhood, the mask must be extended into three dimensions in order to connect neighboring planes. Observe that the 8-connected mask covers the trailing half of the neighborhood (the part that will have been already processed) such that the current pixel can rely on those labels. Thus the mask for the 26-connected neighborhood covers only two out of three potential planes: the entire lower plane (nine voxels), and a mask identical to WOS's (four voxels) on the current plane. While some further optimizations are possible, to begin, the problem can be conceptually decomposed into two parts: establishing a 9-connected link to the bottom plane and then an 8-connected link to the current plane. This works because the current pixel functions as a hub that transmits the connection information from the 9-connected step to the 8-connected step.  

Fig. 1: Mask for an 8-connected plane. If J,K,L, and M are all eliminated, only N remains and a new label is assigned.

| j | k | l |
|---|---|---|
| m | n | . |
| . | . | . |

The very first Z plane (Z=0) the algorithm runs against is special: the edge effect omits the bottom plane of the mask. Therefore, as the remaining mask is only comprosed of the 8-connected 2D mask, after this pass, the bottom of the image is 8-connected. At Z=1, the 9-connected part of the mask kicks in, forming connections to Z=0, making the current plane now (8 + 9) 17-connected. At Z=2, the 9-connected bottom mask now forms connections from Z=1 to Z=2 on the top, making Z=1 (17 + 9) 26-connected. By induction, when this process proceeds to completion it results in a 26-connected labeling of the volume.   

Following inspiration from WOS, we construct a decision tree on the densely labeled bottom plane that minimizes the number of unifications we need to perform.   

Fig 2. The mask for the lower plane in 3D.

| a | b | c |
|---|---|---|
| d | e | f |
| g | h | i |

As `e` is connected to all other voxels, if present, it can simply be copied. If `e` is absent, `b` and `h` fully cover the mask. If `b` is absent, `h`, `a`, `c` comprise a covering. If `h` is absent, `b`, `g`, `i` are one. Below is a list of coverings such that each proceeding entry in the list assumes the first letters in the entries above are background.

1. `e`
2. `b`, (`h` | `g`, `i`)
3. `h`, `a`, `c`
4. `d`, (`f` | `c`, `i`)
5. `f`, `g`, `a`
6. `a`, `c`, `g`, `i`
7. `c`, `g`, `i`
8. `g`, `i`
9. `i`

The decision tree is then constructed such that each of these coverings will be evaluated using the fewest unifications possible. It's possible to further optimize this by noting that `e` and `b` are both fully connected to the upper 2D mask. Therefore, if either of them are present, we can skip the 8-connected unification step. It's also possible to try the DF covering first if B is background, which would save one unification versus HAC given even statistics, but it seems to be slightly slower on the dataset I attempted. To move from binary data to multilabel data, I simply replaced tests for foreground and background with tests for matching labels.

In order to make a reasonably fast implementation, I implemented union-find with path compression. I conservatively used an IDs array qual to the size of the image for the union-find data structure instead of a sparse map. The union-find data structure plus the output labels means the memory consumption will be input + output + rank + equivalences. If your input labels are 32-bit, the memory usage will be 4x the input size. This becomes more problematic when 64-bit labels are used, but if you know something about your data, you can decrease the size of the union-find data structure. I previously used union-by-size but for some reason it merely reduced performance and increased memory usage so it was removed.

For more information on the history of connected components algorithms, and an even faster approach for 2D 8-connected components, consult Grana et al's paper on Block Based Decision Trees. [5]

## References

1. A. Rosenfeld and J. Pfaltz. "Sequential Operations in Digital Picture Processing". Journal of the ACM. Vol. 13, Issue 4, Oct. 1966, Pg. 471-494. doi: 10.1145/321356.321357 ([link](https://dl.acm.org/citation.cfm?id=321357))
2. R. E. Tarjan. "Efficiency of a good but not linear set union algorithm". Journal of the ACM, 22:215-225, 1975. ([link](https://dl.acm.org/citation.cfm?id=321884))
3. K. Wu, E. Otoo, K. Suzuki. "Two Strategies to Speed up Connected Component Labeling Algorithms". Lawrence Berkely National Laboratory. LBNL-29102, 2005. ([link](https://crd-legacy.lbl.gov/~kewu/ps/LBNL-59102.html))
4. S. Selkow. "The Tree-to-Tree Editing Problem". Information Processing Letters. Vol. 6, No. 6. June 1977. doi: 10.1016/0020-0190(77)90064-3 ([link](http://www.grantjenks.com/wiki/_media/ideas:tree-to-tree_editing_problem.pdf))
5. C. Grana, D. Borghesani, R. Cucchiara. "Optimized Block-based Connected Components Labeling with Decision Trees". IEEE Transactions on Image Porcessing. Vol. 19, Iss. 6. June 2010. doi: 10.1109/TIP.2010.2044963 ([link](http://imagelab.ing.unimore.it/imagelab/pubblicazioni/2009TIPlabeling.pdf))
6. P. Sutheebanjard. "Decision Tree for 3-D Connected Components Labeling".  Proc. 2012 International Symposium on Information Technology in Medicine and EEducation. doi: 10.1109/ITiME.2012.6291402 ([link](https://ieeexplore.ieee.org/abstract/document/6291402/authors#authors))
7. C. Grana, D. Borghesani, R. Cucchiara. "Fast Block Based Connected Components Labeling". Proc. 16th IEEE Intl. Conf. on Image Processing. 2009. doi: 10.1109/ICIP.2009.5413731 ([link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=5413731&casa_token=grvS_69THEYAAAAA:DKpVSoo6nUnI6liLel54kiGdK_ee1qMyPaFYXe_9OGKX0iBDtf9p6ks6mf9twZdK0YPM_SQb&tag=1))
8. L. He, Y. Chao and K. Suzuki, "A Linear-Time Two-Scan Labeling Algorithm", IEEE International Conference on Image Processing, vol. 5, pp. 241-244, 2007.
