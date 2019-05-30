# Benchmarks

On an x86_64 3.7 GHz Intel Core i7-4820K CPU @ 3.70GHz, I compared the performance of cc3d to the commonly used `scipy.ndimage.measurements.label` which supports 26-connected binary images. cc3d was designed to handle multilabel datasets more efficiently, and does so. Scipy appears to use a "runs" based algorithm, while cc3d uses a decision tree around each voxel. 

I compared the time and memory performance of both libraries on a 512x512x512 voxel cutout of a dense segmentation of a connectomics dataset at a resolution of 32x32x40 nm<sup>3</sup> containing 2523 labels and 3619 connected components. 

# Multi-Label Comparison

<p style="font-style: italics;" align="center">
<img height=384 src="https://raw.githubusercontent.com/seung-lab/connected-components-3d/master/benchmarks/cc3d_vs_scipy_multilabel.png" alt="Extracting components using SciPy vs cc3d on a 512x512x512 densely labeled connectomics segmentation. (black) 20% of SciPy 1.2.1 (blue) 100% of cc3d 1.1.1" /><br>
Fig. 1: Extracting components using SciPy vs cc3d on a 512x512x512 densely labeled connectomics segmentation. (black) 20% of SciPy 1.2.1 (blue) 100% of cc3d 1.1.1
</p>

```python
import cc3d
from tqdm import tqdm
import scipy.ndimage.measurements
import numpy as np
import fastremap

def cc3d_test(labels):
  labels, remap = fastremap.renumber(labels)
  res = cc3d.connected_components(labels)
  uniques = np.unique(res)[1:]
  for segid in tqdm(uniques):
    extracted = (res == segid)

def ndimage_test(labels):
  s = [
    [[1,1,1], [1,1,1], [1,1,1]],
    [[1,1,1], [1,1,1], [1,1,1]],
    [[1,1,1], [1,1,1], [1,1,1]]
  ]

  uniques = np.unique(labels)[1:]
  for segid in tqdm(uniques):
    extracted = (labels == segid)
    res, N = scipy.ndimage.measurements.label(extracted, structure=s)
    for ccid in tqdm(range(1,N+1)):
      extracted = (res == ccid)
```

In this test, `cc3d_test` was run to completion in 385 seconds after loading the image and processing it. `ndimage_test` was arrested manually after 500 iterations (20%) at 2,866 seconds as `tqdm` projected over three hours of total running time. SciPy's algorithm wins on memory pressure at about 1.7 GB peak usage versus cc3d using 2.8 to 2.9 GB. SciPy performs poorly here because it must be run thousands of times after masking to expose individual labels since it only supports binary data. SciPy's average iteration per label took about 5.7 sec. It then must extract the individual components from the results of connected components, but this is fast. By contrast, since `cc3d` has native multi-label support, it needs to only be run once, with the bulk of time spent querying the resulting image for components.  

`cc3d` has a disadvantage compared with SciPy in that input images must only contain labels smaller than the maximum size of the image or that of a uint32 (whichever is smaller). That is why it is treated with `fastremap.renumber` prior to running. If it is known that the values of the labels are small compared with the image size, this additional step is not needed. The large difference in memory usage is due to the implementation of `cc3d`'s union-find data structure and the run based equivalences in SciPy. If we were to use an `std::unordered_map` in `cc3d`, it would slow down the performance of the initial run, but would reduce memory by about half, and prevent the need for using renumber. However, SciPy would still do better on memory due to its representations.

# 10x Head to Head Comparison  

<p style="font-style: italics;" align="center">
<img height=384 src="https://github.com/seung-lab/connected-components-3d/blob/master/benchmarks/cc3d_vs_scipy_single_label_10x.png" alt="Fig. 2: SciPy vs cc3d run ten times on a 512x512x512 connectomics segmentation masked to only contain one label. (blue) SciPy 1.2.1 (black) cc3d 1.1.1" /><br>
Fig. 2: SciPy vs cc3d run ten times on a 512x512x512 connectomics segmentation masked to only contain one label. (blue) SciPy 1.2.1 (black) cc3d 1.1.1
</p> 

```python
import cc3d
from tqdm import tqdm
import scipy.ndimage.measurements
import numpy as np
import fastremap

labels = ...
labels, remap = fastremap.renumber(labels)
labels[labels != 6] = 0

s = [
  [[1,1,1], [1,1,1], [1,1,1]],
  [[1,1,1], [1,1,1], [1,1,1]],
  [[1,1,1], [1,1,1], [1,1,1]]
]

for i in tqdm(range(10)):
  # cc3d.connected_components(labels)
  scipy.ndimage.measurements.label(labels, structure=s)
```

This comparison was performed to show what happens when SciPy and `cc3d` are run on realistic single-label data. Here, we see again the difference in memory usage in SciPy's favor. However, `cc3d` performs each iteration in 1.2 seconds while SciPy takes about 6.2 seconds. In previous experiments (not shown) on dense labels, `cc3d` takes about 1.7 to 1.9 seconds per an iteration, so it becomes faster when the volume is less dense.
