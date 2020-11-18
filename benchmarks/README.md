# Benchmarks

On an x86_64 3.7 GHz Intel Core i7-4820K CPU @ 3.70GHz with DDR3 1600 MHz RAM, I compared the performance of cc3d to the commonly used `scipy.ndimage.measurements.label` which supports 26-connected binary images. cc3d was designed to handle multilabel datasets more efficiently, and does so. Scipy appears to use a "runs" based algorithm, while cc3d uses a decision tree around each voxel. 

I compared the time and memory performance of both libraries on a 512x512x512 voxel cutout of a dense segmentation of a connectomics dataset at a resolution of 32x32x40 nm<sup>3</sup> containing 2523 labels and 3619 connected components. 

# Multi-Label Comparison

<p style="font-style: italics;" align="center">
<img height=384 src="https://raw.githubusercontent.com/seung-lab/connected-components-3d/master/benchmarks/cc3d_vs_scipy_multilabel.png" alt="Extracting components using SciPy vs cc3d on a 512x512x512 densely labeled connectomics segmentation. (black) 20% of SciPy 1.3.0 (blue) 100% of cc3d 1.2.2" /><br>
Fig. 1: Extracting components using SciPy vs cc3d on a 512x512x512 densely labeled connectomics segmentation. (black) 20% of SciPy 1.3.0 (blue) 100% of cc3d 1.2.2
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
  N = np.max(res)
  for segid in tqdm(range(1, N+1)):
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

In this test, `cc3d_test` was run to completion in 225 seconds after loading the image and processing it. `ndimage_test` was arrested manually after 496 iterations (20%) at 2,745 seconds as `tqdm` projected over three hours of total running time. SciPy's algorithm wins on memory pressure at about 1.7 GB peak usage versus cc3d using about 1.8 to 1.9 GB. SciPy performs poorly here because it must be run thousands of times after masking to expose individual labels since it only supports binary data. SciPy's average iteration per label took about 5.1 sec. It then must extract the individual components from the results of connected components, but this is fast. By contrast, since `cc3d` has native multi-label support, it needs to only be run once, with the bulk of time spent querying the resulting image for components.  

`cc3d` has a disadvantage compared with SciPy in that input images must only contain labels smaller than the maximum size of the image or that of a uint32 (whichever is smaller). That is why it is treated with `fastremap.renumber` prior to running. If it is known that the values of the labels are small compared with the image size, this additional step is not needed. The large difference in memory usage is due to the implementation of `cc3d`'s union-find data structure and the run based equivalences in SciPy. If we were to use an `std::unordered_map` in `cc3d`, it would slow down the performance of the initial run, but would reduce memory by about half, and prevent the need for using renumber. However, SciPy would still do better on memory due to its representations.

# 10x Head to Head: Connectomics Data

<p style="font-style: italics;" align="center">
<img height=384 src="https://github.com/seung-lab/connected-components-3d/blob/master/benchmarks/cc3d_vs_scipy_single_label_10x.png" alt="Fig. 2: SciPy vs cc3d run ten times on a 512x512x512 connectomics segmentation masked to only contain one label. (black) SciPy 1.5.2 (blue) cc3d 2.0.0" /><br>
Fig. 2: SciPy vs cc3d run ten times on a 512x512x512 connectomics segmentation masked to only contain one label. (black) SciPy 1.5.2 (blue) cc3d 2.0.0
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

This comparison was performed to show what happens when SciPy and `cc3d` are run on realistic single-label data. `cc3d` performs each iteration in 0.4 seconds while SciPy takes about 6.1 seconds. In previous experiments (not shown) on dense labels, `cc3d` takes about 7.6 seconds per an iteration, so it becomes faster when the volume is less dense. While in previous versions, cc3d used many times more memory than scipy in this experiment, as of version 2.0.0, the memory usage is now better than SciPy due to estimating the necessary number of provisional labels before executing.

| Trial             | MVx/sec | Rel. Perf. |
|-------------------|---------|------------|
| SciPy 1.5.2       | 17.8    | 1.00x      |
| cc3d 2.0.0        | 323.8   | 18.2x      |


# 10x Head to Head: Random Binary Images  

```python
import numpy as np
import scipy.ndimage.measurements
import cc3d

s = [
  [[1,1,1], [1,1,1], [1,1,1]],
  [[1,1,1], [1,1,1], [1,1,1]],
  [[1,1,1], [1,1,1], [1,1,1]]
]

labels = [ 
  np.random.randint(0,2, size=(384, 384, 384), dtype=np.bool) 
  for _ in range(10)
]

for label in labels:
  # scipy.ndimage.measurements.label(label, structure=s) # black
  cc3d.connected_components(label) # blue
```

## 26-connected

<p style="font-style: italics;" align="center">
<img height=384 src="https://raw.githubusercontent.com/seung-lab/connected-components-3d/master/benchmarks/cc3d_vs_scipy_random_binary_images_26.png" alt="Fig. 3: SciPy vs cc3d run ten times on ten 384x384x384 random binary images using 26-connectivity. (black) SciPy 1.5.2 (blue) cc3d 1.13.0" /><br>
Fig. 3: SciPy vs cc3d run ten times on ten 384x384x384 random binary images. (black) SciPy 1.5.2 (blue) cc3d 1.13.0
</p>   

On random binary images, SciPy marginally wins on memory with a peak memory cosumption of about 790 MB vs. cc3d with a peak consumption of about 800 MB (1.01x). However, SciPy doesn't perform as well as cc3d in running time with an average run time of 2.41 sec versus 1.03 sec per label set. Bear in mind that 10 binary images are stored in memory at once, inflating the baseline. Each image is about 56MB, so 10 of them are about 560MB. With interpreter overhead, the baseline is somewhere around 600 MB. Therefore, they use about 200 MB.

| Trial             | MVx/sec | MB/sec | Rel. Perf. |
|-------------------|---------|--------|------------|
| SciPy 1.5.2       | 23.4    | 23.4   | 1.00x      |
| cc3d 1.13.0       | 54.7    | 54.7   | 2.34x      |

## 6-connected

<p style="font-style: italics;" align="center">
<img height=384 src="https://raw.githubusercontent.com/seung-lab/connected-components-3d/master/benchmarks/cc3d_vs_scipy_random_binary_images_6.png" alt="Fig. 4: SciPy vs cc3d run ten times on ten 384x384x384 random binary images using 6-connectivity. (black) SciPy 1.5.2 (blue) cc3d 1.13.0" /><br>
Fig. 4: SciPy vs cc3d run ten times on ten 384x384x384 random binary images using 6-connectivity. (black) SciPy 1.5.2 (blue) cc3d 1.13.0
</p>

Here there's a slight difference in the memory usage. SciPy uses about 850 MB while cc3d uses 930 MB. Accounting for the ~600 MB of baseline, SciPy uses 250 MB and cc3d uses 330 MB (1.32x). At least for cc3d, the reason for additional memory usage is that 6-connectivity requires a larger union-find datastructure to handle the worst case than 26-connectivity (1/2 vs 1/8). The timings are more favorable though. scipy averages 1.35 seconds per volume vs cc3d averages 0.96 seconds per volume. 

| Trial             | MVx/sec | MB/sec | Rel. Perf. |
|-------------------|---------|--------|------------|
| SciPy 1.5.2       | 42.2    | 42.2   | 1.00x      |
| cc3d 1.13.0       | 59.2    | 59.2   | 1.40x      |

# 10x Head to Head: Black Cube

<p style="font-style: italics;" align="center">
<img height=384 src="https://raw.githubusercontent.com/seung-lab/connected-components-3d/master/benchmarks/cc3d_sparse_black.png" alt="Fig. 4: Different configurations run against a uint64 512x512x512 black cube using 26-connectivity. (black) SciPy 1.5.2 (blue) cc3d 2.0.0 with zeroth_pass off (red) cc3d 2.0.0 with zeroth_pass on." /><br>
Fig. 4: Different configurations run against a uint64 512x512x512 black cube using 26-connectivity. (black) SciPy 1.5.2 (blue) cc3d 2.0.0 with zeroth_pass off (red) cc3d 2.0.0 with zeroth_pass on.
</p>   

Sometimes empty data shows up in your pipeline. Sometimes a lot of it. How do your libraries handle it? At full speed? Slower? Faster than normal?  

Here we show scipy versus cc3d with `zeroth_pass` enabled and disabled using 26 connectivity. cc3d 2.0.0 contains optimizations for handling this case. In all modes, cc3d will skip the relabeling pass if provisional labels total fewer than two. In zeroth_pass mode, it will also skip the decision tree pass and memory allocation of data structures as well if it estimates zero  provisional voxels.  

We can see how this bears out. In black, scipy runs at a brisk and reasonable clip. In data not shown, it appears to have some optimization for black voxels as it runs more slowly on a solid color non-zero cube. cc3d with zeroth_pass off rushes through the decision tree and skips the relabeling. With zeroth_pass on, it skips everything except the scan for foreground labels.

Scipy and cc3d are approximately equal in memory usage with zeroth_pass off, but cc3d wins with it on.  


| Trial             | MVx/sec | Rel. Perf. |
|-------------------|---------|------------|
| SciPy 1.5.2       |  102    |  1.0x      |
| cc3d 2.0.0 off    |  336    |  3.3x      |
| cc3d 2.0.0 on     |  557    |  5.5x      |

# Historical Performance

cc3d has been steadily improving over time. To celebrate the release of 2.0.0, we show plots of peak memory usage and megavoxels per second vs version. Better scores in these charts trend down and right, indicating lower peak memory pressure and faster execution.

<p style="font-style: italics;" align="center">
<img height=512 src="https://raw.githubusercontent.com/seung-lab/connected-components-3d/master/benchmarks/cc3d_26-way_connectomics_over_time.png" alt="Fig. 5: 26-way cc3d peak memory usage and speed in selected releases against a 512x512x512 connectomics dataset." /><br>
Fig. 5: 26-way cc3d peak memory usage and speed in selected releases against a 512x512x512 connectomics dataset.
</p>   

<p style="font-style: italics;" align="center">
<img height=512 src="https://raw.githubusercontent.com/seung-lab/connected-components-3d/master/benchmarks/cc3d_26-way_random_binary_image.png" alt="Fig. 6: 26-way cc3d and scipy peak memory usage and speed in selected releases against a 512x512x512 random binary dataset." /><br>
Fig. 6: 26-way cc3d and scipy peak memory usage and speed in selected releases against a 512x512x512 random binary dataset.
</p>   

<p style="font-style: italics;" align="center">
<img height=512 src="https://raw.githubusercontent.com/seung-lab/connected-components-3d/master/benchmarks/cc3d_6-way_connectomics_over_time.png" alt="Fig. 7: 6-way cc3d peak memory usage and speed in selected releases against a 512x512x512 connectomics dataset." /><br>
Fig. 7: 6-way cc3d peak memory usage and speed in selected releases against a 512x512x512 connectomics dataset.
</p>   







