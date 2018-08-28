Connected Components 3D
=======================

Implementation of connected components in three dimensions using a 26-connected neighborhood.  
Uses a 3D variant of the two pass algorithm by Rosenfeld and Pflatz augmented with Union-Find.

Modern connected components algorithms appear to  do better by 2x-5x depending on the data, but there is
no superlinear improvement. I picked this algorithm mainly because it is easy to understand and implement.

Essentially, you raster scan, and every time you first encounter a foreground pixel, mark it with a new label if the pixels to its top and left are background. If there is a preexisting label in its neighborhood, use that label instead. Whenever you see that two labels are adjacent, record that we should unify them in the next pass. This equivalency table can be constructed in several ways, but some popular approaches are Union-Find with path compression and Selkow's algorithm (which can avoid pipeline stalls). However, Selkow's algorithm is designed for two trees of depth two, appropriate for binary images. We would like to process multiple labels at the same time, making union-find mandatory.

In the next pass, the pixels are relabeled using the equivalency table. Union-Find (disjoint sets) establishes one label as the root label of a tree, and so the root is considered the representative label. Each pixel is labeled with the representative label.
 
There appear to be some modern competing approaches involving decision trees, and an approach called "Light Speed Labeling". I'll try them out if this is too slow.
