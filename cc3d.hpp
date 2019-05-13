/*
 * Connected Components for 3D images. 
 * Implments a 3D variant of the two pass algorithim by
 * Rosenfeld and Pflatz augmented with Union-Find.
 * 
 * Modern connected components algorithms appear to 
 * do better by 2x-5x depending on the data, but there is
 * no superlinear improvement. I picked this algorithm mainly
 * because it is easy to understand and implement.
 *
 * Essentially, you raster scan, and every time you first encounter 
 * a foreground pixel, mark it with a new label if the pixels to its
 * top and left are background. If there is a preexisting label in its
 * neighborhood, use that label instead. Whenever you see that two labels
 * are adjacent, record that we should unify them in the next pass. This
 * equivalency table can be constructed in several ways, but some popular
 * approaches are Union-Find with path compression and Selkow's algorithm
 * (which can avoid pipeline stalls). However, Selkow's algorithm is designed
 * for two trees of depth two, appropriate for binary images. We would like to 
 * process multiple labels at the same time, making union-find mandatory.
 *
 * In the next pass, the pixels are relabeled using the equivalency table.
 * Union-Find (disjoint sets) establishes one label as the root label of a 
 * tree, and so the root is considered the representative label. Each pixel
 * is labeled with the representative label.
 *  
 * There appear to be some modern competing approaches involving decision trees,
 * and an approach called "Light Speed Labeling".
 *
 * Author: William Silversmith
 * Affiliation: Seung Lab, Princeton University
 * Date: August 2018
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>

#include "libdivide.h"

#ifndef CC3D_HPP
#define CC3D_HPP 

#define CC3D_NHOOD 9

namespace cc3d {

template <typename T>
class DisjointSet {
public:
  T *ids;
  size_t *size;
  size_t length;
  DisjointSet () {
    length = 65536;
    ids = new T[length]();
    size = new size_t[length]();
  }

  DisjointSet (size_t len) {
    length = len;
    ids = new T[length]();
    size = new size_t[length]();
  }

  DisjointSet (const DisjointSet &cpy) {
    length = cpy.length;
    ids = new T[length]();
    size = new size_t[length]();

    for (int i = 0; i < length; i++) {
      ids[i] = cpy.ids[i];
      size[i] = cpy.size[i];
    }
  }

  ~DisjointSet () {
    delete []ids;
    delete []size;
  }

  T root (T n) {
    T i = ids[n];
    while (i != ids[i]) {
      ids[i] = ids[ids[i]]; // path compression
      i = ids[i];
    }

    return i;
  }

  bool find (T p, T q) {
    return root(p) == root(q);
  }

  void add(T p) {
    if (p >= length) {
      printf("Connected Components Error: Label %d cannot be mapped to union-find array of length %lu.\n", p, length);
      throw "maximum length exception";
    }

    if (ids[p] == 0) {
      ids[p] = p;
      size[p] = 1;
    }
  }

  void unify (T p, T q) {
    if (p == q) {
      return;
    }

    T i = root(p);
    T j = root(q);

    if (i == 0) {
      add(p);
      i = p;
    }

    if (j == 0) {
      add(q);
      j = q;
    }

    if (size[i] < size[j]) {
      ids[i] = j;
      size[j] += size[i];
    }
    else {
      ids[j] = i;
      size[i] += size[j];
    }
  }

  void print() {
    for (int i = 0; i < 15; i++) {
      printf("%d, ", ids[i]);
    }
    printf("\n");
    for (int i = 0; i < 15; i++) {
      printf("%d, ", size[i]);
    }
    printf("\n");
  }

  // would be easy to write remove. 
  // Will be O(n).
};

template <typename T>
inline void fill(T *arr, const int value, const size_t size) {
  for (size_t i = 0; i < size; i++) {
    arr[i] = value;
  }
}

inline void compute_neighborhood(
  int64_t *neighborhood, 
  const int x, const int y, const int z,
  const size_t sx, const size_t sy, const size_t sz) {

  const int64_t sxy = static_cast<int64_t>(sx) * static_cast<int64_t>(sy);

  fill<int64_t>(neighborhood, 0, CC3D_NHOOD);

  // 6-hood

  if (x > 0) {
    neighborhood[0] = -1;
  }
  if (y > 0) {
    neighborhood[1] = -static_cast<int64_t>(sx);
  }
  if (z > 0) {
    neighborhood[2] = -sxy;
  }

  // xy diagonals
  neighborhood[3] = (neighborhood[0] + neighborhood[1]) * (neighborhood[0] && neighborhood[1]); // up-left

  // yz diagonals
  neighborhood[4] = (neighborhood[1] + neighborhood[2]) * (neighborhood[1] && neighborhood[2]); // up-left
  
  // xz diagonals
  neighborhood[5] = (neighborhood[0] + neighborhood[2]) * (neighborhood[0] && neighborhood[2]); // up-left
  neighborhood[6] = (neighborhood[0] + neighborhood[1] + neighborhood[2]) * (neighborhood[0] && neighborhood[1] && neighborhood[2]);

  // Two forward
  if (x < static_cast<int64_t>(sx) - 1) {
    neighborhood[7] = (1 + neighborhood[1]) * (neighborhood[1] != 0); 
    neighborhood[8] = (1 + neighborhood[1] + neighborhood[2]) * (neighborhood[1] && neighborhood[2]);
  }
}

template <typename T>
uint32_t* connected_components3d(T* in_labels, const int sx, const int sy, const int sz) {
  const int64_t voxels = (int64_t)sx * (int64_t)sy * (int64_t)sz;
  return connected_components3d<T>(in_labels, sx, sy, sz, voxels);
}

template <typename T>
uint32_t* connected_components3d(
    T* in_labels, 
    const int sx, const int sy, const int sz,
    int64_t max_labels
  ) {

	const int sxy = sx * sy;
	const int64_t voxels = (int64_t)sx * (int64_t)sy * (int64_t)sz;

  const libdivide::divider<int64_t> fast_sx(sx); 
  const libdivide::divider<int64_t> fast_sxy(sxy); 

  const bool power_of_two = !((sx & (sx - 1)) || (sy & (sy - 1))); 
  const int xshift = std::log2(sx); // must use log2 here, not lg/lg2 to avoid fp errors
  const int yshift = std::log2(sy);

  max_labels = std::max(std::min(max_labels, voxels), 1L); // can't allocate 0 arrays

  DisjointSet<uint32_t> equivalences(max_labels);

  uint32_t* out_labels = new uint32_t[voxels]();
  int64_t neighborhood[CC3D_NHOOD];
  uint32_t neighbor_values[CC3D_NHOOD];
  
  short int num_neighbor_values = 0;

  uint32_t next_label = 0;

  int x, y, z;

  // Raster Scan 1: Set temporary labels and 
  // record equivalences in a disjoint set.
  for (int64_t loc = 0; loc < voxels; loc++) {
    if (in_labels[loc] == 0) {
      continue;
    }

    if (power_of_two) {
      z = loc >> (xshift + yshift);
      y = (loc - (z << (xshift + yshift))) >> xshift;
      x = loc - ((y + (z << yshift)) << xshift);
    }
    else {
      z = loc / fast_sxy;
      y = (loc - (z * sxy)) / fast_sx;
      x = loc - sx * (y + z * sy);
    }

    compute_neighborhood(neighborhood, x, y, z, sx, sy, sz);
    
    int64_t min_neighbor = voxels; // impossibly high value
    int64_t delta;
    for (int i = 0; i < CC3D_NHOOD; i++) {
      if (neighborhood[i] == 0) {
        continue;
      }

      delta = loc + neighborhood[i];
      if (in_labels[loc] != in_labels[delta]) {
        continue;
      }
      else if (out_labels[delta] == 0) {
        continue;
      }

      min_neighbor = std::min(min_neighbor, static_cast<int64_t>(out_labels[delta]));
      neighbor_values[num_neighbor_values] = out_labels[delta];
      num_neighbor_values++;
    }

    // no labeled neighbors
    if (min_neighbor == voxels) {
      next_label++;
      out_labels[loc] = static_cast<uint32_t>(next_label);
    }
    else {
      out_labels[loc] = static_cast<uint32_t>(min_neighbor);
    }
    
    equivalences.add(out_labels[loc]);
    for (int i = 0; i < num_neighbor_values; i++) {
      equivalences.unify(out_labels[loc], neighbor_values[i]);
    }
    fill<uint32_t>(neighbor_values, 0, num_neighbor_values);
    num_neighbor_values = 0;
  }

  // Raster Scan 2: Write final labels based on equivalences
  for (int64_t loc = 0; loc < voxels; loc++) {
    if (out_labels[loc]) {
      out_labels[loc] = equivalences.root(out_labels[loc]);
    }
  }

  return out_labels;
}

};

#endif











