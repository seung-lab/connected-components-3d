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
inline void unify_ch(
  const int64_t loc, const T cur,
  const int64_t x, const int64_t y, const int64_t z,
  const int64_t sx, const int64_t sy, const int64_t sz,
  const T* in_labels, const uint32_t *out_labels,
  DisjointSet<uint32_t> &equivalences
  ) {

  const int64_t sxy = sx * sy;

  if (x < sx - 1 && y > 0) { // right edge guard
    if (cur == in_labels[loc + 1 - sx]) { // J,H
      equivalences.unify(out_labels[loc], out_labels[loc + 1 - sx]);
    }
    else if (z > 0 && cur == in_labels[loc + 1 - sx - sxy]) { // J,C
      equivalences.unify(out_labels[loc], out_labels[loc + 1 - sx - sxy]);
    }
  }
}

template <typename T>
uint32_t* connected_components3d(T* in_labels, const int64_t sx, const int64_t sy, const int64_t sz) {
  const int64_t voxels = sx * sy * sz;
  return connected_components3d<T>(in_labels, sx, sy, sz, voxels);
}

template <typename T>
uint32_t* connected_components3d(
    T* in_labels, 
    const int64_t sx, const int64_t sy, const int64_t sz,
    int64_t max_labels
  ) {

	const int64_t sxy = sx * sy;
	const int64_t voxels = sxy * sz;

  const libdivide::divider<int64_t> fast_sx(sx); 
  const libdivide::divider<int64_t> fast_sxy(sxy); 

  const bool power_of_two = !((sx & (sx - 1)) || (sy & (sy - 1))); 
  const int xshift = std::log2(sx); // must use log2 here, not lg/lg2 to avoid fp errors
  const int yshift = std::log2(sy);

  max_labels = std::max(std::min(max_labels, voxels), static_cast<int64_t>(1L)); // can't allocate 0 arrays

  DisjointSet<uint32_t> equivalences(max_labels);

  uint32_t* out_labels = new uint32_t[voxels]();
  uint32_t next_label = 0;
  int64_t x, y, z;

  // Raster Scan 1: Set temporary labels and 
  // record equivalences in a disjoint set.
  for (int64_t loc = 0; loc < voxels; loc++) {
    const T cur = in_labels[loc];

    if (cur == 0) {
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

    /*
      Layout of forward pass mask (which faces backwards). 
      J is the current location.

      z = -1     z = 0
      A B C      F G H   y = -1 
      D E        I J     y =  0
     -1 0 +1    -1 0   <-- x axis
    */

    // This is an elaboration of Wu et al's 2005 decision tree algorithm
    // into 3D. Wu worked with a mask of five pixels for an 8 connected
    // 2D image, but we must work with a mask of size ten. 

    // H, E, and B are special in that they are connected to all 
    // other voxels, so check them first. Similar to Fig. 1B in 
    // Wu et al 2005. Check H,B,E in that order to take advantage
    // of the L1 cache. H is sx away, B and E are probably outside 
    // of L1.
    if (y > 0 && z > 0 && cur == in_labels[loc - sx - sxy]) { // E
      out_labels[loc] = out_labels[loc - sx - sxy];

      if (y > 0 && cur == in_labels[loc - sx]) {
        equivalences.unify(out_labels[loc], out_labels[loc - sx]);

        if (x > 0 && cur == in_labels[loc - 1]) {
          equivalences.unify(out_labels[loc], out_labels[loc - 1]);
        }
      }
      else if (x > 0 && cur == in_labels[loc - 1]) {
        equivalences.unify(out_labels[loc], out_labels[loc - 1]); 

        if (x < sx - 1 && y > 0 && cur == in_labels[loc + 1 - sx]) {
          equivalences.unify(out_labels[loc], out_labels[loc + 1 - sx]); 
        }
      }
      else if (x > 0 && y > 0 && cur == in_labels[loc - 1 - sx]) {
        equivalences.unify(out_labels[loc], out_labels[loc - 1 - sx]); 

        if (x < sx - 1 && y > 0 && cur == in_labels[loc + 1 - sx]) {
          equivalences.unify(out_labels[loc], out_labels[loc + 1 - sx]); 
        }
      }
      else if (x < sx - 1 && y > 0 && cur == in_labels[loc + 1 - sx]) {
        equivalences.unify(out_labels[loc], out_labels[loc + 1 - sx]); 
      }
    }
    else if (z > 0 && cur == in_labels[loc - sxy]) { // B
      out_labels[loc] = out_labels[loc - sxy];

      if (y > 0 && cur == in_labels[loc - sx]) { // B,G
        equivalences.unify(out_labels[loc], out_labels[loc - sx]);

        if (x > 0 && cur == in_labels[loc - 1]) { // B,G,I
          equivalences.unify(out_labels[loc], out_labels[loc - 1]);
        }
      }
      else if (x > 0 && cur == in_labels[loc - 1]) { // B,I
        equivalences.unify(out_labels[loc], out_labels[loc - 1]); 

        if (x < sx - 1 && y > 0 && cur == in_labels[loc + 1 - sx]) { // B,I,H
          equivalences.unify(out_labels[loc], out_labels[loc + 1 - sx]); 
        }
      }
      else if (x > 0 && y > 0 && cur == in_labels[loc - 1 - sx]) { // B,F
        equivalences.unify(out_labels[loc], out_labels[loc - 1 - sx]); 

        if (x < sx - 1 && y > 0 && cur == in_labels[loc + 1 - sx]) { // B,F,H
          equivalences.unify(out_labels[loc], out_labels[loc + 1 - sx]); 
        }
      }
      else if (x < sx - 1 && y > 0 && cur == in_labels[loc + 1 - sx]) { // B,H
        equivalences.unify(out_labels[loc], out_labels[loc + 1 - sx]); 
      }
    }
    else if (y > 0 && cur == in_labels[loc - sx]) { // G
      out_labels[loc] = out_labels[loc - sx];
      
      if (x > 0 && cur == in_labels[loc - 1]) { // G,J
        equivalences.unify(out_labels[loc], out_labels[loc - 1]); 
      }
      else if (z > 0 && x > 0 && cur == in_labels[loc - 1 - sxy]) { // G,D
        equivalences.unify(out_labels[loc], out_labels[loc - 1 - sxy]);  
      }
    }
    // Now we move into the next phase of the tree where the two
    // sides of A,D,G,J are potentially connected via K to C,F
    // The test for J is key to advancing new labels at the beginning
    // of the run.
    else if (x > 0 && cur == in_labels[loc - 1]) { // I
      out_labels[loc] = out_labels[loc - 1];
      unify_ch<T>(
        loc, cur, 
        x, y, z, 
        sx, sy, sz, 
        in_labels, out_labels, 
        equivalences
      );
    }
    else if (x > 0 && y > 0 && cur == in_labels[loc - 1 - sx]) { // F
      out_labels[loc] = out_labels[loc - 1 - sx];
      unify_ch<T>(
        loc, cur, 
        x, y, z, 
        sx, sy, sz, 
        in_labels, out_labels, 
        equivalences
      );
    }
    else if (x > 0 && z > 0 && cur == in_labels[loc - 1 - sxy]) { // D
      out_labels[loc] = out_labels[loc - 1 - sxy];
      unify_ch<T>(
        loc, cur, 
        x, y, z, 
        sx, sy, sz, 
        in_labels, out_labels, 
        equivalences
      );
    }
    else if (x > 0 && y > 0 && z > 0 && cur == in_labels[loc - 1 - sx - sxy]) { // A
      out_labels[loc] = out_labels[loc - 1 - sx - sxy];
      unify_ch<T>(
        loc, cur, 
        x, y, z, 
        sx, sy, sz, 
        in_labels, out_labels, 
        equivalences
      );
    }
    else if (x < sx - 1 && y > 0 && cur == in_labels[loc + 1 - sx]) { // H
      out_labels[loc] = out_labels[loc + 1 - sx];
    }
    else if (x < sx - 1 && z > 0 && y > 0 && cur == in_labels[loc + 1 - sx - sxy]) { // C
      out_labels[loc] = out_labels[loc + 1 - sx - sxy];
    }
    else { // New Label (no connected neighbors)
      next_label++;
      out_labels[loc] = next_label;
      equivalences.add(out_labels[loc]);
    }
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











