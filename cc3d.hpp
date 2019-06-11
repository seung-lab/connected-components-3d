/*
 * Connected Components for 3D images. 
 * Implments a 3D variant of the two pass algorithim by
 * Rosenfeld and Pflatz augmented with Union-Find and a decision
 * tree influenced by the work of Wu et al.
 * 
 * Essentially, you raster scan, and every time you first encounter 
 * a foreground pixel, mark it with a new label if the pixels to its
 * top and left are background. If there is a preexisting label in its
 * neighborhood, use that label instead. Whenever you see that two labels
 * are adjacent, record that we should unify them in the next pass. This
 * equivalency table can be constructed in several ways, but we've choseen
 * to use Union-Find with full path compression.
 * 
 * We also use a decision tree that aims to minimize the number of expensive
 * unify operations and replaces them with simple label copies when valid.
 *
 * In the next pass, the pixels are relabeled using the equivalency table.
 * Union-Find (disjoint sets) establishes one label as the root label of a 
 * tree, and so the root is considered the representative label. Each pixel
 * is labeled with the representative label. The representative labels
 * are themselves remapped into an increasing consecutive sequence 
 * starting from one. 
 *
 * Author: William Silversmith
 * Affiliation: Seung Lab, Princeton University
 * Date: August 2018 - June 2019
 */

#ifndef CC3D_HPP
#define CC3D_HPP 

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>

#include "libdivide.h"

namespace cc3d {

template <typename T>
class DisjointSet {
public:
  T *ids;
  size_t length;

  DisjointSet () {
    length = 65536;
    ids = new T[length]();
  }

  DisjointSet (size_t len) {
    length = len;
    ids = new T[length]();
  }

  DisjointSet (const DisjointSet &cpy) {
    length = cpy.length;
    ids = new T[length]();

    for (int i = 0; i < length; i++) {
      ids[i] = cpy.ids[i];
    }
  }

  ~DisjointSet () {
    delete []ids;
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

    ids[i] = j;
  }

  void print() {
    for (int i = 0; i < 15; i++) {
      printf("%d, ", ids[i]);
    }
    printf("\n");
  }

  // would be easy to write remove. 
  // Will be O(n).
};

// This is the original Wu et al decision tree but without
// any copy operations, only union find. We can decompose the problem
// into the z - 1 problem unified with the original 2D algorithm.
// If literally none of the Z - 1 are filled, we can use a faster version
// of this that uses copies.
template <typename T>
inline void unify2d(
    const int64_t loc, const T cur,
    const int64_t x, const int64_t y, 
    const int64_t sx, const int64_t sy, 
    const T* in_labels, const uint32_t *out_labels,
    DisjointSet<uint32_t> &equivalences  
  ) {

  if (y > 0 && cur == in_labels[loc - sx]) {
    equivalences.unify(out_labels[loc], out_labels[loc - sx]);
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

template <typename T>
inline void unify2d_rt(
    const int64_t loc, const T cur,
    const int64_t x, const int64_t y, 
    const int64_t sx, const int64_t sy, 
    const T* in_labels, const uint32_t *out_labels,
    DisjointSet<uint32_t> &equivalences  
  ) {

  if (x < sx - 1 && y > 0 && cur == in_labels[loc + 1 - sx]) {
    equivalences.unify(out_labels[loc], out_labels[loc + 1 - sx]);
  }
}

template <typename T>
inline void unify2d_lt(
    const int64_t loc, const T cur,
    const int64_t x, const int64_t y, 
    const int64_t sx, const int64_t sy, 
    const T* in_labels, const uint32_t *out_labels,
    DisjointSet<uint32_t> &equivalences  
  ) {

  if (x > 0 && cur == in_labels[loc - 1]) {
    equivalences.unify(out_labels[loc], out_labels[loc - 1]);
  }
  else if (x > 0 && y > 0 && cur == in_labels[loc - 1 - sx]) {
    equivalences.unify(out_labels[loc], out_labels[loc - 1 - sx]);
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
    int64_t max_labels, uint32_t *out_labels = NULL
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

  if (out_labels == NULL) {
    out_labels = new uint32_t[voxels]();
  }
  
  uint32_t next_label = 0;
  int64_t x, y, z;
    
  /*
    Layout of forward pass mask (which faces backwards). 
    N is the current location.

    z = -1     z = 0
    A B C      J K L   y = -1 
    D E F      M N     y =  0
    G H I              y = +1
   -1 0 +1    -1 0   <-- x axis
  */

  // Z - 1
  const int64_t A = -1 - sx - sxy;
  const int64_t B = -sx - sxy;
  const int64_t C = +1 - sx - sxy;
  const int64_t D = -1 - sxy;
  const int64_t E = -sxy;
  const int64_t F = +1 - sxy;
  const int64_t G = -1 + sx - sxy;
  const int64_t H = +sx - sxy;
  const int64_t I = +1 + sx - sxy;

  // Current Z
  const int64_t J = -1 - sx;
  const int64_t K = -sx;
  const int64_t L = +1 - sx; 
  const int64_t M = -1;
  // N = 0;

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

    if (z > 0 && cur == in_labels[loc + E]) {
      out_labels[loc] = out_labels[loc + E];
      // unify2d<T>(loc, cur, x, y, sx, sy, in_labels, out_labels, equivalences);
    }
    else if (z > 0 && y > 0 && cur == in_labels[loc + B]) {
      out_labels[loc] = out_labels[loc + B];
      // unify2d<T>(loc, cur, x, y, sx, sy, in_labels, out_labels, equivalences);

      if (y < sy - 1 && z > 0 && cur == in_labels[loc + H]) {
        equivalences.unify(out_labels[loc], out_labels[loc + H]);
      }
      else if (x > 0 && y < sy - 1 && z > 0 && cur == in_labels[loc + G]) {
        equivalences.unify(out_labels[loc], out_labels[loc + G]);
        
        if (x < sx - 1 && y < sy - 1 && z > 0 && cur == in_labels[loc + I]) {
          equivalences.unify(out_labels[loc], out_labels[loc + I]);
        }
      }
      else if (x < sx - 1 && y < sy - 1 && z > 0 && cur == in_labels[loc + I]) {
        equivalences.unify(out_labels[loc], out_labels[loc + I]);
      }
    }
    else if (x > 0 && z > 0 && cur == in_labels[loc + D]) {
      out_labels[loc] = out_labels[loc + D];
      unify2d_rt<T>(loc, cur, x, y, sx, sy, in_labels, out_labels, equivalences);

      if (x < sx - 1 && z > 0 && cur == in_labels[loc + F]) {
        equivalences.unify(out_labels[loc], out_labels[loc + F]);
      }
      else if (x < sx - 1 && y > 0 && z > 0 && cur == in_labels[loc + C]) {
        equivalences.unify(out_labels[loc], out_labels[loc + C]);

        if (x < sx - 1 && y < sy - 1 && z > 0 && cur == in_labels[loc + I]) {
          equivalences.unify(out_labels[loc], out_labels[loc + I]);
        }
      }
      else if (x < sx - 1 && y < sy - 1 && z > 0 && cur == in_labels[loc + I]) {
        equivalences.unify(out_labels[loc], out_labels[loc + I]);
      }
    }
    else if (y < sy - 1 && z > 0 && cur == in_labels[loc + H]) {
      out_labels[loc] = out_labels[loc + H];
      unify2d<T>(loc, cur, x, y, sx, sy, in_labels, out_labels, equivalences);

      if (x > 0 && y > 0 && z > 0 && cur == in_labels[loc + A]) {
        equivalences.unify(out_labels[loc], out_labels[loc + A]);
      }
      if (x < sx - 1 && y > 0 && z > 0 && cur == in_labels[loc + C]) {
        equivalences.unify(out_labels[loc], out_labels[loc + C]);
      }
    }
    else if (x < sx - 1 && z > 0 && cur == in_labels[loc + F]) {
      out_labels[loc] = out_labels[loc + F];
      unify2d_lt<T>(loc, cur, x, y, sx, sy, in_labels, out_labels, equivalences);

      if (x > 0 && y > 0 && z > 0 && cur == in_labels[loc + A]) {
        equivalences.unify(out_labels[loc], out_labels[loc + A]);
      }
      if (x > 0 && y < sy - 1 && z > 0 && cur == in_labels[loc + G]) {
        equivalences.unify(out_labels[loc], out_labels[loc + G]);
      }
    }
    else if (x > 0 && y > 0 && z > 0 && cur == in_labels[loc + A]) {
      out_labels[loc] = out_labels[loc + A];
      unify2d_rt<T>(loc, cur, x, y, sx, sy, in_labels, out_labels, equivalences);

      if (x < sx - 1 && y > 0 && z > 0 && cur == in_labels[loc + C]) {
        equivalences.unify(out_labels[loc], out_labels[loc + C]);
      }
      if (x > 0 && y < sy - 1 && z > 0 && cur == in_labels[loc + G]) {
        equivalences.unify(out_labels[loc], out_labels[loc + G]);
      }      
      if (x < sx - 1 && y < sy - 1 && z > 0 && cur == in_labels[loc + I]) {
        equivalences.unify(out_labels[loc], out_labels[loc + I]);
      }
    }
    else if (x < sx - 1 && y > 0 && z > 0 && cur == in_labels[loc + C]) {
      out_labels[loc] = out_labels[loc + C];
      unify2d_lt<T>(loc, cur, x, y, sx, sy, in_labels, out_labels, equivalences);

      if (x > 0 && y < sy - 1 && z > 0 && cur == in_labels[loc + G]) {
        equivalences.unify(out_labels[loc], out_labels[loc + G]);
      }
      if (x < sx - 1 && y < sy - 1 && z > 0 && cur == in_labels[loc + I]) {
        equivalences.unify(out_labels[loc], out_labels[loc + I]);
      }
    }
    else if (x > 0 && y < sy - 1 && z > 0 && cur == in_labels[loc + G]) {
      out_labels[loc] = out_labels[loc + G];
      unify2d<T>(loc, cur, x, y, sx, sy, in_labels, out_labels, equivalences);

      if (x < sx - 1 && y < sy - 1 && z > 0 && cur == in_labels[loc + I]) {
        equivalences.unify(out_labels[loc], out_labels[loc + I]);
      }
    }
    else if (x < sx - 1 && y < sy - 1 && z > 0 && cur == in_labels[loc + I]) {
      out_labels[loc] = out_labels[loc + I];
      unify2d<T>(loc, cur, x, y, sx, sy, in_labels, out_labels, equivalences);
    }
    // It's the original 2D problem now
    else if (y > 0 && cur == in_labels[loc + K]) {
      out_labels[loc] = out_labels[loc + K];
    }
    else if (x > 0 && cur == in_labels[loc + M]) {
      out_labels[loc] = out_labels[loc + M];

      if (x < sx - 1 && y > 0 && cur == in_labels[loc + L]) {
        equivalences.unify(out_labels[loc], out_labels[loc + L]); 
      }
    }
    else if (x > 0 && y > 0 && cur == in_labels[loc + J]) {
      out_labels[loc] = out_labels[loc + J];

      if (x < sx - 1 && y > 0 && cur == in_labels[loc + L]) {
        equivalences.unify(out_labels[loc], out_labels[loc + L]); 
      }
    }
    else if (x < sx - 1 && y > 0 && cur == in_labels[loc + L]) {
      out_labels[loc] = out_labels[loc + L];
    }
    else {
      next_label++;
      out_labels[loc] = next_label;
      equivalences.add(out_labels[loc]);
    }
  }

  uint32_t label;
  uint32_t* renumber = new uint32_t[next_label + 1]();
  next_label = 1;

  // Raster Scan 2: Write final labels based on equivalences
  for (int64_t loc = 0; loc < voxels; loc++) {
    if (!out_labels[loc]) {
      continue;
    }
   
    label = equivalences.root(out_labels[loc]);

    if (renumber[label]) {
      out_labels[loc] = renumber[label];
    }
    else {
      renumber[label] = next_label;
      out_labels[loc] = next_label;
      next_label++;
    }
  }

  delete[] renumber;

  return out_labels;
}

};

#endif
