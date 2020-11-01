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
 * Date: August 2018 - October 2020
 *
 * ----
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * ----
 */

#ifndef CC3D_HPP
#define CC3D_HPP 

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <unordered_set>
#include <vector>
#include <limits>

namespace cc3d {

static size_t _dummy_N;

template <typename T>
class DisjointSet {
public:
  T *ids;
  size_t length;

  DisjointSet () {
    length = 65536; // 2^16, some "reasonable" starting size
    ids = new T[length]();
    if (!ids) { 
      throw std::runtime_error("Failed to allocate memory for the Union-Find datastructure for connected components.");
    }
  }

  DisjointSet (size_t len) {
    length = len;
    ids = new T[length]();
    if (!ids) { 
      throw std::runtime_error("Failed to allocate memory for the Union-Find datastructure for connected components.");
    }
  }

  DisjointSet (const DisjointSet &cpy) {
    length = cpy.length;
    ids = new T[length]();
    if (!ids) { 
      throw std::runtime_error("Failed to allocate memory for the Union-Find datastructure for connected components.");
    }

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
      printf("Connected Components Error: Label %lli cannot be mapped to union-find array of length %lu.\n", static_cast<long long int>(p), length);
      throw std::runtime_error("maximum length exception");
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

template <typename T>
size_t zeroth_pass(T* in_labels, const int64_t sx, const int64_t voxels) {
  size_t count = 0;
  for (int64_t loc = 0; loc < voxels; loc += sx) {
    count += (in_labels[loc] != 0);
    for (int64_t x = 1; x < sx; x++) {
      count += static_cast<size_t>(in_labels[loc + x] != in_labels[loc + x - 1] && in_labels[loc + x] != 0);
    }
  }
  return count;
}

// This is the original Wu et al decision tree but without
// any copy operations, only union find. We can decompose the problem
// into the z - 1 problem unified with the original 2D algorithm.
// If literally none of the Z - 1 are filled, we can use a faster version
// of this that uses copies.
template <typename T, typename OUT = uint32_t>
inline void unify2d(
    const int64_t loc, const T cur,
    const int64_t x, const int64_t y, 
    const int64_t sx, const int64_t sy, 
    const T* in_labels, const OUT* out_labels,
    DisjointSet<OUT> &equivalences  
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

template <typename T, typename OUT = uint32_t>
inline void unify2d_rt(
    const int64_t loc, const T cur,
    const int64_t x, const int64_t y, 
    const int64_t sx, const int64_t sy, 
    const T* in_labels, const OUT* out_labels,
    DisjointSet<OUT> &equivalences  
  ) {

  if (x < sx - 1 && y > 0 && cur == in_labels[loc + 1 - sx]) {
    equivalences.unify(out_labels[loc], out_labels[loc + 1 - sx]);
  }
}

template <typename T, typename OUT = uint32_t>
inline void unify2d_lt(
    const int64_t loc, const T cur,
    const int64_t x, const int64_t y, 
    const int64_t sx, const int64_t sy, 
    const T* in_labels, const OUT* out_labels,
    DisjointSet<OUT> &equivalences  
  ) {

  if (x > 0 && cur == in_labels[loc - 1]) {
    equivalences.unify(out_labels[loc], out_labels[loc - 1]);
  }
  else if (x > 0 && y > 0 && cur == in_labels[loc - 1 - sx]) {
    equivalences.unify(out_labels[loc], out_labels[loc - 1 - sx]);
  }
}

// This is the second raster pass of the two pass algorithm family.
// The input array (output_labels) has been assigned provisional 
// labels and this resolves them into their final labels. We
// modify this pass to also ensure that the output labels are
// numbered from 1 sequentially.
template <typename OUT = uint32_t>
OUT* relabel(
    OUT* out_labels, const int64_t voxels,
    const int64_t num_labels, DisjointSet<OUT> &equivalences,
    size_t &N
  ) {

  if (num_labels <= 1) {
    return out_labels;
  }

  OUT label;
  OUT* renumber = new OUT[num_labels + 1]();
  OUT next_label = 1;

  for (int64_t i = 1; i <= num_labels; i++) {
    label = equivalences.root(i);
    if (renumber[label] == 0) {
      renumber[label] = next_label;
      renumber[i] = next_label;
      next_label++;
    }
    else {
      renumber[i] = renumber[label];
    }
  }

  // Raster Scan 2: Write final labels based on equivalences
  for (int64_t loc = 0; loc < voxels; loc++) {
    out_labels[loc] = renumber[out_labels[loc]];
  }

  delete[] renumber;

  N = next_label - 1;
  return out_labels;
}

template <typename T, typename OUT = uint32_t>
OUT* connected_components3d_26(
    T* in_labels, 
    const int64_t sx, const int64_t sy, const int64_t sz,
    size_t max_labels, OUT *out_labels = NULL, size_t &N = _dummy_N
  ) {

	const int64_t sxy = sx * sy;
	const int64_t voxels = sxy * sz;

  if (out_labels == NULL) {
    out_labels = new OUT[voxels]();
  }
  if (!out_labels) { 
    throw std::runtime_error("Failed to allocate out_labels memory for connected components.");
  }

  if (max_labels == 0) {
    return out_labels;
  }

  max_labels++;
  max_labels = std::max(std::min(max_labels, static_cast<size_t>(voxels)), static_cast<size_t>(1L)); // can't allocate 0 arrays
  max_labels = std::min(max_labels, static_cast<size_t>(std::numeric_limits<OUT>::max()));
  
  DisjointSet<OUT> equivalences(max_labels);
     
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

  OUT next_label = 0;
  int64_t loc = 0;

  // Raster Scan 1: Set temporary labels and 
  // record equivalences in a disjoint set.
  for (int32_t z = 0; z < sz; z++) {
    for (int32_t y = 0; y < sy; y++) {
      for (int32_t x = 0; x < sx; x++) {
        loc = x + sx * (y + sy * z);
        const T cur = in_labels[loc];

        if (cur == 0) {
          continue;
        }

        if (z > 0 && cur == in_labels[loc + E]) {
          out_labels[loc] = out_labels[loc + E];
        }
        else if (z > 0 && y > 0 && cur == in_labels[loc + B]) {
          out_labels[loc] = out_labels[loc + B];

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
    }
  }

  return relabel<OUT>(out_labels, voxels, next_label, equivalences, N);
}

template <typename T, typename OUT = uint32_t>
OUT* connected_components3d_18(
    T* in_labels, 
    const int64_t sx, const int64_t sy, const int64_t sz,
    size_t max_labels, OUT *out_labels = NULL, size_t &N = _dummy_N
  ) {

  const int64_t sxy = sx * sy;
  const int64_t voxels = sxy * sz;

  if (out_labels == NULL) {
    out_labels = new OUT[voxels]();
  }
  if (!out_labels) { 
    throw std::runtime_error("Failed to allocate out_labels memory for connected components.");
  }

  if (max_labels == 0) {
    return out_labels;
  }

  max_labels++;
  max_labels = std::max(std::min(max_labels, static_cast<size_t>(voxels)), static_cast<size_t>(1L)); // can't allocate 0 arrays
  max_labels = std::min(max_labels, static_cast<size_t>(std::numeric_limits<OUT>::max()));
  
  DisjointSet<OUT> equivalences(max_labels);

     
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
  const int64_t B = -sx - sxy;
  const int64_t D = -1 - sxy;
  const int64_t E = -sxy;
  const int64_t F = +1 - sxy;
  const int64_t H = +sx - sxy;

  // Current Z
  const int64_t J = -1 - sx;
  const int64_t K = -sx;
  const int64_t L = +1 - sx; 
  const int64_t M = -1;
  // N = 0;

  OUT next_label = 0;
  int64_t loc = 0;

  // Raster Scan 1: Set temporary labels and 
  // record equivalences in a disjoint set.
  for (int64_t z = 0; z < sz; z++) {
    for (int64_t y = 0; y < sy; y++) {
      for (int64_t x = 0; x < sx; x++) {
        loc = x + sx * (y + sy * z);
        const T cur = in_labels[loc];

        if (cur == 0) {
          continue;
        }

        if (z > 0 && cur == in_labels[loc + E]) {
          out_labels[loc] = out_labels[loc + E];

          if (x > 0 && y > 0 && cur == in_labels[loc + J]) {
            equivalences.unify(out_labels[loc], out_labels[loc + J]);
          }
          if (x < sx - 1 && y > 0 && cur == in_labels[loc + L]) {
            equivalences.unify(out_labels[loc], out_labels[loc + L]); 
          }
        }
        else if (y > 0 && z > 0 && cur == in_labels[loc + B]) {
          out_labels[loc] = out_labels[loc + B];

          if (x > 0 && cur == in_labels[loc + M]) {
            equivalences.unify(out_labels[loc], out_labels[loc + M]);
          }
          if (y < sy - 1 && z > 0 && cur == in_labels[loc + H]) {
            equivalences.unify(out_labels[loc], out_labels[loc + H]); 
          }
        }
        else if (x > 0 && z > 0 && cur == in_labels[loc + D]) {
          out_labels[loc] = out_labels[loc + D];

          if (x < sx - 1 && y > 0 && cur == in_labels[loc + L]) {
            equivalences.unify(out_labels[loc], out_labels[loc + L]); 
          }
          else {
            if (y > 0 && cur == in_labels[loc + K]) {
              equivalences.unify(out_labels[loc], out_labels[loc + K]); 
            }
            if (x < sx - 1 && z > 0 && cur == in_labels[loc + F]) {
              equivalences.unify(out_labels[loc], out_labels[loc + F]); 
            }
          }
        }
        else if (x < sx - 1 && z > 0 && cur == in_labels[loc + F]) {
          out_labels[loc] = out_labels[loc + F];

          if (x > 0 && y > 0 && cur == in_labels[loc + J]) {
            equivalences.unify(out_labels[loc], out_labels[loc + J]);
          }
          else {
            if (x > 0 && cur == in_labels[loc + M]) {
              equivalences.unify(out_labels[loc], out_labels[loc + M]); 
            }
            if (y > 0 && cur == in_labels[loc + K]) {
              equivalences.unify(out_labels[loc], out_labels[loc + K]); 
            }            
          }
        }
        else if (y < sy - 1 && z > 0 && cur == in_labels[loc + H]) {
          out_labels[loc] = out_labels[loc + H];
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
    }
  }

  return relabel<OUT>(out_labels, voxels, next_label, equivalences, N);
}

template <typename T, typename OUT = uint32_t>
OUT* connected_components3d_6(
    T* in_labels, 
    const int64_t sx, const int64_t sy, const int64_t sz,
    size_t max_labels, OUT *out_labels = NULL, size_t &N = _dummy_N
  ) {

  const int64_t sxy = sx * sy;
  const int64_t voxels = sxy * sz;

  if (out_labels == NULL) {
    out_labels = new OUT[voxels]();
  }
  if (!out_labels) { 
    throw std::runtime_error("Failed to allocate out_labels memory for connected components.");
  }

  if (max_labels == 0) {
    return out_labels;
  }

  max_labels++;
  max_labels = std::max(std::min(max_labels, static_cast<size_t>(voxels)), static_cast<size_t>(1L)); // can't allocate 0 arrays
  max_labels = std::min(max_labels, static_cast<size_t>(std::numeric_limits<OUT>::max()));
  
  DisjointSet<OUT> equivalences(max_labels);

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
  const int64_t B = -sx - sxy;
  const int64_t E = -sxy;
  const int64_t D = -1 - sxy;

  // Current Z
  const int64_t K = -sx;
  const int64_t M = -1;
  const int64_t J = -1 - sx;
  // N = 0;

  int64_t loc = 0;
  OUT next_label = 0;

  // Raster Scan 1: Set temporary labels and 
  // record equivalences in a disjoint set.

  for (int64_t z = 0; z < sz; z++) {
    for (int64_t y = 0; y < sy; y++) {
      for (int64_t x = 0; x < sx; x++) {
        loc = x + sx * (y + sy * z);

        const T cur = in_labels[loc];

        if (cur == 0) {
          continue;
        }

        if (x > 0 && cur == in_labels[loc + M]) {
          out_labels[loc] = out_labels[loc + M];

          if (y > 0 && cur == in_labels[loc + K] && cur != in_labels[loc + J]) {
            equivalences.unify(out_labels[loc], out_labels[loc + K]); 
            if (z > 0 && cur == in_labels[loc + E]) {
              if (cur != in_labels[loc + D] && cur != in_labels[loc + B]) {
                equivalences.unify(out_labels[loc], out_labels[loc + E]);
              }
            }
          }
          else if (z > 0 && cur == in_labels[loc + E] && cur != in_labels[loc + D]) {
            equivalences.unify(out_labels[loc], out_labels[loc + E]); 
          }
        }
        else if (y > 0 && cur == in_labels[loc + K]) {
          out_labels[loc] = out_labels[loc + K];

          if (z > 0 && cur == in_labels[loc + E] && cur != in_labels[loc + B]) {
            equivalences.unify(out_labels[loc], out_labels[loc + E]); 
          }
        }
        else if (z > 0 && cur == in_labels[loc + E]) {
          out_labels[loc] = out_labels[loc + E];
        }
        else {
          next_label++;
          out_labels[loc] = next_label;
          equivalences.add(out_labels[loc]);
        }
      }
    }
  }

  return relabel<OUT>(out_labels, voxels, next_label, equivalences, N);
}


// uses an approach inspired by 2x2 block based decision trees
// by Grana et al that was intended for 8-connected. Here we 
// skip a unify on every other voxel in the horizontal and
// vertical directions.
template <typename T, typename OUT = uint32_t>
OUT* connected_components2d_4(
    T* in_labels, 
    const int64_t sx, const int64_t sy, 
    size_t max_labels, OUT *out_labels = NULL,
    size_t &N = _dummy_N
  ) {

  const int64_t voxels = sx * sy;

  if (out_labels == NULL) {
    out_labels = new OUT[voxels]();
  }
  if (!out_labels) { 
    throw std::runtime_error("Failed to allocate out_labels memory for connected components.");
  }

  if (max_labels == 0) {
    return out_labels;
  }

  max_labels++;
  max_labels = std::max(std::min(max_labels, static_cast<size_t>(voxels)), static_cast<size_t>(1L)); // can't allocate 0 arrays
  max_labels = std::min(max_labels, static_cast<size_t>(std::numeric_limits<OUT>::max()));
  
  DisjointSet<OUT> equivalences(max_labels);
    
  /*
    Layout of forward pass mask. 
    A is the current location.
    D C 
    B A 
  */

  const int64_t A = 0;
  const int64_t B = -1;
  const int64_t C = -sx;
  const int64_t D = -1-sx;

  int64_t loc = 0;
  OUT next_label = 0;

  // Raster Scan 1: Set temporary labels and 
  // record equivalences in a disjoint set.

  T cur = 0;
  for (int64_t y = 0; y < sy; y++) {
    for (int64_t x = 0; x < sx; x++) {
      loc = x + sx * y;
      cur = in_labels[loc];

      if (cur == 0) {
        continue;
      }

      if (x > 0 && cur == in_labels[loc + B]) {
        out_labels[loc + A] = out_labels[loc + B];
        if (y > 0 && cur != in_labels[loc + D] && cur == in_labels[loc + C]) {
          equivalences.unify(out_labels[loc + A], out_labels[loc + C]);
        }
      }
      else if (y > 0 && cur == in_labels[loc + C]) {
        out_labels[loc + A] = out_labels[loc + C];
      }
      else {
        next_label++;
        out_labels[loc + A] = next_label;
        equivalences.add(out_labels[loc + A]);
      }
    }
  }

  return relabel<OUT>(out_labels, voxels, next_label, equivalences, N);
}

// K. Wu, E. Otoo, K. Suzuki. "Two Strategies to Speed up Connected Component Labeling Algorithms". 
// Lawrence Berkely National Laboratory. LBNL-29102, 2005.
// This is the stripped down version of that decision tree algorithm.
// It seems to give up to about 1.18x improvement on some data. No improvement on binary
// vs 18 connected (from 3D).
template <typename T, typename OUT = uint32_t>
OUT* connected_components2d_8(
    T* in_labels, 
    const int64_t sx, const int64_t sy,
    size_t max_labels, OUT *out_labels = NULL,
    size_t &N = _dummy_N
  ) {

  const int64_t voxels = sx * sy;

  if (out_labels == NULL) {
    out_labels = new OUT[voxels]();
  }
  if (!out_labels) { 
    throw std::runtime_error("Failed to allocate out_labels memory for connected components.");
  }

  if (max_labels == 0) {
    return out_labels;
  }

  max_labels++;
  max_labels = std::max(std::min(max_labels, static_cast<size_t>(voxels)), static_cast<size_t>(1L)); // can't allocate 0 arrays
  max_labels = std::min(max_labels, static_cast<size_t>(std::numeric_limits<OUT>::max()));
  
  DisjointSet<OUT> equivalences(max_labels);

  /*
    Layout of mask. We start from e.
      | p |
    a | b | c
    d | e |
  */

  const int64_t A = -1 - sx;
  const int64_t B = -sx;
  const int64_t C = +1 - sx;
  const int64_t D = -1;

  const int64_t P = -2 * sx;

  int64_t loc = 0;
  OUT next_label = 0;

  // Raster Scan 1: Set temporary labels and 
  // record equivalences in a disjoint set.
  for (int64_t y = 0; y < sy; y++) {
    for (int64_t x = 0; x < sx; x++) {
      loc = x + sx * y;

      const T cur = in_labels[loc];

      if (cur == 0) {
        continue;
      }

      if (y > 0 && cur == in_labels[loc + B]) {
        out_labels[loc] = out_labels[loc + B];
      }
      else if (x > 0 && y > 0 && cur == in_labels[loc + A]) {
        out_labels[loc] = out_labels[loc + A];
        if (x < sx - 1 && y > 0 && cur == in_labels[loc + C] 
            && !(y > 1 && cur == in_labels[loc + P])) {

            equivalences.unify(out_labels[loc], out_labels[loc + C]);
        }
      }
      else if (x > 0 && cur == in_labels[loc + D]) {
        out_labels[loc] = out_labels[loc + D];
        if (x < sx - 1 && y > 0 && cur == in_labels[loc + C]) {
          equivalences.unify(out_labels[loc], out_labels[loc + C]);
        }
      }
      else if (x < sx - 1 && y > 0 && cur == in_labels[loc + C]) {
        out_labels[loc] = out_labels[loc + C];
      }
      else {
        next_label++;
        out_labels[loc] = next_label;
        equivalences.add(out_labels[loc]);        
      }
    }
  }

  return relabel<OUT>(out_labels, voxels, next_label, equivalences, N);
}

template <typename T, typename OUT = uint32_t>
OUT* connected_components3d(
    T* in_labels, 
    const int64_t sx, const int64_t sy, const int64_t sz,
    size_t max_labels, const int64_t connectivity,
    OUT *out_labels = NULL, size_t &N = _dummy_N
  ) {

  if (connectivity == 26) {
    return connected_components3d_26<T, OUT>(
      in_labels, sx, sy, sz, 
      max_labels, out_labels, N
    );
  }
  else if (connectivity == 18) {
    return connected_components3d_18<T, OUT>(
      in_labels, sx, sy, sz, 
      max_labels, out_labels, N
    );
  }
  else if (connectivity == 6) {
    return connected_components3d_6<T, OUT>(
      in_labels, sx, sy, sz, 
      max_labels, out_labels, N
    );
  }
  else if (connectivity == 8) {
    if (sz != 1) {
      throw std::runtime_error("sz must be 1 for 2D connectivities.");
    }
    return connected_components2d_8<T,OUT>(
      in_labels, sx, sy,
      max_labels, out_labels, N
    );
  }
  else if (connectivity == 4) {
    if (sz != 1) {
      throw std::runtime_error("sz must be 1 for 2D connectivities.");
    }
    return connected_components2d_4<T, OUT>(
      in_labels, sx, sy, 
      max_labels, out_labels, N
    );
  }
  else {
    throw std::runtime_error("Only 4 and 8 2D and 6, 18, and 26 3D connectivities are supported.");
  }
}

template <typename T, typename OUT = uint32_t>
OUT* connected_components3d(
    T* in_labels, 
    const int64_t sx, const int64_t sy, const int64_t sz,
    const int64_t connectivity=26, size_t &N = _dummy_N
  ) {
  const int64_t voxels = sx * sy * sz;
  return connected_components3d<T, OUT>(in_labels, sx, sy, sz, voxels, connectivity, NULL, N);
}


};



#endif
