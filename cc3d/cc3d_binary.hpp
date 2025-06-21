/*
 * Connected Components for binary 2D and 3D images. 
 *
 * Author: William Silversmith
 * Affiliation: Seung Lab, Princeton University
 * Date: October 2024
 *
 * ----
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the Lesser GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * Lesser GNU General Public License for more details.
 *
 * You should have received a copy of the Lesser GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * ----
 */

#ifndef CC3D_BINARY_HPP
#define CC3D_BINARY_HPP 

#include "cc3d.hpp"

namespace cc3d {

template <typename T, typename OUT = uint32_t>
OUT* connected_components3d_26_binary(
    T* in_labels, 
    const int64_t sx, const int64_t sy, const int64_t sz,
    size_t max_labels, OUT *out_labels = NULL, size_t &N = _dummy_N
  ) {

	const int64_t sxy = sx * sy;
	const int64_t voxels = sxy * sz;

  if (out_labels == NULL) {
    out_labels = new OUT[voxels]();
  }

  if (max_labels == 0) {
    return out_labels;
  }

  max_labels++; // corrects Cython estimation
  max_labels = std::min(max_labels, static_cast<size_t>(voxels) + 1); // + 1L for an array with no zeros
  max_labels = std::min(max_labels, static_cast<size_t>(std::numeric_limits<OUT>::max()));
  
  DisjointSet<OUT> equivalences(max_labels);

  const std::unique_ptr<uint32_t[]> runs(
    compute_foreground_index(in_labels, sx, sy, sz)
  );
     
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
  int64_t row = 0;
  for (int64_t z = 0; z < sz; z++) {
    for (int64_t y = 0; y < sy; y++, row++) {
      const int64_t xstart = runs[row << 1];
      const int64_t xend = runs[(row << 1) + 1];

      for (int64_t x = xstart; x < xend; x++) {
        loc = x + sx * y + sxy * z;
        const T cur = in_labels[loc];

        if (cur == 0) {
          continue;
        }

        if (z > 0 && in_labels[loc + E]) {
          out_labels[loc] = out_labels[loc + E];
        }
        else if (y > 0 && in_labels[loc + K]) {
          out_labels[loc] = out_labels[loc + K];

          if (y < sy - 1 && z > 0 && in_labels[loc + H]) {
            equivalences.unify(out_labels[loc], out_labels[loc + H]);
          }
          else if (x > 0 && y < sy - 1 && z > 0 && in_labels[loc + G]) {
            equivalences.unify(out_labels[loc], out_labels[loc + G]);
            
            if (x < sx - 1 && y < sy - 1 && z > 0 && in_labels[loc + I]) {
              equivalences.unify(out_labels[loc], out_labels[loc + I]);
            }
          }
          else if (x < sx - 1 && y < sy - 1 && z > 0 && in_labels[loc + I]) {
            equivalences.unify(out_labels[loc], out_labels[loc + I]);
          }
        }
        else if (z > 0 && y > 0 && in_labels[loc + B]) {
          out_labels[loc] = out_labels[loc + B];

          if (y < sy - 1 && z > 0 && in_labels[loc + H]) {
            equivalences.unify(out_labels[loc], out_labels[loc + H]);
          }
          else if (x > 0 && y < sy - 1 && z > 0 && in_labels[loc + G]) {
            equivalences.unify(out_labels[loc], out_labels[loc + G]);
            
            if (x < sx - 1 && y < sy - 1 && z > 0 && in_labels[loc + I]) {
              equivalences.unify(out_labels[loc], out_labels[loc + I]);
            }
          }
          else if (x < sx - 1 && y < sy - 1 && z > 0 && in_labels[loc + I]) {
            equivalences.unify(out_labels[loc], out_labels[loc + I]);
          }
        }
        else if (x > 0 && in_labels[loc + M]) {
          out_labels[loc] = out_labels[loc + M];

          if (x < sx - 1 && z > 0 && in_labels[loc + F]) {
            equivalences.unify(out_labels[loc], out_labels[loc + F]);
          }
          else if (x < sx - 1 && y > 0 && in_labels[loc + L]) {
            equivalences.unify(out_labels[loc], out_labels[loc + L]);

            if (x < sx - 1 && y < sy - 1 && z > 0 && in_labels[loc + I]) {
              equivalences.unify(out_labels[loc], out_labels[loc + I]);
            }
          }
          else if (x < sx - 1 && y > 0 && z > 0 && in_labels[loc + C]) {
            equivalences.unify(out_labels[loc], out_labels[loc + C]);

            if (x < sx - 1 && y < sy - 1 && z > 0 && in_labels[loc + I]) {
              equivalences.unify(out_labels[loc], out_labels[loc + I]);
            }
          }
          else if (x < sx - 1 && y < sy - 1 && z > 0 && in_labels[loc + I]) {
            equivalences.unify(out_labels[loc], out_labels[loc + I]);
          }
        }
        else if (x > 0 && z > 0 && in_labels[loc + D]) {
          out_labels[loc] = out_labels[loc + D];

          if (x < sx - 1 && z > 0 && in_labels[loc + F]) {
            equivalences.unify(out_labels[loc], out_labels[loc + F]);
          }
          else if (x < sx - 1 && y > 0 && in_labels[loc + L]) {
            equivalences.unify(out_labels[loc], out_labels[loc + L]);

            if (x < sx - 1 && y < sy - 1 && z > 0 && in_labels[loc + I]) {
              equivalences.unify(out_labels[loc], out_labels[loc + I]);
            }
          }
          else if (x < sx - 1 && y > 0 && z > 0 && in_labels[loc + C]) {
            equivalences.unify(out_labels[loc], out_labels[loc + C]);

            if (x < sx - 1 && y < sy - 1 && z > 0 && in_labels[loc + I]) {
              equivalences.unify(out_labels[loc], out_labels[loc + I]);
            }
          }
          else if (x < sx - 1 && y < sy - 1 && z > 0 && in_labels[loc + I]) {
            equivalences.unify(out_labels[loc], out_labels[loc + I]);
          }
        }
        else if (y < sy - 1 && z > 0 && in_labels[loc + H]) {
          out_labels[loc] = out_labels[loc + H];
          unify2d_ac<T>(loc, cur, x, y, sx, sy, in_labels, out_labels, equivalences);

          if (x > 0 && y > 0 && z > 0 && in_labels[loc + A]) {
            equivalences.unify(out_labels[loc], out_labels[loc + A]);
          }
          if (x < sx - 1 && y > 0 && z > 0 && in_labels[loc + C]) {
            equivalences.unify(out_labels[loc], out_labels[loc + C]);
          }
        }
        else if (x < sx - 1 && z > 0 && in_labels[loc + F]) {
          out_labels[loc] = out_labels[loc + F];
          unify2d_lt<T>(loc, cur, x, y, sx, sy, in_labels, out_labels, equivalences);

          if (x > 0 && y > 0 && z > 0 && in_labels[loc + A]) {
            equivalences.unify(out_labels[loc], out_labels[loc + A]);
          }
          if (x > 0 && y < sy - 1 && z > 0 && in_labels[loc + G]) {
            equivalences.unify(out_labels[loc], out_labels[loc + G]);
          }
        }
        else if (x > 0 && y > 0 && z > 0 && in_labels[loc + A]) {
          out_labels[loc] = out_labels[loc + A];
          unify2d_rt<T>(loc, cur, x, y, sx, sy, in_labels, out_labels, equivalences);

          if (x < sx - 1 && y > 0 && z > 0 && in_labels[loc + C]) {
            equivalences.unify(out_labels[loc], out_labels[loc + C]);
          }
          if (x > 0 && y < sy - 1 && z > 0 && in_labels[loc + G]) {
            equivalences.unify(out_labels[loc], out_labels[loc + G]);
          }      
          if (x < sx - 1 && y < sy - 1 && z > 0 && in_labels[loc + I]) {
            equivalences.unify(out_labels[loc], out_labels[loc + I]);
          }
        }
        else if (x < sx - 1 && y > 0 && z > 0 && in_labels[loc + C]) {
          out_labels[loc] = out_labels[loc + C];
          unify2d_lt<T>(loc, cur, x, y, sx, sy, in_labels, out_labels, equivalences);

          if (x > 0 && y < sy - 1 && z > 0 && in_labels[loc + G]) {
            equivalences.unify(out_labels[loc], out_labels[loc + G]);
          }
          if (x < sx - 1 && y < sy - 1 && z > 0 && in_labels[loc + I]) {
            equivalences.unify(out_labels[loc], out_labels[loc + I]);
          }
        }
        else if (x > 0 && y < sy - 1 && z > 0 && in_labels[loc + G]) {
          out_labels[loc] = out_labels[loc + G];
          unify2d_ac<T>(loc, cur, x, y, sx, sy, in_labels, out_labels, equivalences);

          if (x < sx - 1 && y < sy - 1 && z > 0 && in_labels[loc + I]) {
            equivalences.unify(out_labels[loc], out_labels[loc + I]);
          }
        }
        else if (x < sx - 1 && y < sy - 1 && z > 0 && in_labels[loc + I]) {
          out_labels[loc] = out_labels[loc + I];
          unify2d_ac<T>(loc, cur, x, y, sx, sy, in_labels, out_labels, equivalences);
        }
        // It's the original 2D problem now
        else if (y > 0 && in_labels[loc + K]) {
          out_labels[loc] = out_labels[loc + K];
        }
        else if (x > 0 && in_labels[loc + M]) {
          out_labels[loc] = out_labels[loc + M];

          if (x < sx - 1 && y > 0 && in_labels[loc + L]) {
            equivalences.unify(out_labels[loc], out_labels[loc + L]); 
          }
        }
        else if (x > 0 && y > 0 && in_labels[loc + J]) {
          out_labels[loc] = out_labels[loc + J];

          if (x < sx - 1 && y > 0 && in_labels[loc + L]) {
            equivalences.unify(out_labels[loc], out_labels[loc + L]); 
          }
        }
        else if (x < sx - 1 && y > 0 && in_labels[loc + L]) {
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
  
  return relabel<OUT>(out_labels, sx, sy, sz, next_label, equivalences, N, runs.get());
}

template <typename T, typename OUT = uint32_t>
OUT* connected_components3d_18_binary(
    T* in_labels, 
    const int64_t sx, const int64_t sy, const int64_t sz,
    size_t max_labels, 
    OUT *out_labels = NULL, size_t &N = _dummy_N
  ) {

  const int64_t sxy = sx * sy;
  const int64_t voxels = sxy * sz;

  if (out_labels == NULL) {
    out_labels = new OUT[voxels]();
  }

  if (max_labels == 0) {
    return out_labels;
  }

  max_labels++; // corrects Cython estimation
  max_labels = std::min(max_labels, static_cast<size_t>(voxels) + 1); // + 1L for an array with no zeros
  max_labels = std::min(max_labels, static_cast<size_t>(std::numeric_limits<OUT>::max()));

  DisjointSet<OUT> equivalences(max_labels);

  const std::unique_ptr<uint32_t[]> runs(
    compute_foreground_index(in_labels, sx, sy, sz)
  );
     
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
  int64_t row = 0;

  // Raster Scan 1: Set temporary labels and 
  // record equivalences in a disjoint set.
  for (int64_t z = 0; z < sz; z++) {
    for (int64_t y = 0; y < sy; y++, row++) {
      const int64_t xstart = runs[row << 1];
      const int64_t xend = runs[(row << 1) + 1];

      for (int64_t x = xstart; x < xend; x++) {
        loc = x + sx * (y + sy * z);
        const T cur = in_labels[loc];

        if (cur == 0) {
          continue;
        }

        if (z > 0 && in_labels[loc + E]) {
          out_labels[loc] = out_labels[loc + E];

          if (x > 0 && y > 0 && in_labels[loc + J]) {
            equivalences.unify(out_labels[loc], out_labels[loc + J]);
          }
          if (x < sx - 1 && y > 0 && in_labels[loc + L]) {
            equivalences.unify(out_labels[loc], out_labels[loc + L]); 
          }
        }
        else if (y > 0 && z > 0 && in_labels[loc + B]) {
          out_labels[loc] = out_labels[loc + B];

          if (x > 0 && in_labels[loc + M]) {
            equivalences.unify(out_labels[loc], out_labels[loc + M]);
          }
          if (y < sy - 1 && z > 0 && in_labels[loc + H]) {
            equivalences.unify(out_labels[loc], out_labels[loc + H]); 
          }
        }
        else if (x > 0 && z > 0 && in_labels[loc + D]) {
          out_labels[loc] = out_labels[loc + D];

          if (x < sx - 1 && y > 0 && in_labels[loc + L]) {
            equivalences.unify(out_labels[loc], out_labels[loc + L]); 
          }
          else {
            if (y > 0 && in_labels[loc + K]) {
              equivalences.unify(out_labels[loc], out_labels[loc + K]); 
            }
            if (x < sx - 1 && z > 0 && in_labels[loc + F]) {
              equivalences.unify(out_labels[loc], out_labels[loc + F]); 
            }
          }
        }
        else if (x < sx - 1 && z > 0 && in_labels[loc + F]) {
          out_labels[loc] = out_labels[loc + F];

          if (x > 0 && y > 0 && in_labels[loc + J]) {
            equivalences.unify(out_labels[loc], out_labels[loc + J]);
          }
          else {
            if (x > 0 && in_labels[loc + M]) {
              equivalences.unify(out_labels[loc], out_labels[loc + M]); 
            }
            if (y > 0 && in_labels[loc + K]) {
              equivalences.unify(out_labels[loc], out_labels[loc + K]); 
            }            
          }
        }
        else if (y < sy - 1 && z > 0 && in_labels[loc + H]) {
          out_labels[loc] = out_labels[loc + H];
          unify2d<T>(loc, cur, x, y, sx, sy, in_labels, out_labels, equivalences);
        }
        // It's the original 2D problem now
        else if (y > 0 && in_labels[loc + K]) {
          out_labels[loc] = out_labels[loc + K];
        }
        else if (x > 0 && in_labels[loc + M]) {
          out_labels[loc] = out_labels[loc + M];

          if (x < sx - 1 && y > 0 && in_labels[loc + L]) {
            equivalences.unify(out_labels[loc], out_labels[loc + L]); 
          }
        }
        else if (x > 0 && y > 0 && in_labels[loc + J]) {
          out_labels[loc] = out_labels[loc + J];

          if (x < sx - 1 && y > 0 && in_labels[loc + L]) {
            equivalences.unify(out_labels[loc], out_labels[loc + L]); 
          }
        }
        else if (x < sx - 1 && y > 0 && in_labels[loc + L]) {
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

  return relabel<OUT>(out_labels, sx, sy, sz, next_label, equivalences, N, runs.get());
}

template <typename T, typename OUT = uint32_t>
OUT* connected_components3d_6_binary(
    T* in_labels, 
    const int64_t sx, const int64_t sy, const int64_t sz,
    size_t max_labels, 
    OUT *out_labels = NULL, size_t &N = _dummy_N,
    bool periodic_boundary = false
  ) {

  const int64_t sxy = sx * sy;
  const int64_t voxels = sxy * sz;

  if (out_labels == NULL) {
    out_labels = new OUT[voxels]();
  }

  if (max_labels == 0) {
    return out_labels;
  }

  max_labels++; // corrects Cython estimation
  max_labels = std::min(max_labels, static_cast<size_t>(voxels) + 1); // + 1L for an array with no zeros
  max_labels = std::min(max_labels, static_cast<size_t>(std::numeric_limits<OUT>::max()));

  DisjointSet<OUT> equivalences(max_labels);

  const std::unique_ptr<uint32_t[]> runs(
    compute_foreground_index(in_labels, sx, sy, sz)
  );

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
  int64_t row = 0;
  OUT next_label = 0;

  // Raster Scan 1: Set temporary labels and 
  // record equivalences in a disjoint set.

  for (int64_t z = 0; z < sz; z++) {
    for (int64_t y = 0; y < sy; y++, row++) {
      const int64_t xstart = runs[row << 1];
      const int64_t xend = runs[(row << 1) + 1];

      for (int64_t x = xstart; x < xend; x++) {
        loc = x + sx * (y + sy * z);

        const T cur = in_labels[loc];

        if (cur == 0) {
          continue;
        }

        if (x > 0 && in_labels[loc + M]) {
          out_labels[loc] = out_labels[loc + M];

          if (y > 0 && in_labels[loc + K] && cur != in_labels[loc + J]) {
            equivalences.unify(out_labels[loc], out_labels[loc + K]); 
            if (z > 0 && in_labels[loc + E]) {
              if (cur != in_labels[loc + D] && cur != in_labels[loc + B]) {
                equivalences.unify(out_labels[loc], out_labels[loc + E]);
              }
            }
          }
          else if (z > 0 && in_labels[loc + E] && cur != in_labels[loc + D]) {
            equivalences.unify(out_labels[loc], out_labels[loc + E]); 
          }
        }
        else if (y > 0 && in_labels[loc + K]) {
          out_labels[loc] = out_labels[loc + K];

          if (z > 0 && in_labels[loc + E] && cur != in_labels[loc + B]) {
            equivalences.unify(out_labels[loc], out_labels[loc + E]); 
          }
        }
        else if (z > 0 && in_labels[loc + E]) {
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

  if (periodic_boundary) {
    for (int64_t z = 0; z < sz; z++) {
      for (int64_t y = 0; y < sy; y++) {
        loc = sx * (y + sy * z);
        if (in_labels[loc] != 0 && in_labels[loc + sx - 1]) {
          equivalences.unify(out_labels[loc], out_labels[loc + sx - 1]);
        }
      }
    }
    for (int64_t z = 0; z < sz; z++) {
      for (int64_t x = 0; x < sx; x++) {
        loc = x + sxy * z;
        if (in_labels[loc] != 0 && in_labels[loc + sx * (sy - 1)]) {
          equivalences.unify(out_labels[loc], out_labels[loc + sx * (sy - 1)]);
        }
      }
    }
    for (int64_t y = 0; y < sy; y++) {
      for (int64_t x = 0; x < sx; x++) {
        loc = x + sx * y;
        if (in_labels[loc] != 0 && in_labels[loc + sxy * (sz - 1)]) {
          equivalences.unify(out_labels[loc], out_labels[loc + sxy * (sz - 1)]);
        }
      }
    }
  }

  return relabel<OUT>(out_labels, sx, sy, sz, next_label, equivalences, N, runs.get());
}


// uses an approach inspired by 2x2 block based decision trees
// by Grana et al that was intended for 8-connected. Here we 
// skip a unify on every other voxel in the horizontal and
// vertical directions.
template <typename T, typename OUT = uint32_t>
OUT* connected_components2d_4_binary(
    T* in_labels, 
    const int64_t sx, const int64_t sy, 
    size_t max_labels, 
    OUT *out_labels = NULL, size_t &N = _dummy_N,
    const bool periodic_boundary = false
  ) {

  const int64_t voxels = sx * sy;

  if (out_labels == NULL) {
    out_labels = new OUT[voxels]();
  }

  if (max_labels == 0) {
    return out_labels;
  }

  max_labels++; // corrects Cython estimation
  max_labels = std::min(max_labels, static_cast<size_t>(voxels) + 1); // + 1L for an array with no zeros
  max_labels = std::min(max_labels, static_cast<size_t>(std::numeric_limits<OUT>::max()));

  DisjointSet<OUT> equivalences(max_labels);

  const std::unique_ptr<uint32_t[]> runs(
    compute_foreground_index(in_labels, sx, sy, /*sz=*/1)
  );
    
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
  int64_t row = 0;
  OUT next_label = 0;

  // Raster Scan 1: Set temporary labels and 
  // record equivalences in a disjoint set.

  T cur = 0;
  for (int64_t y = 0; y < sy; y++, row++) {
    const int64_t xstart = runs[row << 1];
    const int64_t xend = runs[(row << 1) + 1];

    for (int64_t x = xstart; x < xend; x++) {
      loc = x + sx * y;
      cur = in_labels[loc];

      if (cur == 0) {
        continue;
      }

      if (x > 0 && in_labels[loc + B]) {
        out_labels[loc + A] = out_labels[loc + B];
        if (y > 0 && cur != in_labels[loc + D] && in_labels[loc + C]) {
          equivalences.unify(out_labels[loc + A], out_labels[loc + C]);
        }
      }
      else if (y > 0 && in_labels[loc + C]) {
        out_labels[loc + A] = out_labels[loc + C];
      }
      else {
        next_label++;
        out_labels[loc + A] = next_label;
        equivalences.add(out_labels[loc + A]);
      }
    }
  }

  if (periodic_boundary) {
    for (int64_t x = 0; x < sx; x++) {
      if (in_labels[x] && in_labels[x + sx * (sy - 1)]) {
        equivalences.unify(out_labels[x], out_labels[x + sx * (sy - 1)]);
      }
    }
    for (int64_t y = 0; y < sy; y++) {
      loc = sx * y;
      if (in_labels[loc] && in_labels[loc + (sx - 1)]) {
        equivalences.unify(out_labels[loc], out_labels[loc + (sx - 1)]);
      }
    }
  }

  return relabel<OUT>(out_labels, sx, sy, /*sz=*/1, next_label, equivalences, N, runs.get());
}

// K. Wu, E. Otoo, K. Suzuki. "Two Strategies to Speed up Connected Component Labeling Algorithms". 
// Lawrence Berkely National Laboratory. LBNL-29102, 2005.
// This is the stripped down version of that decision tree algorithm.
// It seems to give up to about 1.18x improvement on some data. No improvement on binary
// vs 18 connected (from 3D).
template <typename T, typename OUT = uint32_t>
OUT* connected_components2d_8_binary(
    T* in_labels, 
    const int64_t sx, const int64_t sy,
    size_t max_labels, 
    OUT *out_labels = NULL, size_t &N = _dummy_N,
    bool periodic_boundary = false
  ) {

  const int64_t voxels = sx * sy;

  if (out_labels == NULL) {
    out_labels = new OUT[voxels]();
  }

  if (max_labels == 0) {
    return out_labels;
  }

  max_labels++; // corrects Cython estimation
  max_labels = std::max(std::min(max_labels, static_cast<size_t>(voxels) + 1), static_cast<size_t>(1L)); // can't allocate 0 arrays
  max_labels = std::min(max_labels, static_cast<size_t>(std::numeric_limits<OUT>::max()));
  
  DisjointSet<OUT> equivalences(max_labels);

  const std::unique_ptr<uint32_t[]> runs(
    compute_foreground_index(in_labels, sx, sy, /*sz=*/1)
  );

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
  int64_t row = 0;
  OUT next_label = 0;

  // Raster Scan 1: Set temporary labels and 
  // record equivalences in a disjoint set.
  for (int64_t y = 0; y < sy; y++, row++) {
    const int64_t xstart = runs[row << 1];
    const int64_t xend = runs[(row << 1) + 1];

    for (int64_t x = xstart; x < xend; x++) {
      loc = x + sx * y;

      const T cur = in_labels[loc];

      if (cur == 0) {
        continue;
      }

      if (y > 0 && in_labels[loc + B]) {
        out_labels[loc] = out_labels[loc + B];
      }
      else if (x > 0 && y > 0 && in_labels[loc + A]) {
        out_labels[loc] = out_labels[loc + A];
        if (x < sx - 1 && y > 0 && in_labels[loc + C] 
            && !(y > 1 && in_labels[loc + P])) {

            equivalences.unify(out_labels[loc], out_labels[loc + C]);
        }
      }
      else if (x > 0 && in_labels[loc + D]) {
        out_labels[loc] = out_labels[loc + D];
        if (x < sx - 1 && y > 0 && in_labels[loc + C]) {
          equivalences.unify(out_labels[loc], out_labels[loc + C]);
        }
      }
      else if (x < sx - 1 && y > 0 && in_labels[loc + C]) {
        out_labels[loc] = out_labels[loc + C];
      }
      else {
        next_label++;
        out_labels[loc] = next_label;
        equivalences.add(out_labels[loc]);      
      }
    }
  }

  if (periodic_boundary) {
    for (int64_t x = 0; x < sx; x++) {
      if (in_labels[x] == 0) {
        continue;
      }

      if (x > 0 && in_labels[x] && in_labels[x - 1 + sx * (sy - 1)]) {
        equivalences.unify(out_labels[x], out_labels[x - 1 + sx * (sy - 1)]);
      }
      if (in_labels[x] && in_labels[x + sx * (sy - 1)]) {
        equivalences.unify(out_labels[x], out_labels[x + sx * (sy - 1)]);
      }
      if (x < sx - 1 && in_labels[x] && in_labels[x + 1 + sx * (sy - 1)]) {
        equivalences.unify(out_labels[x], out_labels[x + 1 + sx * (sy - 1)]);
      }
    }

    if (in_labels[0] && in_labels[voxels - 1]) {
      equivalences.unify(out_labels[0], out_labels[voxels - 1]);
    }
    if (in_labels[sx - 1] && in_labels[sx * (sy - 1)]) {
      equivalences.unify(out_labels[sx - 1], out_labels[sx * (sy - 1)]);
    }

    for (int64_t y = 0; y < sy; y++) {
      loc = sx * y;
      if (in_labels[loc] && in_labels[loc + (sx - 1)]) {
        equivalences.unify(out_labels[loc], out_labels[loc + (sx - 1)]);
      }
    }
  }

  return relabel<OUT>(out_labels, sx, sy, /*sz=*/1, next_label, equivalences, N, runs.get());
}

template <typename T, typename OUT = uint32_t>
OUT* connected_components3d_binary(
    T* in_labels, 
    const int64_t sx, const int64_t sy, const int64_t sz,
    size_t max_labels, const int64_t connectivity,
    OUT *out_labels = NULL, size_t &N = _dummy_N, 
    bool periodic_boundary = false
  ) {

  if (connectivity == 26) {
    return connected_components3d_26_binary<T, OUT>(
      in_labels, sx, sy, sz, 
      max_labels, out_labels, N
    );
  }
  else if (connectivity == 18) {
    return connected_components3d_18_binary<T, OUT>(
      in_labels, sx, sy, sz, 
      max_labels, out_labels, N
    );
  }
  else if (connectivity == 6) {
    return connected_components3d_6_binary<T, OUT>(
      in_labels, sx, sy, sz, 
      max_labels, out_labels, N, periodic_boundary
    );
  }
  else if (connectivity == 8) {
    if (sz != 1) {
      throw std::runtime_error("sz must be 1 for 2D connectivities.");
    }
    return connected_components2d_8_binary<T,OUT>(
      in_labels, sx, sy,
      max_labels, out_labels, N, periodic_boundary
    );
  }
  else if (connectivity == 4) {
    if (sz != 1) {
      throw std::runtime_error("sz must be 1 for 2D connectivities.");
    }
    return connected_components2d_4_binary<T, OUT>(
      in_labels, sx, sy, 
      max_labels, out_labels, N, periodic_boundary
    );
  }
  else {
    throw std::runtime_error("Only 4 and 8 2D and 6, 18, and 26 3D connectivities are supported.");
  }
}

template <typename T, typename OUT = uint32_t>
OUT* connected_components3d_binary(
    T* in_labels, 
    const int64_t sx, const int64_t sy, const int64_t sz,
    const int64_t connectivity=26, size_t &N = _dummy_N,
    const bool periodic_boundary = false
  ) {
  const size_t voxels = sx * sy * sz;
  size_t max_labels = std::min(estimate_provisional_label_count(in_labels, sx, voxels), voxels);
  return connected_components3d<T, OUT>(
    in_labels, sx, sy, sz, 
    max_labels, connectivity, 
    NULL, N, periodic_boundary
  );
}

};

#endif