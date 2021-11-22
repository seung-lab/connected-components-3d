/* cc3d_continuous.hpp
 *
 * Defines connected components on continuously
 * valued images such as photographs.
 *
 * The algorithms for this seem to be more limited
 * because there is less structure in the image to 
 * use.
 *
 * Author: William Silversmith
 * Date: November 2021
 * Affiliation: Princeton Neuroscience Institute
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

#ifndef CC3D_CONTINUOUS_HPP
#define CC3D_CONTINUOUS_HPP 

#include "cc3d.hpp"

namespace {

inline void compute_neighborhood(
  int *neighborhood, 
  const int x, const int y, const int z,
  const int sx, const int sy, const int sz,
  const int connectivity = 26
) {

  const int sxy = sx * sy;

  // 6-hood
  neighborhood[0] = -1 * (x > 0); // -x
  neighborhood[1] = -sx * (y > 0); // -y
  neighborhood[2] = -sxy * (z > 0); // -z

  // 18-hood

  // xy diagonals
  neighborhood[3] = (connectivity > 6) * (x > 0 && y > 0) * (-1 - sx); // up-left
  neighborhood[4] = (connectivity > 6) * (x < sx - 1 && y > 0) * (1 - sx); // up-right

  // yz diagonals
  neighborhood[5] = (connectivity > 6) * (y > 0 && z > 0) * (-sx - sxy); // down-left
  neighborhood[6] = (connectivity > 6) * (y < sy - 1 && z > 0) * (sx - sxy); // down-right

  // xz diagonals
  neighborhood[7] = (connectivity > 6) * (x > 0 && z > 0) * (-1 - sxy); // down-left
  neighborhood[8] = (connectivity > 6) * (x < sx - 1 && z > 0) * (1 - sxy); // down-right

  // 26-hood

  // Now the four corners of the bottom plane
  neighborhood[ 9] = (connectivity > 18) * (x > 0 && y > 0 && z > 0) * (-1 - sx - sxy);
  neighborhood[10] = (connectivity > 18) * (x < sx - 1 && y > 0 && z > 0) * (1 - sx - sxy);
  neighborhood[11] = (connectivity > 18) * (x > 0 && y < sy - 1 && z > 0) * (-1 + sx - sxy);
  neighborhood[12] = (connectivity > 18) * (x < sx - 1 && y < sy - 1 && z > 0) * (1 + sx - sxy);
}

};

namespace cc3d {

#define MATCH(cur, val) (std::max((cur), (val)) - std::min((cur), (val)) <= delta)

template <typename T, typename OUT = uint32_t>
OUT* connected_components3d_continuous(
    T* in_labels, 
    const int64_t sx, const int64_t sy, const int64_t sz,
    size_t max_labels, const int64_t connectivity, const T delta,
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

  const uint32_t *runs = compute_foreground_index(in_labels, sx, sy, sz);


  int64_t loc = 0;
  int64_t row = 0;
  OUT next_label = 0;

  int neighborhood[13];

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

        compute_neighborhood(neighborhood, x, y, z, sx, sy, sz, connectivity);
        bool any = false;

        for (int64_t i = 0; i < connectivity / 2; i++) {
          int64_t neighbor = neighborhood[i];

          if (neighbor == 0 || in_labels[loc + neighbor] == 0) {
            continue;
          }

          if (MATCH(cur, in_labels[loc + neighbor])) {
            if (any) {
              equivalences.unify(out_labels[loc], out_labels[loc + neighbor]);
            }
            else {
              out_labels[loc] = out_labels[loc + neighbor];  
            }
            any = true;
          }
        }

        if (!any) {
          next_label++;
          out_labels[loc] = next_label;
          equivalences.add(out_labels[loc]);        
        }
      }
    }
  }

  out_labels = relabel<OUT>(out_labels, sx, sy, sz, next_label, equivalences, N, runs);
  delete[] runs;
  return out_labels;
}

template <typename T, typename OUT = uint32_t>
OUT* connected_components2d_4(
    T* in_labels, 
    const int64_t sx, const int64_t sy, 
    size_t max_labels, const T delta,
    OUT *out_labels = NULL, size_t &N = _dummy_N
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

  const uint32_t *runs = compute_foreground_index(in_labels, sx, sy, /*sz=*/1);
    
  /*
    Layout of forward pass mask. 
    A is the current location.
    D C 
    B A 
  */

  const int64_t A = 0;
  const int64_t B = -1;
  const int64_t C = -sx;

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

      if (x > 0 && in_labels[loc + B] && MATCH(cur, in_labels[loc + B])) {
        out_labels[loc + A] = out_labels[loc + B];
        if (y > 0 && in_labels[loc + C] && MATCH(cur, in_labels[loc + C])) {
          equivalences.unify(out_labels[loc + A], out_labels[loc + C]);
        }
      }
      else if (y > 0 && in_labels[loc + C] && MATCH(cur, in_labels[loc + C])) {
        out_labels[loc + A] = out_labels[loc + C];
      }
      else {
        next_label++;
        out_labels[loc + A] = next_label;
        equivalences.add(out_labels[loc + A]);
      }
    }
  }

  out_labels = relabel<OUT>(
    out_labels, sx, sy, /*sz=*/1, next_label, 
    equivalences, N, runs
  );
  delete[] runs;
  return out_labels;
}

template <typename T, typename OUT = uint32_t>
OUT* connected_components2d_8(
    T* in_labels, 
    const int64_t sx, const int64_t sy,
    size_t max_labels, const T delta,
    OUT *out_labels = NULL, size_t &N = _dummy_N
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

  const uint32_t *runs = compute_foreground_index(in_labels, sx, sy, /*sz=*/1);

  /*
    Layout of mask. We start from e.

    a | b | c
    d | e |
  */

  const int64_t A = -1 - sx;
  const int64_t B = -sx;
  const int64_t C = +1 - sx;
  const int64_t D = -1;

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

      bool any = false;
      if (x > 0 && y > 0 && in_labels[loc + A] && MATCH(cur, in_labels[loc + A])) {
        out_labels[loc] = out_labels[loc + A];
        any = true;
      }
      if (y > 0 && in_labels[loc + B] && MATCH(cur, in_labels[loc + B])) {
        if (any) {
          equivalences.unify(out_labels[loc], out_labels[loc + B]);
        }
        else {
          out_labels[loc] = out_labels[loc + B];
        }
        any = true;
      }
      if (x < sx - 1 && y > 0 && in_labels[loc + C] && MATCH(cur, in_labels[loc + C])) {
        if (any) {
          equivalences.unify(out_labels[loc], out_labels[loc + C]);
        }
        else {
          out_labels[loc] = out_labels[loc + C];
        }
        any = true;
      }
      if (x > 0 && in_labels[loc + D] && MATCH(cur, in_labels[loc + D])) {
        if (any) {
          equivalences.unify(out_labels[loc], out_labels[loc + D]);
        }
        else {
          out_labels[loc] = out_labels[loc + D];
        }
        any = true;
      }


      if (!any) {
        next_label++;
        out_labels[loc] = next_label;
        equivalences.add(out_labels[loc]);        
      }
    }
  }

  out_labels = relabel<OUT>(out_labels, sx, sy, /*sz=*/1, next_label, equivalences, N, runs);
  delete[] runs;
  return out_labels;
}

template <typename T, typename OUT = uint32_t>
OUT* connected_components3d(
    T* in_labels, 
    const int64_t sx, const int64_t sy, const int64_t sz,
    size_t max_labels, const int64_t connectivity, const T delta,
    OUT *out_labels = NULL, size_t &N = _dummy_N
  ) {

  // for performance, shouldn't be "more correct"
  if (delta == 0) {
    return connected_components3d<T,OUT>(
      in_labels, sx, sy, sz, 
      max_labels, connectivity, 
      out_labels, N
    );
  }

  if (connectivity == 26 || connectivity == 18 || connectivity == 6) {
    return connected_components3d_continuous<T, OUT>(
      in_labels, sx, sy, sz, 
      max_labels, connectivity, delta,
      out_labels, N
    );
  }
  else if (connectivity == 8) {
    if (sz != 1) {
      throw std::runtime_error("sz must be 1 for 2D connectivities.");
    }
    return connected_components2d_8<T,OUT>(
      in_labels, sx, sy,
      max_labels, delta, out_labels, N
    );
  }
  else if (connectivity == 4) {
    if (sz != 1) {
      throw std::runtime_error("sz must be 1 for 2D connectivities.");
    }
    return connected_components2d_4<T, OUT>(
      in_labels, sx, sy, 
      max_labels, delta, out_labels, N
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
    const T delta,
    const int64_t connectivity=26, size_t &N = _dummy_N
  ) {
  const size_t voxels = sx * sy * sz;
  size_t max_labels = std::min(estimate_provisional_label_count(in_labels, sx, voxels), voxels);
  return connected_components3d<T, OUT>(
    in_labels, sx, sy, sz, 
    max_labels, connectivity, delta, 
    NULL, N
  );
}

#undef MATCH

};

#endif