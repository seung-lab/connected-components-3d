#ifndef CC3D_CONTINUOUS_HPP
#define CC3D_CONTINUOUS_HPP 

#include "cc3d.hpp"

namespace {

inline void compute_neighborhood(
  int *neighborhood, 
  const int x, const int y, const int z,
  const uint64_t sx, const uint64_t sy, const uint64_t sz,
  const int connectivity = 26
) {

  const int sxy = sx * sy;

  // 6-hood
  neighborhood[0] = -1 * (x > 0); // -x
  neighborhood[1] = (x < (static_cast<int>(sx) - 1)); // +x
  neighborhood[2] = -static_cast<int>(sx) * (y > 0); // -y
  neighborhood[3] = static_cast<int>(sx) * (y < static_cast<int>(sy) - 1); // +y
  neighborhood[4] = -sxy * static_cast<int>(z > 0); // -z
  neighborhood[5] = sxy * (z < static_cast<int>(sz) - 1); // +z

  // 18-hood

  // xy diagonals
  neighborhood[6] = (connectivity > 6) * (neighborhood[0] + neighborhood[2]) * (neighborhood[0] && neighborhood[2]); // up-left
  neighborhood[7] = (connectivity > 6) * (neighborhood[0] + neighborhood[3]) * (neighborhood[0] && neighborhood[3]); // up-right
  neighborhood[8] = (connectivity > 6) * (neighborhood[1] + neighborhood[2]) * (neighborhood[1] && neighborhood[2]); // down-left
  neighborhood[9] = (connectivity > 6) * (neighborhood[1] + neighborhood[3]) * (neighborhood[1] && neighborhood[3]); // down-right

  // yz diagonals
  neighborhood[10] = (connectivity > 6) * (neighborhood[2] + neighborhood[4]) * (neighborhood[2] && neighborhood[4]); // up-left
  neighborhood[11] = (connectivity > 6) * (neighborhood[2] + neighborhood[5]) * (neighborhood[2] && neighborhood[5]); // up-right
  neighborhood[12] = (connectivity > 6) * (neighborhood[3] + neighborhood[4]) * (neighborhood[3] && neighborhood[4]); // down-left
  neighborhood[13] = (connectivity > 6) * (neighborhood[3] + neighborhood[5]) * (neighborhood[3] && neighborhood[5]); // down-right

  // xz diagonals
  neighborhood[14] = (connectivity > 6) * (neighborhood[0] + neighborhood[4]) * (neighborhood[0] && neighborhood[4]); // up-left
  neighborhood[15] = (connectivity > 6) * (neighborhood[0] + neighborhood[5]) * (neighborhood[0] && neighborhood[5]); // up-right
  neighborhood[16] = (connectivity > 6) * (neighborhood[1] + neighborhood[4]) * (neighborhood[1] && neighborhood[4]); // down-left
  neighborhood[17] = (connectivity > 6) * (neighborhood[1] + neighborhood[5]) * (neighborhood[1] && neighborhood[5]); // down-right

  // 26-hood

  // Now the eight corners of the cube
  neighborhood[18] = (connectivity > 18) * (neighborhood[0] + neighborhood[2] + neighborhood[4]) * (neighborhood[2] && neighborhood[4]);
  neighborhood[19] = (connectivity > 18) * (neighborhood[1] + neighborhood[2] + neighborhood[4]) * (neighborhood[2] && neighborhood[4]);
  neighborhood[20] = (connectivity > 18) * (neighborhood[0] + neighborhood[3] + neighborhood[4]) * (neighborhood[3] && neighborhood[4]);
  neighborhood[21] = (connectivity > 18) * (neighborhood[0] + neighborhood[2] + neighborhood[5]) * (neighborhood[2] && neighborhood[5]);
  neighborhood[22] = (connectivity > 18) * (neighborhood[1] + neighborhood[3] + neighborhood[4]) * (neighborhood[3] && neighborhood[4]);
  neighborhood[23] = (connectivity > 18) * (neighborhood[1] + neighborhood[2] + neighborhood[5]) * (neighborhood[2] && neighborhood[5]);
  neighborhood[24] = (connectivity > 18) * (neighborhood[0] + neighborhood[3] + neighborhood[5]) * (neighborhood[3] && neighborhood[5]);
  neighborhood[25] = (connectivity > 18) * (neighborhood[1] + neighborhood[3] + neighborhood[5]) * (neighborhood[3] && neighborhood[5]);
}

};

namespace cc3d {

#define MATCH(cur, val) (std::max((cur), (val)) - std::min((cur), (val)) <= delta)

// uses an approach inspired by 2x2 block based decision trees
// by Grana et al that was intended for 8-connected. Here we 
// skip a unify on every other voxel in the horizontal and
// vertical directions.
template <typename T, typename OUT = uint32_t>
OUT* connected_components2d_4(
    T* in_labels, 
    const int64_t sx, const int64_t sy, 
    size_t max_labels, 
    OUT *out_labels = NULL, size_t &N = _dummy_N,
    const T delta = 0
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
    size_t max_labels, 
    OUT *out_labels = NULL, size_t &N = _dummy_N,
    const T delta = 0
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

  // if (connectivity == 26) {
  //   return connected_components3d_26<T, OUT>(
  //     in_labels, sx, sy, sz, 
  //     max_labels, out_labels, N
  //   );
  // }
  // else if (connectivity == 18) {
  //   return connected_components3d_18<T, OUT>(
  //     in_labels, sx, sy, sz, 
  //     max_labels, out_labels, N
  //   );
  // }
  // else if (connectivity == 6) {
  //   return connected_components3d_6<T, OUT>(
  //     in_labels, sx, sy, sz, 
  //     max_labels, out_labels, N
  //   );
  // }
  if (connectivity == 8) {
    if (sz != 1) {
      throw std::runtime_error("sz must be 1 for 2D connectivities.");
    }
    return connected_components2d_8<T,OUT>(
      in_labels, sx, sy,
      max_labels, out_labels, N, delta
    );
  }
  else if (connectivity == 4) {
    if (sz != 1) {
      throw std::runtime_error("sz must be 1 for 2D connectivities.");
    }
    return connected_components2d_4<T, OUT>(
      in_labels, sx, sy, 
      max_labels, out_labels, N, delta
    );
  }
  else {
    throw std::runtime_error("Only 4 and 8 2D and 6, 18, and 26 3D connectivities are supported.");
  }
}

// template <typename T, typename OUT = uint32_t>
// OUT* connected_components3d(
//     T* in_labels, 
//     const int64_t sx, const int64_t sy, const int64_t sz,
//     const T delta,
//     const int64_t connectivity=26, size_t &N = _dummy_N
//   ) {
//   const size_t voxels = sx * sy * sz;
//   size_t max_labels = std::min(estimate_provisional_label_count(in_labels, sx, voxels), voxels);
//   return connected_components3d<T, OUT>(
//     in_labels, sx, sy, sz, max_labels, 
//     connectivity, NULL, N, delta
//   );
// }

// // template <typename T, typename OUT = uint32_t>
// // OUT* connected_components3d_continuous(
// //     T* in_labels, 
// //     const int64_t sx, const int64_t sy, const int64_t sz,
// //     size_t max_labels, const int64_t connectivity,
// //     OUT *out_labels = NULL, size_t &N = _dummy_N,
// //     const T delta = 0
// //   ) {
// //   return connected_components3d<T,OUT>(
// //     in_labels, sx, sy, sz,
// //     max_labels, connectivity,
// //     out_labels, N, delta
// //   );
// // }

#undef MATCH

};

#endif