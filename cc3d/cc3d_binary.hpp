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

template <typename T>
uint8_t* create_2x2x2_minor_image(
    T* in_labels, 
    const int64_t sx, const int64_t sy, const int64_t sz
) {

  const int64_t sxy = sx * sy;

  const int64_t msx = (sx + 1) >> 1;
  const int64_t msy = (sy + 1) >> 1;
  const int64_t msz = (sz + 1) >> 1;

  const int64_t minor_voxels = msx * msy * msz;

  uint8_t* minor = new uint8_t[minor_voxels]();

  int64_t i = 0;
  for (int64_t z = 0; z < sz; z += 2) {
    for (int64_t y = 0; y < sy; y += 2) {
      for (int64_t x = 0; x < sx; x += 2, i++) {
        int64_t loc = x + sx * y + sxy * z;
        minor[i] = (
          (in_labels[loc] > 0)
          | (((x < sx - 1) && (in_labels[loc+1] > 0)) << 1)
          | (((y < sy - 1) && (in_labels[loc+sx] > 0)) << 2)
          | (((x < sx - 1 && y < sy - 1) && (in_labels[loc+sx+1] > 0)) << 3)
          | (((z < sz - 1) && (in_labels[loc+sxy] > 0)) << 4)
          | (((x < sx - 1 && z < sz - 1) && (in_labels[loc+sxy+1] > 0)) << 5)
          | (((y < sy - 1 && z < sz - 1) && (in_labels[loc+sxy+sx] > 0)) << 6)
          | (((x < sx - 1 && y < sy - 1 && z < sz - 1) && (in_labels[loc+sxy+sx+1] > 0)) << 7)
        );
      }
    }
  }

  return minor;
}

bool is_26_connected(
  const uint8_t center, const uint8_t candidate, 
  const int x, const int y, const int z
) {
  if (x < 0) {
    if (y < 0) {
      if (z < 0) {
        return (candidate & 0b00000100) && (center & 0b00010000);
      }
      else if (z == 0) {
        return (candidate & 0b00010100) && (center & 0b00010001);
      }
      else {
        return (candidate & 0b00010000) && (center & 0b00000001);
      }
    }
    else if (y == 0) {
      if (z < 0) {
        return (candidate & 0b10100000) && (center & 0b00000101);
      }
      else if (z == 0) {
        return (candidate & 0b10101010) && (center & 0b01010101);
      }
      else {
        return (candidate & 0b00001010) && (center & 0b01010000);
      }
    }
    else {
      if (z < 0) {
        return (candidate & 0b00100000) && (center & 0b00000100);
      }
      else if (z == 0) {
        return (candidate & 0b00100010) && (center & 0b01000010);
      }
      else {
        return (candidate & 0b00000010) && (center & 0b01000000);
      }
    }
  }
  else if (x == 0) {
    if (y < 0) {
      if (z < 0) {
        return (candidate & 0b11000000) && (center & 0b00000011);
      }
      else if (z == 0) {
        return (candidate & 0b11001100) && (center & 0b00110011);
      }
      else {
        return (candidate & 0b00001100) && (center & 0b00110000);
      }
    }
    else if (y == 0) {
      if (z < 0) {
        return (candidate & 0b11110000) && (center & 0b00001111);
      }
      else if (z == 0) {
        return true;
      }
      else {
        return (candidate & 0b00001111) && (center & 0b11110000);
      }
    }
    else {
      if (z < 0) {
        return (candidate & 0b00110000) && (center & 0b00001100);
      }
      else if (z == 0) {
        return (candidate & 0b00110011) && (center & 0b11001100);
      }
      else {
        return (candidate & 0b00000011) && (center & 0b11000000);
      }
    }
  }
  else {
    if (y < 0) {
      if (z < 0) {
        return (candidate & 0b01000000) && (center & 0b00000010);
      }
      else if (z == 0) {
        return (candidate & 0b01000100) && (center & 0b00100010);
      }
      else {
        return (candidate & 0b00000100) && (center & 0b00100000);
      }
    }
    else if (y == 0) {
      if (z < 0) {
        return (candidate & 0b01010000) && (center & 0b00001010);
      }
      else if (z == 0) {
        return (candidate & 0b01010101) && (center & 0b10101010);
      }
      else {
        return (candidate & 0b00000101) && (center & 0b10100000);
      }
    }
    else {
      if (z < 0) {
        return (candidate & 0b00010000) && (center & 0b00001000);
      }
      else if (z == 0) {
        return (candidate & 0b00010001) && (center & 0b10001000);
      }
      else {
        return (candidate & 0b00000001) && (center & 0b10000000);
      }
    }
  }
}

// This is the second raster pass of the two pass algorithm family.
// The input array (output_labels) has been assigned provisional 
// labels and this resolves them into their final labels. We
// modify this pass to also ensure that the output labels are
// numbered from 1 sequentially.

// This special version for the 2x2x2 block based decision tree
// exploits the fact that the downsampled version of the out labels
// is written to the front of the array and then writes backwards
// erasing them as it goes.

// The fact that the first pass does not touch every voxel in out_labels
// unlike the multi-label version means we can't early exit on 
// a few special conditions like before.
template <typename T, typename OUT = uint32_t>
OUT* relabel_2x2x2(
  T* in_labels, OUT* out_labels, 
  const int64_t sx, const int64_t sy, const int64_t sz,
  const int64_t num_labels, DisjointSet<OUT> &equivalences,
  size_t &N
) {
  OUT label;
  std::unique_ptr<OUT[]> renumber(new OUT[num_labels + 1]());
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

  N = next_label - 1;

  const int64_t msx = (sx + 1) >> 1;
  const int64_t msy = (sy + 1) >> 1;
  const int64_t msz = (sz + 1) >> 1;
  const int64_t voxels = sx * sy * sz;


  uint64_t loc = voxels - 1;
  uint64_t oloc = 0;
  for (int64_t z = sz - 1; z >= 0; z--) {
    for (int64_t y = sy - 1; y >= 0; y--) {
      for (int64_t x = sx - 1; x >= 0; x--, loc--) {
        oloc = (x >> 1) + msx * ((y >> 1) + msy * (z >> 1));
        out_labels[loc] = (static_cast<OUT>(in_labels[loc] == 0) - 1) & renumber[out_labels[oloc]];
      }
    }
  }

  return out_labels;
}

// This is the original Wu et al decision tree but without
// any copy operations, only union find. We can decompose the problem
// into the z - 1 problem unified with the original 2D algorithm.
// If literally none of the Z - 1 are filled, we can use a faster version
// of this that uses copies.
template <typename OUT = uint32_t>
inline void unify2d_2x2x2(
    const int64_t loc, const uint8_t cur,
    const int64_t x, const int64_t y, 
    const int64_t msx, const int64_t msy, 
    const uint8_t* minor, const OUT* out_labels,
    DisjointSet<OUT> &equivalences  
  ) {

  if (y > 0 && cur == minor[loc - msx] && is_26_connected(cur, minor[loc-msx], 0, -1, 0)) {
    equivalences.unify(out_labels[loc], out_labels[loc - msx]);
  }
  else if (x > 0 && cur == minor[loc - 1] && is_26_connected(cur, minor[loc-1], -1, 0, 0)) {
    equivalences.unify(out_labels[loc], out_labels[loc - 1]); 

    if (x < msx - 1 && y > 0 && cur == minor[loc + 1 - msx] && is_26_connected(cur, minor[loc + 1 - msx], 1, -1, 0)) {
      equivalences.unify(out_labels[loc], out_labels[loc + 1 - msx]); 
    }
  }
  else if (x > 0 && y > 0 && cur == minor[loc - 1 - msx] && is_26_connected(cur, minor[loc-1-msx], -1, -1, 0)) {
    equivalences.unify(out_labels[loc], out_labels[loc - 1 - msx]); 

    if (x < msx - 1 && y > 0 && cur == minor[loc + 1 - msx] && is_26_connected(cur, minor[loc+1-msx], 1, -1, 0)) {
      equivalences.unify(out_labels[loc], out_labels[loc + 1 - msx]); 
    }
  }
  else if (x < msx - 1 && y > 0 && cur == minor[loc + 1 - msx] && is_26_connected(cur, minor[loc+1-msx], 1, -1, 0)) {
    equivalences.unify(out_labels[loc], out_labels[loc + 1 - msx]);
  }
}

template <typename OUT = uint32_t>
inline void unify2d_ac_2x2x2(
    const int64_t loc, const uint8_t cur,
    const int64_t x, const int64_t y, 
    const int64_t msx, const int64_t msy, 
    const uint8_t* minor, const OUT* out_labels,
    DisjointSet<OUT> &equivalences  
  ) {

  if (x > 0 && y > 0 && cur == minor[loc - 1 - msx] && is_26_connected(cur, minor[loc-1-msx], -1, -1, 0)) {
    equivalences.unify(out_labels[loc], out_labels[loc - 1 - msx]); 

    if (x < msx - 1 && y > 0 && cur == minor[loc + 1 - msx] && !(y > 1 && cur == minor[loc - msx - msx]) && is_26_connected(cur, minor[loc+1-msx], 1, -1, 0)) {
      equivalences.unify(out_labels[loc], out_labels[loc + 1 - msx]); 
    }
  }
  else if (x < msx - 1 && y > 0 && cur == minor[loc + 1 - msx] && is_26_connected(cur, minor[loc+1-msx], 1, -1, 0)) {
    equivalences.unify(out_labels[loc], out_labels[loc + 1 - msx]);
  }
}

template <typename OUT = uint32_t>
inline void unify2d_rt_2x2x2(
    const int64_t loc, const uint8_t cur,
    const int64_t x, const int64_t y, 
    const int64_t msx, const int64_t msy, 
    const uint8_t* minor, const OUT* out_labels,
    DisjointSet<OUT> &equivalences  
  ) {

  if (x < msx - 1 && y > 0 && cur == minor[loc + 1 - msx] && is_26_connected(cur, minor[loc+1-msx], 1, -1, 0)) {
    equivalences.unify(out_labels[loc], out_labels[loc + 1 - msx]);
  }
}

template <typename OUT = uint32_t>
inline void unify2d_lt_2x2x2(
    const int64_t loc, const uint8_t cur,
    const int64_t x, const int64_t y, 
    const int64_t msx, const int64_t msy, 
    const uint8_t* minor, const OUT* out_labels,
    DisjointSet<OUT> &equivalences  
  ) {

  if (x > 0 && cur == minor[loc - 1] && is_26_connected(cur, minor[loc - 1], -1, 0, 0)) {
    equivalences.unify(out_labels[loc], out_labels[loc - 1]);
  }
  else if (x > 0 && y > 0 && cur == minor[loc - 1 - msx] && is_26_connected(cur, minor[loc - 1 - msx], -1, -1, 0)) {
    equivalences.unify(out_labels[loc], out_labels[loc - 1 - msx]);
  }
}

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

  // const std::unique_ptr<uint32_t[]> runs(
  //   compute_foreground_index(in_labels, sx, sy, sz)
  // );

  const int64_t msx = (sx + 1) >> 1;
  const int64_t msy = (sy + 1) >> 1;
  const int64_t msz = (sz + 1) >> 1;
  const int64_t msxy = msx * msy;

  uint8_t* minor = create_2x2x2_minor_image(in_labels, sx, sy, sz);
  
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
  const int64_t A = -1 - msx - msxy;
  const int64_t B = -msx - msxy;
  const int64_t C = +1 - msx - msxy;
  const int64_t D = -1 - msxy;
  const int64_t E = -msxy;
  const int64_t F = +1 - msxy;
  const int64_t G = -1 + msx - msxy;
  const int64_t H = +msx - msxy;
  const int64_t I = +1 + msx - msxy;

  // Current Z
  const int64_t J = -1 - msx;
  const int64_t K = -msx;
  const int64_t L = +1 - msx; 
  const int64_t M = -1;
  // N = 0;

  OUT next_label = 0;
  int64_t loc = 0;

  // Raster Scan 1: Set temporary labels and 
  // record equivalences in a disjoint set.
  // int64_t row = 0;
  for (int64_t z = 0; z < msz; z++) {
    for (int64_t y = 0; y < msy; y++) { //, row++) {
      // const int64_t xstart = runs[row << 1];
      // const int64_t xend = runs[(row << 1) + 1];

      for (int64_t x = 0; x < msx; x++) {
        loc = x + msx * y + msxy * z;
        const uint8_t cur = minor[loc];

        if (cur == 0) {
          continue;
        }

        if (z > 0 && minor[loc + E] && is_26_connected(cur, minor[loc+E], 0, 0, -1)) {
          out_labels[loc] = out_labels[loc + E];
        }
        else if (y > 0 && minor[loc + K] && is_26_connected(cur, minor[loc+K], 0, -1, 0)) {
          out_labels[loc] = out_labels[loc + K];

          if (y < msy - 1 && z > 0 && minor[loc + H] && is_26_connected(cur, minor[loc+H], 0, 1, -1)) {
            equivalences.unify(out_labels[loc], out_labels[loc + H]);
          }
          else if (x > 0 && y < msy - 1 && z > 0 && minor[loc + G] && is_26_connected(cur, minor[loc+G], -1, 1, -1)) {
            equivalences.unify(out_labels[loc], out_labels[loc + G]);
            
            if (x < msx - 1 && y < msy - 1 && z > 0 && minor[loc + I] && is_26_connected(cur, minor[loc+I], 1, 1, -1)) {
              equivalences.unify(out_labels[loc], out_labels[loc + I]);
            }
          }
          else if (x < msx - 1 && y < msy - 1 && z > 0 && minor[loc + I] && is_26_connected(cur, minor[loc+I], 1, 1, -1)) {
            equivalences.unify(out_labels[loc], out_labels[loc + I]);
          }
        }
        else if (z > 0 && y > 0 && minor[loc + B] && is_26_connected(cur, minor[loc+B], 0, -1, -1)) {
          out_labels[loc] = out_labels[loc + B];

          if (y < msy - 1 && z > 0 && minor[loc + H] && is_26_connected(cur, minor[loc+H], 0, 1, -1)) {
            equivalences.unify(out_labels[loc], out_labels[loc + H]);
          }
          else if (x > 0 && y < msy - 1 && z > 0 && minor[loc + G] && is_26_connected(cur, minor[loc+G], -1, 1, -1)) {
            equivalences.unify(out_labels[loc], out_labels[loc + G]);
            
            if (x < msx - 1 && y < msy - 1 && z > 0 && minor[loc + I] && is_26_connected(cur, minor[loc+I], 1, 1, -1)) {
              equivalences.unify(out_labels[loc], out_labels[loc + I]);
            }
          }
          else if (x < msx - 1 && y < msy - 1 && z > 0 && minor[loc + I] && is_26_connected(cur, minor[loc+I], 1, 1, -1)) {
            equivalences.unify(out_labels[loc], out_labels[loc + I]);
          }
        }
        else if (x > 0 && minor[loc + M] && is_26_connected(cur, minor[loc+M], -1, 0, 0)) {
          out_labels[loc] = out_labels[loc + M];

          if (x < msx - 1 && z > 0 && minor[loc + F] && is_26_connected(cur, minor[loc+F], 1, 0, -1)) {
            equivalences.unify(out_labels[loc], out_labels[loc + F]);
          }
          else if (x < msx - 1 && y > 0 && minor[loc + L] && is_26_connected(cur, minor[loc+L], 1, -1, 0)) {
            equivalences.unify(out_labels[loc], out_labels[loc + L]);

            if (x < msx - 1 && y < msy - 1 && z > 0 && minor[loc + I] && is_26_connected(cur, minor[loc+I], 1, 1, -1)) {
              equivalences.unify(out_labels[loc], out_labels[loc + I]);
            }
          }
          else if (x < msx - 1 && y > 0 && z > 0 && minor[loc + C] && is_26_connected(cur, minor[loc+C], 1, -1, -1)) {
            equivalences.unify(out_labels[loc], out_labels[loc + C]);

            if (x < msx - 1 && y < msy - 1 && z > 0 && minor[loc + I] && is_26_connected(cur, minor[loc+I], 1, 1, -1)) {
              equivalences.unify(out_labels[loc], out_labels[loc + I]);
            }
          }
          else if (x < msx - 1 && y < msy - 1 && z > 0 && minor[loc + I] && is_26_connected(cur, minor[loc+I], 1, 1, -1)) {
            equivalences.unify(out_labels[loc], out_labels[loc + I]);
          }
        }
        else if (x > 0 && z > 0 && minor[loc + D] && is_26_connected(cur, minor[loc+D], -1, 0, -1)) {
          out_labels[loc] = out_labels[loc + D];

          if (x < msx - 1 && z > 0 && minor[loc + F] && is_26_connected(cur, minor[loc+F], 1, 0, -1)) {
            equivalences.unify(out_labels[loc], out_labels[loc + F]);
          }
          else if (x < msx - 1 && y > 0 && minor[loc + L] && is_26_connected(cur, minor[loc+L], 1, -1, 0)) {
            equivalences.unify(out_labels[loc], out_labels[loc + L]);

            if (x < msx - 1 && y < msy - 1 && z > 0 && minor[loc + I] && is_26_connected(cur, minor[loc+I], 1, 1, -1)) {
              equivalences.unify(out_labels[loc], out_labels[loc + I]);
            }
          }
          else if (x < msx - 1 && y > 0 && z > 0 && minor[loc + C] && is_26_connected(cur, minor[loc+C], 1, -1, -1)) {
            equivalences.unify(out_labels[loc], out_labels[loc + C]);

            if (x < msx - 1 && y < msy - 1 && z > 0 && minor[loc + I] && is_26_connected(cur, minor[loc+I], 1, 1, -1)) {
              equivalences.unify(out_labels[loc], out_labels[loc + I]);
            }
          }
          else if (x < msx - 1 && y < msy - 1 && z > 0 && minor[loc + I] && is_26_connected(cur, minor[loc+I], 1, 1, -1)) {
            equivalences.unify(out_labels[loc], out_labels[loc + I]);
          }
        }
        else if (y < msy - 1 && z > 0 && minor[loc + H] && is_26_connected(cur, minor[loc+H], 0, 1, -1)) {
          out_labels[loc] = out_labels[loc + H];
          unify2d_ac_2x2x2(loc, cur, x, y, msx, msy, minor, out_labels, equivalences);

          if (x > 0 && y > 0 && z > 0 && minor[loc + A] && is_26_connected(cur, minor[loc+A], -1, -1, -1)) {
            equivalences.unify(out_labels[loc], out_labels[loc + A]);
          }
          if (x < msx - 1 && y > 0 && z > 0 && minor[loc + C] && is_26_connected(cur, minor[loc+C], 1, -1, -1)) {
            equivalences.unify(out_labels[loc], out_labels[loc + C]);
          }
        }
        else if (x < msx - 1 && z > 0 && minor[loc + F] && is_26_connected(cur, minor[loc+F], 1, 0, -1)) {
          out_labels[loc] = out_labels[loc + F];
          unify2d_lt_2x2x2(loc, cur, x, y, msx, msy, minor, out_labels, equivalences);

          if (x > 0 && y > 0 && z > 0 && minor[loc + A] && is_26_connected(cur, minor[loc+A], -1, -1, -1)) {
            equivalences.unify(out_labels[loc], out_labels[loc + A]);
          }
          if (x > 0 && y < msy - 1 && z > 0 && minor[loc + G] && is_26_connected(cur, minor[loc+G], -1, 1, -1)) {
            equivalences.unify(out_labels[loc], out_labels[loc + G]);
          }
        }
        else if (x > 0 && y > 0 && z > 0 && minor[loc + A] && is_26_connected(cur, minor[loc+A], -1, -1, -1)) {
          out_labels[loc] = out_labels[loc + A];
          unify2d_rt_2x2x2(loc, cur, x, y, msx, msy, minor, out_labels, equivalences);

          if (x < msx - 1 && y > 0 && z > 0 && minor[loc + C] && is_26_connected(cur, minor[loc+C], 1, -1, -1)) {
            equivalences.unify(out_labels[loc], out_labels[loc + C]);
          }
          if (x > 0 && y < msy - 1 && z > 0 && minor[loc + G] && is_26_connected(cur, minor[loc+G], -1, 1, -1)) {
            equivalences.unify(out_labels[loc], out_labels[loc + G]);
          }      
          if (x < msx - 1 && y < msy - 1 && z > 0 && minor[loc + I] && is_26_connected(cur, minor[loc+I], 1, 1, -1)) {
            equivalences.unify(out_labels[loc], out_labels[loc + I]);
          }
        }
        else if (x < msx - 1 && y > 0 && z > 0 && minor[loc + C] && is_26_connected(cur, minor[loc+C], 1, -1, -1)) {
          out_labels[loc] = out_labels[loc + C];
          unify2d_lt_2x2x2(loc, cur, x, y, msx, msy, minor, out_labels, equivalences);

          if (x > 0 && y < msy - 1 && z > 0 && minor[loc + G] && is_26_connected(cur, minor[loc+G], -1, 1, -1)) {
            equivalences.unify(out_labels[loc], out_labels[loc + G]);
          }
          if (x < msx - 1 && y < msy - 1 && z > 0 && minor[loc + I] && is_26_connected(cur, minor[loc+I], 1, 1, -1)) {
            equivalences.unify(out_labels[loc], out_labels[loc + I]);
          }
        }
        else if (x > 0 && y < msy - 1 && z > 0 && minor[loc + G] && is_26_connected(cur, minor[loc+G], -1, 1, -1)) {
          out_labels[loc] = out_labels[loc + G];
          unify2d_ac_2x2x2(loc, cur, x, y, msx, msy, minor, out_labels, equivalences);

          if (x < msx - 1 && y < msy - 1 && z > 0 && minor[loc + I] && is_26_connected(cur, minor[loc+I], 1, 1, -1)) {
            equivalences.unify(out_labels[loc], out_labels[loc + I]);
          }
        }
        else if (x < msx - 1 && y < msy - 1 && z > 0 && minor[loc + I] && is_26_connected(cur, minor[loc+I], 1, 1, -1)) {
          out_labels[loc] = out_labels[loc + I];
          unify2d_ac_2x2x2(loc, cur, x, y, msx, msy, minor, out_labels, equivalences);
        }
        // It's the original 2D problem now
        else if (y > 0 && minor[loc + K] && is_26_connected(cur, minor[loc+K], 0, -1, 0)) {
          out_labels[loc] = out_labels[loc + K];
        }
        else if (x > 0 && minor[loc + M] && is_26_connected(cur, minor[loc+M], -1, 0, 0)) {
          out_labels[loc] = out_labels[loc + M];

          if (x < msx - 1 && y > 0 && minor[loc + L] && is_26_connected(cur, minor[loc+L], 1, -1, 0)) {
            equivalences.unify(out_labels[loc], out_labels[loc + L]); 
          }
        }
        else if (x > 0 && y > 0 && minor[loc + J] && is_26_connected(cur, minor[loc+J], 1, 1, 0)) {
          out_labels[loc] = out_labels[loc + J];

          if (x < msx - 1 && y > 0 && minor[loc + L] && is_26_connected(cur, minor[loc+L], 1, -1, 0)) {
            equivalences.unify(out_labels[loc], out_labels[loc + L]); 
          }
        }
        else if (x < msx - 1 && y > 0 && minor[loc + L] && is_26_connected(cur, minor[loc+L], 1, -1, 0)) {
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
  
  delete[] minor;

  return relabel_2x2x2<T,OUT>(in_labels, out_labels, sx, sy, sz, next_label, equivalences, N);
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
  size_t max_labels = voxels;

  if (connectivity == 26) {
    max_labels = (voxels >> 3) + 1;
  } 
  else if (connectivity == 8 || connectivity == 18) {
    max_labels = (voxels >> 2) + 1;
  }
  else {
    max_labels = (voxels >> 1) + 1;
  }

  return connected_components3d_binary<T, OUT>(
    in_labels, sx, sy, sz, 
    max_labels, connectivity, 
    NULL, N, periodic_boundary
  );
}

};

#endif