#ifndef CC3D_GRAPHS_HPP
#define CC3D_GRAPHS_HPP 

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace cc3d {

// The voxel connectivity graph is specified as a directed graph
// written as a bitfield for each voxel. 
// The ordering of the field is corners, edges, faces, where the faces
// are located at the least significant bits.

// i.e. for a 4-connected pixel -y,+y,-x,+x
//      for an 8-connected pixel: -x-y,x-y,-xy,xy,-y,y,-x,x
// Note that for 2D connectivities, this bitfield requires only 8-bits.

//      8      7      6      5      4      3      2      1
// ------ ------ ------ ------ ------ ------ ------ ------
//   -x-y    x-y    -xy     xy     -x     +y     -x     +x

template <typename T, typename OUT = uint8_t>
OUT* extract_voxel_connectivity_graph_2d(
		T* labels, 
		const int64_t sx, const int64_t sy,
		OUT* graph = NULL
	) {

	const int64_t voxels = sx * sy;

	if (graph == NULL) {
		graph = new OUT[voxels];
	}
	for (int64_t i = 0; i < voxels; i++) {
		graph[i] = 0b11111111;
	}

	T cur = 0;

	for (int64_t y = 0; y < sy; y++) {
		for (int64_t x = 0; x < sx; x++) {
			int64_t loc = x + sx * y;
			cur = labels[loc];

			if (x > 0 && cur != labels[loc - 1]) {
				graph[loc  ] &= 0b11111101;
				graph[loc-1] &= 0b11111110;
			}
			if (y > 0 && cur != labels[loc - sx]) {
				graph[loc   ] &= 0b11110111;
				graph[loc-sx] &= 0b11111011;
			}
			if (x > 0 && y > 0 && cur != labels[loc - sx - 1]) {
				graph[loc     ] &= 0b01111111;
				graph[loc-sx-1] &= 0b11101111;					
			}
			if (x < sx - 1 && y > 0 && cur != labels[loc - sx + 1]) {
				graph[loc     ] &= 0b10111111;
				graph[loc-sx+1] &= 0b11011111;					
			}
		}
	}

	return graph;
} 

// For a 3D 26 connectivity, we need to use a 32-bit integer.
// 6x unused, 8 corners, 12 edges, 6 faces

//     32     31     30     29     28     27     26     25     24     23     
// ------ ------ ------ ------ ------ ------ ------ ------ ------ ------
// unused unused unused unused unused unused -x-y-z  x-y-z -x+y-z +x+y-z
//
//     22     21     20     19     18     17     16     15     14     13
// ------ ------ ------ ------ ------ ------ ------ ------ ------ ------
// -x-y+z +x-y+z -x+y+z    xyz   -y-z    y-z   -x-z    x-z    -yz     yz
//
//     12     11     10      9      8      7      6      5      4      3
// ------ ------ ------ ------ ------ ------ ------ ------ ------ ------
//    -xz     xz   -x-y    x-y    -xy     xy     -z     +z     -y     +y  
//      2      1
// ------ ------
//     -x     +x
//
// 6 connected uses bits 1-6, 18 connected uses 1-18, 26 connected uses 1-26
//  

template <typename T, typename OUT = uint32_t>
OUT* extract_voxel_connectivity_graph_3d(
		T* labels, 
		const int64_t sx, const int64_t sy, const int64_t sz,
		OUT* graph = NULL
	) {

	const int64_t sxy = sx * sy;
	const int64_t voxels = sx * sy * sz;

	const OUT ALL = static_cast<OUT>(0x3ffffff); // 26 free directions

	if (graph == NULL) {
		graph = new OUT[voxels];
	}
	for (int64_t i = 0; i < voxels; i++) {
		graph[i] = ALL;
	}

	T cur = 0;

	for (int64_t z = 0; z < sz; z++) {
		for (int64_t y = 0; y < sy; y++) {
			for (int64_t x = 0; x < sx; x++) {
				int64_t loc = x + sx * y + sxy * z;
				cur = labels[loc];

				// 0b11111111111111111111111111111111;
				if (x > 0 && cur != labels[loc - 1]) {
					graph[loc  ] &= 0b11111111111111111111111111111101; // 2
					graph[loc-1] &= 0b11111111111111111111111111111110; // 1
				}
				if (y > 0 && cur != labels[loc - sx]) {
					graph[loc   ] &= 0b11111111111111111111111111110111; // 4
					graph[loc-sx] &= 0b11111111111111111111111111111011; // 3
				}
				if (z > 0 && cur != labels[loc - sxy]) {
					graph[loc    ] &= 0b11111111111111111111111111011111; // 6
					graph[loc-sxy] &= 0b11111111111111111111111111101111; // 5
				}
				if (x > 0 && y > 0 && cur != labels[loc - sx - 1]) {
					graph[loc     ] &= 0b11111111111111111111110111111111; // 10
					graph[loc-sx-1] &= 0b11111111111111111111111110111111; // 7
				}
				if (x < sx - 1 && y > 0 && cur != labels[loc - sx + 1]) {
					graph[loc     ] &= 0b11111111111111111111111011111111; // 9
 					graph[loc-sx+1] &= 0b11111111111111111111111101111111; // 8
				}
				if (x > 0 && y > 0 && z > 0 && cur != labels[loc - sxy - sx - 1]) {
					graph[loc         ] &= 0b11111101111111111111111111111111; // 26
					graph[loc-sxy-sx-1] &= 0b11111111111110111111111111111111; // 19
				}
				if (y > 0 && z > 0 && cur != labels[loc - sxy - sx]) {
					graph[loc       ] &= 0b11111111111111011111111111111111; // 18
					graph[loc-sxy-sx] &= 0b11111111111111111110111111111111; // 13
				}
				if (x < sx - 1 && y > 0 && z > 0 && cur != labels[loc - sxy - sx + 1]) {
					graph[loc         ] &= 0b11111110111111111111111111111111; // 25
					graph[loc-sxy-sx+1] &= 0b11111111111101111111111111111111; // 20
				}
				if (x > 0 && z > 0 && cur != labels[loc - sxy - 1]) {
					graph[loc      ] &= 0b11111111111111110111111111111111; // 16
					graph[loc-sxy-1] &= 0b11111111111111111111101111111111; // 11
				}
				if (x < sx - 1 && z > 0 && cur != labels[loc - sxy + 1]) {
					graph[loc      ] &= 0b11111111111111111011111111111111; // 15
					graph[loc-sxy+1] &= 0b11111111111111111111011111111111; // 12
				}
				if (x > 0 && y < sy - 1 && z > 0 && cur != labels[loc - sxy + sx - 1]) {
					graph[loc         ] &= 0b11111111011111111111111111111111; // 24
					graph[loc-sxy+sx-1] &= 0b11111111111011111111111111111111; // 21
				}
				if (y < sy - 1 && z > 0 && cur != labels[loc - sxy + sx]) {
					graph[loc       ] &= 0b11111111111111101111111111111111; // 17
					graph[loc-sxy+sx] &= 0b11111111111111111101111111111111; // 14
				}
				if (x < sx - 1 && y < sy - 1 && z > 0 && cur != labels[loc - sxy + sx + 1]) {
					graph[loc         ] &= 0b11111111101111111111111111111111; // 23
					graph[loc-sxy+sx+1] &= 0b11111111110111111111111111111111; // 22
				}
			}
		}
	}

	return graph;
} 

template <typename T>
inline T* and_mask(T mask, T* graph, int64_t sx, int64_t sy, int64_t sz = 1) {
	for (int64_t i = 0; i < sx*sy*sz; i++) {
		graph[i] &= mask;
	}
	return graph;
}

template <typename T, typename OUT = uint32_t>
OUT* extract_voxel_connectivity_graph(
    T* in_labels, 
    const int64_t sx, const int64_t sy, const int64_t sz,
    const int64_t connectivity, OUT *graph = NULL
  ) {


  if (connectivity == 26) {
    return extract_voxel_connectivity_graph_3d<T, OUT>(
      in_labels, sx, sy, sz, graph
    );
  }
  else if (connectivity == 18) {
    graph = extract_voxel_connectivity_graph_3d<T, OUT>(
      in_labels, sx, sy, sz, graph
    );
  	return and_mask<OUT>(static_cast<OUT>(0x3ffff), graph, sx, sy, sz);
  }
  else if (connectivity == 6) {
    graph = extract_voxel_connectivity_graph_3d<T, OUT>(
      in_labels, sx, sy, sz, graph
    );
   return and_mask<OUT>(0b00111111, graph, sx, sy, sz);
  }
  else if (connectivity == 8) {
    if (sz != 1) {
      throw std::runtime_error("sz must be 1 for 2D connectivities.");
    }
    return extract_voxel_connectivity_graph_2d<T, OUT>(
      in_labels, sx, sy, graph
    );
  }
  else if (connectivity == 4) {
    if (sz != 1) {
      throw std::runtime_error("sz must be 1 for 2D connectivities.");
    }
    graph = extract_voxel_connectivity_graph_2d<T, OUT>(
      in_labels, sx, sy, graph
    );
    return and_mask<OUT>(0b00001111, graph, sx, sy);
  }
  else {
    throw std::runtime_error("Only 4 and 8 2D and 6, 18, and 26 3D connectivities are supported.");
  }
}

inline void compute_neighborhood(
	int *neighborhood, 
	const int x, const int y, const int z,
	const uint64_t sx, const uint64_t sy, const uint64_t sz,
	const int connectivity = 26
) {

	const int sxy = sx * sy;

	const int plus_x = (x < (static_cast<int>(sx) - 1)); // +x
	const int minus_x = -1 * (x > 0); // -x
	const int plus_y = static_cast<int>(sx) * (y < static_cast<int>(sy) - 1); // +y
	const int minus_y = -static_cast<int>(sx) * (y > 0); // -y
	const int minus_z = -sxy * static_cast<int>(z > 0); // -z

	// 6-hood
	neighborhood[0] = minus_x;
	neighborhood[1] = minus_y;
	neighborhood[2] = minus_z;
	
	// 18-hood

	// xy diagonals
	neighborhood[3] = (connectivity > 6) * (minus_x + minus_y) * (minus_x && minus_y); // up-left
	neighborhood[4] = (connectivity > 6) * (plus_x + minus_y) * (plus_x && minus_y); // up-right

	// yz diagonals
	neighborhood[5] = (connectivity > 6) * (minus_x + minus_z) * (minus_x && minus_z); // down-left
	neighborhood[6] = (connectivity > 6) * (plus_x + minus_z) * (plus_x && minus_z); // down-right

	// xz diagonals
	neighborhood[7] = (connectivity > 6) * (minus_y + minus_z) * (minus_y && minus_z); // down-left
	neighborhood[8] = (connectivity > 6) * (plus_y + minus_z) * (plus_y && minus_z); // down-right

	// 26-hood

	// Now the eight corners of the cube
	neighborhood[9] = (connectivity > 18) * (minus_x + minus_y + minus_z) * (minus_y && minus_z);
	neighborhood[10] = (connectivity > 18) * (plus_x + minus_y + minus_z) * (minus_y && minus_z);
	neighborhood[11] = (connectivity > 18) * (minus_x + plus_y + minus_z) * (plus_y && minus_z);
	neighborhood[12] = (connectivity > 18) * (plus_x + plus_y + minus_z) * (plus_y && minus_z);
}

struct pair_hash {
	inline std::size_t operator()(const std::pair<int,int> & v) const {
		return v.first * 31 + v.second; // arbitrary hash fn
	}
};

template <typename T>
std::vector<T> extract_region_graph(
		T* labels, 
		const int64_t sx, const int64_t sy, const int64_t sz,
		const int64_t connectivity=26
	) {

	if (connectivity != 6 && connectivity != 18 && connectivity != 26) {
		throw std::runtime_error("Only 6, 18, and 26 connectivities are supported.");
	}

	const int64_t sxy = sx * sy;

	int neighborhood[13];

	T cur = 0;
	T label = 0;
	T last_label = 0;

	std::unordered_set<std::pair<T,T>, pair_hash> edges;

	for (int64_t z = 0; z < sz; z++) {
		for (int64_t y = 0; y < sy; y++) {
			for (int64_t x = 0; x < sx; x++) {
				int64_t loc = x + sx * y + sxy * z;
				cur = labels[loc];

				if (cur == 0) {
					continue;
				}

				compute_neighborhood(neighborhood, x, y, z, sx, sy, sz, connectivity);
				
				last_label = cur;

				for (int i = 0; i < connectivity / 2; i++) {
					int64_t neighboridx = loc + neighborhood[i];
					label = labels[neighboridx];

					if (label == 0 || label == last_label) {
						continue;
					}
					else if (label != cur) {
						if (cur > label) {
							edges.emplace(std::pair<T,T>(label, cur));
						}
						else {
							edges.emplace(std::pair<T,T>(cur, label)); 
						}
						last_label = label;
					}
				}
			}
		}
	}

	std::vector<T> output;
	output.reserve(edges.size() * 2);

	for (std::pair<T,T> edge : edges) {
		output.push_back(edge.first);
		output.push_back(edge.second);
	}

	return output;
}

};

#endif