#ifndef CC3D_GRAPHS_HPP
#define CC3D_GRAPHS_HPP 

#include <memory>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <map>
#include <unordered_map>
#include <vector>

#include "cc3d.hpp"

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
//   -x-y    x-y    -xy     xy     -y     +y     -x     +x

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
	inline std::size_t operator()(const std::pair<uint64_t,uint64_t> & v) const {
		return v.first * 31 + v.second; // arbitrary hash fn
	}
};

template <typename T>
const std::unordered_map<std::pair<T,T>, float, pair_hash> 
extract_region_graph(
	T* labels, 
	const int64_t sx, const int64_t sy, const int64_t sz,
	const float wx=1, const float wy=1, const float wz=1,
	const int64_t connectivity=26,
	const bool surface_area=true
) {

	if (connectivity != 6 && connectivity != 18 && connectivity != 26) {
		throw std::runtime_error("Only 6, 18, and 26 connectivities are supported.");
	}

	const int64_t sxy = sx * sy;

	int neighborhood[13];
	float areas[13]; // all zero except faces

	if (surface_area) {
		for (int i = 3; i < 13; i++) {
			areas[i] = 0;
		}
		areas[0] = wy * wz; // x axis
		areas[1] = wx * wz; // y axis
		areas[2] = wx * wy; // z axis
	}
	else { // voxel counts
		for (int i = 0; i < 13; i++) {
			areas[i] = 1;
		}
	}

	T cur = 0;
	T label = 0;

	std::unordered_map<std::pair<T,T>, float, pair_hash> edges;

	for (int64_t z = 0; z < sz; z++) {
		for (int64_t y = 0; y < sy; y++) {
			for (int64_t x = 0; x < sx; x++) {
				int64_t loc = x + sx * y + sxy * z;
				cur = labels[loc];

				if (cur == 0) {
					continue;
				}

				compute_neighborhood(neighborhood, x, y, z, sx, sy, sz, connectivity);

				for (int i = 0; i < connectivity / 2; i++) {
					int64_t neighboridx = loc + neighborhood[i];
					label = labels[neighboridx];

					if (label == 0) {
						continue;
					}
					else if (cur > label) {
						edges[std::pair<T,T>(label, cur)] += areas[i];
					}
					else if (cur < label) {
						edges[std::pair<T,T>(cur, label)] += areas[i];
					}
				}
			}
		}
	}

	return edges;
}

template <typename T>
std::map<T, std::vector<std::pair<size_t, size_t>>> 
extract_runs(T* labels, const size_t voxels) {
	std::map<T, std::vector<std::pair<size_t, size_t>>> runs;
	if (voxels == 0) {
		return runs;
	}

	size_t cur = labels[0];
	size_t start = 0; // of run

	if (voxels == 1) {
		runs[cur].push_back(std::pair<size_t,size_t>(0,1));
		return runs;
	}

	size_t loc = 1;
	for (loc = 1; loc < voxels; loc++) {
		if (labels[loc] != cur) {
			runs[cur].push_back(std::pair<size_t,size_t>(start,loc));
			cur = labels[loc];
			start = loc;
		}
	}

	if (loc > start) {
		runs[cur].push_back(std::pair<size_t,size_t>(start,voxels));
	}

	return runs;
}

template <typename T>
void set_run_voxels(
	const T val,
	const std::vector<std::pair<size_t, size_t>> runs,
	T* labels, const size_t voxels
) {
	for (std::pair<size_t, size_t> run : runs) {
		if (
			run.first < 0 || run.second > voxels 
			|| run.second < 0 || run.second > voxels
			|| run.first >= run.second
		) {
			throw std::runtime_error("Invalid run.");
		}

		for (size_t loc = run.first; loc < run.second; loc++) {
			labels[loc] = val;
		}
	}
}

template <typename OUT>
OUT* simplified_relabel(
    OUT* out_labels, const int64_t voxels,
    const int64_t num_labels, DisjointSet<OUT> &equivalences,
    size_t &N = _dummy_N
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

  // Raster Scan 2: Write final labels based on equivalences
  N = next_label - 1;
  for (int64_t loc = 0; loc < voxels; loc++) {
    out_labels[loc] = renumber[out_labels[loc]];
  }

  return out_labels;
}

template <typename OUT>
OUT* color_connectivity_graph_26(
	const uint8_t* vcg, // voxel connectivity graph
	const int64_t sx, const int64_t sy, const int64_t sz,
	OUT* out_labels = NULL,
	size_t &N = _dummy_N
) {
	throw new std::runtime_error("26-connectivity requires a 32-bit voxel graph for color_connectivity_graph_26.");
}

template <typename T>
uint64_t estimate_vcg_provisional_label_count(
  T* vcg, const int64_t sx, const int64_t voxels
) {
  uint64_t count = 0; // number of transitions between labels

  for (int64_t row = 0, loc = 0; loc < voxels; loc += sx, row++) {
    count++;
    for (int64_t x = 1; x < sx; x++) {
      count += static_cast<uint64_t>((vcg[loc + x] & 0b10) == 0);
    }
  }

  return count;
}

template <typename OUT>
OUT* color_connectivity_graph_26(
	const uint32_t* vcg, // voxel connectivity graph
	const int64_t sx, const int64_t sy, const int64_t sz,
	OUT* out_labels = NULL,
	size_t &N = _dummy_N
) {
	const int64_t sxy = sx * sy;
	const int64_t voxels = sx * sy * sz;

	uint64_t max_labels = static_cast<uint64_t>(voxels) + 1; // + 1L for an array with no zeros
	max_labels = std::min(max_labels, static_cast<uint64_t>(std::numeric_limits<OUT>::max()));
	max_labels = std::min(max_labels, estimate_vcg_provisional_label_count(vcg, sx, voxels) + 1);

	if (out_labels == NULL) {
		out_labels = new OUT[voxels]();
	}

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
	const int64_t A_mask = 0b10000000000000000000000000;
	const int64_t B = -sx - sxy;
	const int64_t B_mask = 0b100000000000000000;
	const int64_t C = +1 - sx - sxy;
	const int64_t C_mask = 0b1000000000000000000000000;
	const int64_t D = -1 - sxy;
	const int64_t D_mask = 0b1000000000000000;
	const int64_t E = -sxy;
	const int64_t E_mask = 0b100000;
	const int64_t F = +1 - sxy;
	const int64_t F_mask = 0b100000000000000;
	const int64_t G = -1 + sx - sxy;
	const int64_t G_mask = 0b100000000000000000000000;
	const int64_t H = +sx - sxy;
	const int64_t H_mask = 0b10000000000000000;
	const int64_t I = +1 + sx - sxy;
	const int64_t I_mask = 0b10000000000000000000000;

	// Current Z
	const int64_t J = -1 - sx;
	const int64_t J_mask = 0b1000000000;
	const int64_t K = -sx;
	const int64_t K_mask = 0b1000;
	const int64_t L = +1 - sx;
	const int64_t L_mask = 0b100000000; 
	const int64_t M = -1;
	const int64_t M_mask = 0b10;
	// N = 0;

	OUT new_label = 0;

	for (int64_t z = 0; z < sz; z++) {
		new_label++;
		equivalences.add(new_label);

		for (int64_t x = 0; x < sx; x++) {
			if (x > 0 && (vcg[x + sxy * z] & 0b0010) == 0) {
				new_label++;
				equivalences.add(new_label);
			}
			out_labels[x + sxy * z] = new_label;
		}

		for (int64_t y = 1; y < sy; y++) {
			for (int64_t x = 0; x < sx; x++) {
				int64_t loc = x + sx * y + sxy * z;

				bool stolen = false;

				if (vcg[loc] & K_mask) {
					out_labels[loc] = out_labels[loc+K];
					stolen = true;
				}
				if (x > 0 && (vcg[loc] & J_mask)) {
					if (!stolen) {
						out_labels[loc] = out_labels[loc+J];
						stolen = true;	
					}
					else {
						equivalences.unify(out_labels[loc], out_labels[loc+J]);
					}
				}
				if (x > 0 && (vcg[loc] & M_mask)) {
					if (!stolen) {
						out_labels[loc] = out_labels[loc+M];
						stolen = true;	
					}
					else {
						equivalences.unify(out_labels[loc], out_labels[loc+M]);
					}
				}
				if (x < sx - 1 && (vcg[loc] & L_mask)) {
					if (!stolen) {
						out_labels[loc] = out_labels[loc+L];
						stolen = true;	
					}
					else {
						equivalences.unify(out_labels[loc], out_labels[loc+L]);
					}
				}

				if (!stolen) {
					new_label++;
					out_labels[loc] = new_label;
					equivalences.add(out_labels[loc]);
				}
			}
		}
	}

	for (int64_t z = 1; z < sz; z++) {
		for (int64_t y = 0; y < sy; y++) {
			for (int64_t x = 0; x < sx; x++) {  
				int64_t loc = x + sx * y + sxy * z;

				if (x > 0 && y > 0 && (vcg[loc] & A_mask)) {
					equivalences.unify(out_labels[loc], out_labels[loc+A]);
				}
				if (y > 0 && (vcg[loc] & B_mask)) {
					equivalences.unify(out_labels[loc], out_labels[loc+B]);
				}
				if (x < sx - 1 && y > 0 && (vcg[loc] & C_mask)) {
					equivalences.unify(out_labels[loc], out_labels[loc+C]);
				}
				if (x > 0 && (vcg[loc] & D_mask)) {
					equivalences.unify(out_labels[loc], out_labels[loc+D]);
				}
				if (vcg[loc] & E_mask) {
					equivalences.unify(out_labels[loc], out_labels[loc+E]);
				}
				if (x < sx - 1 && (vcg[loc] & F_mask)) {
					equivalences.unify(out_labels[loc], out_labels[loc+F]);
				}
				if (x > 0 && y < sy - 1 && (vcg[loc] & G_mask)) {
					equivalences.unify(out_labels[loc], out_labels[loc+G]);
				}
				if (y < sy - 1 && (vcg[loc] & H_mask)) {
					equivalences.unify(out_labels[loc], out_labels[loc+H]);
				}
				if (x < sx - 1 && y < sy - 1 && (vcg[loc] & I_mask)) {
					equivalences.unify(out_labels[loc], out_labels[loc+I]);
				}
			}
		}
	}

	return simplified_relabel<OUT>(out_labels, voxels, new_label, equivalences, N);
}

template <typename OUT>
OUT* color_connectivity_graph_6(
	const uint8_t* vcg, // voxel connectivity graph
	const int64_t sx, const int64_t sy, const int64_t sz,
	OUT* out_labels = NULL,
	size_t &N = _dummy_N
) {
	throw new std::runtime_error("6-connectivity requires a 32-bit voxel graph for color_connectivity_graph_6.");
}

template <typename OUT>
OUT* color_connectivity_graph_6(
	const uint32_t* vcg, // voxel connectivity graph
	const int64_t sx, const int64_t sy, const int64_t sz,
	OUT* out_labels = NULL,
	size_t &N = _dummy_N
) {
	const int64_t sxy = sx * sy;
	const int64_t voxels = sx * sy * sz;

	uint64_t max_labels = static_cast<uint64_t>(voxels) + 1; // + 1L for an array with no zeros
	max_labels = std::min(max_labels, static_cast<uint64_t>(std::numeric_limits<OUT>::max()));
	max_labels = std::min(max_labels, estimate_vcg_provisional_label_count(vcg, sx, voxels) + 1);

	if (out_labels == NULL) {
		out_labels = new OUT[voxels]();
	}

	DisjointSet<OUT> equivalences(max_labels);

	const int64_t B = -1;
	const int64_t B_mask = 0b10;
	const int64_t C = -sx;
	const int64_t C_mask = 0b1000;


	OUT new_label = 0;
	for (int64_t z = 0; z < sz; z++) {
		new_label++;
		equivalences.add(new_label);

		for (int64_t x = 0; x < sx; x++) {
			if (x > 0 && (vcg[x + sxy * z] & B_mask) == 0) {
				new_label++;
				equivalences.add(new_label);
			}
			out_labels[x + sxy * z] = new_label;
		}

		for (int64_t y = 1; y < sy; y++) {
			for (int64_t x = 0; x < sx; x++) {
				int64_t loc = x + sx * y + sxy * z;

				bool stolen = false;

				if (vcg[loc] & C_mask) {
					out_labels[loc] = out_labels[loc+C];
					stolen = true;
				}
				if (x > 0 && (vcg[loc] & B_mask)) {
					if (!stolen) {
						out_labels[loc] = out_labels[loc+B];
						stolen = true;	
					}
					else {
						equivalences.unify(out_labels[loc], out_labels[loc+B]);
					}
				}

				if (!stolen) {
					new_label++;
					out_labels[loc] = new_label;
					equivalences.add(out_labels[loc]);
				}
			}
		}
	}

	for (int64_t z = 1; z < sz; z++) {
		for (int64_t y = 0; y < sy; y++) {
			for (int64_t x = 0; x < sx; x++) {  
				int64_t loc = x + sx * y + sxy * z;

				if (vcg[loc] & 0b100000) {
					equivalences.unify(out_labels[loc], out_labels[loc-sxy]);
				}
			}
		}
	}

	return simplified_relabel<OUT>(out_labels, voxels, new_label, equivalences, N);
}

// 8 and 32 bit inputs have different encodings so 
// need to write this two ways
template <typename OUT>
OUT* color_connectivity_graph_8(
  const uint8_t* vcg, // voxel connectivity graph
  const int64_t sx, const int64_t sy,
  OUT* out_labels = NULL,
  size_t &N = _dummy_N
) {
	const int64_t sxy = sx * sy;

	uint64_t max_labels = static_cast<uint64_t>(sxy) + 1; // + 1L for an array with no zeros
	max_labels = std::min(max_labels, static_cast<uint64_t>(std::numeric_limits<OUT>::max()));

	if (out_labels == NULL) {
		out_labels = new OUT[sxy]();
	}

	DisjointSet<OUT> equivalences(max_labels);

	OUT new_label = 1;
	equivalences.add(new_label);

	for (int64_t x = 0; x < sx; x++) {
		if (x > 0 && (vcg[x] & 0b0010) == 0) {
			new_label++;
			equivalences.add(new_label);
		}
		out_labels[x] = new_label;
	}

	/*
	Layout of mask. We start from e.
	a | b | c
	d | e |
	*/

	const int64_t A = -1 - sx;
	const int64_t A_mask = 0b10000000;
	const int64_t B = -sx;
	const int64_t B_mask = 0b1000;
	const int64_t C = +1 - sx;
	const int64_t C_mask = 0b1000000;
	const int64_t D = -1;
	const int64_t D_mask = 0b10;

	for (int64_t y = 1; y < sy; y++) {
		for (int64_t x = 0; x < sx; x++) {
			int64_t loc = x + sx * y;

			bool stolen = false;

			if (vcg[loc] & B_mask) {
				out_labels[loc] = out_labels[loc+B];
				stolen = true;
			}
			if (x > 0 && (vcg[loc] & A_mask)) {
				if (!stolen) {
					out_labels[loc] = out_labels[loc+A];
					stolen = true;	
				}
				else {
					equivalences.unify(out_labels[loc], out_labels[loc+A]);
				}
			}
			if (x > 0 && (vcg[loc] & D_mask)) {
				if (!stolen) {
					out_labels[loc] = out_labels[loc+D];
					stolen = true;	
				}
				else {
					equivalences.unify(out_labels[loc], out_labels[loc+D]);
				}
			}
			if (x < sx - 1 && (vcg[loc] & C_mask)) {
				if (!stolen) {
					out_labels[loc] = out_labels[loc+C];
					stolen = true;	
				}
				else {
					equivalences.unify(out_labels[loc], out_labels[loc+C]);
				}
			}

			if (!stolen) {
				new_label++;
				out_labels[loc] = new_label;
				equivalences.add(out_labels[loc]);
			}
		}
	}

	return simplified_relabel<OUT>(out_labels, sxy, new_label, equivalences, N);
}

// 8 and 32 bit inputs have different encodings so 
// need to write this two ways
template <typename OUT>
OUT* color_connectivity_graph_8(
  const uint32_t* vcg, // voxel connectivity graph
  const int64_t sx, const int64_t sy,
  OUT* out_labels = NULL,
  size_t &N = _dummy_N
) {
	const int64_t sxy = sx * sy;

	uint64_t max_labels = static_cast<uint64_t>(sxy) + 1; // + 1L for an array with no zeros
	max_labels = std::min(max_labels, static_cast<uint64_t>(std::numeric_limits<OUT>::max()));

	if (out_labels == NULL) {
		out_labels = new OUT[sxy]();
	}

	DisjointSet<OUT> equivalences(max_labels);

	OUT new_label = 1;
	equivalences.add(new_label);

	for (int64_t x = 0; x < sx; x++) {
		if (x > 0 && (vcg[x] & 0b0010) == 0) {
			new_label++;
			equivalences.add(new_label);
		}
		out_labels[x] = new_label;
	}

	/*
	Layout of mask. We start from e.
	a | b | c
	d | e |
	*/

	const int64_t A = -1 - sx;
	const int64_t A_mask = 0b1000000000;
	const int64_t B = -sx;
	const int64_t B_mask = 0b1000;
	const int64_t C = +1 - sx;
	const int64_t C_mask = 0b100000000;
	const int64_t D = -1;
	const int64_t D_mask = 0b10;

	for (int64_t y = 1; y < sy; y++) {
		for (int64_t x = 0; x < sx; x++) {
			int64_t loc = x + sx * y;

			bool stolen = false;

			if (vcg[loc] & B_mask) {
				out_labels[loc] = out_labels[loc+B];
				stolen = true;
			}
			if (x > 0 && (vcg[loc] & A_mask)) {
				if (!stolen) {
					out_labels[loc] = out_labels[loc+A];
					stolen = true;	
				}
				else {
					equivalences.unify(out_labels[loc], out_labels[loc+A]);
				}
			}
			if (x > 0 && (vcg[loc] & D_mask)) {
				if (!stolen) {
					out_labels[loc] = out_labels[loc+D];
					stolen = true;	
				}
				else {
					equivalences.unify(out_labels[loc], out_labels[loc+D]);
				}
			}
			if (x < sx - 1 && (vcg[loc] & C_mask)) {
				if (!stolen) {
					out_labels[loc] = out_labels[loc+C];
					stolen = true;	
				}
				else {
					equivalences.unify(out_labels[loc], out_labels[loc+C]);
				}
			}

			if (!stolen) {
				new_label++;
				out_labels[loc] = new_label;
				equivalences.add(out_labels[loc]);
			}
		}
	}

	return simplified_relabel<OUT>(out_labels, sxy, new_label, equivalences, N);
}

template <typename VCG_t, typename OUT>
OUT* color_connectivity_graph_4(
  const VCG_t* vcg, // voxel connectivity graph
  const int64_t sx, const int64_t sy,
  OUT* out_labels = NULL,
  size_t &N = _dummy_N
) {
	const int64_t sxy = sx * sy;

	uint64_t max_labels = static_cast<uint64_t>(sxy) + 1; // + 1L for an array with no zeros
	max_labels = std::min(max_labels, static_cast<uint64_t>(std::numeric_limits<OUT>::max()));

	if (out_labels == NULL) {
		out_labels = new OUT[sxy]();
	}

	DisjointSet<OUT> equivalences(max_labels);

	OUT new_label = 1;
	equivalences.add(new_label);

	for (int64_t x = 0; x < sx; x++) {
		if (x > 0 && (vcg[x] & 0b0010) == 0) {
			new_label++;
			equivalences.add(new_label);
		}
		out_labels[x] = new_label;
	}

	const int64_t B = -1;
	const int64_t B_mask = 0b0010;
	const int64_t C = -sx;
	const int64_t C_mask = 0b1000;

	for (int64_t y = 1; y < sy; y++) {
		for (int64_t x = 0; x < sx; x++) {
			int64_t loc = x + sx * y;

			if (x > 0 && (vcg[loc] & B_mask)) {
				out_labels[loc] = out_labels[loc+B];
				if (vcg[loc] & C_mask) {
					equivalences.unify(out_labels[loc], out_labels[loc+C]);
				}
			}
			else if (vcg[loc] & C_mask) {
				out_labels[loc] = out_labels[loc+C];
			}
			else {
				new_label++;
				out_labels[loc] = new_label;
				equivalences.add(new_label);
			}
		}
	}

	return simplified_relabel<OUT>(out_labels, sxy, new_label, equivalences, N);
}

template <typename VCG_t, typename OUT>
OUT* color_connectivity_graph_N(
  const VCG_t* vcg, // voxel connectivity graph
  const int64_t sx, const int64_t sy, const int64_t sz,
  const int connectivity,
  OUT* out_labels = NULL,
  size_t &N = _dummy_N
) {
	if (connectivity != 26 && connectivity != 6 && connectivity != 4 && connectivity != 8) {
		throw std::runtime_error("Only 4, 8, 6 and 26 connectivities are supported.");
	}

	if (sz > 1 && (connectivity != 6 && connectivity != 26)) {
		throw std::runtime_error("Only 6 and 26 connectivity is supported in 3D.");
	}

	if (sz == 1) {
		if (connectivity == 6 || connectivity == 4) {
			return color_connectivity_graph_4<VCG_t, OUT>(vcg, sx, sy, out_labels, N);
		}
		else {
			return color_connectivity_graph_8(vcg, sx, sy, out_labels, N);
		}
	}
	else if (connectivity == 6) {
		return color_connectivity_graph_6<OUT>(vcg, sx, sy, sz, out_labels, N);
	}
	else {
		return color_connectivity_graph_26<OUT>(vcg, sx, sy, sz, out_labels, N);
	}
}

};

#endif