#include "cc3d.hpp"
#include <chrono>
#include <iostream>

void print(uint16_t *input, int sx, int sy, int sz) {
	int i = 0;

	for (int z = 0; z < sz; z++) {
		for (int y = 0; y < sy; y++) {
			for (int x = 0; x < sx; x++, i++) {
				printf("%d, ", input[i]);
			}
			printf("\n");
		}
		printf("\n");
	}
}

void print(int *input, int sx, int sy, int sz) {
	int i = 0;

	for (int z = 0; z < sz; z++) {
		for (int y = 0; y < sy; y++) {
			for (int x = 0; x < sx; x++, i++) {
				printf("%d, ", input[i]);
			}
			printf("\n");
		}
		printf("\n");
	}
}

uint32_t* read_subvol() {
	const size_t voxels = 512 * 512 * 512;
	uint32_t* input = new uint32_t[voxels]();
  FILE *readPtr = fopen("subvol.bin", "rb");
  fread(input, sizeof(uint32_t), voxels, readPtr);
  fclose(readPtr);
  return input;
}

uint32_t* randomvol() {
	size_t sx = 512;
	size_t sy = 512;
	size_t sz = 512;
	size_t voxels = sx * sy * sz;

	uint32_t *big = new uint32_t[sx*sy*sz]();
	for (uint32_t i = 0; i < voxels; i++) {
		big[i] = rand() % 10;
	}
	return big;
}

uint32_t* black() {
	size_t sx = 512;
	size_t sy = 512;
	size_t sz = 512;
	size_t voxels = sx * sy * sz;

	return new uint32_t[sx*sy*sz]();
}

int main () {

	size_t sx = 512;
	size_t sy = 512;
	size_t sz = 512;
	size_t voxels = sx * sy * sz;

	uint32_t* subvol = black();

  typedef std::chrono::high_resolution_clock Time;
  typedef std::chrono::milliseconds ms;
  typedef std::chrono::duration<float> fsec;
  auto t0 = Time::now();
  int N = 1;
  for (int i = 0; i < N; i++) {
		cc3d::connected_components3d<uint32_t, uint32_t>(subvol, sx,sy,sz, 26);
	}

  auto t1 = Time::now();
  fsec fs = t1 - t0;
  ms d = std::chrono::duration_cast<ms>(fs);
  std::cout << d.count() / N << "ms\n";

	return 0;
}
