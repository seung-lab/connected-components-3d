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



int main () {

	size_t sx = 512;
	size_t sy = 512;
	size_t sz = 512;
	size_t voxels = sx * sy * sz;

	int *big = new int[sx*sy*sz]();
	for (int i = 0; i < voxels; i++) {
		big[i] = rand() % 10;
	}

	// printf("%d\n", big[5]);
	
	// printf("%u %u\n", p.first, p.second);
    typedef std::chrono::high_resolution_clock Time;
    typedef std::chrono::milliseconds ms;
    typedef std::chrono::duration<float> fsec;
    auto t0 = Time::now();
	// size_t p = cc3d::zeroth_pass<int>(big, voxels);
	std::pair<size_t, bool> p = cc3d::zeroth_pass<int>(big, voxels);
    auto t1 = Time::now();
    fsec fs = t1 - t0;
    ms d = std::chrono::duration_cast<ms>(fs);
    std::cout << fs.count() << "s\n";
    std::cout << d.count() << "ms\n";
	
    printf("%d\n", p.first);
	// cc3d::connected_components3d<int, uint32_t>(big, sx,sy,sz, 26);	


	// int twod[25] = {
	// 	1,1,0,1,1,
	// 	0,0,1,0,0,
	// 	1,1,0,1,1,
	// 	0,0,1,0,0,
	// 	2,2,2,2,2
	// };

	// int threed[27] = {
	// 	1,1,0,
	// 	0,0,1,
	// 	0,0,1,

	// 	0,0,0,
	// 	0,0,1,
	// 	1,0,1,

	// 	0,0,0,
	// 	0,1,1,
	// 	0,0,1
	// };

	// printf("INPUT\n");
	// print(twod, 5,5,1);

	// uint16_t *y = cc3d::connected_components3d(twod, 5,5,1);
	// printf("OUTPUT\n");
	// print(y, 5,5,1);



	// cc3d::DisjointSet<int> djs;
	// printf("\nDISJOINT SET\n");
	// printf("root 5: %d\n", djs.root(5));

	// djs.unify(1,2);
	// printf("root 1: %d\n", djs.root(1));
	// printf("root 2: %d\n", djs.root(2));
	// printf("root 1: %d\n", djs.root(1));
	// printf("root 2: %d\n", djs.root(2));
	// printf("root 1: %d\n", djs.root(1));
	// printf("root 2: %d\n", djs.root(2));

	return 0;
}