#include "cc3d.hpp"

void print(uint32_t *input, int sx, int sy, int sz) {
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

	// int *big = new int[512*512*512]();
	// for (int i = 0; i < 512*512*512; i++) {
	// 	big[i] = 1;
	// }
	//
	// cc3d::connected_components3d<int>(big, 512,512,512);


	// int twod[25] = {
	// 	1,1,0,1,1,
	// 	0,0,1,0,0,
	// 	1,1,0,1,1,
	// 	0,0,1,0,0,
	// 	2,2,2,2,2
	// };

	int td[75] = {
		1,1,0,0,2,
		1,1,0,0,0,
		1,1,1,0,0,
		0,0,0,1,1,
		3,2,2,1,1,

		1,1,0,0,2,
		1,1,0,0,0,
		1,1,1,0,0,
		0,0,0,1,1,
		3,2,2,1,1,

		1,1,0,0,2,
		1,1,0,0,0,
		1,1,1,0,0,
		0,0,0,1,1,
		3,2,2,1,6


	};

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

	printf("INPUT\n");
	print(td, 5,5,3);

	uint32_t *y = cc3d::connected_components3d(td, 5,5,3,10,6);
	printf("OUTPUT\n");
	print(y, 5,5,3);


	// compile with: gcc test.cpp -o out_test -lstdc++

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
