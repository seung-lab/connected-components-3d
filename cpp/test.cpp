#include "cc3d.hpp"

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

	int *big = new int[512*512*512]();
	for (int i = 0; i < 512*512*512; i++) {
		big[i] = 1;
	}

	cc3d::connected_components3d(big, 512,512,512);	


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