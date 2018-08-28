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

	int x[27] = {
		1,1,0,
		0,0,1,
		0,0,1,

		0,0,0,
		0,0,1,
		1,0,1,

		0,0,0,
		0,1,1,
		0,0,1
	};

	printf("INPUT\n");
	print(x, 3,3,3);

	uint16_t *y = cc3d::connected_components3d(x, 3, 3, 3);
	printf("OUTPUT\n");
	print(y, 3,3,3);

	return 0;
}