#include "cc3d.hpp"

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

	uint16_t *y = cc3d::connected_components3d(x, 3, 3, 3);

	for (int i = 0; i < 27; i++) {
		printf("%d ", y[i]);
	}

	return 0;
}