all: debug

debug:
	g++ -pg -std=c++11 -g -O3 cc3d.hpp -ffast-math -o cc3d 

shared:
	g++ -fPIC -shared -std=c++11 -O3 cc3d.hpp -ffast-math -o cc3d.so

test: FORCE
	g++ -pg -std=c++11 -g -O3 test.cpp -ffast-math -o test

FORCE: