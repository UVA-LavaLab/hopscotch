.PHONY: all clean roofline

all: roofline

roofline: hs_roofline.cpp common.cpp include/common.h
	g++ -O3 -o roofline hs_roofline.cpp common.cpp -Iinclude/ -fopenmp -std=c++14 -march=native $(USER_DEFS)

clean:
	@rm -rf roofline
