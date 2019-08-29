.PHONY: all clean bj

all: clean bj


clean:
	@rm -rf bj
	@rm -rf *.o

%.o: %.cpp common.h
	g++ -c -O3 -fopenmp -march=native -o $@ $< -std=c++11 -DFLOPS_PER_ELEM=$(FP)

bj: bj.o common.o
	g++ -O3 -fopenmp -march=native -o $@ $^ -std=c++11
