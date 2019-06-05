.PHONY: all clean preprocess postprocess throughput 

CPPFLAGS = -Wall -Ofast -g -fopenmp -I./include/ -Wno-unused-variable -std=c++11
BUILD_DIR = build
INC_DIR = include

KERN_SRC = $(wildcard src/kernels/*.cpp) $(wildcard src/utils/*.cpp)
KERN_OBJ = $(KERN_SRC:.cpp=.o)

BENCH_SRC = $(wildcard src/*.cpp)
BENCH_OBJ = $(BENCH_SRC:.cpp=.o)
	
all: preprocess throughput peak_bandwidth latency postprocess
	
clean: postprocess
	@rm -rf $(BUILD_DIR)/throughput
	@echo "Cleaned..."
	
preprocess:
	@mkdir -p $(BUILD_DIR)

postprocess:
	@rm -rf $(KERN_OBJ)
	@rm -rf $(BENCH_OBJ)

throughput: $(KERN_OBJ) src/hs_throughput.o
	g++ $(CPPFLAGS) $^ -o $(BUILD_DIR)/$@
	
peak_bandwidth: $(KERN_OBJ) src/hs_peak_bandwidth.o
	g++ $(CPPFLAGS) $^ -o $(BUILD_DIR)/$@
	
latency: $(KERN_OBJ) src/hs_latency.o
	g++ $(CPPFLAGS) $^ -o $(BUILD_DIR)/$@
