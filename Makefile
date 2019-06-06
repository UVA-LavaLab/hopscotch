.PHONY: all clean preprocess 

CPPFLAGS = -Wall -Ofast -g -fopenmp -I./include/ -Wno-unused-variable -std=c++11 $(USER_DEF)
BUILD_DIR = build
INC_DIR = include
HEADERS = $(wildcard $(INC_DIR)/*.h)

KERN_SRC = $(wildcard src/kernels/*.cpp) $(wildcard src/utils/*.cpp)
KERN_OBJ = $(KERN_SRC:.cpp=.o)

BENCH_SRC = $(wildcard src/*.cpp)
BENCH_OBJ = $(BENCH_SRC:.cpp=.o)
	
all: preprocess 					\
	$(BUILD_DIR)/throughput 		\
	$(BUILD_DIR)/peak_bandwidth 	\
	$(BUILD_DIR)/latency 			\
	$(BUILD_DIR)/test
	
clean:
	@rm -rf $(KERN_OBJ)
	@rm -rf $(BENCH_OBJ)
	@rm -rf $(BUILD_DIR)/throughput
	@echo "Cleaned..."
	
preprocess:
	@mkdir -p $(BUILD_DIR)

%.o: %.cpp $(HEADERS)
	g++ $(CPPFLAGS) -c -o $@ $<

$(BUILD_DIR)/throughput: $(KERN_OBJ) src/hs_throughput.o
	g++ $(CPPFLAGS) $^ -o $@
	
$(BUILD_DIR)/peak_bandwidth: $(KERN_OBJ) src/hs_peak_bandwidth.o
	g++ $(CPPFLAGS) $^ -o $@
	
$(BUILD_DIR)/latency: $(KERN_OBJ) src/hs_latency.o
	g++ $(CPPFLAGS) $^ -o $@

$(BUILD_DIR)/test: $(KERN_OBJ) src/test.o
	g++ $(CPPFLAGS) $^ -o $@

