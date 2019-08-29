###############################################################################
#
# Makefile for building $(TARGET) benchmark. $(USER_DEFS) is used to pass the
# definitions to the compiler.
#
# Author: 	Alif Ahmed
# email: 	alifahmed@virginia.edu
# Updated: 	Aug 06, 2019
#
###############################################################################

TARGET = fairness

.PHONY: all clean $(TARGET)

INC_DIR = ../include
HEADERS = $(wildcard $(INC_DIR)/*.h)

KERN_DIR = ../kernels
CMN_DIR = ../common
KERN_SRC = $(wildcard $(KERN_DIR)/*.cpp) $(wildcard $(CMN_DIR)/*.cpp)
KERN_OBJ = $(KERN_SRC:.cpp=.o)

CPPFLAGS = -Ofast -fopenmp -march=native -I$(INC_DIR) -Wno-unused-variable -std=c++11 $(USER_DEF)
	
BENCH_SRC = $(wildcard *.cpp)

#all: $(TARGET)
all: workload back
	
clean:
	@rm -rf $(KERN_OBJ)
	@rm -rf $(TARGET)
	@rm -rf back
	@rm -rf workload
	
%.o: %.cpp $(HEADERS)
	@g++ $(CPPFLAGS) -c -o $@ $<

$(TARGET): $(KERN_OBJ) $(BENCH_SRC)
	@g++ $(CPPFLAGS) $^ -o $@

back: background_job.o $(CMN_DIR)/common.o
	@g++ $(CPPFLAGS) $^ -o $@

workload: workload.o $(CMN_DIR)/common.o
	@g++ $(CPPFLAGS) $^ -o $@
	
