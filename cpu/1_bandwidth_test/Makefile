###############################################################################
#
# Makefile for building $(TARGET) benchmark. $(USER_DEFS) is used to pass the
# definitions to the compiler. $(KERNELS) defines the kernels required by the
# benchmark.
#
# Author: 	Alif Ahmed
# email: 	alifahmed@virginia.edu
# Updated: 	Sep 16, 2019
#
###############################################################################

TARGET = 	bandwidth

.PHONY: all clean $(TARGET)

# directories
INC_DIR = ../include
KERN_DIR = ../kernels
CMN_DIR = ../common
OBJ_DIR = obj


# compiler flags
CXX = g++
#CXXFLAGS = -O3 -fopenmp -march=native -I$(INC_DIR) -std=c++14 $(USER_DEFS)
CXXFLAGS = -O3 -I$(INC_DIR) -std=c++14 $(USER_DEFS)


# header files
HEADERS = $(wildcard $(INC_DIR)/*.h)


# src files
SRC = $(wildcard *.cpp) $(wildcard $(KERN_DIR)/*.cpp) $(wildcard $(CMN_DIR)/*.cpp)

# object files
OBJ = $(SRC:.cpp=.o)

all: $(TARGET)
	
clean:
	@rm -rf $(OBJ)
	@rm -rf $(TARGET)
	@echo "Cleaned..."
	
%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@

