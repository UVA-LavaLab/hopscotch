###############################################################################
#
# Makefile for building latency benchmark. $(USER_DEFS) is used to pass the
# definitions to the compiler.
#
# Author: 	Alif Ahmed
# email: 	alifahmed@virginia.edu
# Updated: 	Sep 16, 2019
#
###############################################################################

.PHONY: all clean kernel

INC_DIR = ../include
CMN_DIR = ../common
KERN_DIR = ../kernels

CXX = g++
CXXFLAGS = -O3 -fopenmp -std=c++14 -march=native $(USER_DEFS) -I$(INC_DIR)

all:
	# Default target does nothing. Use latency.py instead.

kernel: hs_latency.cpp $(CMN_DIR)/common.cpp $(KERN_DIR)/r_rand_pchase.cpp $(INC_DIR)/common.h
	@$(CXX) -o kernel hs_latency.cpp $(CMN_DIR)/common.cpp $(KERN_DIR)/r_rand_pchase.cpp $(CXXFLAGS)

clean:
	@rm -rf kernel
