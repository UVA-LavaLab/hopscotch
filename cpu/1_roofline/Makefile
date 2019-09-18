###############################################################################
#
# Makefile for building roofline kernel. Used by roofline.py to build for
# different configurations. $(USER_DEFS) is used to pass the definitions to the
# compiler.
#
# Author: 	Alif Ahmed
# email: 	alifahmed@virginia.edu
# Updated: 	Aug 06, 2019
#
###############################################################################

.PHONY: all clean kernel

INC_DIR = ../include
CMN_DIR = ../common
KERN_DIR = ../kernels

CXX = g++
CXXFLAGS = -O3 -fopenmp -std=c++14 -march=native $(USER_DEFS) -I$(INC_DIR)

all:
	# Default target does nothing. Use roofline.py instead.

kernel: hs_roofline.cpp $(CMN_DIR)/common.cpp $(KERN_DIR)/rw_roofline.cpp $(INC_DIR)/common.h
	@$(CXX) -o kernel hs_roofline.cpp $(CMN_DIR)/common.cpp $(KERN_DIR)/rw_roofline.cpp $(CXXFLAGS)

clean:
	@rm -rf kernel
