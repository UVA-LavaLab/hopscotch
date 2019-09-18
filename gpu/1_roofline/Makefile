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

# Kernel is built every time, even no source code is changed.
# This is because the kernel is compiled by a script with different configurations.
.PHONY: all clean kernel

INC_DIR = ../include
CMN_DIR = ../common

NVCCFLAGS += -I$(INC_DIR) -std=c++11 $(USER_DEFS)

all:
#Do nothing for default make target. Run roofline.py script for roofline plot.

query: $(CMN_DIR)/common.cu $(CMN_DIR)/query.cu $(INC_DIR)/common.h 
	nvcc $(CMN_DIR)/common.cu $(CMN_DIR)/query.cu -o $@ $(NVCCFLAGS)

kernel: $(CMN_DIR)/common.cu $(INC_DIR)/common.h roofline.cu
	nvcc $(CMN_DIR)/common.cu roofline.cu -o $@ $(NVCCFLAGS)

clean:
	rm -rf query
	rm -rf kernel
	
