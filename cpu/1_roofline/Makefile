###############################################################################
#
# Makefile for building roofline kernel. Used by roofline_cpu.py to build for
# different configurations. $(USER_DEFS) is used to pass the definitions to the
# compiler.
#
# Author: 	Alif Ahmed
# email: 	alifahmed@virginia.edu
# Updated: 	Aug 06, 2019
#
###############################################################################

.PHONY: all clean kernel

CPPFLAGS += -O3 -fopenmp -std=c++14 -march=native $(USER_DEFS)

all: kernel

kernel: hs_roofline.cpp common.cpp common.h
	@g++ -o kernel hs_roofline.cpp common.cpp $(CPPFLAGS)

clean:
	@rm -rf kernel
