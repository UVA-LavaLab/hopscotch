#!/bin/bash

cd ..
make clean > log.tmp
if ! make USER_DEF="-DHS_ARRAY_ELEM=\(32*1024*1024UL\)" > log.tmp; then
	exit
fi
./build/throughput
	
