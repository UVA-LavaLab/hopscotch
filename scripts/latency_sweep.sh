#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: ./latency_sweep min_pow max_pow"
	exit
fi

let start=2**$1
let end=2**$2

cd ..

while [ $start -le $end ]
do
	make clean > log.tmp
	if ! make USER_DEF="-DHS_ARRAY_ELEM=\($start*1UL\)" > log.tmp; then
		exit
	fi
	echo "WSS: $((start*8)) bytes"
	./build/latency
	start=$((start*2))
done
