#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: ./roofline min_flop_per_iter max_flop_per_iter"
	exit
fi

cd ..

while [ $start -le $end ]
do
	make clean > log.tmp
	if ! make USER_DEF="-DFLOP_PER_ITER=$start" > log.tmp; then
		exit
	fi
	#echo "WSS: $((start*8)) bytes"
	./build/roofline
	start=$((start*2))
done
