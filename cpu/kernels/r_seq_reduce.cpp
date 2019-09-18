/******************************************************************************
 *
 * File: r_seq_reduce.cpp
 * Description: Sequential read with loop-carried dependency (reduction)
 * 
 * Author: Alif Ahmed
 * Date: Sep 16, 2019
 *
 *****************************************************************************/
#include <common.h>
#include <cstdint>


data_t r_seq_reduce(data_t* __restrict__ a){
	data_t sum = 0;
    #pragma omp parallel for simd reduction(+ : sum) aligned(a)
    for(uint64_t i = 0; i < WSS_ELEMS; ++i) {
        sum += a[i];
    }
	return sum;
}


res_t run_r_seq_reduce(double allowed_time, data_t* a){
	res_t result;
	run_kernel(r_seq_reduce(a), allowed_time, result);
	result.bytes_read = result.iters * WSS_ELEMS * sizeof(data_t);
	result.bytes_write = 0;
	return result;
}
