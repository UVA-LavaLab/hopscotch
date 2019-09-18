/******************************************************************************
 *
 * File: rw_seq_inc.cpp
 * Description: Kernel with sequential mixed access increment.
 * 
 * Author: Alif Ahmed
 * Date: Sep 16, 2019
 *
 *****************************************************************************/
#include <common.h>
#include <cstdint>


void rw_seq_inc(data_t* __restrict__ a){
    #pragma omp parallel for simd aligned(a)
    for(uint64_t i = 0; i < WSS_ELEMS; ++i) {
        a[i]++;
    }
}


res_t run_rw_seq_inc(double allowed_time, data_t* a){
	res_t result;
	run_kernel(rw_seq_inc(a), allowed_time, result);
	result.bytes_read = result.iters * WSS_ELEMS * sizeof(data_t);
	result.bytes_write = result.iters * WSS_ELEMS * sizeof(data_t);
	return result;
}
