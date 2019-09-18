/******************************************************************************
 *
 * File: rw_scatter_gather.cpp
 * Description: Kernel with scatter-gather pattern.
 * 
 * Author: Alif Ahmed
 * Date: Sep 16, 2019
 *
 *****************************************************************************/
#include <common.h>
#include <cstdint>


void rw_scatter_gather(data_t* __restrict__ a, data_t* __restrict__ b,
		uint64_t* __restrict__ idx1, uint64_t* __restrict__ idx2){
    #pragma omp parallel for simd aligned(a,b,idx1,idx2)
    for(uint64_t i = 1; i < WSS_ELEMS; ++i) {
        a[idx1[i]] = b[idx2[i]];
    }
}


res_t run_rw_scatter_gather(double allowed_time, data_t* a, data_t* b,
		uint64_t* idx1, uint64_t* idx2){
	res_t result;
	run_kernel(rw_scatter_gather(a,b,idx1,idx2), allowed_time, result);
	result.bytes_read = result.iters * WSS_ELEMS * (sizeof(data_t) + 16);	// read = idx1 + idx2 + b
	result.bytes_write = result.iters * WSS_ELEMS * sizeof(data_t);			// write = a
	return result;
}
