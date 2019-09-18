/******************************************************************************
 *
 * File: rw_scatter.cpp
 * Description: Kernel with scatter pattern.
 * 
 * Author: Alif Ahmed
 * Date: Sep 16, 2019
 *
 *****************************************************************************/
#include <common.h>
#include <cstdint>


void rw_scatter(data_t* __restrict__ a, data_t* __restrict__ b, uint64_t* __restrict__ idx){
    #pragma omp parallel for simd aligned(a,b,idx)
    for(uint64_t i = 1; i < WSS_ELEMS; ++i) {
        a[idx[i]] = b[i];
    }
}


res_t run_rw_scatter(double allowed_time, data_t* a, data_t* b, uint64_t* idx){
	res_t result;
	run_kernel(rw_scatter(a,b,idx), allowed_time, result);
	result.bytes_read = result.iters * WSS_ELEMS * (sizeof(data_t) + 8);	// read = idx + b
	result.bytes_write = result.iters * WSS_ELEMS * sizeof(data_t);			// write = a
	return result;
}
