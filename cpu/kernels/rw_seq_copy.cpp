/******************************************************************************
 *
 * File: rw_seq_copy.cpp
 * Description: Kernel for sequentially copying one array to another.
 * 
 * Author: Alif Ahmed
 * Date: Sep 16, 2019
 *
 *****************************************************************************/
#include <common.h>
#include <cstdint>


void rw_seq_copy(data_t* __restrict__ a, data_t* __restrict__ b){
    #pragma omp parallel for simd aligned(a,b)
    for(uint64_t i = 0; i < WSS_ELEMS; ++i) {
        a[i] = b[i];
    }
}


res_t run_rw_seq_copy(double allowed_time, data_t* a, data_t* b){
	res_t result;
	run_kernel(rw_seq_copy(a,b), allowed_time, result);
	result.bytes_read = result.iters * WSS_ELEMS * sizeof(data_t);
	result.bytes_write = result.iters * WSS_ELEMS * sizeof(data_t);
	return result;
}
