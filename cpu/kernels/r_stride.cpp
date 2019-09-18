/******************************************************************************
 *
 * File: r_stride.cpp
 * Description: Strided read. Since it is a template, it should be included in
 *              the benchmark source.
 * 
 * Author: Alif Ahmed
 * Date: Sep 16, 2019
 *
 *****************************************************************************/
#include <common.h>
#include <cstdint>

template <uint64_t stride>
void r_stride(data_t* __restrict__ a){
    volatile data_t * vol_a = a;
    #pragma omp parallel for simd aligned (vol_a)
    for(uint64_t i = 0; i < WSS_ELEMS; i += stride) {
        data_t res = vol_a[i];
    }
}

template <uint64_t stride>
res_t run_r_stride(double allowed_time, data_t* a){
	res_t result;
	run_kernel(r_stride<stride>(a), allowed_time, result);
	result.bytes_read = result.iters * WSS_ELEMS * sizeof(data_t) / stride;
	result.bytes_write = 0;
	return result;
}

