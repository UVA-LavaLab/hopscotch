/******************************************************************************
 *
 * File: w_stride.cpp
 * Description: Strided write.
 * 
 * Author: Alif Ahmed
 * Date: Sep 16, 2019
 *
 *****************************************************************************/
#include <common.h>
#include <cstdint>

template <uint64_t stride>
void w_stride(data_t* __restrict__ a){
    #pragma omp parallel for simd aligned (a)
    for(uint64_t i = 0; i < WSS_ELEMS; i += stride) {
        a[i] = 7;
    }
}


template <uint64_t stride>
res_t run_w_stride(double allowed_time, data_t* a){
    res_t result;
    run_kernel(w_stride<stride>(a), allowed_time, result);
    result.bytes_read = 0;
    result.bytes_write = result.iters * WSS_ELEMS * sizeof(data_t) / stride;
    return result;
}

