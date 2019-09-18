/******************************************************************************
 *
 * File: w_seq_fill.cpp
 * Description: Fill an array with a constant value.
 * 
 * Author: Alif Ahmed
 * Date: Sep 16, 2019
 *
 *****************************************************************************/
#include <common.h>
#include <cstdint>


void w_seq_fill(data_t* __restrict__ a){
    #pragma omp parallel for simd aligned (a)
    for(uint64_t i = 0; i < WSS_ELEMS; ++i) {
        a[i] = 7;
    }
}


res_t run_w_seq_fill(double allowed_time, data_t* a){
    res_t result;
    run_kernel(w_seq_fill(a), allowed_time, result);
    result.bytes_read = 0;
    result.bytes_write = result.iters * WSS_ELEMS * sizeof(data_t);
    return result;
}
