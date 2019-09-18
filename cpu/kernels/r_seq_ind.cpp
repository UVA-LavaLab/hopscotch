/******************************************************************************
 *
 * File: r_seq_ind.cpp
 * Description: Sequential read without loop-carried dependency
 * 
 * Author: Alif Ahmed
 * Date: Sep 16, 2019
 *
 *****************************************************************************/
#include <common.h>
#include <cstdint>


void r_seq_ind(data_t* __restrict__ a){
    volatile data_t * vol_a = a;
    #pragma omp parallel for simd
    for(uint64_t i = 0; i < WSS_ELEMS; ++i) {
        data_t res = vol_a[i];
    }
}


res_t run_r_seq_ind(double allowed_time, data_t* a){
    res_t result;
    run_kernel(r_seq_ind(a), allowed_time, result);
    result.bytes_read = result.iters * WSS_ELEMS * sizeof(data_t);
    result.bytes_write = 0;
    return result;
}
