/******************************************************************************
 *
 * File: w_rand_ind.cpp
 * Description: Randomly fill an array with a constant value. The random index
 *              is generated within loop.
 * 
 * Author: Alif Ahmed
 * Date: Sep 16, 2019
 *
 *****************************************************************************/
#include <common.h>
#include <cstdint>


void w_rand_ind(data_t* __restrict__ a){
    #pragma omp parallel for simd aligned (a)
    for(uint64_t i = 0; i < WSS_ELEMS; ++i) {
        uint64_t idx = ((i * 0xDEADBEEF) ^ 0xC0FFEE0B) % WSS_ELEMS;
        a[idx] = 7;
    }
}


res_t run_w_rand_ind(double allowed_time, data_t* a){
    res_t result;
    run_kernel(w_rand_ind(a), allowed_time, result);
    result.bytes_read = 0;
    result.bytes_write = result.iters * WSS_ELEMS * sizeof(data_t);;
    return result;
}
