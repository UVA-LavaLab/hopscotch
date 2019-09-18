/******************************************************************************
 *
 * File: r_tile.cpp
 * Description: Tile kernel (single-threaded) with only read access.
 * 
 * Author: Alif Ahmed
 * Date: Sep 16, 2019
 *
 *****************************************************************************/
#include <common.h>
#include <cstdint>


void r_tile(data_t* __restrict__ a, uint64_t L, uint64_t K){
    volatile data_t * vol_a = a;
    for(uint64_t i = 0; i < WSS_ELEMS; i += K) {
        for(uint64_t j = 0; j < L; j++) {
            const uint64_t idx = i + j;
            if(idx >= WSS_ELEMS)
                break;
            data_t res = vol_a[idx];
        }
    }
}


res_t run_r_tile(double allowed_time, data_t* a, uint64_t L, uint64_t K){
    res_t result;
    run_kernel(r_tile(a, L, K), allowed_time, result);
    result.bytes_read = result.iters * WSS_ELEMS * sizeof(data_t) * L / K;
    result.bytes_write = 0;
    return result;
}
