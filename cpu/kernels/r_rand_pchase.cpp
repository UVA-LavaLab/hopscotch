/******************************************************************************
 *
 * File: r_rand_pchase.cpp
 * Description: Single-threaded pointer chasing
 * 
 * Author: Alif Ahmed
 * Date: Sep 16, 2019
 *
 *****************************************************************************/
#include "../include/common.h"
#include <cstdint>


inline void r_rand_pchase(void** ptr){
    void **curr = ptr;
    void* saved_val = *ptr;
    for(uint64_t i = 0; i < WSS_ELEMS; ++i) {
        curr = (void**)*curr;
    }
    *curr = saved_val; //do a write to prevent optimization
}


res_t run_r_rand_pchase(double allowed_time, void** ptr){
    res_t result;
    run_kernel(r_rand_pchase(ptr), allowed_time, result);
    result.bytes_read = result.iters * WSS_ELEMS * sizeof(data_t);
    result.bytes_write = 0;
    return result;
}
