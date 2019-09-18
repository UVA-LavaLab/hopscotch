/*******************************************************************************
 *
 * File: hs_cache.cpp
 * Description: Measures cache performance with varying spatial and temporal
 * locality.
 * 
 * Author: Alif Ahmed
 * Date: Aug 06, 2019
 *
 ******************************************************************************/
#include "common.h"
#include <cstdlib>

// allowed runtime per kernel
#define ALLOWED_RUNTIME     3

int main(){
    uint64_t K, L;
    res_t result;

    //print header
    print_bw_header();
        
    //allocate and initialize working set
    data_t* a = (data_t*)hs_alloc(WSS_BYTES);
    init_const(a, WSS_ELEMS, 7);

    //spatial locality = low
    //temporal locality = low
    L = 1;
    K = 32;
    result = run_rw_tile(ALLOWED_RUNTIME, a, L, K);
    print_max_bw("Spatial=low, Temporal=low", result);
    
    
    //spatial locality = low
    //temporal locality = high
    L = 2;
    K = 1;
    result = run_rw_tile(ALLOWED_RUNTIME, a, L, K);
    print_max_bw("Spatial=low, Temporal=high", result);


    //spatial locality = high
    //temporal locality = low
    L = WSS_ELEMS;
    K = WSS_ELEMS;
    result = run_rw_tile(ALLOWED_RUNTIME, a, L, K);
    print_max_bw("Spatial=high, Temporal=low", result);


    //spatial locality = high
    //temporal locality = high
    L = 32;
    K = 1;
    result = run_rw_tile(ALLOWED_RUNTIME, a, L, K);
    print_max_bw("Spatial=high, Temporal=high", result);


    free(a);

    return 0;
}
