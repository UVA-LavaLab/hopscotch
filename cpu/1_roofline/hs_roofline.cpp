/******************************************************************************
 *
 * File: hs_roofline.cpp
 * Description: Runs roofline kernel.
 * 
 * Author: Alif Ahmed
 * Date: Aug 06, 2019
 *
 *****************************************************************************/
#include "common.h"
#include <cstdlib>
#include <cstdio>

// allowed runtime for each configuration
#define ALLOWED_RUNTIME     5


#ifndef FLOPS_PER_ELEM
#define FLOPS_PER_ELEM       512
#endif


int main(){
    // Allocate and initialize working set
    data_t* a = (data_t*)hs_alloc(WSS_BYTES);
    init_const(a, WSS_ELEMS, 1.0);

    // Run kernel and collect result
    res_t result = run_rw_roofline(ALLOWED_RUNTIME, a);
  
    // Print result
    printf("%-8s%11d%17.2f%19.2f%19.2f\n",
            sizeof(data_t) == 4 ? "float" : "double",
            FLOPS_PER_ELEM,
            FLOPS_PER_ELEM / 2.0 / sizeof(data_t),
            (WSS_BYTES * 2.0) / result.min_time / 1e9,
            (WSS_ELEMS * FLOPS_PER_ELEM) / result.min_time / 1e9);

    // Done, free allocated memory.
    free(a);

    return 0;
}
