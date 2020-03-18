/******************************************************************************
 *
 * File: hs_bandwidth.cpp
 * Description: Measures bandwidth with different types of access patterns.
 * 
 * Author: Alif Ahmed
 * Date: Aug 06, 2019
 *
 *****************************************************************************/
#include "common.h"
#include <cstdlib>

// Stride kernels use template. Must include to use.
#include "../kernels/r_stride.cpp"
#include "../kernels/w_stride.cpp"


// allowed minimum runtime of each kernel (in seconds)
#define ALLOWED_RUNTIME     2

int main(){
    res_t result;

    // print result header
    print_bw_header();

    // allocate memory for working sets and indexes
    data_t* a = (data_t*)hs_alloc(WSS_ELEMS * sizeof(data_t));
    data_t* b = (data_t*)hs_alloc(WSS_ELEMS * sizeof(data_t));
    uint64_t* idx1 = (uint64_t*)hs_alloc(WSS_ELEMS * sizeof(uint64_t));
    uint64_t* idx2 = (uint64_t*)hs_alloc(WSS_ELEMS * sizeof(uint64_t));

    // initialize arrays
    init_const(a, WSS_ELEMS, 3);
    init_const(b, WSS_ELEMS, 5);
    init_linear(idx1, WSS_ELEMS, true);
    init_linear(idx2, WSS_ELEMS, true);

    // read kernels
    result = run_r_seq_ind(ALLOWED_RUNTIME, a);
    print_max_bw("r_seq_ind", result);
    
    result = run_r_seq_reduce(ALLOWED_RUNTIME, a);
    print_max_bw("r_seq_reduce", result);

    result = run_r_rand_ind(ALLOWED_RUNTIME, a);
    print_max_bw("r_rand_ind", result); 

    result = run_r_stride<2>(ALLOWED_RUNTIME, a);
    print_max_bw("r_stride_2", result);

    result = run_r_stride<4>(ALLOWED_RUNTIME, a);
    print_max_bw("r_stride_4", result);

    result = run_r_stride<8>(ALLOWED_RUNTIME, a);
    print_max_bw("r_stride_8", result);

    result = run_r_stride<16>(ALLOWED_RUNTIME, a);
    print_max_bw("r_stride_16", result);

    result = run_r_stride<32>(ALLOWED_RUNTIME, a);
    print_max_bw("r_stride_32", result);
    
    
    // write kernels
    result = run_w_seq_fill(ALLOWED_RUNTIME, a);
    print_max_bw("w_seq_fill", result);

    result = run_w_seq_memset(ALLOWED_RUNTIME, a);
    print_max_bw("w_seq_memset", result);

    result = run_w_rand_ind(ALLOWED_RUNTIME, a);
    print_max_bw("w_rand_ind", result);

    result = run_w_stride<2>(ALLOWED_RUNTIME, a);
    print_max_bw("w_stride_2", result);

    result = run_w_stride<4>(ALLOWED_RUNTIME, a);
    print_max_bw("w_stride_4", result);

    result = run_w_stride<8>(ALLOWED_RUNTIME, a);
    print_max_bw("w_stride_8", result);

    result = run_w_stride<16>(ALLOWED_RUNTIME, a);
    print_max_bw("w_stride_16", result);

    result = run_w_stride<32>(ALLOWED_RUNTIME, a);
    print_max_bw("w_stride_32", result);
 

    // mixed kernels
    result = run_rw_seq_copy(ALLOWED_RUNTIME, a, b);
    print_max_bw("rw_seq_copy", result);

    result = run_rw_seq_inc(ALLOWED_RUNTIME, a);
    print_max_bw("rw_seq_inc", result);

    result = run_rw_scatter(ALLOWED_RUNTIME, a, b, idx1);
    print_max_bw("rw_scatter", result);

    result = run_rw_gather(ALLOWED_RUNTIME, a, b, idx2);
    print_max_bw("rw_gather", result);

    result = run_rw_scatter_gather(ALLOWED_RUNTIME, a, b, idx1, idx2);
    print_max_bw("rw_scatter_gather", result);


    // deallocate
    free(a);
    free(b);
    free(idx1);
    free(idx2);
    
    return 0;
}
