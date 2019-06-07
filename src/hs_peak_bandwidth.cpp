#include "common.h"

int main(){
    double elapsed;
    uint64_t iter;
    alloc_a();
    
    print_bw_header();
    
    elapsed = kernel_min_time(&iter, r_seq_ind, init_a);
    print_bw("r_seq_ind", iter, elapsed, HS_ARRAY_SIZE_MB / elapsed);
    
    elapsed = kernel_min_time(&iter, r_seq_reduce, init_a);
    print_bw("r_seq_reduce", iter, elapsed, HS_ARRAY_SIZE_MB / elapsed);
    
    elapsed = kernel_min_time(&iter, w_seq_memset, init_a);
    print_bw("w_seq_memset", iter, elapsed, HS_ARRAY_SIZE_MB / elapsed);
    
    elapsed = kernel_min_time(&iter, w_seq_fill, init_a);
    print_bw("w_seq_fill", iter, elapsed, HS_ARRAY_SIZE_MB  / elapsed);
    
    alloc_b();
    
    elapsed = kernel_min_time(&iter, rw_seq_memcpy, init_ab);
    print_bw("rw_seq_memcpy", iter, elapsed, HS_ARRAY_SIZE_MB * 2 / elapsed);
    
    elapsed = kernel_min_time(&iter, rw_seq_copy, init_ab);
    print_bw("rw_seq_copy", iter, elapsed, HS_ARRAY_SIZE_MB * 2 / elapsed);
    
    free_a();
    free_b();
    
    return 0;
}