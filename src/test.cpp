#include "common.h"

int main(){
    double elapsed;
    uint64_t iter;
    alloc_arrays();
    print_bw_header();
    
    elapsed = kernel_min_time(&iter, rw_seq_rmw, init_a);
    print_bw("rw_seq_rmw", iter, elapsed, HS_ARRAY_SIZE_MB * 2 / elapsed);
    
    elapsed = kernel_min_time(&iter, rw_seq_memcpy, init_ab);
    print_bw("rw_seq_copy", iter, elapsed, HS_ARRAY_SIZE_MB * 2 / elapsed);
    
    return 0;
}