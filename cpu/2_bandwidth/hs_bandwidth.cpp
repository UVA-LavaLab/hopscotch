#include "common.h"
#include "stdio.h"

int main(){
    double elapsed;
    uint64_t iter;
    print_bw_header();
    
    
    alloc_a();
     
    //read kernels
    elapsed = kernel_min_time(&iter, r_seq_ind, init_a);
    print_bw("r_seq_ind", iter, elapsed, HS_ARRAY_SIZE_MB / elapsed);
    
    elapsed = kernel_min_time(&iter, r_seq_reduce, init_a);
    print_bw("r_seq_reduce", iter, elapsed, HS_ARRAY_SIZE_MB / elapsed);
    
    elapsed = kernel_min_time(&iter, r_rand_ind, init_a);
    print_bw("r_rand_ind", iter, elapsed, HS_ARRAY_SIZE_MB / elapsed);
    
    elapsed = kernel_min_time(&iter, r_stride_2, init_a);
    print_bw("r_stride_2", iter, elapsed, HS_ARRAY_SIZE_MB / 2 / elapsed);
    
    elapsed = kernel_min_time(&iter, r_stride_4, init_a);
    print_bw("r_stride_4", iter, elapsed, HS_ARRAY_SIZE_MB / 4 / elapsed);
    
    elapsed = kernel_min_time(&iter, r_stride_8, init_a);
    print_bw("r_stride_8", iter, elapsed, HS_ARRAY_SIZE_MB / 8 / elapsed);
    
    elapsed = kernel_min_time(&iter, r_stride_16, init_a);
    print_bw("r_stride_16", iter, elapsed, HS_ARRAY_SIZE_MB / 16 / elapsed);
    
    elapsed = kernel_min_time(&iter, r_stride_32, init_a);
    print_bw("r_stride_32", iter, elapsed, HS_ARRAY_SIZE_MB / 32 / elapsed);
    
	//write kernels
    elapsed = kernel_min_time(&iter, w_seq_fill, init_a);
    print_bw("w_seq_fill", iter, elapsed, HS_ARRAY_SIZE_MB / elapsed);

    elapsed = kernel_min_time(&iter, w_seq_memset, init_a);
    print_bw("w_seq_memset", iter, elapsed, HS_ARRAY_SIZE_MB / elapsed);
    
    elapsed = kernel_min_time(&iter, w_rand_ind, init_a);
    print_bw("w_rand_ind", iter, elapsed, HS_ARRAY_SIZE_MB / elapsed);
    
    elapsed = kernel_min_time(&iter, w_stride_2, init_a);
    print_bw("w_stride_2", iter, elapsed, HS_ARRAY_SIZE_MB / 2 / elapsed);
    
    elapsed = kernel_min_time(&iter, w_stride_4, init_a);
    print_bw("w_stride_4", iter, elapsed, HS_ARRAY_SIZE_MB / 4 / elapsed);
    
    elapsed = kernel_min_time(&iter, w_stride_8, init_a);
    print_bw("w_stride_8", iter, elapsed, HS_ARRAY_SIZE_MB / 8 / elapsed);
    
    elapsed = kernel_min_time(&iter, w_stride_16, init_a);
    print_bw("w_stride_16", iter, elapsed, HS_ARRAY_SIZE_MB / 16 / elapsed);
    
    elapsed = kernel_min_time(&iter, w_stride_32, init_a);
    print_bw("w_stride_32", iter, elapsed, HS_ARRAY_SIZE_MB / 32 / elapsed);
    

	//mixed kernels
    alloc_b();
    
    elapsed = kernel_min_time(&iter, rw_seq_copy, init_ab);
    print_bw("rw_seq_copy", iter, elapsed, HS_ARRAY_SIZE_MB * 2 / elapsed);
    
    elapsed = kernel_min_time(&iter, rw_seq_inc, init_a);
    print_bw("rw_seq_inc", iter, elapsed, HS_ARRAY_SIZE_MB * 2 / elapsed);

    alloc_idx1();
    
    init_ab();
    init_idx1();
    elapsed = kernel_min_time(&iter, rw_scatter, no_init);
    print_bw("rw_scatter", iter, elapsed, HS_ARRAY_SIZE_MB * 3 / elapsed);
    
    init_ab();
    init_idx1();
    elapsed = kernel_min_time(&iter, rw_gather, no_init);
    print_bw("rw_gather", iter, elapsed, HS_ARRAY_SIZE_MB * 3 / elapsed);
    
    alloc_idx2();
    init_ab();
    init_idx12();
    elapsed = kernel_min_time(&iter, rw_scatter_gather, no_init);
    print_bw("rw_scatter_gather", iter, elapsed, HS_ARRAY_SIZE_MB * 4 / elapsed);
    
    free_a();
    free_b();
    free_idx1();
    free_idx2();
    
	return 0;
}
