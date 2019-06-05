#include "common.h"
#include "stdio.h"

int main(){
    print_thread_num();
    print_bw_header();
    double elapsed;
    uint64_t iter;
    
    /*elapsed = run_kernel(&iter, kern_seq_read, default_init);
    printf("Seq Read: Iterations: %lu\t\tElapsed time: %lf s\t\tRead BW: %0.2lf MB/s\n", iter, elapsed, HS_ARRAY_SIZE_MB * iter / elapsed);
    
    elapsed = run_kernel(&iter, kern_seq_reduction, default_init);
    printf("Seq Read (reduction): Iterations: %lu\t\tElapsed time: %lf s\t\tRead BW: %0.2lf MB/s\n", iter, elapsed, HS_ARRAY_SIZE_MB * iter / elapsed);
    
    elapsed = run_kernel(&iter, kern_stride2_read, default_init);
    printf("Stride 2: Iterations: %lu\t\tElapsed time: %lf s\t\tRead BW: %0.2lf MB/s\n", iter, elapsed, HS_ARRAY_SIZE_MB * iter / 2 / elapsed);
    
    elapsed = run_kernel(&iter, kern_stride4_read, default_init);
    printf("Stride 4: Iterations: %lu\t\tElapsed time: %lf s\t\tRead BW: %0.2lf MB/s\n", iter, elapsed, HS_ARRAY_SIZE_MB * iter / 4 / elapsed);
    
    elapsed = run_kernel(&iter, kern_stride8_read, default_init);
    printf("Stride 8: Iterations: %lu\t\tElapsed time: %lf s\t\tRead BW: %0.2lf MB/s\n", iter, elapsed, HS_ARRAY_SIZE_MB * iter / 8 / elapsed);
    
    elapsed = run_kernel(&iter, kern_stride16_read, default_init);
    printf("Stride 16: Iterations: %lu\t\tElapsed time: %lf s\t\tRead BW: %0.2lf MB/s\n", iter, elapsed, HS_ARRAY_SIZE_MB * iter / 16 / elapsed);
    
    elapsed = run_kernel(&iter, kern_stride32_read, default_init);
    printf("Stride 32: Iterations: %lu\t\tElapsed time: %lf s\t\tRead BW: %0.2lf MB/s\n", iter, elapsed, HS_ARRAY_SIZE_MB * iter / 32 / elapsed);
    
    elapsed = run_kernel(&iter, kern_read_pointer_chasing, init_pointer_chasing);
    printf("Pointer-Chasing: Iterations: %lu\t\tElapsed time: %lf s\t\tRead BW: %0.2lf MB/s\n", iter, elapsed, HS_ARRAY_SIZE_MB * iter / elapsed);
    
    elapsed = run_kernel(&iter, kern_rand_read, default_init);
    printf("Random Read: Iterations: %lu\t\tElapsed time: %lf s\t\tRead BW: %0.2lf MB/s\n", iter, elapsed, HS_ARRAY_SIZE_MB * iter / elapsed);*/
    
    elapsed = kernel_min_time(&iter, w_seq_fill, init_a);
    print_bw("w_seq_fill", iter, elapsed, HS_ARRAY_SIZE_MB / elapsed);

    elapsed = kernel_min_time(&iter, w_seq_memset, init_a);
    print_bw("w_seq_memset", iter, elapsed, HS_ARRAY_SIZE_MB / elapsed);
    
    elapsed = kernel_min_time(&iter, w_rand_ind, init_a);
    print_bw("w_rand_ind", iter, elapsed, HS_ARRAY_SIZE_MB / elapsed);
    
	return 0;
}
