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


/**
 * Initializes and runs kernel
 */
static double kernel_min_time_2(uint64_t* iter, double (*func)(uint64_t,uint64_t), void (*init)(), uint64_t arg1, uint64_t arg2) {
    init();
    func(arg1, arg2);             //warm up
    double elapsed = 0;
    double min = DBL_MAX;
    uint64_t i;
    for(i = 0; (i < ITER_MIN) || (elapsed < ITER_TIMEOUT_SEC); ++i) {
        double curr = func(arg1, arg2);
        min = curr < min ? curr : min;
        elapsed += curr;
    }
    *iter = i;
    return min;
}

int main(){
    double elapsed, min;
    uint64_t iter;
	uint64_t K, L;
	print_bw_header();
        
    alloc_a();

	//spatial locality = low
	//temporal locality = low
	L = 1;
	K = 32;
	elapsed = kernel_min_time_2(&iter, rw_tile, init_a, L, K);
    print_bw("Spatial=low, Temporal=low", iter, elapsed, HS_ARRAY_SIZE_MB * 2 * L / K / elapsed);
	
	
	//spatial locality = low
	//temporal locality = high
	L = 2;
	K = 1;
	elapsed = kernel_min_time_2(&iter, rw_tile, init_a, L, K);
    print_bw("Spatial=low, Temporal=high", iter, elapsed, HS_ARRAY_SIZE_MB * 2 * L / K / elapsed);


	//spatial locality = high
	//temporal locality = low
	L = HS_ARRAY_ELEM;
	K = HS_ARRAY_ELEM;
	elapsed = kernel_min_time_2(&iter, rw_tile, init_a, L, K);
    print_bw("Spatial=high, Temporal=low", iter, elapsed,  HS_ARRAY_SIZE_MB * 2 * L / K / elapsed);


	//spatial locality = high
	//temporal locality = high
	L = 32;
	K = 1;
	elapsed = kernel_min_time_2(&iter, rw_tile, init_a, L, K);
    print_bw("Spatial=high, Temporal=high", iter, elapsed,  HS_ARRAY_SIZE_MB * 2 * L / K / elapsed);


	free_a();

	return 0;
}
