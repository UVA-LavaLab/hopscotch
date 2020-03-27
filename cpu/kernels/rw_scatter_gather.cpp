/******************************************************************************
 *
 * File: rw_scatter_gather.cpp
 * Description: Kernel with scatter-gather pattern.
 * 
 * Author: Alif Ahmed
 * Date: Sep 16, 2019
 *
 *****************************************************************************/
#include "../include/common.h"
#include <cstdint>

void rw_scatter_gather(data_t* __restrict__ a, data_t* __restrict__ b, uint64_t* __restrict__ idx1, uint64_t* __restrict__ idx2) {
//#pragma omp parallel for simd aligned(a,b,idx1,idx2)
	for (uint64_t i = 1; i < WSS_ELEMS; i = i + 16) {
		a[idx1[i + 0]] = b[idx2[i + 0]];
		a[idx1[i + 1]] = b[idx2[i + 1]];
		a[idx1[i + 2]] = b[idx2[i + 2]];
		a[idx1[i + 3]] = b[idx2[i + 3]];
		a[idx1[i + 4]] = b[idx2[i + 4]];
		a[idx1[i + 5]] = b[idx2[i + 5]];
		a[idx1[i + 6]] = b[idx2[i + 6]];
		a[idx1[i + 7]] = b[idx2[i + 7]];
		a[idx1[i + 8]] = b[idx2[i + 8]];
		a[idx1[i + 9]] = b[idx2[i + 9]];
		a[idx1[i + 10]] = b[idx2[i + 10]];
		a[idx1[i + 11]] = b[idx2[i + 11]];
		a[idx1[i + 12]] = b[idx2[i + 12]];
		a[idx1[i + 13]] = b[idx2[i + 13]];
		a[idx1[i + 14]] = b[idx2[i + 14]];
		a[idx1[i + 15]] = b[idx2[i + 15]];
//		a[idx1[i + 16]] = b[idx2[i + 16]];
//		a[idx1[i + 17]] = b[idx2[i + 17]];
//		a[idx1[i + 18]] = b[idx2[i + 18]];
//		a[idx1[i + 19]] = b[idx2[i + 19]];
//		a[idx1[i + 20]] = b[idx2[i + 20]];
//		a[idx1[i + 21]] = b[idx2[i + 21]];
//		a[idx1[i + 22]] = b[idx2[i + 22]];
//		a[idx1[i + 23]] = b[idx2[i + 23]];
//		a[idx1[i + 24]] = b[idx2[i + 24]];
//		a[idx1[i + 25]] = b[idx2[i + 25]];
//		a[idx1[i + 26]] = b[idx2[i + 26]];
//		a[idx1[i + 27]] = b[idx2[i + 27]];
//		a[idx1[i + 28]] = b[idx2[i + 28]];
//		a[idx1[i + 29]] = b[idx2[i + 29]];
//		a[idx1[i + 30]] = b[idx2[i + 30]];
//		a[idx1[i + 31]] = b[idx2[i + 31]];
	}
}

res_t run_rw_scatter_gather(double allowed_time, data_t* a, data_t* b, uint64_t* idx1, uint64_t* idx2) {
	res_t result;
	run_kernel(rw_scatter_gather(a, b, idx1, idx2), allowed_time, result);
	result.bytes_read = result.iters * WSS_ELEMS * (sizeof(data_t) + 16);   // read = idx1 + idx2 + b
	result.bytes_write = result.iters * WSS_ELEMS * sizeof(data_t);         // write = a
	return result;
}
