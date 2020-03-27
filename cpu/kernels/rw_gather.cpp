/******************************************************************************
 *
 * File: rw_gather.cpp
 * Description: Kernel with gather pattern.
 * 
 * Author: Alif Ahmed
 * Date: Sep 16, 2019
 *
 *****************************************************************************/
#include "../include/common.h"
#include <cstdint>

void rw_gather(data_t* __restrict__ a, data_t* __restrict__ b, uint64_t* __restrict__ idx) {
//#pragma omp parallel for simd aligned(a,b,idx)
	for (uint64_t i = 1; i < WSS_ELEMS; i = i + 16) {
		a[i + 0] = b[idx[i + 0]];
		a[i + 1] = b[idx[i + 1]];
		a[i + 2] = b[idx[i + 2]];
		a[i + 3] = b[idx[i + 3]];
		a[i + 4] = b[idx[i + 4]];
		a[i + 5] = b[idx[i + 5]];
		a[i + 6] = b[idx[i + 6]];
		a[i + 7] = b[idx[i + 7]];
		a[i + 8] = b[idx[i + 8]];
		a[i + 9] = b[idx[i + 9]];
		a[i + 10] = b[idx[i + 10]];
		a[i + 11] = b[idx[i + 11]];
		a[i + 12] = b[idx[i + 12]];
		a[i + 13] = b[idx[i + 13]];
		a[i + 14] = b[idx[i + 14]];
		a[i + 15] = b[idx[i + 15]];
//		a[i + 16] = b[idx[i + 16]];
//		a[i + 17] = b[idx[i + 17]];
//		a[i + 18] = b[idx[i + 18]];
//		a[i + 19] = b[idx[i + 19]];
//		a[i + 20] = b[idx[i + 20]];
//		a[i + 21] = b[idx[i + 21]];
//		a[i + 22] = b[idx[i + 22]];
//		a[i + 23] = b[idx[i + 23]];
//		a[i + 24] = b[idx[i + 24]];
//		a[i + 25] = b[idx[i + 25]];
//		a[i + 26] = b[idx[i + 26]];
//		a[i + 27] = b[idx[i + 27]];
//		a[i + 28] = b[idx[i + 28]];
//		a[i + 29] = b[idx[i + 29]];
//		a[i + 30] = b[idx[i + 30]];
//		a[i + 31] = b[idx[i + 31]];
	}
}

res_t run_rw_gather(double allowed_time, data_t* a, data_t* b, uint64_t* idx) {
	res_t result;
	run_kernel(rw_gather(a, b, idx), allowed_time, result);
	result.bytes_read = result.iters * WSS_ELEMS * (sizeof(data_t) + 8);    // read = idx + b
	result.bytes_write = result.iters * WSS_ELEMS * sizeof(data_t);         // write = a
	return result;
}
