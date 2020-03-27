/******************************************************************************
 *
 * File: w_seq_fill.cpp
 * Description: Fill an array with a constant value.
 * 
 * Author: Alif Ahmed
 * Date: Sep 16, 2019
 *
 *****************************************************************************/
#include "../include/common.h"
#include <cstdint>

void w_seq_fill(data_t* __restrict__ a) {
//#pragma omp parallel for simd aligned (a)
	for (uint64_t i = 0; i < WSS_ELEMS; i = i + 16) {
		a[i + 0] = 7;
		a[i + 1] = 7;
		a[i + 2] = 7;
		a[i + 3] = 7;
		a[i + 4] = 7;
		a[i + 5] = 7;
		a[i + 6] = 7;
		a[i + 7] = 7;
		a[i + 8] = 7;
		a[i + 9] = 7;
		a[i + 10] = 7;
		a[i + 11] = 7;
		a[i + 12] = 7;
		a[i + 13] = 7;
		a[i + 14] = 7;
		a[i + 15] = 7;
//		a[i + 16] = 7;
//		a[i + 17] = 7;
//		a[i + 18] = 7;
//		a[i + 19] = 7;
//		a[i + 20] = 7;
//		a[i + 21] = 7;
//		a[i + 22] = 7;
//		a[i + 23] = 7;
//		a[i + 24] = 7;
//		a[i + 25] = 7;
//		a[i + 26] = 7;
//		a[i + 27] = 7;
//		a[i + 28] = 7;
//		a[i + 29] = 7;
//		a[i + 30] = 7;
//		a[i + 31] = 7;
	}
}

res_t run_w_seq_fill(double allowed_time, data_t* a) {
	res_t result;
	run_kernel(w_seq_fill(a), allowed_time, result);
	result.bytes_read = 0;
	result.bytes_write = result.iters * WSS_ELEMS * sizeof(data_t);
	return result;
}
