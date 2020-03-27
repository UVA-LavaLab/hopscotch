/******************************************************************************
 *
 * File: rw_seq_copy.cpp
 * Description: Kernel for sequentially copying one array to another.
 * 
 * Author: Alif Ahmed
 * Date: Sep 16, 2019
 *
 *****************************************************************************/
#include "../include/common.h"
#include <cstdint>

void rw_seq_copy(data_t* __restrict__ a, data_t* __restrict__ b) {
//#pragma omp parallel for simd aligned(a,b)
	for (uint64_t i = 0; i < WSS_ELEMS; i = i + 16) {
		a[i + 0] = b[i + 0];
		a[i + 1] = b[i + 1];
		a[i + 2] = b[i + 2];
		a[i + 3] = b[i + 3];
		a[i + 4] = b[i + 4];
		a[i + 5] = b[i + 5];
		a[i + 6] = b[i + 6];
		a[i + 7] = b[i + 7];
		a[i + 8] = b[i + 8];
		a[i + 9] = b[i + 9];
		a[i + 10] = b[i + 10];
		a[i + 11] = b[i + 11];
		a[i + 12] = b[i + 12];
		a[i + 13] = b[i + 13];
		a[i + 14] = b[i + 14];
		a[i + 15] = b[i + 15];
//		a[i + 16] = b[i + 16];
//		a[i + 17] = b[i + 17];
//		a[i + 18] = b[i + 18];
//		a[i + 19] = b[i + 19];
//		a[i + 20] = b[i + 20];
//		a[i + 21] = b[i + 21];
//		a[i + 22] = b[i + 22];
//		a[i + 23] = b[i + 23];
//		a[i + 24] = b[i + 24];
//		a[i + 25] = b[i + 25];
//		a[i + 26] = b[i + 26];
//		a[i + 27] = b[i + 27];
//		a[i + 28] = b[i + 28];
//		a[i + 29] = b[i + 29];
//		a[i + 30] = b[i + 30];
//		a[i + 31] = b[i + 31];
	}
}

res_t run_rw_seq_copy(double allowed_time, data_t* a, data_t* b) {
	res_t result;
	run_kernel(rw_seq_copy(a, b), allowed_time, result);
	result.bytes_read = result.iters * WSS_ELEMS * sizeof(data_t);
	result.bytes_write = result.iters * WSS_ELEMS * sizeof(data_t);
	return result;
}
