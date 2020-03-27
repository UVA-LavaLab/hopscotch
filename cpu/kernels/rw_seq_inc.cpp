/******************************************************************************
 *
 * File: rw_seq_inc.cpp
 * Description: Kernel with sequential mixed access increment.
 * 
 * Author: Alif Ahmed
 * Date: Sep 16, 2019
 *
 *****************************************************************************/
#include "../include/common.h"
#include <cstdint>

void rw_seq_inc(data_t* __restrict__ a) {
//#pragma omp parallel for simd aligned(a)
	for (uint64_t i = 0; i < WSS_ELEMS; i = i + 16) {
		a[i + 0]++;
		a[i + 1]++;
		a[i + 2]++;
		a[i + 3]++;
		a[i + 4]++;
		a[i + 5]++;
		a[i + 6]++;
		a[i + 7]++;
		a[i + 8]++;
		a[i + 9]++;
		a[i + 10]++;
		a[i + 11]++;
		a[i + 12]++;
		a[i + 13]++;
		a[i + 14]++;
		a[i + 15]++;
//		a[i + 16]++;
//		a[i + 17]++;
//		a[i + 18]++;
//		a[i + 19]++;
//		a[i + 20]++;
//		a[i + 21]++;
//		a[i + 22]++;
//		a[i + 23]++;
//		a[i + 24]++;
//		a[i + 25]++;
//		a[i + 26]++;
//		a[i + 27]++;
//		a[i + 28]++;
//		a[i + 29]++;
//		a[i + 30]++;
//		a[i + 31]++;
	}
}

res_t run_rw_seq_inc(double allowed_time, data_t* a) {
	res_t result;
	run_kernel(rw_seq_inc(a), allowed_time, result);
	result.bytes_read = result.iters * WSS_ELEMS * sizeof(data_t);
	result.bytes_write = result.iters * WSS_ELEMS * sizeof(data_t);
	return result;
}
