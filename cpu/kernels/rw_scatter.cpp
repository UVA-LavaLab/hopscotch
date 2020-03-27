/******************************************************************************
 *
 * File: rw_scatter.cpp
 * Description: Kernel with scatter pattern.
 * 
 * Author: Alif Ahmed
 * Date: Sep 16, 2019
 *
 *****************************************************************************/
#include "../include/common.h"
#include <cstdint>

void rw_scatter(data_t* __restrict__ a, data_t* __restrict__ b, uint64_t* __restrict__ idx) {
//#pragma omp parallel for simd aligned(a,b,idx)
	for (uint64_t i = 1; i < WSS_ELEMS; i = i + 16) {
		a[idx[i + 0]] = b[i + 0];
		a[idx[i + 1]] = b[i + 1];
		a[idx[i + 2]] = b[i + 2];
		a[idx[i + 3]] = b[i + 3];
		a[idx[i + 4]] = b[i + 4];
		a[idx[i + 5]] = b[i + 5];
		a[idx[i + 6]] = b[i + 6];
		a[idx[i + 7]] = b[i + 7];
		a[idx[i + 8]] = b[i + 8];
		a[idx[i + 9]] = b[i + 9];
		a[idx[i + 10]] = b[i + 10];
		a[idx[i + 11]] = b[i + 11];
		a[idx[i + 12]] = b[i + 12];
		a[idx[i + 13]] = b[i + 13];
		a[idx[i + 14]] = b[i + 14];
		a[idx[i + 15]] = b[i + 15];
//		a[idx[i + 16]] = b[i + 16];
//		a[idx[i + 17]] = b[i + 17];
//		a[idx[i + 18]] = b[i + 18];
//		a[idx[i + 19]] = b[i + 19];
//		a[idx[i + 20]] = b[i + 20];
//		a[idx[i + 21]] = b[i + 21];
//		a[idx[i + 22]] = b[i + 22];
//		a[idx[i + 23]] = b[i + 23];
//		a[idx[i + 24]] = b[i + 24];
//		a[idx[i + 25]] = b[i + 25];
//		a[idx[i + 26]] = b[i + 26];
//		a[idx[i + 27]] = b[i + 27];
//		a[idx[i + 28]] = b[i + 28];
//		a[idx[i + 29]] = b[i + 29];
//		a[idx[i + 30]] = b[i + 30];
//		a[idx[i + 31]] = b[i + 31];
	}
}

res_t run_rw_scatter(double allowed_time, data_t* a, data_t* b, uint64_t* idx) {
	res_t result;
	run_kernel(rw_scatter(a, b, idx), allowed_time, result);
	result.bytes_read = result.iters * WSS_ELEMS * (sizeof(data_t) + 8);    // read = idx + b
	result.bytes_write = result.iters * WSS_ELEMS * sizeof(data_t);         // write = a
	return result;
}
