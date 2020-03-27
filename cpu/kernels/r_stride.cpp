/******************************************************************************
 *
 * File: r_stride.cpp
 * Description: Strided read. Since it is a template, it should be included in
 *              the benchmark source.
 * 
 * Author: Alif Ahmed
 * Date: Sep 16, 2019
 *
 *****************************************************************************/
#include "../include/common.h"
#include <cstdint>

template<uint64_t stride>
void r_stride(data_t* __restrict__ a) {
	volatile data_t * vol_a = a;
//#pragma omp parallel for simd aligned (vol_a)
	for (uint64_t i = 0; i < WSS_ELEMS; i += 16 * stride) {
		data_t res00 = vol_a[i + 0 * stride];
		data_t res01 = vol_a[i + 1 * stride];
		data_t res02 = vol_a[i + 2 * stride];
		data_t res03 = vol_a[i + 3 * stride];
		data_t res04 = vol_a[i + 4 * stride];
		data_t res05 = vol_a[i + 5 * stride];
		data_t res06 = vol_a[i + 6 * stride];
		data_t res07 = vol_a[i + 7 * stride];
		data_t res08 = vol_a[i + 8 * stride];
		data_t res09 = vol_a[i + 9 * stride];
		data_t res10 = vol_a[i + 10 * stride];
		data_t res11 = vol_a[i + 11 * stride];
		data_t res12 = vol_a[i + 12 * stride];
		data_t res13 = vol_a[i + 13 * stride];
		data_t res14 = vol_a[i + 14 * stride];
		data_t res15 = vol_a[i + 15 * stride];
//		data_t res16 = vol_a[i + 16];
//		data_t res17 = vol_a[i + 17];
//		data_t res18 = vol_a[i + 18];
//		data_t res19 = vol_a[i + 19];
//		data_t res20 = vol_a[i + 20];
//		data_t res21 = vol_a[i + 21];
//		data_t res22 = vol_a[i + 22];
//		data_t res23 = vol_a[i + 23];
//		data_t res24 = vol_a[i + 24];
//		data_t res25 = vol_a[i + 25];
//		data_t res26 = vol_a[i + 26];
//		data_t res27 = vol_a[i + 27];
//		data_t res28 = vol_a[i + 28];
//		data_t res29 = vol_a[i + 29];
//		data_t res30 = vol_a[i + 30];
//		data_t res31 = vol_a[i + 31];
	}
}

template<uint64_t stride>
res_t run_r_stride(double allowed_time, data_t* a) {
	res_t result;
	run_kernel(r_stride<stride>(a), allowed_time, result);
	result.bytes_read = result.iters * WSS_ELEMS * sizeof(data_t) / stride;
	result.bytes_write = 0;
	return result;
}

