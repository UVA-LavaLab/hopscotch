/******************************************************************************
 *
 * File: w_stride.cpp
 * Description: Strided write.
 * 
 * Author: Alif Ahmed
 * Date: Sep 16, 2019
 *
 *****************************************************************************/
#include "../include/common.h"
#include <cstdint>

template<uint64_t stride>
void w_stride(data_t* __restrict__ a) {
//#pragma omp parallel for simd aligned (a)
	for (uint64_t i = 0; i < WSS_ELEMS; i += 16 * stride) {
		a[i + 0 * stride] = 7;
		a[i + 1 * stride] = 7;
		a[i + 2 * stride] = 7;
		a[i + 3 * stride] = 7;
		a[i + 4 * stride] = 7;
		a[i + 5 * stride] = 7;
		a[i + 6 * stride] = 7;
		a[i + 7 * stride] = 7;
		a[i + 8 * stride] = 7;
		a[i + 9 * stride] = 7;
		a[i + 10 * stride] = 7;
		a[i + 11 * stride] = 7;
		a[i + 12 * stride] = 7;
		a[i + 13 * stride] = 7;
		a[i + 14 * stride] = 7;
		a[i + 15 * stride] = 7;
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

template<uint64_t stride>
res_t run_w_stride(double allowed_time, data_t* a) {
	res_t result;
	run_kernel(w_stride<stride>(a), allowed_time, result);
	result.bytes_read = 0;
	result.bytes_write = result.iters * WSS_ELEMS * sizeof(data_t) / stride;
	return result;
}

