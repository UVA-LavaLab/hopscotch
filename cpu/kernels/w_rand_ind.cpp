/******************************************************************************
 *
 * File: w_rand_ind.cpp
 * Description: Randomly fill an array with a constant value. The random index
 *              is generated within loop.
 * 
 * Author: Alif Ahmed
 * Date: Sep 16, 2019
 *
 *****************************************************************************/
#include "../include/common.h"
#include <cstdint>

void w_rand_ind(data_t* __restrict__ a) {
//#pragma omp parallel for simd aligned (a)
	for (uint64_t i = 0; i < WSS_ELEMS; i = i + 16) {
		uint64_t idx0 = (((i + 0) * 0xDEADBEEF) ^ 0xC0FFEE0B) % WSS_ELEMS;
		uint64_t idx1 = (((i + 1) * 0xDEADBEEF) ^ 0xC0FFEE0B) % WSS_ELEMS;
		uint64_t idx2 = (((i + 2) * 0xDEADBEEF) ^ 0xC0FFEE0B) % WSS_ELEMS;
		uint64_t idx3 = (((i + 3) * 0xDEADBEEF) ^ 0xC0FFEE0B) % WSS_ELEMS;
		uint64_t idx4 = (((i + 4) * 0xDEADBEEF) ^ 0xC0FFEE0B) % WSS_ELEMS;
		uint64_t idx5 = (((i + 5) * 0xDEADBEEF) ^ 0xC0FFEE0B) % WSS_ELEMS;
		uint64_t idx6 = (((i + 6) * 0xDEADBEEF) ^ 0xC0FFEE0B) % WSS_ELEMS;
		uint64_t idx7 = (((i + 7) * 0xDEADBEEF) ^ 0xC0FFEE0B) % WSS_ELEMS;
		uint64_t idx8 = (((i + 8) * 0xDEADBEEF) ^ 0xC0FFEE0B) % WSS_ELEMS;
		uint64_t idx9 = (((i + 9) * 0xDEADBEEF) ^ 0xC0FFEE0B) % WSS_ELEMS;
		uint64_t idx10 = (((i + 10) * 0xDEADBEEF) ^ 0xC0FFEE0B) % WSS_ELEMS;
		uint64_t idx11 = (((i + 11) * 0xDEADBEEF) ^ 0xC0FFEE0B) % WSS_ELEMS;
		uint64_t idx12 = (((i + 12) * 0xDEADBEEF) ^ 0xC0FFEE0B) % WSS_ELEMS;
		uint64_t idx13 = (((i + 13) * 0xDEADBEEF) ^ 0xC0FFEE0B) % WSS_ELEMS;
		uint64_t idx14 = (((i + 14) * 0xDEADBEEF) ^ 0xC0FFEE0B) % WSS_ELEMS;
		uint64_t idx15 = (((i + 15) * 0xDEADBEEF) ^ 0xC0FFEE0B) % WSS_ELEMS;
		a[idx0] = 7;
		a[idx1] = 7;
		a[idx2] = 7;
		a[idx3] = 7;
		a[idx4] = 7;
		a[idx5] = 7;
		a[idx6] = 7;
		a[idx7] = 7;
		a[idx8] = 7;
		a[idx9] = 7;
		a[idx10] = 7;
		a[idx11] = 7;
		a[idx12] = 7;
		a[idx13] = 7;
		a[idx14] = 7;
		a[idx15] = 7;
//		a[idx + 16] = 7;
//		a[idx + 17] = 7;
//		a[idx + 18] = 7;
//		a[idx + 19] = 7;
//		a[idx + 20] = 7;
//		a[idx + 21] = 7;
//		a[idx + 22] = 7;
//		a[idx + 23] = 7;
//		a[idx + 24] = 7;
//		a[idx + 25] = 7;
//		a[idx + 26] = 7;
//		a[idx + 27] = 7;
//		a[idx + 28] = 7;
//		a[idx + 29] = 7;
//		a[idx + 30] = 7;
//		a[idx + 31] = 7;
	}
}

res_t run_w_rand_ind(double allowed_time, data_t* a) {
	res_t result;
	run_kernel(w_rand_ind(a), allowed_time, result);
	result.bytes_read = 0;
	result.bytes_write = result.iters * WSS_ELEMS * sizeof(data_t);
	;
	return result;
}
