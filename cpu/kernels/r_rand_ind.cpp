/******************************************************************************
 *
 * File: r_rand_ind.cpp
 * Description: Random read without loop-carried dependency
 * 
 * Author: Alif Ahmed
 * Date: Sep 16, 2019
 *
 *****************************************************************************/
#include "../include/common.h"
#include <cstdint>

void r_rand_ind(data_t* __restrict__ a) {
	volatile data_t * vol_a = a;
//#pragma omp parallel for simd aligned (vol_a)
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
		data_t res00 = vol_a[idx0];
		data_t res01 = vol_a[idx1];
		data_t res02 = vol_a[idx2];
		data_t res03 = vol_a[idx3];
		data_t res04 = vol_a[idx4];
		data_t res05 = vol_a[idx5];
		data_t res06 = vol_a[idx6];
		data_t res07 = vol_a[idx7];
		data_t res08 = vol_a[idx8];
		data_t res09 = vol_a[idx9];
		data_t res10 = vol_a[idx10];
		data_t res11 = vol_a[idx11];
		data_t res12 = vol_a[idx12];
		data_t res13 = vol_a[idx13];
		data_t res14 = vol_a[idx14];
		data_t res15 = vol_a[idx15];
//		data_t res16 = vol_a[idx + 16];
//		data_t res17 = vol_a[idx + 17];
//		data_t res18 = vol_a[idx + 18];
//		data_t res19 = vol_a[idx + 19];
//		data_t res20 = vol_a[idx + 20];
//		data_t res21 = vol_a[idx + 21];
//		data_t res22 = vol_a[idx + 22];
//		data_t res23 = vol_a[idx + 23];
//		data_t res24 = vol_a[idx + 24];
//		data_t res25 = vol_a[idx + 25];
//		data_t res26 = vol_a[idx + 26];
//		data_t res27 = vol_a[idx + 27];
//		data_t res28 = vol_a[idx + 28];
//		data_t res29 = vol_a[idx + 29];
//		data_t res30 = vol_a[idx + 30];
//		data_t res31 = vol_a[idx + 31];
	}
}

res_t run_r_rand_ind(double allowed_time, data_t* a) {
	res_t result;
	run_kernel(r_rand_ind(a), allowed_time, result);
	result.bytes_read = result.iters * WSS_ELEMS * sizeof(data_t);
	result.bytes_write = 0;
	return result;
}
