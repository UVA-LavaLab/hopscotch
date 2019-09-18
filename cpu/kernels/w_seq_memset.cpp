/******************************************************************************
 *
 * File: w_seq_memset.cpp
 * Description: Fill an array with memset (non-temporal store in most
 *              implementations).
 * 
 * Author: Alif Ahmed
 * Date: Sep 16, 2019
 *
 *****************************************************************************/
#include <common.h>
#include <cstring>


void w_seq_memset(data_t* __restrict__ a){
    memset(a, 7, WSS_BYTES);
}


res_t run_w_seq_memset(double allowed_time, data_t* a){
	res_t result;
	run_kernel(w_seq_memset(a), allowed_time, result);
	result.bytes_read = 0;
	result.bytes_write = result.iters * WSS_BYTES;
	return result;
}
