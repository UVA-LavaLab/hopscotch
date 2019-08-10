/*******************************************************************************
 *
 * File: common.cu
 * Description: Contains definitions of common utility functions.
 * 
 * Author: Alif Ahmed
 * Date: Aug 06, 2019
 *
 ******************************************************************************/
#include "common.h"
#include <string>
#include <iostream>

using namespace std;

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err) {
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}
