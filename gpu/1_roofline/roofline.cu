/*******************************************************************************
 *
 * File: roofline.cu
 * Description: Contains CUDA roofline kernel. roofline.py compiles it for 
 *              different FLOPs and DATA_T.
 * 
 * Author: Alif Ahmed
 * Date: Aug 06, 2019
 *
 ******************************************************************************/

#include "common.h"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cfloat>

//Data type. It is complied for float and double by roofline.py.
#ifndef DATA_T
#define DATA_T 					float
#endif

// Number of tries. Maximum performance is taken among the tries.
#ifndef NTRIES
#define NTRIES					10
#endif

// Array size in byte
#ifndef HS_ARRAY_SIZE_BYTE
#define HS_ARRAY_SIZE_BYTE		(1024UL*1024*1024)
#endif

// FLOPs per array element
#ifndef FLOPS_PER_ELEM
#define FLOPS_PER_ELEM			8192
#endif

// GPU Device
#ifndef HS_DEVICE
#define HS_DEVICE				0
#endif

// Number of elements in the working set
#define HS_ARRAY_ELEM			(HS_ARRAY_SIZE_BYTE/sizeof(DATA_T))


/**
 * Roofline kernel.
 */
__global__
void roofline_kernel(DATA_T* __restrict__ a){
	const DATA_T p = 1e-6;
	const DATA_T q = 1.0 + p;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// Read an array element
	DATA_T val = a[idx];
	
	// Do fused multiply accumulate operation on the 
    // element FLOPS_PER_ELEM/2 times
	for(int i = 0; i < (FLOPS_PER_ELEM/2); ++i) {
		val = val * q + p;
	}

	// If FLOPS_PER_ELEM is odd, do another operation
#if (FLOPS_PER_ELEM & 1) == 1
	val = val + p;
#endif

	// Store the new value
	a[idx] = val;
}



int main(int argc, char** argv) {
	DATA_T* dataHost;
	DATA_T* dataDev;
	cudaEvent_t start, end;
	float elapsedTime;
	float minTime = FLT_MAX;
	int grid, block;

	// Select CUDA device in multi-GPU environment
	CUDA_CHECK_RETURN(cudaSetDevice(HS_DEVICE));

	// Allocate host and device memory
	CUDA_CHECK_RETURN(cudaMallocHost(&dataHost, HS_ARRAY_SIZE_BYTE));
	CUDA_CHECK_RETURN(cudaMalloc(&dataDev, HS_ARRAY_SIZE_BYTE));

	// Create events for time measurement
	CUDA_CHECK_RETURN(cudaEventCreate(&start));
	CUDA_CHECK_RETURN(cudaEventCreate(&end));

	// Initialize host memory
	for(int64_t i = 0; i < HS_ARRAY_ELEM; ++i){
		dataHost[i] = 1.0;
	}

	// Use CUDA Runtime API to get optimum block size
	CUDA_CHECK_RETURN(cudaOccupancyMaxPotentialBlockSize(&grid, &block, roofline_kernel, 0, 0));

	// Copy host memory to device
	CUDA_CHECK_RETURN(cudaMemcpy(dataDev, dataHost, HS_ARRAY_SIZE_BYTE, cudaMemcpyHostToDevice));

	// Run the kernel NTRIES times, and take the best time
	for(int i = 0; i < NTRIES; ++i){
		// Start timer
		CUDA_CHECK_RETURN(cudaEventRecord(start));

		// Launch kernel. HS_ARRAY_ELEM is set to values by roofline.py such that
		// it is a multiple of block.
		roofline_kernel<<<HS_ARRAY_ELEM/block,block>>>(dataDev);

		// Stop timer
		CUDA_CHECK_RETURN(cudaEventRecord(end));

		// Calculate elapsed time
		CUDA_CHECK_RETURN(cudaEventSynchronize(end));
		CUDA_CHECK_RETURN(cudaEventElapsedTime(&elapsedTime, start, end));
		if(elapsedTime < minTime) {
			minTime = elapsedTime;
		}
	}

	// Copy results to host
	CUDA_CHECK_RETURN(cudaMemcpy(dataHost, dataDev, HS_ARRAY_SIZE_BYTE, cudaMemcpyDeviceToHost));

	// Free events that were used for profiling
	CUDA_CHECK_RETURN(cudaEventDestroy(start));
	CUDA_CHECK_RETURN(cudaEventDestroy(end));

	// Print result
	printf("%-8s%11d%17.2f%19.2f%19.2f\n",
			XSTR(DATA_T),
			FLOPS_PER_ELEM,
			FLOPS_PER_ELEM / 2.0f / sizeof(DATA_T),
			(HS_ARRAY_SIZE_BYTE * 2.0) / minTime / 1000.0 / 1000.0,
			(HS_ARRAY_ELEM * FLOPS_PER_ELEM) / minTime / 1000.0 / 1000.0);


	// Free allocated memory
	CUDA_CHECK_RETURN(cudaFreeHost(dataHost));
	CUDA_CHECK_RETURN(cudaFree(dataDev));

	return 0;
}

