/*
 ============================================================================
 Name        : roofline.cu
 Author      : Alif Ahmed
 Version     :
 Copyright   : 
 Description : Computes achievable Single/Double precision FLOPs
 ============================================================================
 */

#include "common.h"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cfloat>

#ifndef DATA_T
#define DATA_T 					float
#endif

#ifndef NTRIES
#define NTRIES					100
#endif

#ifndef HS_ARRAY_SIZE_BYTE
#define HS_ARRAY_SIZE_BYTE		(1024UL*1024*1024)
#endif

#ifndef FLOPS_PER_ELEM
#define FLOPS_PER_ELEM			8192
#endif

#ifndef HS_DEVICE
#define HS_DEVICE				0
#endif

#define HS_ARRAY_ELEM			(HS_ARRAY_SIZE_BYTE/sizeof(DATA_T))

__global__
void roofline_kernel(DATA_T* __restrict__ a){
	const DATA_T p = 1e-6;
	const DATA_T q = 1.0 + p;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	DATA_T val = a[idx];
	for(int i = 0; i < (FLOPS_PER_ELEM/2); ++i) {
		val = val * q + p;
	}
#if (FLOPS_PER_ELEM & 1) == 1
	val = val + p;
#endif
	a[idx] = val;
}

/*static void printDeviceInfo(int device){
	cudaDeviceProp prop;
	CUDA_CHECK_RETURN(cudaGetDeviceProperties(&prop, device));
	cout << "SELECTED [device " << device << "] " << prop.name << endl;
	cout << "  Total global memory: " 		<< prop.totalGlobalMem				<< endl;
	cout << "  Warp size: " 				<< prop.warpSize 					<< endl;
	cout << "  Max threads per block: "		<< prop.maxThreadsPerBlock			<< endl;
	cout << "  Clock rate (KHz): "			<< prop.clockRate					<< endl;
	cout << "  Compute capability: "		<< prop.major << "." << prop.minor 	<< endl;
	cout << "  SM: "						<< prop.multiProcessorCount			<< endl;
	cout << "  Memory clock rate (KHz): "	<< prop.memoryClockRate				<< endl;
	cout << "  Memory bus width: "			<< prop.memoryBusWidth				<< endl;
	cout << "  Max threads per SM: "		<< prop.maxThreadsPerMultiProcessor << endl;
	cout << "  L2 cache size: "				<< prop.l2CacheSize					<< endl;
	cout << "  Shared memory per block: " 	<< prop.sharedMemPerBlock			<< endl;
	cout << "  Shared memory per SM: " 		<< prop.sharedMemPerMultiprocessor	<< endl;
	cout << "  Single to double ratio: "	<< prop.singleToDoublePrecisionPerfRatio << endl;
	int grid, block;
	CUDA_CHECK_RETURN(cudaOccupancyMaxPotentialBlockSize(&grid, &block, roofline_kernel, 0, 0));
	cout << "Minimum grid size for max occupancy: " << grid << endl;
	cout << "Optimum block size: " << block << endl;
	cout << "Theoretical max performace: " << 128UL * prop.multiProcessorCount * 2 * prop.clockRate / 1000 / 1000 << " GFLOP/s" << endl;
}*/

int main(int argc, char** argv) {
	DATA_T* dataHost;
	DATA_T* dataDev;
	cudaEvent_t start, end;
	float elapsedTime[NTRIES];
//	int deviceNum;
	int grid, block;

	/*CUDA_CHECK_RETURN(cudaGetDeviceCount(&deviceCount));
	for(int i = 0; i < deviceCount; ++i) {
		cudaDeviceProp prop;
		CUDA_CHECK_RETURN(cudaGetDeviceProperties(&prop, i));
		cout << "[device " << i << "] " <<  prop.name << endl;
	}*/

	//printDeviceInfo(deviceNum);

	CUDA_CHECK_RETURN(cudaSetDevice(HS_DEVICE));

	CUDA_CHECK_RETURN(cudaMallocHost(&dataHost, HS_ARRAY_SIZE_BYTE));
	CUDA_CHECK_RETURN(cudaMalloc(&dataDev, HS_ARRAY_SIZE_BYTE));

	CUDA_CHECK_RETURN(cudaEventCreate(&start));
	CUDA_CHECK_RETURN(cudaEventCreate(&end));

	for(int64_t i = 0; i < HS_ARRAY_ELEM; ++i){
		dataHost[i] = 1.0;
	}

	CUDA_CHECK_RETURN(cudaOccupancyMaxPotentialBlockSize(&grid, &block, roofline_kernel, 0, 0));
	CUDA_CHECK_RETURN(cudaMemcpy(dataDev, dataHost, HS_ARRAY_SIZE_BYTE, cudaMemcpyHostToDevice));

	for(int i = 0; i < NTRIES; ++i){
		// Start timer
		CUDA_CHECK_RETURN(cudaEventRecord(start));

		// Launch kernel
		//roofline_kernel<<<HS_ARRAY_ELEM/THREAD_PER_BLOCK,THREAD_PER_BLOCK>>>(dataDev);
		roofline_kernel<<<HS_ARRAY_ELEM/block,block>>>(dataDev);

		// Stop timer
		CUDA_CHECK_RETURN(cudaEventRecord(end));

		// Check validity of results

		// Calculate elapsed time
		CUDA_CHECK_RETURN(cudaEventSynchronize(end));
		CUDA_CHECK_RETURN(cudaEventElapsedTime(&elapsedTime[i], start, end));
	}

	// Copy results to host
	CUDA_CHECK_RETURN(cudaMemcpy(dataHost, dataDev, HS_ARRAY_SIZE_BYTE, cudaMemcpyDeviceToHost));

	// Free events that were used for profiling
	CUDA_CHECK_RETURN(cudaEventDestroy(start));
	CUDA_CHECK_RETURN(cudaEventDestroy(end));

	float time_total = 0;
	float time_min = FLT_MAX;
	float time_max = 0;
	for(int i = 0; i < NTRIES; ++i){
		const float t = elapsedTime[i];
		time_total += t;
		if(t < time_min) time_min = t;
		if(t > time_max) time_max = t;
	}
	
	/*printf("%-8s%8d%14.2f%16.2f%16.2f%16.2f\n",
			XSTR(DATA_T),
			FLOPS_PER_ELEM,
			FLOPS_PER_ELEM * 1.0f / sizeof(DATA_T) / 2.0f,
			(HS_ARRAY_ELEM * FLOPS_PER_ELEM) / time_max / 1000.0 / 1000.0,
			(HS_ARRAY_ELEM * FLOPS_PER_ELEM) * NTRIES / time_total / 1000.0 / 1000.0,
			(HS_ARRAY_ELEM * FLOPS_PER_ELEM) / time_min / 1000.0 / 1000.0);*/
	
	printf("%-8s%11d%17.2f%19.2f%19.2f\n",
			XSTR(DATA_T),
			FLOPS_PER_ELEM,
			FLOPS_PER_ELEM * 1.0f / sizeof(DATA_T) / 2.0f,
			(HS_ARRAY_SIZE_BYTE * 2.0 * 1000.0) / time_min / 1024 / 1024 / 1024,
			(HS_ARRAY_ELEM * FLOPS_PER_ELEM) / time_min / 1000.0 / 1000.0);


	// Free allocated memory
	CUDA_CHECK_RETURN(cudaFreeHost(dataHost));
	CUDA_CHECK_RETURN(cudaFree(dataDev));

	return 0;
}

