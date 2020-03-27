/******************************************************************************
 *
 * File: util.cpp
 * Description: Utility functions.
 * 
 * Author: Alif Ahmed
 * Date: Sep 16, 2019
 *
 *****************************************************************************/
#include "../include/common.h"

/******************************************************************************
 * Print formatting
 *****************************************************************************/
void print_bw_header() {
	printf("=======================================================================\n");
	printf("%-30s%-15s%-15s%-15s\n", "Kernel", "Iterations", "Time (s)", "BW (MB/s)");
	printf("=======================================================================\n");
}

void print_max_bw(const char* kernel, const res_t &result) {
//	printf("%-30s%-15lu%-15.2lf%-15.0lf\n", kernel, result.iters, result.iters * result.avg_time, (result.bytes_read + result.bytes_write) / result.iters / result.min_time / 1e6);
	float totaltime = result.iters * result.avg_time;
	float bw = (result.bytes_read + result.bytes_write) / result.iters / result.min_time / 1e6;
	int iterations = result.iters;
	printf("%s | %d | %f | %f\n", kernel, iterations, totaltime, bw);
}

void print_complete() {
	printf("=======================================================================\n");
	printf("==================================END==================================\n");
	printf("=======================================================================\n");
}

/******************************************************************************
 * Allocates 4096 bytes aligned memory. Portable.
 *****************************************************************************/
void* hs_alloc(size_t size) {
	// void* ptr = aligned_alloc(4096, size);
	void* ptr = malloc(size);
	if (ptr == nullptr) {
		fprintf(stderr, "Memory allocation of size %lu bytes failed\n", size);
		exit(-1);
	}
	return ptr;
}

/******************************************************************************
 * Initializes the Xilinx AXI Timer
 *****************************************************************************/
XTmrCtr AxiTimerInit() {
	XTmrCtr m_AxiTimer;
	XTmrCtr_Initialize(&m_AxiTimer, XPAR_TMRCTR_0_DEVICE_ID);
	return m_AxiTimer;
}

/******************************************************************************
 * Resets and Starts the AXI Timer and returns the start tick count
 *****************************************************************************/
unsigned int AxiTimerStart(XTmrCtr m_AxiTimer) {
	XTmrCtr_Reset(&m_AxiTimer, 0);
	XTmrCtr_Start(&m_AxiTimer, 0);
	unsigned int tStart = XTmrCtr_GetValue(&m_AxiTimer, 0);
	return tStart;
}

/******************************************************************************
 * Stops the AXI Timer and returns the end tick count
 *****************************************************************************/
unsigned int AxiTimerStop(XTmrCtr m_AxiTimer) {
	XTmrCtr_Stop(&m_AxiTimer, 0);
	unsigned int tEnd = XTmrCtr_GetValue(&m_AxiTimer, 0);
	return tEnd;
}

/******************************************************************************
 * Converts a AXI Timer tick count to duration in seconds
 *****************************************************************************/
double getDuration(unsigned int tStart, unsigned int tEnd) {
	return (double) (tEnd - tStart) / CLK_FREQ;
}
void output_totaltime(double total_time) {
	printf("total time=%f \n", total_time);
}
;

/******************************************************************************
 * Captures current time as a time_point object. Use hs_duration() to get elapsed
 * time. Portable.
 *****************************************************************************/
//std::chrono::high_resolution_clock::time_point get_time() {
//    return std::chrono::high_resolution_clock::now();
//}

/******************************************************************************
 * Measures the elasped time (in seconds) since 'start'. Can support upto
 * nanosecond resoultion depending on hardware support.
 *****************************************************************************/
//double get_duration(const std::chrono::high_resolution_clock::time_point &start) {
//    std::chrono::duration<double> t = get_time() - start;
//    return t.count();
//}

/******************************************************************************
 * Initializes an array with constant value.
 *****************************************************************************/
void init_const(data_t* arr, uint64_t num_elem, const data_t val) {
#pragma omp parallel for
	for (uint64_t i = 0; i < num_elem; ++i) {
		arr[i] = val;
	}
}

// Initializes an index array with [0,1,...,(num_elem-1)].
// If suffle is true, randomize the generated array.
void init_linear(uint64_t* arr, uint64_t num_elem, bool shuffle) {
#pragma omp parallel for
	for (uint64_t i = 0; i < num_elem; ++i) {
		arr[i] = i;
	}
	if (shuffle) {
		std::random_shuffle(arr, arr + num_elem);
	}
}

//init pointer chasing to array 'ptr' with hamiltonian cycle
void init_pointer_chasing(void ** ptr, uint64_t num_elem) {
	std::vector<void*> unused(num_elem - 1);
#pragma omp parallel for
	for (uint64_t i = 1; i < num_elem; ++i) {
		unused[i - 1] = ptr + i;
	}
	std::random_shuffle(unused.begin(), unused.end());
	void** curr = ptr;
	for (uint64_t i = 0; i < num_elem - 1; ++i) {
		*curr = unused[i];
		curr = (void**) unused[i];
	}
	*curr = ptr;
}

