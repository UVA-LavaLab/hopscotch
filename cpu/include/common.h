/******************************************************************************
 *
 * File: common.h
 * Description: Common header file.
 * 
 * Author: Alif Ahmed
 * Date: Sep 16, 2019
 *
 *****************************************************************************/
#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdio.h>
#include <cstdint>
#include <cfloat>
#include <cstdlib>
#include <cstdio>
#include <string>
//#include <chrono>
#include <algorithm>
#include <vector>

#include <xil_types.h>
#include <xtmrctr.h>
#include <xparameters.h>

// Data type of the working set
#ifndef data_t
#define data_t      double
#define ELEM_SIZE   8
#endif

// Working set size = (2 ^ WSS_EXP) bytes
// Default is 2 ^ 30 = 1GiB
#ifndef WSS_EXP
#define WSS_EXP         4
#endif

// Minimum set size before time is measured
// Default is 64MB
#ifndef WSS_MIN_EXP
#define WSS_MIN_EXP         26
#endif

// Working set size in bytes
#define WSS_BYTES       (1UL << WSS_EXP)

// Number of elements in working set
#define WSS_ELEMS       (WSS_BYTES / sizeof(data_t))

// macro expansion helper
#define XSTR(s) STR(s)
#define STR(s)  #s

// Data structure for results
typedef struct {
	uint64_t iters;
	double min_time;
	double max_time;
	double avg_time;
	uint64_t bytes_read;
	uint64_t bytes_write;
} res_t;

// Helper for print formatting
void print_bw_header();
void print_max_bw(const char* kernel, const res_t &result);
void print_complete();

// AXI Timer global variables
#define CLK_FREQ XPAR_AXI_TIMER_0_CLOCK_FREQ_HZ

// Xilinx AXI Timer functions
XTmrCtr AxiTimerInit();
unsigned int AxiTimerStart(XTmrCtr m_AxiTimer);
unsigned int AxiTimerStop(XTmrCtr m_AxiTimer);
double getDuration(unsigned int tStart, unsigned int tEnd);
void output_totaltime(double total_time);

// Timer functions
//std::chrono::high_resolution_clock::time_point get_time();
//double get_duration(const std::chrono::high_resolution_clock::time_point &start);

// Allocates 4K aligned memory. Portable.
void* hs_alloc(size_t size);

// Initializes a data array with a constant value
void init_const(data_t* arr, uint64_t num_elem, const data_t val);

// Initializes an index array with [0,1,...,(num_elem-1)].
// If suffle is true, randomize the generated array.
void init_linear(uint64_t* arr, uint64_t num_elem, bool shuffle);

//init pointer chasing to array 'ptr' with hamiltonian cycle
void init_pointer_chasing(void ** ptr, uint64_t num_elem);

// Runs kernel till allowed_time is elapsed, and then stores the following results:
//     result.iters
//     result.min_time
//     result.max_time
//     result.avg_time
// Other variables of results are not changed.
#define run_kernel(kernel, allowed_time, result)                            \
        {                                                                   \
            double total_time = 0;                                          \
            double min_time = DBL_MAX;                                      \
            double max_time = 0;                                            \
            uint64_t iters = 0;                                             \
            XTmrCtr m_AxiTimer = AxiTimerInit();                            \
            static const uint64_t min_iter = (WSS_MIN_EXP <= WSS_EXP) ? 1 : \
                                   (1UL << WSS_MIN_EXP) / (1UL << WSS_EXP); \
                                                                            \
            kernel; /*warm up*/                                             \
            while(total_time < allowed_time) {                              \
                unsigned int t_start = AxiTimerStart(m_AxiTimer);           \
                for(uint64_t i = 0; i < min_iter; i++) {                    \
                    kernel;                                                 \
                }                                                           \
                unsigned int t_end = AxiTimerStop(m_AxiTimer);              \
                double t = getDuration(t_start, t_end);                     \
                if(t < min_time){                                           \
                    min_time = t;                                           \
                }                                                           \
                if(t > max_time){                                           \
                    max_time = t;                                           \
                }                                                           \
                total_time += t;                                            \
                iters += min_iter;                                          \
            }                                                               \
                                                                            \
            result.iters = iters;                                           \
            result.min_time = min_time;                                     \
            result.max_time = max_time;                                     \
            result.avg_time = total_time / iters;                           \
        }

/******************************************************************************
 * Kernels
 *****************************************************************************/
extern void r_rand_ind(data_t* __restrict__ a);
extern res_t run_r_rand_ind(double allowed_time, data_t* a);

extern void r_rand_pchase(void** ptr);
extern res_t run_r_rand_pchase(double allowed_time, void** ptr);

extern void r_seq_ind(data_t* __restrict__ a);
extern res_t run_r_seq_ind(double allowed_time, data_t* a);

extern data_t r_seq_reduce(data_t* __restrict__ a);
extern res_t run_r_seq_reduce(double allowed_time, data_t* a);

extern void r_tile(data_t* __restrict__ a, uint64_t L, uint64_t K);
extern res_t run_r_tile(double allowed_time, data_t* a, uint64_t L, uint64_t K);

extern void rw_gather(data_t* __restrict__ a, data_t* __restrict__ b,
		uint64_t* __restrict__ idx);
extern res_t run_rw_gather(double allowed_time, data_t* a, data_t* b,
		uint64_t* idx);

extern void rw_scatter(data_t* __restrict__ a, data_t* __restrict__ b,
		uint64_t* __restrict__ idx);
extern res_t run_rw_scatter(double allowed_time, data_t* a, data_t* b,
		uint64_t* idx);

extern void rw_scatter_gather(data_t* __restrict__ a, data_t* __restrict__ b,
		uint64_t* __restrict__ idx1, uint64_t* __restrict__ idx2);
extern res_t run_rw_scatter_gather(double allowed_time, data_t* a, data_t* b,
		uint64_t* idx1, uint64_t* idx2);

extern void rw_seq_copy(data_t* __restrict__ a, data_t* __restrict__ b);
extern res_t run_rw_seq_copy(double allowed_time, data_t* a, data_t* b);

extern void rw_seq_inc(data_t* __restrict__ a);
extern res_t run_rw_seq_inc(double allowed_time, data_t* a);

extern void rw_tile(data_t* __restrict__ a, uint64_t L, uint64_t K);
extern res_t run_rw_tile(double allowed_time, data_t* a, uint64_t L,
		uint64_t K);

extern void w_rand_ind(data_t* __restrict__ a);
extern res_t run_w_rand_ind(double allowed_time, data_t* a);

extern void w_seq_fill(data_t* __restrict__ a);
extern res_t run_w_seq_fill(double allowed_time, data_t* a);

extern void w_seq_memset(data_t* __restrict__ a);
extern res_t run_w_seq_memset(double allowed_time, data_t* a);

extern void w_tile(data_t* __restrict__ a, uint64_t L, uint64_t K);
extern res_t run_w_tile(double allowed_time, data_t* a, uint64_t L, uint64_t K);

extern void rw_roofline(data_t* __restrict__ x);
extern res_t run_rw_roofline(double allowed_time, data_t* a);

#endif /* _COMMON_H_ */

