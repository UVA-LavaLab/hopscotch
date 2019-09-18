/*******************************************************************************
 *
 * File: hs_fairness.cpp
 * Description: Measures fairness of memory controller.
 * 
 * Author: Alif Ahmed
 * Date: Aug 06, 2019
 *
 ******************************************************************************/
#include "common.h"
#include <cstdint>
#include <iostream>
#include <omp.h>
#include <cfloat>
#include <unistd.h>

using namespace std;

#define NUM_THREADS		16
#define NTRIES			10UL

int main(){
	double elapsed_seq, elapsed_rand;
    uint64_t iter;
	#pragma omp parallel
	#pragma omp single
	cout << "Default number of threads: " << omp_get_num_threads() << endl;
    print_bw_header();

	omp_set_num_threads(NUM_THREADS);
	
	alloc_a();
	alloc_b();

	init_ab();

    elapsed_seq = kernel_min_time(&iter, r_seq_reduce, init_a);
    print_bw("unloaded r_seq_reduce", iter, elapsed_seq, HS_ARRAY_SIZE_MB / elapsed_seq);

    elapsed_rand = kernel_min_time(&iter, r_rand_ind, init_a);
    print_bw("unloaded r_rand_ind", iter, elapsed_rand, HS_ARRAY_SIZE_MB / elapsed_rand);

	cout << "Unloaded ratio: " << elapsed_rand / elapsed_seq << endl;

	volatile bool done = false;
	
	data_t tot_sum = 0;
	#pragma omp parallel shared(done, tot_sum)
	{
		int tid = omp_get_thread_num();
		if(tid == 0){
			data_t sum = 0;
			double min_time = FLT_MAX;
			sleep(1);
			for(uint64_t n = 0; n < NTRIES; ++n) {
				double t = get_time();
				for(uint64_t i = 0; i < HS_ARRAY_ELEM; ++i) {
					sum += a[i];
				}
				t = get_time() - t;
				if(t < min_time) min_time = t;
			}

			cout << "Seq sum: " << sum << endl;
			print_bw("loaded seq", NTRIES, min_time, HS_ARRAY_SIZE_MB / min_time);
			done = true;
		}
		else {
			const uint64_t elem_per_thread = HS_ARRAY_ELEM / NUM_THREADS;
			uint64_t lb = tid * elem_per_thread;
			uint64_t ub = lb + elem_per_thread;
			data_t sum = 0;
			while(!done) {
				for(uint64_t i = lb; i < ub; ++i) {
					sum += b[i];
				}
			}
			#pragma omp critical
			tot_sum += sum;
		}	
	}

	cout << "Total sum: " << tot_sum << endl;
	
	done = false;
	#pragma omp parallel shared(done, tot_sum)
	{
		int tid = omp_get_thread_num();
		if(tid == 0){
			//data_t sum = 0;
			double min_time = FLT_MAX;
			sleep(1);
			for(uint64_t n = 0; n < NTRIES; ++n) {
				volatile data_t* vol_a = a;
				double t = get_time();
				for(uint64_t i = 0; i < HS_ARRAY_ELEM; ++i) {
					uint64_t idx = ((i * 0xDEADBEEF) ^ 0xC0FFEE0B) % HS_ARRAY_ELEM;
					data_t res = vol_a[idx];
				}
				t = get_time() - t;
				if(t < min_time) min_time = t;
			}

			//cout << "Rand sum: " << sum << endl;
			print_bw("loaded rand", NTRIES, min_time, HS_ARRAY_SIZE_MB / min_time);
			done = true;
		}
		else {
			const uint64_t elem_per_thread = HS_ARRAY_ELEM / NUM_THREADS;
			uint64_t lb = tid * elem_per_thread;
			uint64_t ub = lb + elem_per_thread;
			data_t sum = 0;
			while(!done) {
				for(uint64_t i = lb; i < ub; ++i) {
					sum += b[i];
				}
			}
			#pragma omp critical
			tot_sum += sum;
		}	
	}

	free_a();
	free_b();

	return 0;
}
