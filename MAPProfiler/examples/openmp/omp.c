/*******************************************************************************
 *
 * File: omp.c
 * Description: Example for memory access pattern of sequential access with 
 * openmp.
 * 
 * Author:  Alif Ahmed
 * Email:   alifahmed@virginia.edu
 * Updated: Aug 06, 2019
 *
 ******************************************************************************/
#include <stdlib.h>
#include <stdio.h>

#define N	500000

static volatile int arr[N];

int main(int argc, const char** argv) {

	//sequential write
	#pragma omp parallel for
	for(int i = 0; i < N; ++i) {
		arr[i] = 2;
	}
	
	//sequential read
	int sum = 0;
	#pragma omp parallel for reduction(+:sum)
	for(int i = 0; i < N; ++i) {
		sum += arr[i];
	}

	//sequential read-modify-write
	#pragma omp parallel for
	for(int i=0; i < N; ++i){
		arr[i]++;
	}

	return 0;
}
