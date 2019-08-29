/*******************************************************************************
 *
 * File: seq.c
 * Description: Simplest example with sequential access patterns.
 * 
 * Author:  Alif Ahmed
 * Email:   alifahmed@virginia.edu
 * Updated: Aug 06, 2019
 *
 ******************************************************************************/
#include <stdlib.h>
#include <stdio.h>

#define N	10000

static volatile int arr[N];

int main(int argc, const char** argv) {
	//sequential write
	for(int i = 0; i < N; ++i) {
		arr[i] = 2;
	}

	//sequential read
	int sum = 0;
	for(int i = 0; i < N; ++i) {
		sum += arr[i];
	}

	//sequential read-modify-write
	for(int i=0; i < N; ++i){
		arr[i]++;
	}

	return 0;
}
