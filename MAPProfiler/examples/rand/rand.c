/*******************************************************************************
 *
 * File: rand.c
 * Description: Example with random access patterns.
 * 
 * Author:  Alif Ahmed
 * Email:   alifahmed@virginia.edu
 * Updated: Aug 06, 2019
 *
 ******************************************************************************/
#include <stdlib.h>

#define N	8192

static volatile int arr[N];
static volatile int idx[N];

int main(int argc, const char** argv) {
	//create index (sequential write)
	for(int i = 0; i < N; ++i) {
		idx[i] = rand() % N;
	}

	//sequential read (idx) + random write (arr)
	for(int i = 0; i < N; ++i) {
		arr[idx[i]] = 2;
	}

	//sequential read (idx) + random read (arr)
	int sum = 0;
	for(int i = 0; i < N; ++i) {
		sum += arr[idx[i]];
	}
	
	//sequential read (idx) + random read-modify-write (arr)
	for(int i=0; i < N; ++i){
		arr[idx[i]]++;
	}

	return 0;
}
