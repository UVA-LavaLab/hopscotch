/*******************************************************************************
 *
 * File: scatter_gather.cpp
 * Description: Scatter gather access pattern.
 * 
 * Author:  Alif Ahmed
 * Email:   alifahmed@virginia.edu
 * Updated: Aug 06, 2019
 *
 ******************************************************************************/
#include <stdlib.h>
#include <stdio.h>

#define N	10000

static int a[N];
static int b[N];
static int idx1[N];
static int idx2[N];

// initialize
void init(){
	for(int i = 0; i < N; ++i){
		b[i] = 2;
		idx1[i] = rand() % N;
		idx2[i] = rand() % N;
	}
}

//gather kernel
void scatter_gather(){
	for(int i = 0; i < N; ++i){
		a[idx1[i]] = b[idx2[i]];
	}
}

//sum
int getSum(const int* arr){
	int sum = 0;
	for(int i = 0; i < N; ++i){
		sum += arr[i];
	}
	return sum;
}


int main(int argc, const char** argv) {
	// initialize
	init();

	// call gather kernel
	scatter_gather();

	// print result
	printf("%d\n", getSum(a));

	return 0;
}
