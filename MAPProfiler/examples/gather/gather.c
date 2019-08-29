/*******************************************************************************
 *
 * File: gather.c
 * Description: Gather access pattern.
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
static int idx[N];

// initialize
void init(){
	for(int i = 0; i < N; ++i){
		b[i] = 2;
		idx[i] = rand() % N;
	}
}

//gather kernel
void gather(){
	for(int i = 0; i < N; ++i){
		a[i] = b[idx[i]];
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
	gather();

	// print result
	printf("%d\n", getSum(a));

	return 0;
}
