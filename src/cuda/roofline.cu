#include <iostream>
#include <cstdint>

typedef double data_t;

#define HS_ARRAY_SIZE_BYTE	(1024UL*1024*1024)
#define HS_ARRAY_SIZE_KB	(HS_ARRAY_SIZE_BYTE/1024)
#define HS_ARRAY_SIZE_MB	(HS_ARRAY_SIZE_KB/1024)
#define HS_ARRAY_SIZE_GB	(HS_ARRAY_SIZE_GB/1024)

#define HS_ARRAY_ELEM		(HS_ARRAY_SIZE_BYTE/sizeof(data_t))

__global__
void roofline(data_t* __restrict__ a){
	
}


int main(){
	data_t* a_d;

	data_t* a_h = (data_t*)malloc(HS_ARRAY_SIZE_BYTE);
	for(uint64_t i = 0; i < HS_ARRAY_ELEM; ++i){
		a_h[i] = 1.0 + 1e-8;
	}


	cudaMalloc((void**)&a_d, HS_ARRAY_SIZE_BYTE);

	cudaMemcpy(a_d, a_h, HS_ARRAY_SIZE_BYTE, cudaMemcpyHostToDevice);

	roofline<<<HS_ARRAY_ELEM,1>>>(a_d);

	cudaMemcpy(a_h, a_d, HS_ARRAY_SIZE_BYTE, cudaMemcpyDeviceToHost);

	cudaFree(a_d);
	free(a_h);
	return 0;
}

