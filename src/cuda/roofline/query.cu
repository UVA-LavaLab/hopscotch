#include "common.h"
#include <iostream>
#include <cuda.h>

using namespace std;

void printDeviceInfo(int device) {
	cudaDeviceProp prop;
	CUDA_CHECK_RETURN(cudaGetDeviceProperties(&prop, device));
	cout << "    {" << endl;
	cout << "      \"name\": \"" 				<< prop.name << "\",\n";
	cout << "      \"totalGlobalMem\": " 	<< prop.totalGlobalMem << ",\n";
	cout << "      \"clockRate\": " 		<< prop.clockRate << ",\n";
	cout << "      \"computeCapability\": \"" << prop.major << "." << prop.minor << "\",\n";
	cout << "      \"multiProcessorCount\": "	<< prop.multiProcessorCount	<< ",\n";
	cout << "      \"memoryClockRate\": "	<< prop.memoryClockRate	<< ",\n";
	cout << "      \"memoryBusWidth\": "	<< prop.memoryBusWidth	<< ",\n";
	cout << "      \"warpSize\": "	<< prop.warpSize	<< "\n";
	cout << "    }";
}

int main() {
	int deviceCount;
	CUDA_CHECK_RETURN(cudaGetDeviceCount(&deviceCount));
	cout << "{\n" << "  \"cudaDevices\": [\n";
	for(int i = 0; i < deviceCount; ++i) {
		if(i) cout << ",\n";
		printDeviceInfo(i);
	}
	cout << "\n  ]\n}\n";
	return 0;
}
