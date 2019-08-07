#include "common.h"
#include <cstdio>
#include <cfloat>
#include <immintrin.h>

#ifndef FLOPS_PER_ELEM
#define FLOPS_PER_ELEM       512
#endif

#if defined(__AVX512F__)
	#if DATA_T == double
		double kernel_roofline(DATA_T* __restrict__ x, DATA_T r, DATA_T s) {
			const __m512d rv = _mm512_set1_pd(r);
			const __m512d sv = _mm512_set1_pd(s);
		
			auto start = hs_get_time();
			#pragma omp parallel for
			for(uint64_t i = 0; i < HS_ARRAY_ELEM; i+=32){
				__m512d xv1 = _mm512_load_pd(&x[i]);
				__m512d xv2 = _mm512_load_pd(&x[i+4]);
				__m512d xv3 = _mm512_load_pd(&x[i+8]);
				__m512d xv4 = _mm512_load_pd(&x[i+12]);
				__m512d xv5 = _mm512_load_pd(&x[i+16]);
				__m512d xv6 = _mm512_load_pd(&x[i+20]);
				__m512d xv7 = _mm512_load_pd(&x[i+24]);
				__m512d xv8 = _mm512_load_pd(&x[i+28]);
				
				for(uint64_t j = 0; j < FLOPS_PER_ELEM/2; ++j) {
				    xv1 = _mm256_fmadd_pd(xv1, rv, sv);
				    xv2 = _mm256_fmadd_pd(xv2, rv, sv);
				    xv3 = _mm256_fmadd_pd(xv3, rv, sv);
				    xv4 = _mm256_fmadd_pd(xv4, rv, sv);
				    xv5 = _mm256_fmadd_pd(xv5, rv, sv);
				    xv6 = _mm256_fmadd_pd(xv6, rv, sv);
				    xv7 = _mm256_fmadd_pd(xv7, rv, sv);
				    xv8 = _mm256_fmadd_pd(xv8, rv, sv);
				}
				       
				#if (FLOPS_PER_ELEM & 1) == 1
				xv1 = _mm256_add_pd(xv1, sv);
				xv2 = _mm256_add_pd(xv2, sv);
				xv3 = _mm256_add_pd(xv3, sv);
				xv4 = _mm256_add_pd(xv4, sv);
				xv5 = _mm256_add_pd(xv5, sv);
				xv6 = _mm256_add_pd(xv6, sv);
				xv7 = _mm256_add_pd(xv7, sv);
				xv8 = _mm256_add_pd(xv8, sv);
				#endif

				_mm256_store_pd(&x[i], xv1);
				_mm256_store_pd(&x[i+4], xv2);
				_mm256_store_pd(&x[i+8], xv3);
				_mm256_store_pd(&x[i+12], xv4);
				_mm256_store_pd(&x[i+16], xv5);
				_mm256_store_pd(&x[i+20], xv6);
				_mm256_store_pd(&x[i+24], xv7);
				_mm256_store_pd(&x[i+28], xv8);
			}
			return hs_duration(start);
		}
	#elif DATA_T == float
		double kernel_roofline(DATA_T* __restrict__ x, DATA_T r, DATA_T s) {
			const __m256d rv = _mm256_set1_pd(r);
			const __m256d sv = _mm256_set1_pd(s);
		
			auto start = hs_get_time();
			#pragma omp parallel for
			for(uint64_t i = 0; i < HS_ARRAY_ELEM; i+=32){
				__m256d xv1 = _mm256_load_pd(&x[i]);
				__m256d xv2 = _mm256_load_pd(&x[i+4]);
				__m256d xv3 = _mm256_load_pd(&x[i+8]);
				__m256d xv4 = _mm256_load_pd(&x[i+12]);
				__m256d xv5 = _mm256_load_pd(&x[i+16]);
				__m256d xv6 = _mm256_load_pd(&x[i+20]);
				__m256d xv7 = _mm256_load_pd(&x[i+24]);
				__m256d xv8 = _mm256_load_pd(&x[i+28]);
				
				for(uint64_t j = 0; j < FLOPS_PER_ELEM/2; ++j) {
				    xv1 = _mm256_fmadd_pd(xv1, rv, sv);
				    xv2 = _mm256_fmadd_pd(xv2, rv, sv);
				    xv3 = _mm256_fmadd_pd(xv3, rv, sv);
				    xv4 = _mm256_fmadd_pd(xv4, rv, sv);
				    xv5 = _mm256_fmadd_pd(xv5, rv, sv);
				    xv6 = _mm256_fmadd_pd(xv6, rv, sv);
				    xv7 = _mm256_fmadd_pd(xv7, rv, sv);
				    xv8 = _mm256_fmadd_pd(xv8, rv, sv);
				}
				       
				#if (FLOPS_PER_ELEM & 1) == 1
				xv1 = _mm256_add_pd(xv1, sv);
				xv2 = _mm256_add_pd(xv2, sv);
				xv3 = _mm256_add_pd(xv3, sv);
				xv4 = _mm256_add_pd(xv4, sv);
				xv5 = _mm256_add_pd(xv5, sv);
				xv6 = _mm256_add_pd(xv6, sv);
				xv7 = _mm256_add_pd(xv7, sv);
				xv8 = _mm256_add_pd(xv8, sv);
				#endif

				_mm256_store_pd(&x[i], xv1);
				_mm256_store_pd(&x[i+4], xv2);
				_mm256_store_pd(&x[i+8], xv3);
				_mm256_store_pd(&x[i+12], xv4);
				_mm256_store_pd(&x[i+16], xv5);
				_mm256_store_pd(&x[i+20], xv6);
				_mm256_store_pd(&x[i+24], xv7);
				_mm256_store_pd(&x[i+28], xv8);
			}
			return hs_duration(start);
		}
	#endif

#elif defined(__AVX2__) || defined(__AVX__)
	#if DATA_T == double
		double kernel_roofline(DATA_T* __restrict__ x, DATA_T r, DATA_T s) {
			const __m256d rv = _mm256_set1_pd(r);
			const __m256d sv = _mm256_set1_pd(s);
		
			auto start = hs_get_time();
			#pragma omp parallel for
			for(uint64_t i = 0; i < HS_ARRAY_ELEM; i+=32){
				__m256d xv1 = _mm256_load_pd(&x[i]);
				__m256d xv2 = _mm256_load_pd(&x[i+4]);
				__m256d xv3 = _mm256_load_pd(&x[i+8]);
				__m256d xv4 = _mm256_load_pd(&x[i+12]);
				__m256d xv5 = _mm256_load_pd(&x[i+16]);
				__m256d xv6 = _mm256_load_pd(&x[i+20]);
				__m256d xv7 = _mm256_load_pd(&x[i+24]);
				__m256d xv8 = _mm256_load_pd(&x[i+28]);
				
				for(uint64_t j = 0; j < FLOPS_PER_ELEM/2; ++j) {
				    xv1 = _mm256_fmadd_pd(xv1, rv, sv);
				    xv2 = _mm256_fmadd_pd(xv2, rv, sv);
				    xv3 = _mm256_fmadd_pd(xv3, rv, sv);
				    xv4 = _mm256_fmadd_pd(xv4, rv, sv);
				    xv5 = _mm256_fmadd_pd(xv5, rv, sv);
				    xv6 = _mm256_fmadd_pd(xv6, rv, sv);
				    xv7 = _mm256_fmadd_pd(xv7, rv, sv);
				    xv8 = _mm256_fmadd_pd(xv8, rv, sv);
				}
				       
				#if (FLOPS_PER_ELEM & 1) == 1
				xv1 = _mm256_add_pd(xv1, sv);
				xv2 = _mm256_add_pd(xv2, sv);
				xv3 = _mm256_add_pd(xv3, sv);
				xv4 = _mm256_add_pd(xv4, sv);
				xv5 = _mm256_add_pd(xv5, sv);
				xv6 = _mm256_add_pd(xv6, sv);
				xv7 = _mm256_add_pd(xv7, sv);
				xv8 = _mm256_add_pd(xv8, sv);
				#endif

				_mm256_store_pd(&x[i], xv1);
				_mm256_store_pd(&x[i+4], xv2);
				_mm256_store_pd(&x[i+8], xv3);
				_mm256_store_pd(&x[i+12], xv4);
				_mm256_store_pd(&x[i+16], xv5);
				_mm256_store_pd(&x[i+20], xv6);
				_mm256_store_pd(&x[i+24], xv7);
				_mm256_store_pd(&x[i+28], xv8);
			}
			return hs_duration(start);
		}
	#elif DATA_T == float
	double kernel_roofline(DATA_T* __restrict__ x, DATA_T r, DATA_T s) {
		const __m256d rv = _mm256_set1_pd(r);
		const __m256d sv = _mm256_set1_pd(s);
		
		auto start = hs_get_time();
		#pragma omp parallel for
		for(uint64_t i = 0; i < HS_ARRAY_ELEM; i+=32){
		    __m256d xv1 = _mm256_load_pd(&x[i]);
		    __m256d xv2 = _mm256_load_pd(&x[i+4]);
		    __m256d xv3 = _mm256_load_pd(&x[i+8]);
		    __m256d xv4 = _mm256_load_pd(&x[i+12]);
		    __m256d xv5 = _mm256_load_pd(&x[i+16]);
		    __m256d xv6 = _mm256_load_pd(&x[i+20]);
		    __m256d xv7 = _mm256_load_pd(&x[i+24]);
		    __m256d xv8 = _mm256_load_pd(&x[i+28]);
		    
		    for(uint64_t j = 0; j < FLOPS_PER_ELEM/2; ++j) {
		        xv1 = _mm256_fmadd_pd(xv1, rv, sv);
		        xv2 = _mm256_fmadd_pd(xv2, rv, sv);
		        xv3 = _mm256_fmadd_pd(xv3, rv, sv);
		        xv4 = _mm256_fmadd_pd(xv4, rv, sv);
		        xv5 = _mm256_fmadd_pd(xv5, rv, sv);
		        xv6 = _mm256_fmadd_pd(xv6, rv, sv);
		        xv7 = _mm256_fmadd_pd(xv7, rv, sv);
		        xv8 = _mm256_fmadd_pd(xv8, rv, sv);
		    }
		           
			#if (FLOPS_PER_ELEM & 1) == 1
			xv1 = _mm256_add_pd(xv1, sv);
		    xv2 = _mm256_add_pd(xv2, sv);
		    xv3 = _mm256_add_pd(xv3, sv);
		    xv4 = _mm256_add_pd(xv4, sv);
		    xv5 = _mm256_add_pd(xv5, sv);
		    xv6 = _mm256_add_pd(xv6, sv);
		    xv7 = _mm256_add_pd(xv7, sv);
		    xv8 = _mm256_add_pd(xv8, sv);
			#endif

		    _mm256_store_pd(&x[i], xv1);
		    _mm256_store_pd(&x[i+4], xv2);
		    _mm256_store_pd(&x[i+8], xv3);
		    _mm256_store_pd(&x[i+12], xv4);
		    _mm256_store_pd(&x[i+16], xv5);
		    _mm256_store_pd(&x[i+20], xv6);
		    _mm256_store_pd(&x[i+24], xv7);
		    _mm256_store_pd(&x[i+28], xv8);
		}
		return hs_duration(start);
	}
	#endif

#elif defined(__FMA__)
	#if DATA_T == double
	double kernel_roofline(DATA_T* __restrict__ x, DATA_T r, DATA_T s) {
		const __m256d rv = _mm256_set1_pd(r);
		const __m256d sv = _mm256_set1_pd(s);
		
		auto start = hs_get_time();
		#pragma omp parallel for
		for(uint64_t i = 0; i < HS_ARRAY_ELEM; i+=32){
		    __m256d xv1 = _mm256_load_pd(&x[i]);
		    __m256d xv2 = _mm256_load_pd(&x[i+4]);
		    __m256d xv3 = _mm256_load_pd(&x[i+8]);
		    __m256d xv4 = _mm256_load_pd(&x[i+12]);
		    __m256d xv5 = _mm256_load_pd(&x[i+16]);
		    __m256d xv6 = _mm256_load_pd(&x[i+20]);
		    __m256d xv7 = _mm256_load_pd(&x[i+24]);
		    __m256d xv8 = _mm256_load_pd(&x[i+28]);
		    
		    for(uint64_t j = 0; j < FLOPS_PER_ELEM/2; ++j) {
		        xv1 = _mm256_fmadd_pd(xv1, rv, sv);
		        xv2 = _mm256_fmadd_pd(xv2, rv, sv);
		        xv3 = _mm256_fmadd_pd(xv3, rv, sv);
		        xv4 = _mm256_fmadd_pd(xv4, rv, sv);
		        xv5 = _mm256_fmadd_pd(xv5, rv, sv);
		        xv6 = _mm256_fmadd_pd(xv6, rv, sv);
		        xv7 = _mm256_fmadd_pd(xv7, rv, sv);
		        xv8 = _mm256_fmadd_pd(xv8, rv, sv);
		    }
		           
			#if (FLOPS_PER_ELEM & 1) == 1
			xv1 = _mm256_add_pd(xv1, sv);
		    xv2 = _mm256_add_pd(xv2, sv);
		    xv3 = _mm256_add_pd(xv3, sv);
		    xv4 = _mm256_add_pd(xv4, sv);
		    xv5 = _mm256_add_pd(xv5, sv);
		    xv6 = _mm256_add_pd(xv6, sv);
		    xv7 = _mm256_add_pd(xv7, sv);
		    xv8 = _mm256_add_pd(xv8, sv);
			#endif

		    _mm256_store_pd(&x[i], xv1);
		    _mm256_store_pd(&x[i+4], xv2);
		    _mm256_store_pd(&x[i+8], xv3);
		    _mm256_store_pd(&x[i+12], xv4);
		    _mm256_store_pd(&x[i+16], xv5);
		    _mm256_store_pd(&x[i+20], xv6);
		    _mm256_store_pd(&x[i+24], xv7);
		    _mm256_store_pd(&x[i+28], xv8);
		}
		return hs_duration(start);
	}
	#elif DATA_T == float
	double kernel_roofline(DATA_T* __restrict__ x, DATA_T r, DATA_T s) {
		const __m256d rv = _mm256_set1_pd(r);
		const __m256d sv = _mm256_set1_pd(s);
		
		auto start = hs_get_time();
		#pragma omp parallel for
		for(uint64_t i = 0; i < HS_ARRAY_ELEM; i+=32){
		    __m256d xv1 = _mm256_load_pd(&x[i]);
		    __m256d xv2 = _mm256_load_pd(&x[i+4]);
		    __m256d xv3 = _mm256_load_pd(&x[i+8]);
		    __m256d xv4 = _mm256_load_pd(&x[i+12]);
		    __m256d xv5 = _mm256_load_pd(&x[i+16]);
		    __m256d xv6 = _mm256_load_pd(&x[i+20]);
		    __m256d xv7 = _mm256_load_pd(&x[i+24]);
		    __m256d xv8 = _mm256_load_pd(&x[i+28]);
		    
		    for(uint64_t j = 0; j < FLOPS_PER_ELEM/2; ++j) {
		        xv1 = _mm256_fmadd_pd(xv1, rv, sv);
		        xv2 = _mm256_fmadd_pd(xv2, rv, sv);
		        xv3 = _mm256_fmadd_pd(xv3, rv, sv);
		        xv4 = _mm256_fmadd_pd(xv4, rv, sv);
		        xv5 = _mm256_fmadd_pd(xv5, rv, sv);
		        xv6 = _mm256_fmadd_pd(xv6, rv, sv);
		        xv7 = _mm256_fmadd_pd(xv7, rv, sv);
		        xv8 = _mm256_fmadd_pd(xv8, rv, sv);
		    }
		           
			#if (FLOPS_PER_ELEM & 1) == 1
			xv1 = _mm256_add_pd(xv1, sv);
		    xv2 = _mm256_add_pd(xv2, sv);
		    xv3 = _mm256_add_pd(xv3, sv);
		    xv4 = _mm256_add_pd(xv4, sv);
		    xv5 = _mm256_add_pd(xv5, sv);
		    xv6 = _mm256_add_pd(xv6, sv);
		    xv7 = _mm256_add_pd(xv7, sv);
		    xv8 = _mm256_add_pd(xv8, sv);
			#endif

		    _mm256_store_pd(&x[i], xv1);
		    _mm256_store_pd(&x[i+4], xv2);
		    _mm256_store_pd(&x[i+8], xv3);
		    _mm256_store_pd(&x[i+12], xv4);
		    _mm256_store_pd(&x[i+16], xv5);
		    _mm256_store_pd(&x[i+20], xv6);
		    _mm256_store_pd(&x[i+24], xv7);
		    _mm256_store_pd(&x[i+28], xv8);
		}
		return hs_duration(start);
	}
	#endif

#else
	#if DATA_T == double

	#elif DATA_T == float

	#endif

#endif



int main(){
    DATA_T* a = (DATA_T*)hs_alloc(HS_ARRAY_SIZE_BYTE);
    hs_init_const(a, HS_ARRAY_ELEM, 1.0);

    double time_min = DBL_MAX;
    for(uint64_t i = 0; i < NTRIES; ++i){
        double time_curr = kernel_roofline(a, 1.0 + 1e-6, 1e-6);
        if(time_curr < time_min){
            time_min = time_curr;
        }
    }
    
	printf("%-8s%11d%17.2f%19.2f%19.2f\n",
			XSTR(DATA_T),
			FLOPS_PER_ELEM,
			FLOPS_PER_ELEM * 1.0f / sizeof(DATA_T) / 2.0f,
			(HS_ARRAY_SIZE_BYTE * 2.0) / time_min / 1024 / 1024 / 1024,
			(HS_ARRAY_ELEM * FLOPS_PER_ELEM) / time_min / 1000.0 / 1000.0 / 1000.0);

    free(a);

    return 0;
}
