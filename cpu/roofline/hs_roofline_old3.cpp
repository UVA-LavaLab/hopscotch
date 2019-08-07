#include "common.h"
#include <cstdio>
#include <cfloat>
#include <immintrin.h>

#ifndef FLOPS_PER_ELEM
#define FLOPS_PER_ELEM       512
#endif

#if defined(__AVX512F__)
	#define VEC_WIDTH		512
	#define VEC_FUNC_PREFIX	512
#elif defined(__AVX__) || defined(__AVX2__)
	#define VEC_WIDTH		256
	#define VEC_FUNC_PREFIX	256
#elif defined(__SSE__)
	#define VEC_WIDTH		128
	#define VEC_FUNC_PREFIX
#endif


#if DATA_T == float
	#define VEC_DATA_SUFFIX
	#define VEC_FUNC_SUFFIX	s
#elif DATA_T == double
	#define VEC_DATA_SUFFIX d
	#define VEC_FUNC_SUFFIX	d
#endif


#define VEC_ELEMS			(VEC_WIDTH/8/sizeof(DATA_T))

#define CAT3_I(a,b)			a##b##c
#define CAT3(a,b)			CAT3_I(a,b,c)

#define VEC_DATA_T			CAT3(__m,VEC_WIDTH,VEC_DATA_SUFFIX)
#define VEC_SET(x)			CAT3(_mm,VEC_FUNC_PREFIX,_set_


double kernel_roofline(DATA_T* __restrict__ x, DATA_T r, DATA_T s) {

/*#if defined(__INTEL_COMPILER)
	DATA_T* __restrict__ x_al = x;
	__assume_aligned(x_al, 4096);
#elif defined(__GNUC__)
	DATA_T* __restrict__ x_al = (DATA_T*)__builtin_assume_aligned(x, 4096);
#endif*/
 
    const VEC_DATA_T rv = VEC_SET(r);
    const VEC_DATA_T sv = VEC_SET(s);

    auto start = hs_get_time();
    #pragma omp parallel for
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; i+=VEC_ELEMS*8){
        VEC_DATA_T xv1 = VEC_LOAD(&x[i+(VEC_ELEMS*0)]);
        VEC_DATA_T xv2 = VEC_LOAD(&x[i+(VEC_ELEMS*1)]);
        VEC_DATA_T xv3 = VEC_LOAD(&x[i+(VEC_ELEMS*2)]);
        VEC_DATA_T xv4 = VEC_LOAD(&x[i+(VEC_ELEMS*3)]);
        VEC_DATA_T xv5 = VEC_LOAD(&x[i+(VEC_ELEMS*4)]);
        VEC_DATA_T xv6 = VEC_LOAD(&x[i+(VEC_ELEMS*5)]);
        VEC_DATA_T xv7 = VEC_LOAD(&x[i+(VEC_ELEMS*6)]);
        VEC_DATA_T xv8 = VEC_LOAD(&x[i+(VEC_ELEMS*7)]);
        
        for(uint64_t j = 0; j < FLOPS_PER_ELEM/2; ++j) {
            xv1 = VEC_FMA(xv1, rv, sv);
            xv2 = VEC_FMA(xv2, rv, sv);
            xv3 = VEC_FMA(xv3, rv, sv);
            xv4 = VEC_FMA(xv4, rv, sv);
            xv5 = VEC_FMA(xv5, rv, sv);
            xv6 = VEC_FMA(xv6, rv, sv);
            xv7 = VEC_FMA(xv7, rv, sv);
            xv8 = VEC_FMA(xv8, rv, sv);
        }
       
		#if (FLOPS_PER_ELEM & 1) == 1
		/*xv1 = VEC_FMA(xv1, rv, sv);
        xv2 = VEC_FMA(xv2, rv, sv);
        xv3 = VEC_FMA(xv3, rv, sv);
        xv4 = VEC_FMA(xv4, rv, sv);
        xv5 = VEC_FMA(xv5, rv, sv);
        xv6 = VEC_FMA(xv6, rv, sv);
        xv7 = VEC_FMA(xv7, rv, sv);
        xv8 = VEC_FMA(xv8, rv, sv);*/
		#endif

        VEC_STORE(&x[i+(VEC_ELEMS*0)], xv1);
        VEC_STORE(&x[i+(VEC_ELEMS*1)], xv2);
        VEC_STORE(&x[i+(VEC_ELEMS*2)], xv3);
        VEC_STORE(&x[i+(VEC_ELEMS*3)], xv4);
        VEC_STORE(&x[i+(VEC_ELEMS*4)], xv5);
        VEC_STORE(&x[i+(VEC_ELEMS*5)], xv6);
        VEC_STORE(&x[i+(VEC_ELEMS*6)], xv7);
        VEC_STORE(&x[i+(VEC_ELEMS*7)], xv8);
    }
    return hs_duration(start);
}

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
