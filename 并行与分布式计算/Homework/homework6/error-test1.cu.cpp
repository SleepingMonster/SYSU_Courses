#include <cstdio>
#include <cmath>
//#include "error_checks.h" // Macros CUDA_CHECK and CHECK_ERROR_MSG

#ifndef COURSE_UTIL_H_
#define COURSE_UTIL_H_

#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(errarg)   __checkErrorFunc(errarg, __FILE__, __LINE__)
#define CHECK_ERROR_MSG(errstr) __checkErrMsgFunc(errstr, __FILE__, __LINE__)

inline void __checkErrorFunc(cudaError_t errarg, const char* file, 
			     const int line)
{
    if(errarg) {
	fprintf(stderr, "Error at %s(%i)\n", file, line);
	exit(EXIT_FAILURE);
    }
}


inline void __checkErrMsgFunc(const char* errstr, const char* file, 
			      const int line)
{
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
	fprintf(stderr, "Error: %s at %s(%i): %s\n", 
		errstr, file, line, cudaGetErrorString(err));
	exit(EXIT_FAILURE);
    }
}

#endif

__global__ void vector_add(double *C, const double *A, const double *B, int N)
{
    // Add the kernel code
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Do not try to access past the allocated memory
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}


int main(void)
{
    const int N = 20;
    const int ThreadsInBlock = 128;
    double *dA, *dB, *dC;
    double hA[N], hB[N], hC[N];
  
    for(int i = 0; i < N; ++i) {
        hA[i] = (double) i;
        hB[i] = (double) i * i;
    }

    /* 
       Add memory allocations and copies. Wrap your runtime function
       calls with CUDA_CHECK( ) macro
    */
    CUDA_CHECK( cudaMalloc((void**)&dA, sizeof(double)*N) );
    CUDA_CHECK( cudaMalloc((void**)&dB, sizeof(double)*N) );
    CUDA_CHECK( cudaMalloc((void**)&dC, sizeof(double)*N) );
    CUDA_CHECK( cudaMemcpy(dA, hA, sizeof(double)*N, cudaMemcpyHostToDevice));//传数据到GPU 
    CUDA_CHECK( cudaMemcpy(dB, hB, sizeof(double)*N, cudaMemcpyHostToDevice));
    //#error Add the remaining memory allocations and copies

    // Note the maximum size of threads in a block
    dim3 grid, threads;

    //// Add the kernel call here
    vector_add<<<1, 32>>>(dC,dA,dB,N);
    //#error Add the CUDA kernel call

    // Here we add an explicit synchronization so that we catch errors
    // as early as possible. Don't do this in production code!
    cudaDeviceSynchronize();
    CHECK_ERROR_MSG("vector_add kernel");

    //// Copy back the results and free the device memory
    CUDA_CHECK(cudaMemcpy(hC, dC, sizeof(double) *N, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
	//#error Copy back the results and free the allocated memory

    for (int i = 0; i < N; i++)
        printf("%5.1f\n", hC[i]);
    return 0;
}
