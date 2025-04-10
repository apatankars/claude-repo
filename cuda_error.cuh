#ifndef CUDA_ERROR
#define CUDA_ERROR

#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#define cudaCheckError(ans) cudaAssert((ans), __FILE__, __LINE__)
inline void cudaAssert(cudaError_t code, const char *file, int line) {
   if (code != cudaSuccess) {
      fprintf(stderr, "CUDA error in %s at line %d: %s\n", file, line, cudaGetErrorString(code));
      exit(EXIT_FAILURE);
   }
}

inline void checkCublasStatus(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:
            // No error
            break;
        case CUBLAS_STATUS_NOT_INITIALIZED:
            fprintf(stderr, "cuBLAS error: Not initialized\n");
            exit(EXIT_FAILURE);
        case CUBLAS_STATUS_ALLOC_FAILED:
            fprintf(stderr, "cuBLAS error: Resource allocation failed\n");
            exit(EXIT_FAILURE);
        case CUBLAS_STATUS_INVALID_VALUE:
            fprintf(stderr, "cuBLAS error: Invalid value\n");
            exit(EXIT_FAILURE);
        case CUBLAS_STATUS_ARCH_MISMATCH:
            fprintf(stderr, "cuBLAS error: Architecture mismatch\n");
            exit(EXIT_FAILURE);
        case CUBLAS_STATUS_EXECUTION_FAILED:
            fprintf(stderr, "cuBLAS error: Execution failed\n");
            exit(EXIT_FAILURE);
        case CUBLAS_STATUS_INTERNAL_ERROR:
            fprintf(stderr, "cuBLAS error: Internal error\n");
            exit(EXIT_FAILURE);
        default:
            fprintf(stderr, "cuBLAS error: Unknown error code: %d\n", status);
            exit(EXIT_FAILURE);
    }
}

#endif // CUDA_ERROR