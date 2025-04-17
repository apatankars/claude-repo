#ifndef KERNELS
#define KERNELS

#include <stdio.h>

// Kernel implementations of SGEMM (Single precision GEneral Matrix Multiply)
// Performs the operation C = α*(A@B)+β*C for matrices A, B, C and scalars α, β
// where A is MxK, B is KxN, and C is MxN

// -------------------------------------------------------------------------------------
// Sequential implementations

void sgemm_sequential(int M, int K, int N, float alpha, const float *A,
                      const float *B, float beta, float *C) {
    // A = MxK
    // B = KxN
    // C = MxN
    for (int row = 0; row < M; ++row) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[row * K + k] * B[k * N + j];
            }
            C[row * N + j] = alpha * sum + beta * C[row * N + j];
        }
    }
}

// -------------------------------------------------------------------------------------
// Kernel 1 implementation

__global__ void sgemm_naive(int M, int K, int N, float alpha, const float *A,
                            const float *B, float beta, float *C) {
    // TODO (Part 1.1): Implement!
    // A = MxK
    // B = KxN
    // C = MxN
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

// -------------------------------------------------------------------------------------
// Kernel 2 implementation

__global__ void sgemm_global_coalescing(int M, int K, int N, float alpha, const float *A,
                                       const float *B, float beta, float *C) {
    // TODO (Part 1.2): Implement!
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

// -------------------------------------------------------------------------------------
// Kernel 3 implementation

template <const int BLOCKDIM>
__global__ void sgemm_shared_mem_cache(int M, int K, int N, float alpha, const float *A,
                                       const float *B, float beta, float *C) {
    __shared__ float blockA[BLOCKDIM * BLOCKDIM];
    __shared__ float blockB[BLOCKDIM * BLOCKDIM];

    // TODO (Part 1.3): Implement!
    int row = blockIdx.y * BLOCKDIM + threadIdx.y;
    int col = blockIdx.x * BLOCKDIM + threadIdx.x;
    float sum = 0.0f;

    for (int k = 0; k < (K + BLOCKDIM - 1) / BLOCKDIM; ++k) {
        if (row < M && k * BLOCKDIM + threadIdx.x < K) {
            blockA[threadIdx.y * BLOCKDIM + threadIdx.x] = A[row * K + k * BLOCKDIM + threadIdx.x];
        } else {
            blockA[threadIdx.y * BLOCKDIM + threadIdx.x] = 0.0f;
        }
        if (col < N && k * BLOCKDIM + threadIdx.y < K) {
            blockB[threadIdx.y * BLOCKDIM + threadIdx.x] = B[(k * BLOCKDIM + threadIdx.y) * N + col];
        } else {
            blockB[threadIdx.y * BLOCKDIM + threadIdx.x] = 0.0f;
        }
        __syncthreads();

        for (int j = 0; j < BLOCKDIM; ++j) {
            sum += blockA[threadIdx.y * BLOCKDIM + j] * blockB[j * BLOCKDIM + threadIdx.x];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

// -------------------------------------------------------------------------------------
// Kernel 4 implementation

template <const int BM, const int BN, const int BK, const int TM>
__global__ void __launch_bounds__(1024, 1)
sgemm_1D_thread_tiling(int M, int N, int K, float alpha, const float *A, 
                      const float *B, float beta, float *C) {
    __shared__ float blockA[BM * BK];
    __shared__ float blockB[BK * BN];

    // TODO (Part 1.4): Implement!
}

// -------------------------------------------------------------------------------------
// Kernel 5 implementation

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__(1024, 1)
sgemm_2D_thread_tiling(int M, int N, int K, float alpha, const float *A,
                      const float *B, float beta, float *C) {
    __shared__ float blockA[BM * BK];
    __shared__ float blockB[BK * BN];

    // TODO (Part 1.5): Implement!
}

// -------------------------------------------------------------------------------------
// Kernel launcher

inline void launch_kernel(long kernel_num, int M, int K, int N, float alpha, 
                          const float *d_A, const float *d_B, float beta, float *d_C) {
    if(kernel_num == 1) {
        // TODO (Part 1.1): Set execution configuration parameters
        dim3 thr_per_blk(32, 32);
        dim3 blk_in_grid((M+31/32), (N+31/32));
        
        sgemm_naive<<<blk_in_grid, thr_per_blk>>>(M, K, N, alpha, d_A, d_B, beta, d_C);
    } else if(kernel_num == 2) {
        // TODO (Part 1.2): Set execution configuration parameters
        dim3 thr_per_blk(32, 32);
        dim3 blk_in_grid((N + 31) / 32, (M + 31) / 32);

        sgemm_global_coalescing<<<blk_in_grid, thr_per_blk>>>(M, K, N, alpha, d_A, d_B, beta, d_C);
    } else if(kernel_num == 3) {
        const uint BLOCKDIM = 32;

        // TODO (Part 1.3): Set execution configuration parameters
        dim3 thr_per_blk(BLOCKDIM, BLOCKDIM);
        dim3 blk_in_grid((N + BLOCKDIM - 1) / BLOCKDIM, (M + BLOCKDIM - 1) / BLOCKDIM);

        sgemm_shared_mem_cache<BLOCKDIM><<<blk_in_grid, thr_per_blk>>>(M, K, N, alpha, d_A, d_B, beta, d_C);
    } else if(kernel_num == 4) {
        const uint BM = 64;
        const uint BN = 64;
        const uint BK = 8;
        const uint TM = 8;

        // TODO (Part 1.4): Set execution configuration parameters
        dim3 blk_in_grid;
        dim3 thr_per_blk;

        sgemm_1D_thread_tiling<BM, BN, BK, TM><<<blk_in_grid, thr_per_blk>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    } else if(kernel_num == 5) {
        const uint BM = 128;
        const uint BN = 128;
        const uint BK = 8;
        const uint TM = 8;
        const uint TN = 8;

        // TODO (Part 1.5): Set execution configuration parameters
        dim3 blk_in_grid;
        dim3 thr_per_blk;

        sgemm_2D_thread_tiling<BM, BN, BK, TM, TN><<<blk_in_grid, thr_per_blk>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    } else {
        fprintf(stderr, "Error: %lu is not a valid kernel number.\n", kernel_num);
        exit(1);
    }
}

#endif // KERNELS