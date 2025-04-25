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
__global__ void
sgemm_shared_mem_cache(int   M, int K, int N,
                       float alpha,
                       const float* __restrict__ A,
                       const float* __restrict__ B,
                       float beta,
                       float*       __restrict__ C)
{
    // A = MxK
    // B = KxN
    // C = MxN
   
    // ?? this is shared amongst all the threads in a block
    __shared__ float blockA[BLOCKDIM * BLOCKDIM];
    __shared__ float blockB[BLOCKDIM * BLOCKDIM];


    const int col = blockIdx.x * BLOCKDIM + threadIdx.x;  // column in C
    const int row = blockIdx.y * BLOCKDIM + threadIdx.y;  // row in C

    float val = 0.f;

    // each iteration, we want to move 32 threads along the K dimension
    for (int blockOffset = 0; blockOffset < K; blockOffset += BLOCKDIM) {

        // ?? need to do blockOffset + threadIdx.x becuase blockOffset starting column of the current tile chunk along the K‑dimension
        // ?? threadIdx.x runs from 0 to BLOCKDIM–1 within that tile
        int aCol = blockOffset + threadIdx.x;
        int bRow = blockOffset + threadIdx.y;
        int idx  = threadIdx.y * BLOCKDIM + threadIdx.x;

        // no warp-divergence here, all threads in the block will execute this code
        if (row < M && aCol < K) {
            // ?? only load A and B if the row and column are within bounds
            // ?? this is to avoid out of bounds memory access
            // ?? we can use a_ok and b_ok to check if the indices are valid
            blockA[idx] = A[row * K + aCol];
        } else {
            blockA[idx] = 0.f;
        }

        if (bRow < K && col < N) {
            blockB[idx] = B[bRow * N + col];
        } else {
            blockB[idx] = 0.f;
        }

        __syncthreads(); // cache full

        for (int k = 0; k < BLOCKDIM; ++k) {
            val += blockA[threadIdx.y * BLOCKDIM + k] *
                   blockB[k * BLOCKDIM + threadIdx.x];
        }

        __syncthreads(); 
    }

    if (row < M && col < N) {
        const int c_idx = row * N + col;
        C[c_idx] = alpha * val + beta * C[c_idx];
    }
}

// -------------------------------------------------------------------------------------
// Kernel 4 implementation

template <const int BM, const int BN, const int BK, const int TM>
__global__ void __launch_bounds__((BM / TM) * BN, 1)
sgemm_1D_thread_tiling(int M, int N, int K, float alpha, const float *A, 
                      const float *B, float beta, float *C) {
    __shared__ float blockA[BM * BK];
    __shared__ float blockB[BK * BN];

    // ?? BM = shared block rows; 64
    // ?? BN = shared block columns; 64
    // ?? BK = number of columns in A and rows in B; 8
    // ?? TM = number of rows in C produced by each thread; 8

    // ?? threadidx.x range from 0-512 for the current config

    // ?? tBlock is BK x BN (each thread is responsible for a TM x BN tile of C)
    int tBlockCol = threadIdx.x % BN; // ranges from 0 - 63
    int tBlockRow = threadIdx.x / BN; // ranges from 0 - 7

    int cCol = blockIdx.x * BN + tBlockCol; // absolute column in C
    int cSRow = blockIdx.y * BM + tBlockRow * TM; // first row this thread owns

    float val[TM] = {0.f};

    // ?? iterate over the K dimension in BK-wide slices so we can load each row of A and column of B
    for (int k = 0; k < K; k += BK) {

        // blockA = BM x BK
        // A = MxK
        int aRow = blockIdx.y * BM + (threadIdx.x / BK);
        int aCol = k + (threadIdx.x % BK);
        int aBlockRow = threadIdx.x / BK; 
        int aBlockCol = threadIdx.x % BK; 
        // ?? blockA is BM x BK, so we need to compute the index into the blockA array
        int aBlockIdx = aBlockRow * BK + aBlockCol; // index into the blockA array
        int aIdx = aRow * K + aCol; // absolute index into A
        blockA[aBlockIdx] = ((aRow < M) && (aCol < K))? A[aIdx] : 0.f; // use aIdx for loading A

        // blockB = BK x BN
        // B = KxN
        int bRow = k + tBlockRow;
        int bCol = cCol;
        int bBlockIdx = tBlockRow * BN + tBlockCol;
        int bIdx = bRow * N + bCol; // absolute index into B
        blockB[bBlockIdx] = ((bRow < K) && (bCol < N)) ? B[bIdx] : 0.f;
        
        __syncthreads(); 
        // ?? Now we have a BM x BK tile of A and a BK x BN tile of B in shared memory for all
        // ?? threads in the block to use for the dot product

        // !! want to make sure to only load in memory when needed to optimize

        // ?? Now we need to compute the dot product for each entry in the TM rows of this C column
        // ?? We have BM rows in the block, and each thread computes TM rows
        
        // ?? For each TM value in this column of C, we need to calculte the dot product

        // ?? each thread now needs to compute a TM stripe by iterating over the BK-wide tile
        // ?? each iteration moves over column in A and row in B
        for (int tileCol = 0; tileCol < BK; ++tileCol) {

            // blockB = BK x BN
            float b_val = blockB[tileCol * BN + tBlockCol]; // get a single value from the bRow, tBlockCol position

            int aBase = (tBlockRow * TM) * BK + tileCol;   // first part of the index into blockA and the second part is the row in the BK-wide tile

            // since blockA is BM x BK, we can compute the dot product for each TM row
            // and move down a row by incrementing aBase by BK
            for (int r = 0; r < TM; ++r) {
                float a_val = blockA[aBase + r * BK];
                val[r] += a_val * b_val;
            }
        }

        __syncthreads();
    }

    // ?? Now we have the accumulated values in val for each TM row in this column of C
    for (int r = 0; r < TM; ++r) {
        int cRow = cSRow + r;
        if (cRow < M && cCol < N) {
            int cIdx  = cRow * N + cCol;
            C[cIdx] = alpha * val[r] + beta * C[cIdx];
        }
    }
}

// -------------------------------------------------------------------------------------
// Kernel 5 implementation

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM / TM) * (BN / TN), 1)
sgemm_2D_thread_tiling(int M, int N, int K, float alpha, const float *A,
                      const float *B, float beta, float *C) {
    __shared__ float blockA[BM * BK];
    __shared__ float blockB[BK * BN];

    // ?? compute TM x TN tiles of C
    // ?? each thread is responsible 
                      
    int numTiles = (BM / TM) * (BN / TN);

    // first calculate what row in the block and then the column in the block
    // ?? threadIdx.y and threadIdx.x [0, 15] for this config
    int threadBlockIdx = threadIdx.y * (BN / TN) + threadIdx.x; // range from 0-256

    int blockRow = blockIdx.y; // range from 0-16
    int blockCol = blockIdx.x; // range from 0-16

    // multiply the block row and column by the tile size to get the starting row and column in C
    int rowBase = blockRow * BM + threadIdx.y * TM;
    int colBase = blockCol * BN + threadIdx.x * TN;

    float val[TM][TN] = {0.f};

    for (int k = 0; k < K; k += BK) {

        // ?? now each thread block loads a BM x BK tile of A and a BK x BN tile of B
        // ?? each thread in the block loads a single row of A and a single column of B

        // ?? coalesce A loads across threads in the block
        // ?? consecutive threads load consecutive rows of A

        // ?? fewer threads than elements now so we need to use a threadBlockIdx to iterate over the tiles

        // ?? now we move in strides over the elements to load in all of the necessary tiles
        // ?? coleasce calls across threads in the block, and then move in strides over the elements
        
        for (int idx = threadBlockIdx; idx < BK * BN; idx += numTiles) {
            // blockA = BM x BK
            int aBlockRow = idx / BK;
            int aBlockCol = idx % BK;
            // A = MxK so we calculate the row and column indices
            int aRow = blockRow * BM + aBlockRow;
            int aCol = k + aBlockCol;

            // blockB = BK x BN
            int bBlockRow = idx / BN;
            int bBlockCol = idx % BN;
            // B = KxN so we calculate the row and column indices
            int bRow = k + bBlockRow;
            int bCol = blockCol * BN + bBlockCol;
            
            // !! add a bounds check to make this pass the divisible test
            blockA[idx] = ((aRow < M) && (aCol < K)) ? A[aRow * K + aCol] : 0.f;
            blockB[idx] = ((bRow < K) && (bCol < N)) ? B[bRow * N + bCol] : 0.f;
        }

        __syncthreads();

        for (int innerCol = 0; innerCol < BK; ++innerCol) {
            // !! blockA = BM x BK ; blockB = BK x BN

            // ?? since we want to load TM x TN tiles of C, we load 
            // ?? TM elements from A from a single column in A
            // ?? and TN elements from a row in B

            for (int m = 0; m < TM; ++m) {
                // ?? each thread loads in a single column of A so first calculate
                // ?? the row of A within the block using the TM slice
                int aIdx = (threadIdx.y * TM + m) * BK + innerCol;
                // ?? then compute the column within the block using
                for (int n = 0; n < TN; ++n) {
                    // ?? first compute the row within the block
                    // ?? then compute the column within the block using
                    // ?? each TN slice across the row
                    int bIdx = innerCol * BN + threadIdx.x * TN + n;
                    val[m][n] += blockA[aIdx] * blockB[bIdx];
                }
            }
        }
        __syncthreads(); 
    }

    for (int m = 0; m < TM; ++m) {
        int cRow = rowBase + m;
        if (cRow < M) {
            for (int n = 0; n < TN; ++n) {
                int cCol = colBase + n;
                if (cCol < N){
                    int cIdx = cRow * N + cCol;
                    C[cIdx] = alpha * val[m][n] + beta * C[cIdx];
                }
            }
        }
    }
}

// -------------------------------------------------------------------------------------
// Kernel launcher

inline void launch_kernel(long kernel_num, int M, int K, int N, float alpha, 
                          const float *d_A, const float *d_B, float beta, float *d_C) {
    if(kernel_num == 1) {
        dim3 thr_per_blk(32, 32);
        dim3 blk_in_grid(((N + 31) / 32), ((M + 31) / 32));
        
        sgemm_naive<<<blk_in_grid, thr_per_blk>>>(M, K, N, alpha, d_A, d_B, beta, d_C);
    } else if(kernel_num == 2) {
        dim3 thr_per_blk(32, 32);
        dim3 blk_in_grid((N + 31) / 32, (M + 31) / 32);

        sgemm_global_coalescing<<<blk_in_grid, thr_per_blk>>>(M, K, N, alpha, d_A, d_B, beta, d_C);
    } else if(kernel_num == 3) {
        const uint BLOCKDIM = 32;

        dim3 thr_per_block(BLOCKDIM, BLOCKDIM);
        dim3 blk_in_grid( (N + BLOCKDIM - 1) / BLOCKDIM,   // one block produces a BLOCKDIM blo,ck
                          (M + BLOCKDIM - 1) / BLOCKDIM );

        sgemm_shared_mem_cache<BLOCKDIM><<<blk_in_grid, thr_per_block>>>(M, K, N, alpha, d_A, d_B, beta, d_C);
    } else if(kernel_num == 4) {
        const uint BM = 64;
        const uint BN = 64;
        const uint BK = 8;
        const uint TM = 8;

        dim3 thr_per_blk((BM / TM) * BN); // 512 threads
        dim3 blk_in_grid((N + BN - 1) / BN, // columns of tiles
                         (M + BM - 1) / BM); // rows   of tiles

        sgemm_1D_thread_tiling<BM, BN, BK, TM><<<blk_in_grid, thr_per_blk>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    } else if(kernel_num == 5) {
        const uint BM = 128;
        const uint BN = 128;
        const uint BK = 8;
        const uint TM = 8;
        const uint TN = 8;

        dim3 thr_per_blk(BN / TN, // 16 for this config
                     BM / TM);

        dim3 blk_in_grid((N + BN - 1) / BN, 
                        (M + BM - 1) / BM);

        sgemm_2D_thread_tiling<BM, BN, BK, TM, TN><<<blk_in_grid, thr_per_blk>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    } else {
        fprintf(stderr, "Error: %lu is not a valid kernel number.\n", kernel_num);
        exit(1);
    }
}

#endif // KERNELS