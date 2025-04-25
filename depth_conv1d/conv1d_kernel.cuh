#include <cuda_runtime.h>
#include "../cuda_error.cuh"

const int MAX_K = 5;

template <const int BLOCKDIM>
__global__ void conv1d_kernel(
    const float* __restrict__ u,
    const float* __restrict__ filter,
    const float* __restrict__ bias,
    float*       __restrict__ out,
    int B, int L, int D, int K)
{
    // ?? Each block is calculating over a 1024 length segment
    // ?? need to account for the overlap padding on the left and right
    __shared__ float convChunk[BLOCKDIM + MAX_K - 1];
    
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    int d = blockIdx.y;
    int b = blockIdx.z;
    
    // ?? size of padding on each end
    int pad = K / 2;
    // ?? from where this chunk of shared memory starts / threadblock
    int blockStart = blockIdx.x * blockDim.x - pad;
    
    // ?? need to iterate over the input up to the side of the shared block
    // ?? increment by blockDim incase some threads need to load to multiple elements
    for (int i = threadIdx.x; i < blockDim.x + K - 1; i += blockDim.x) {
        // ?? will be negative for block 0 and > L for the last blocks
        int loadIdx = blockStart + i;
        
        // ?? handles the padding cases
        if (loadIdx >= 0 && loadIdx < L) {
            convChunk[i] = u[b * D * L + d * L + loadIdx];
        } else {
            // ?? pads the input
            convChunk[i] = 0.0f;
        }
    }
    __syncthreads();
    
    if (l < L) {
        // ?? accumulate result for this thread (each one only responsible for one dimension)
        float result = 0.0f;
        // ?? iterate over the fitler length
        for (int k = 0; k < K; k++) {
            result += convChunk[threadIdx.x + k] * filter[d * K + k];
        }
        // ?? lol just keep nesting the multiplications for index
        out[b * D * L + d * L + l] = result + bias[d];
    }
}

inline void launch_kernel(
    const float* u,
    const float* filter,
    const float* bias,
    float*       out,
    uint B, uint L, uint D, uint K)
{
    // !! make each thread block responsible for a part of the output sequence
    const uint BLOCKDIM = 1024;
    
    dim3 blockDims(BLOCKDIM);
    dim3 gridDims(
        (L + BLOCKDIM - 1) / BLOCKDIM, // ?? set to 256 but change to 512
        D,
        B
    );
    
    cudaFuncSetCacheConfig(conv1d_kernel<BLOCKDIM>, cudaFuncCachePreferL1);

    conv1d_kernel<BLOCKDIM><<<gridDims, blockDims>>>(
        u, filter, bias, out, B, L, D, K
    );

    cudaCheckError(cudaPeekAtLastError());
}