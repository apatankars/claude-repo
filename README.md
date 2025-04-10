[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/R86Zsw65)
# Conceptual Questions

## Warm-up: SAXPY

1. Suppose that we are performing SAXPY on two vectors of length $N$, with $T$ threads per block and where a single thread performs SAXPY for one entry of the output vector. Derive an expression in terms of $N$ and $T$ for how many thread blocks are needed to perform SAXPY (note that $T$ doesn't necessarily divide $N$!).

    <!-- Your answer here --->

2. Recall that our kernel will perform SAXPY for a unique entry of the output vector. Determine an expression, in terms of `gridDim.x`, `blockIdx.x`, `blockDim.x`, and/or `threadIdx.x`, to obtain a unique index into the output vector that the kernel will perform SAXPY for.

    <!-- Your answer here --->

3. Research the theoretical memory bandwidths for each of the 3 above transfers for the GPU you are running on. Running the SAXPY kernel (e.g. `./saxpy -n 1024`) will display the name of your GPU model. You can also determine this from the Hydra ID of the GPU (given at the start of your shell prompt in a terminal), which you can match to the corresponding GPU model on [Hydra's webpage](https://cs.brown.edu/about/system/services/hpc/resources/).

    We recommend using [this site](https://www.techpowerup.com/gpu-specs/) for researching the specifications of GPUs. *(Hint: The bandwidth of host-to-device and device-to-host transfers is determined by its bus interface. PCIe 3.0 has a bandwidth of 1GB/s per lane, while PCIe 4.0 has a bandwidth of 2GB/s per lane. For a PCIe specification, the number after the x refers to the number of lanes.)*

    <!-- Your answer here --->

4. Determine an expression for the number of bytes that the SAXPY kernel reads from/stores to memory, in terms of $N$, the number of elements in each input vector. Assume the kernel uses single precision (FP32), i.e. uses elements with a size of 4 bytes. Only consider the input and output vectors; you can ignore the loading of the constant parameters $n$ and $\alpha$ in your solution.

    <!-- Your answer here --->

5. Determine an expression for the number of bytes is moved from/to host memory with each `cudaMemcpy`, in terms of $N$, the number of elements in each input vector. Assume the kernel uses single precision (FP32), i.e. uses elements with a size of 4 bytes.

    <!-- Your answer here --->

6. Research the theoretical compute bandwidth for the GPU you are running on. Running the SAXPY kernel (e.g. `./saxpy -n 1024`) will display the name of your GPU model. Assume the kernel uses single precision (FP32), i.e. uses elements with a size of 4 bytes. We recommend using [this site](https://www.techpowerup.com/gpu-specs/) for researching the specifications of GPUs. 

    <!-- Your answer here --->

7. Determine an expression for the number of FLOPs that the SAXPY kernel performs, in terms of $N$, the number of elements in each input vector.

    <!-- Your answer here --->

8. Determine the theoretical arithmetic intensity of the GPU model you're running on, in FLOPs/byte. Based on the displayed effective arithmetic intensity from the previous task, is your kernel compute or memory bound for $N = 10,000,000$? Why does this make sense, intuitively?

    <!-- Your answer here --->

## Part 1: SGEMM

1. We'll first conceptually walk through how to work with row-major order matrices. For all of the below questions, assume that you are working with a $M \times N$ matrix, i.e. one with $M$ rows and $N$ columns.

    1. Determine an expression that gives the index $i$ in the 1D array representation of a matrix for an entry located at row $r$ and column $c$. (Note that $r$, $c$, and $i$ are all zero-indexed.)

        <!-- Your answer here --->

    2. Determine expressions that give (1) the row $r$ and (2) the column $c$ of an entry in the matrix located at index $i$ in its 1D array representation. (Note that $r$, $c$, and $i$ are all zero-indexed.)

        <!-- Your answer here --->

2. For the following questions, assume that we are performing SGEMM for a $M \times K$ matrix $A$, a $K \times N$ matrix $B$, and a $M \times N$ matrix $C$, just as described above.

    1. Determine an expression for the number of bytes that the SGEMM algorithm reads from/stores to memory, in terms of $M$, $K$, and $N$. Assume that (1) matrix multiplication can be done by reading each input matrix once and writing down the output matrix once, (2) that the operation is fused (no intermediate computation is stored and then reread), and (3) that the kernel uses single precision (FP32), i.e. uses elements with a size of 4 bytes.

        <!-- Your answer here --->

    2. Determine an expression for the number of FLOPs that the SAXPY kernel performs, in terms of $M$, $K$, and $N$.

        <!-- Your answer here --->

3. Based on the displayed effective arithmetic intensity from the previous task and the theoretical arithmetic intensity of the GPU model you're running on (that you determined in the warm-up), is the SGEMM algorithm compute or memory bound when $M = N = K = 1024$? Why does this make sense, intuitively?

    <!-- Your answer here --->

4. Given that each thread block computes a 32×32 block of the output $M \times N$ matrix $C$, derive the formulas for:

    1. The number of thread blocks $B_x$ required along the `x`-dimension, corresponding to the rows of the output matrix. Your answer should be in terms of $M$ and/or $N$.

        <!-- Your answer here --->

    2. The number of thread blocks $B_y$ required in the `y`-dimension, corresponding to the columns of the output matrix. Your answer should be in terms of $M$ and/or $N$.

        <!-- Your answer here --->

5. Recall that the kernel will perform SGEMM for a single entry of the output matrix. Determine expressions that give:

    1. a unique row `r` of the output matrix, in terms of `gridDim.x`, `blockIdx.x`, `blockDim.x`, and/or `threadIdx.x`.

        <!-- Your answer here --->

    2. a unique column `c` in terms of in terms of `gridDim.y`, `blockIdx.y`, `blockDim.y`, and/or `threadIdx.y`.

        <!-- Your answer here --->

6. Let's explore how our memory access pattern changes with this optimization.

    1. In *our previous kernel*, how many times is an entry of $A$ loaded from *global memory*? an entry of $B$? Your answer should be in terms of $M$, $K$, and/or $N$.

        <!-- Your answer here --->

    2. In *this described kernel*, how many times is an entry of $A$ loaded from *global memory*? an entry of $B$? Your answer should be in terms of $M$, $K$, and/or $N$.

        <!-- Your answer here --->

    3. In *this described kernel*, how many times is an entry of $A$ loaded from *shared memory*? an entry of $B$? Your answer should be in terms of $M$, $K$, and/or $N$.

        <!-- Your answer here --->

    4. Considering just your answers to the above questions, how much lower does the latency for a memory load from shared memory ($t_s$) have to be, compared to the latency for a memory load from global memory ($t_g$), for the runtime of our kernel to decrease compared to the previous one? (Assume no other overhead is contributing to runtime.) Your answer will be a fraction representing $t_g/t_s$.

        <!-- Your answer here --->

7. Suppose we had a kernel that performs matrix addition, with the same distribution of threads and thread blocks - each thread performs the operation for one entry of the output matrix, and each thread block computes over a 32x32 block of the output matrix. Would such an optimization (loading chunks of the input matrices that a thread block operates over into shared memory, to then be computed over) provide any performance boost? Why or why not?

    <!-- Your answer here --->

8. Let's determine the occupancy of our previous kernel. To start with, we need to determine the resource requirements of our kernel:
    1. The number of threads per thread block
    2. The amount of shared memory each thread block allocates
    3. The number of registers each thread requires

    The first we already know is 1024 threads per thread block, as defined by our launch parameters. Your job is to determine the latter two requirements for your `sgemm_shared_mem_cache` kernel. 

    To do so, uncomment the line `CCFLAGS += --ptxas-options=-v` in your `Makefile`, and recompile your `sgemm` executable by running `make clean all`. The PTX assembler will then output metadata about your compiled kernels, in particular the number of registers and amount of shared memory that each uses. (Consider what shared memory allocations the kernel makes - make sure the outputted number aligns with your expectation!)

    <!-- Your answer here --->

9. Using the three resource requirements you determined in the previous question, determine the maximum number of thread blocks that a single SM of the GPU you're running on can support, based on each constraint individually. That is:
    - Calculate how many thread blocks a single SM could support if only the thread count were limiting (ignoring shared memory and registers).
    - Calculate how many thread blocks a single SM could support if only shared memory were limiting (ignoring thread count and registers).
    - Calculate how many thread blocks a single SM could support if only register usage were limiting (ignoring thread count and shared memory).

    To help you, we've provided a script that provides all the hardware statistics of the GPU you're running on. To get them, run `python3 query_config.py`. Use these statistics along with your kernel's resource requirements to calculate how many thread blocks a single SM can support under each individual constraint. Be sure to show your work!

    <!-- Your answer here --->

10. In the previous question, you calculated the maximum number of thread blocks a single SM could support under each resource constraint (thread count, shared memory, and register usage). We calculated this constraint using the unit of thread blocks as work is scheduled on an SM at the granularity of thread blocks, not warps.

    Now, you'll use your calculations to determine how many warps can be active on each SM. First, identify the limiting resource - that which determines the minimum number of thread blocks the SM can support. Then, using this number, calculate the total number of warps that can be scheduled at the same time on an SM, based on the limiting resource.

    Using the statistics outputted by `query_config.py`, then determine the maximum possible number of warps that a single SM could support. Then, use these two values to determine the occupancy of your kernel! 

    Your answer should provide (1) the number of warps that can be active on each SM at once, (2) which of the 3 resource constraints this was limited by, and (3) your calculated occupancy. Again, be sure to show your work for each of these answers!

    <!-- Your answer here --->

11. Suppose that we have a kernel where each thread computes one entry in $C$. How many entries of $A$, $B$, and $C$ does that thread have to load from (shared) memory? How many entries are loaded per computed result? Your answers should both be in terms of $M$, $K$, and/or $N$.

    Suppose that we instead have a kernel where each thread computes $TM$ consecutive entries in the same column of $C$. How many entries of $A$, $B$, and $C$ does that thread have to load from (shared) memory? How many entries are loaded per computed result? Your answers should both be in terms of $M$, $K$, $N$, and/or $TM$.

    How does the number of entries loaded made per computed result change between the two scenarios? How does arithmetic intensity change?  

    <!-- Your answer here --->


12. With this new configuration, how many threads should each thread block contain? Your answer should be in terms of $BM$, $BN$, $BK$, and/or $TM$.

    <!-- Your answer here --->

13. Suppose that we instead have a kernel where each thread computes a $TM \times TN$ section of $C$. How many total loads from $A$, $B$, and $C$ does that thread have to make from (shared) memory? How many loads are made per computed result? Your answers should both be in terms of $M$, $K$, $N$, $TM$, and/or $TN$.

    Comparing your result to those from CQ 11, how does the number of entries loaded made per computed result change with this scenario? How does arithmetic intensity change?

    <!-- Your answer here --->

14. With this new configuration, how many threads should each thread block contain? Your answer should be in terms of $BM$, $BN$, $BK$, $TM$, and/or $TN$.

    <!-- Your answer here --->

## Part 2: 1D Convolution

1. What is the GPU memory bandwidth for the T4? What is the single precision compute bandwidth for the T4? A datasheet for the NVIDIA T4 GPU can be found [at this link](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf).

    <!-- Your answer here --->

2. What is the theoretical arithmetic intensity for the T4, using the information above? Your answer should be given in FLOPs/byte.

    <!-- Your answer here --->

3. What is the total number of floating point operations required to compute the convolution?

    <!-- Your answer here --->

4. What is the total number of bytes accessed (read + written) to compute the convolution? For this question, assume that the input is first read from memory, padded, and the padded input is written back to memory. As our filter has a length of 3, we will have to pad the original input along the $L$ dimension with 4 zeros, 2 on both sides. The padded input, filter, and bias are then read from memory, the convolution is computed, and the output is then written back to memory.

    <!-- Your answer here --->

5. Using your answers from the previous two questions, what is the arithmetic intensity of the operation?

    <!-- Your answer here --->

6. Is this operation compute or memory bound on a T4 GPU? Justify your answer.

    <!-- Your answer here --->

7. Given your previous responses, why is having two separate kernels (instead of a single one): one for padding and another for computing the convolution a bad idea?

    <!-- Your answer here --->

8. As mentioned in lecture and part 1, there a few simple principles that can help us achieve good performance (fusion, tiling, pipelining, caching/recomputation, etc.). What opportunities exist to apply (1) fusion and (2) tiling to a CUDA kernel that implements 1D depthwise convolution?

    <!-- Your answer here --->

9. We want to expose enough parallel work to the GPU in order to achieve good performance. How will you parallelize the work in the algorithm (i.e. how will you split work along individual threads)?

    <!-- Your answer here --->