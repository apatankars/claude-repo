
# Conceptual Questions

## Warm-up: SAXPY

1. Suppose that we are performing SAXPY on two vectors of length $N$, with $T$ threads per block and where a single thread performs SAXPY for one entry of the output vector. Derive an expression in terms of $N$ and $T$ for how many thread blocks are needed to perform SAXPY (note that $T$ doesn't necessarily divide $N$!).

   In order to calculate the number of thread blocks necessary to perform SAXPY given that $N$ is not directly divisible by $T$, we need to calculate the cieling division. This is because if there are any threads that aren't easily divided into one our blocks, we allocate them to a final block that may not be filled entirely. For example, if $N=100$ and $T=32$, $N$ is not easily divisible by $T$ and if round, we would only get 3 thread blocks, where $4$ threads are unaccounted for. Therefore, we use cieling division to always ensure we have more than enough thread blocks. The number of thread blocks, $B$ can be expressed as
   $$ B = \frac{N+T-1}{T} $$ 
   where we are assuming integer division. Now calculating this from the same example above, $N=100$ and $T=32$, $B=4$ which is the correct and expected output.

2. Recall that our kernel will perform SAXPY for a unique entry of the output vector. Determine an expression, in terms of `gridDim.x`, `blockIdx.x`, `blockDim.x`, and/or `threadIdx.x`, to obtain a unique index into the output vector that the kernel will perform SAXPY for.

   In order to calculate the index into our final output array, we need to properly calculate this based on the index of the block, the number of threads in each block, and then the index of the thread within the block. For example, consider an input array where $N=1000$. Therefore, our output array for SAXPY will also be $N=1000$. If we assume that $T=256$, then we calculate (using the formula above), we need $4$ thread blocks. However, each individual `threadIdx.x` will now only span from `[0, 256)`, therefore we need to incorporate the information about which block it is in so it properly scales to the output. We know which block we are in using `blockIdx.x` and know the number of threads in each block `blockDim.x`. Using the combination of these factors, we get that final index is
    $$\texttt{outputIdx}=(\texttt{blockIdx.x} \cdot \texttt{blockDim.x})+\texttt{threadIdx.x}$$

3. Research the theoretical memory bandwidths for each of the 3 above transfers for the GPU you are running on. Running the SAXPY kernel (e.g. `./saxpy -n 1024`) will display the name of your GPU model. You can also determine this from the Hydra ID of the GPU (given at the start of your shell prompt in a terminal), which you can match to the corresponding GPU model on [Hydra's webpage](https://cs.brown.edu/about/system/services/hpc/resources/).

    We recommend using [this site](https://www.techpowerup.com/gpu-specs/) for researching the specifications of GPUs. *(Hint: The bandwidth of host-to-device and device-to-host transfers is determined by its bus interface. PCIe 3.0 has a bandwidth of 1GB/s per lane, while PCIe 4.0 has a bandwidth of 2GB/s per lane. For a PCIe specification, the number after the x refers to the number of lanes.)*

    The bandwidth of host-to-device and device-to-host transfers is determined by its bus interface as mentioned. I am working on NVIDIA GeForce GTX 1080 where it uses PCIe 3.0x16 that has a a bandwidth of 1GB/s per lane, with 16 lanes. That gives us the transfer rates between device and host (in the ideal sense) of 16GB/s in total. As for device-to-device transfer rates, this is determined by its memory bandwidth internally which is 320GB/s. 

4. Determine an expression for the number of bytes that the SAXPY kernel reads from/stores to memory, in terms of $N$, the number of elements in each input vector. Assume the kernel uses single precision (FP32), i.e. uses elements with a size of 4 bytes. Only consider the input and output vectors; you can ignore the loading of the constant parameters $n$ and $\alpha$ in your solution.

    The number of bytes to read or write a singular input vector of size $N$ where each element is represented by FP32, where each element is $4$ bytes is represented as
    $$ \texttt{inputBytesRead}= 4N$$
    Now, we have to read two input vectors of size $N$ and write on output vector of size $N$ as well. Therefore, the total number of bytes is
    $$ \texttt{totalBytesReadAndWrite}= 12N $$ 

5. Determine an expression for the number of bytes is moved from/to host memory with each `cudaMemcpy`, in terms of $N$, the number of elements in each input vector. Assume the kernel uses single precision (FP32), i.e. uses elements with a size of 4 bytes.

   Since each operation is reading or writing a vector of size $N$ where each element is $4$ btes, each `cudeMemcpy` is responsible for $4N$ bytes.

6. Research the theoretical compute bandwidth for the GPU you are running on. Running the SAXPY kernel (e.g. `./saxpy -n 1024`) will display the name of your GPU model. Assume the kernel uses single precision (FP32), i.e. uses elements with a size of 4 bytes. We recommend using [this site](https://www.techpowerup.com/gpu-specs/) for researching the specifications of GPUs. 

    The NVIDIA GeForce GTX 1080 has a compute bandwidth of 8.873 TFLOPS or $8.873 \times 10^{12} $ FLOPS.

7. Determine an expression for the number of FLOPs that the SAXPY kernel performs, in terms of $N$, the number of elements in each input vector.

    For every element $N$, in SAXPY we are computing $ z = \alpha \cdot x + y$. Therefore, each element is multiplied by $ \alpha$ and then needs to be added. Therefore, we get two operations per elements. This gets us a total of 
    $$ \texttt{totalFLOPS} = 2N $$

8. Determine the theoretical arithmetic intensity of the GPU model you're running on, in FLOPs/byte. Based on the displayed effective arithmetic intensity from the previous task, is your kernel compute or memory bound for $N = 10,000,000$? Why does this make sense, intuitively?

    The arithmetic internsity is calculated as 
    $$ \text{arithInten} = \frac{\texttt{\# of operations}}{\texttt{\# of bytes}}$$
    Therefore, the theoretical arithemtic intensity can be calculated as
    $$ \text{arithInten}_{\texttt{theoretical}}= \frac{8.873 \times 10^{12}}{320 \times 10^9} = 27.7 \texttt{ FLOPs/byte}$$
    The effective arithmetic indensity reported is $0.167 \texttt{ FLOPS/byte}$ which shows this is way below the GPU's theoretical arithmetic intensity. 
    This makes sense here because we are only performing two operations per element, but each element requires three memory accesses (read both inputs and write one output). Therefore, it makes sense that SAXPY is memory-intensive.

## Part 1: SGEMM

1. We'll first conceptually walk through how to work with row-major order matrices. For all of the below questions, assume that you are working with a $M \times N$ matrix, i.e. one with $M$ rows and $N$ columns.

    1. Determine an expression that gives the index $i$ in the 1D array representation of a matrix for an entry located at row $r$ and column $c$. (Note that $r$, $c$, and $i$ are all zero-indexed.)

       In order to calculate the index into the 1D representation of a 2D matrix using row-major ordering, we can calculate this as 
       $$ i = N \times r + c

    2. Determine expressions that give (1) the row $r$ and (2) the column $c$ of an entry in the matrix located at index $i$ in its 1D array representation. (Note that $r$, $c$, and $i$ are all zero-indexed.)

        In order to find the row index given the index into the 1D row-major vector, we must integer floor-divide the given index, $i$, by the number of columns as this tells us how many complete rows of $N$ fit. This is expressed as 
        $$ r = i // N$$
        In order to find the column, we can do the modulo of the the index by the number of columns, as this will tell us what is left over (which is the column index)
        $$ c = i \mod N$$

2. For the following questions, assume that we are performing SGEMM for a $M \times K$ matrix $A$, a $K \times N$ matrix $B$, and a $M \times N$ matrix $C$, just as described above.

    1. Determine an expression for the number of bytes that the SGEMM algorithm reads from/stores to memory, in terms of $M$, $K$, and $N$. Assume that (1) matrix multiplication can be done by reading each input matrix once and writing down the output matrix once, (2) that the operation is fused (no intermediate computation is stored and then reread), and (3) that the kernel uses single precision (FP32), i.e. uses elements with a size of 4 bytes.

        In order to compute SGEMM, which is computed $$ C = \alpha (A \times B) + \beta C $$ 
        Therefore, we need to read out inputs are $A, B,$ and $C$. We can calculate this as
        $$ \texttt{bytesRead} = 4MK + 4KN + 4MN $$
        Then, we need to write out a matrix, $C$ which can be calulated as 
        $$ \texttt{bytesWritten} = 4MN $$
        Therefore, we can calculate the total number of bytes in order to compute SGEMM is 
        $$ \texttt{bytesTotal} = 4MK + 4KN + 8MN $$
        

    2. Determine an expression for the number of FLOPs that the SGEMM kernel performs, in terms of $M$, $K$, and $N$.

        The total number of floating point operations required to compute SGEMM is is the number of operations in  $$ C = \alpha (A \times B) + \beta C $$ 
        We first need to compute $ A \times B$ which from looking at the code (or our previous assigment), we need to do $K$ multiplications and $K$ ($K-1$ from prev. assignment) additions. This gets us a total of $MN(2K)$ FLOPs for matrix multiplication. Then, we need to multiply all elements of $AB$ and $C$ by the scalars $\alpha$ and $\beta$ respectively. This requires $MN$ operations each. We then need to add these results together which is another $MN$ FLOPs. The total number of FLOPs can therefore be expressed
        $$ \texttt{totalFLOPs} = MN(2K) + 3MN = MN(2K+3)

3. Based on the displayed effective arithmetic intensity from the previous task and the theoretical arithmetic intensity of the GPU model you're running on (that you determined in the warm-up), is the SGEMM algorithm compute or memory bound when $M = N = K = 1024$? Why does this make sense, intuitively?

        Ended up having to restart my SSH, so my GPU changed to a NVIDIA GeForce RTX 2080 Ti. The device-to-device memory bandwidth is 616.0 GB/s and host-device bandwidth is 16GB/s as well (since PCIe 3.0 x16). The compute bandwidth is 13.45 TFLOPS ($13.45 \times 10^{12} $ FLOPS). Therefore, we calculate the theoretical arithmetic intensity as
        $$ \texttt{arithInten}_{\texttt{2080}} = \frac{13.45 \times 10^{12} \texttt{ operations}}{616 \times 10^9 \texttt{ bytes}}=21.834$$

        When look at the effective arithmetic intensity, we can see it is $128.188$ FLOPs/byte which results means our implementation is compute-bound. This is obvious because we doing $1024^2(2051)$ operations for only $4(1024^2 + 1024^2 + 2048^2)$ bytes of memory read. 

4. Given that each thread block computes a 32×32 block of the output $M \times N$ matrix $C$, derive the formulas for:

    1. The number of thread blocks $B_x$ required along the `x`-dimension, corresponding to the rows of the output matrix. Your answer should be in terms of $M$ and/or $N$.

         We need to groups the rows of the matrix, $M$, into blocks of 32 threads each. From above, we calulated the number of blocks we need is calculated as  the cieling division expressed as 
        $$ B = \frac{N+T-1}{T} $$

        Therefore, in order to calcuate $B_x$, it will be 
        $$ B_x = \frac{M+31}{32} $$

    2. The number of thread blocks $B_y$ required in the `y`-dimension, corresponding to the columns of the output matrix. Your answer should be in terms of $M$ and/or $N$.

        From the same logic as above, in order to calcuate $B_y$, it will be 
        $$ B_y = \frac{N+31}{32} $$

5. Recall that the kernel will perform SGEMM for a single entry of the output matrix. Determine expressions that give:

    1. a unique row `r` of the output matrix, in terms of `gridDim.x`, `blockIdx.x`, `blockDim.x`, and/or `threadIdx.x`.

        Similar to the calculations above, in order to calculate the row of the output matrix, we write this as 
        $$ \texttt{r = blockIdx.x * blockDim.x + threadIdx.x}

    2. a unique column `c` in terms of in terms of `gridDim.y`, `blockIdx.y`, `blockDim.y`, and/or `threadIdx.y`.

        Similar logic to above, 
        $$ \texttt{c = blockIdx.y * blockDim.y + threadIdx.y} $$

6. Let's explore how our memory access pattern changes with this optimization.

    1. In *our previous kernel*, how many times is an entry of $A$ loaded from *global memory*? an entry of $B$? Your answer should be in terms of $M$, $K$, and/or $N$.

        In our previous kernel for the global memory coalesced kernel, we need to load each entire of $A$, $N$ times and and $B$, $M$ times. This is because there are $M \times N$ threads that loop over the $K$ dimension, loading one entry from $A$ and one from $B$. Each element from $A$ needs to be loaded in from the global memory once for every column, so $N$ times. And similarly with $B$, we load each element one for every row, so $M$ times. 

    2. In *this described kernel*, how many times is an entry of $A$ loaded from *global memory*? an entry of $B$? Your answer should be in terms of $M$, $K$, and/or $N$.

        Now, in the shared memory kernel, each thread block shared a tiled chunk of A and B and then performs the calculations on the loaded chunk to accumulate the value, and the moves on to the next chunk. Therefore, each element from $A$ is loaded in $\frac{N}{32}$ and each element from $B$ is loaded in $\frac{M}{32}$ times.

    3. In *this described kernel*, how many times is an entry of $A$ loaded from *shared memory*? an entry of $B$? Your answer should be in terms of $M$, $K$, and/or $N$.

        Just like the global memory coalesced version, we still need multiply each element in $A$ by an element in $B$, $\mathbf{N}$ times and $B$ by $A$, $\mathbf{M}$ times.

    4. Considering just your answers to the above questions, how much lower does the latency for a memory load from shared memory ($t_s$) have to be, compared to the latency for a memory load from global memory ($t_g$), for the runtime of our kernel to decrease compared to the previous one? (Assume no other overhead is contributing to runtime.) Your answer will be a fraction representing $t_g/t_s$.

        If we are always loading from global memory, then we need to do one global load for every multiply–add and we have both $A$ and $B$ so it comes out do $2MNK$ operations, where the total latency becomes $2MNK(t_g)$. 

        Now, if we use the shared memory, we can decrease the reads from global memory by a factor of $\texttt{BLOCK\_DIM}$. Therefore, the total latency time to load from global memory for the shared memory is $\frac{2MNK(t_g)}{B}$. We then need to read each element again from the shared memory to compute the actual product, and again we have $A$ and $B$ so we get a total latency of $2MNK(t_s)$. This gets us a total latency of $\frac{2MNK(t_g)}{\texttt{BLOCK\_DIM}}+2MNK(t_s)$ for the shared memory. 

        Now, we want $$\frac{2MNK(t_g)}{B}+2MNK(t_s) < 2MNK(t_g)$$ We can then divide both sides by $2MNK$ and rearrange to get 
        $$t_s \;<\; t_g \;\Bigl(1 - \tfrac1B\Bigr)$$
        $$\frac{t_g}{t_s} \;>\; \frac{1}{1 - \tfrac1B}\;=\;\frac{B}{B-1}$$
        Apply this to where $\texttt{BLOCK\_DIM} = B = 32 $, we find 
        $$\frac{32}{31} \approx 1.0322580645 $$  
        as long as a shared‑memory load is even ~3 % faster than a global‑memory load, the tiled kernel will beat the basic one.
        
7. Suppose we had a kernel that performs matrix addition, with the same distribution of threads and thread blocks - each thread performs the operation for one entry of the output matrix, and each thread block computes over a 32x32 block of the output matrix. Would such an optimization (loading chunks of the input matrices that a thread block operates over into shared memory, to then be computed over) provide any performance boost? Why or why not?

    <!-- Your answer here --->

8. Let's determine the occupancy of our previous kernel. To start with, we need to determine the resource requirements of our kernel:
    1. The number of threads per thread block
    2. The amount of shared memory each thread block allocates
    3. The number of registers each thread requires

    The first we already know is 1024 threads per thread block, as defined by our launch parameters. Your job is to determine the latter two requirements for your `sgemm_shared_mem_cache` kernel. 

    To do so, uncomment the line `CCFLAGS += --ptxas-options=-v` in your `Makefile`, and recompile your `sgemm` executable by running `make clean all`. The PTX assembler will then output metadata about your compiled kernels, in particular the number of registers and amount of shared memory that each uses. (Consider what shared memory allocations the kernel makes - make sure the outputted number aligns with your expectation!)

    From looking at the output of the run with the CCFLAGS enabled, we get `Used 26 registers, 8192 bytes smem, 368 bytes cmem[0]`, which means that our final  list is
    1. The number of threads per thread block - `1024 threads/block`
    2. The amount of shared memory each thread block allocates - `8192 bytes/block`
    3. The number of registers each thread requires - `26 registers/thread`

    This does match up with the expectations since the shared memory for each block is a $32 \times 32$ which are each $4$ bytes so we get $2\bigg((32 \times 32) \cdot 4\bigg)=8192$ bytes.

9. Using the three resource requirements you determined in the previous question, determine the maximum number of thread blocks that a single SM of the GPU you're running on can support, based on each constraint individually. That is:
    - Calculate how many thread blocks a single SM could support if only the thread count were limiting (ignoring shared memory and registers).
    - Calculate how many thread blocks a single SM could support if only shared memory were limiting (ignoring thread count and registers).
    - Calculate how many thread blocks a single SM could support if only register usage were limiting (ignoring thread count and shared memory).

    To help you, we've provided a script that provides all the hardware statistics of the GPU you're running on. To get them, run `python3 query_config.py`. Use these statistics along with your kernel's resource requirements to calculate how many thread blocks a single SM can support under each individual constraint. Be sure to show your work!

    So the results were 

    1. If only the thread count were the limiting factor for each SM, then since the number of threads per multiprocessor is $2048$, then we could:
    $$\frac{2048 \texttt{ threads/SM}}{1048 \texttt{ threads/block}}=2 \texttt{ blocks/SM}$$
    
    2. If only shared memory were limiting and each multiprocessor has $98304$ bytes of shared memory:
    $$\frac{98304 \texttt{ bytes/SM}}{8192 \texttt{ bytes/block}}=12 \texttt{ blocks/SM}$$

    3. If only register usage were limiting and each SM has $65536$ registers, first we need to calculate the total number of registers in each SM by multiplying the number of registers per thread by the number of registers, so we get:
    $$26 \texttt{ registers/thread} \cdot 1024 \texttt{ threads} = 26624 \texttt{ threads}$$
    $$\frac{65536 \texttt{ registers/SM}}{26624 \texttt{ registers/block}}\approx 2 \texttt{ block/SM}$$
    


10. In the previous question, you calculated the maximum number of thread blocks a single SM could support under each resource constraint (thread count, shared memory, and register usage). We calculated this constraint using the unit of thread blocks as work is scheduled on an SM at the granularity of thread blocks, not warps.

    Now, you'll use your calculations to determine how many warps can be active on each SM. First, identify the limiting resource - that which determines the minimum number of thread blocks the SM can support. Then, using this number, calculate the total number of warps that can be scheduled at the same time on an SM, based on the limiting resource.

    Using the statistics outputted by `query_config.py`, then determine the maximum possible number of warps that a single SM could support. Then, use these two values to determine the occupancy of your kernel! 

    Your answer should provide (1) the number of warps that can be active on each SM at once, (2) which of the 3 resource constraints this was limited by, and (3) your calculated occupancy. Again, be sure to show your work for each of these answers!

    1. The number of warps that can be active on an each block is calculated as:
    - We have $32$ threads per warp and $1024$ threads per block
        - This means we have $\frac{1024}{32}=32$ warps/block
    2. Our most restrictive constraints are both the thread count and register count. Both of these limit us to just 2 blocks/SM. Therefore, we get a total of 
    $$ 32 \texttt{ warps/block} \cdot 2 \texttt{ blocks} = 64 \texttt{ warps} $$
    3. Each SM can have up to $64$ warps so our occupancy is
    $$ \frac{64 \texttt{ warps/SM}}{64 \texttt{ warps/SM}}=100\% \texttt{ occupancy} $$

11. Suppose that we have a kernel where each thread computes one entry in $C$. How many entries of $A$, $B$, and $C$ does that thread have to load from (shared) memory? How many entries are loaded per computed result? Your answers should both be in terms of $M$, $K$, and/or $N$.

    Suppose that we instead have a kernel where each thread computes $TM$ consecutive entries in the same column of $C$. How many entries of $A$, $B$, and $C$ does that thread have to load from (shared) memory? How many entries are loaded per computed result? Your answers should both be in terms of $M$, $K$, $N$, and/or $TM$.

    How does the number of entries loaded made per computed result change between the two scenarios? How does arithmetic intensity change?  

    For the first kernel where each thread computes one entry in $C$, then for each entry it has to load the entire row from $A$ so $K$ entries. It also has to read the entire column from $B$ so $K$ entries as well. Finally, we read one entry from $C$, so the total entries loaded from shared memory and loaded per computed result is $2K+1$.

    For the secodn kernel where each thread computes $TM$ consecutive entries in the same column of $C$, since each thread is now calculated $TM$ entries, we need to load $TM$ rows now so $TM \times K$ entries from $A$, still one column from $B$ for $K$ entries. Finally, we need to read $TM$ entries from $C$. The total memory loads is now $(TM \times K)+K+TM$. However, the total number of entries loaded per computed result is divided by $TM$ so we get $K + \frac{K}{TM} + 1$.

    The total number of entires loaded per computed result changed by $K-\frac{K}{TM}$ which approaches $K$ as $TM$ increases. For the first kernel, we can see that the arithmetic intensity is
    $$ \frac{ 2K+1 \texttt{ ops}}{ 2K+1 \texttt{ mem}} \approx 1 $$
    whereas for the second kernel, we can see 
    $$ \frac{ 2TM \cdot K \texttt{ ops}}{ TM \cdot K \texttt{ mem}} \approx 2 $$
    which shows the arithmetic intensity doubles for the tiling approach meaning we can do twice as many operations per memory load.

12. With this new configuration, how many threads should each thread block contain? Your answer should be in terms of $BM$, $BN$, $BK$, and/or $TM$.

    Each thread is now computing $TM$ entires for a column in the output $C$. Each thread block is now computing a $BM \times BN$ section of $C$. Therefore, each column will $\frac{BM}{TM}$ $ threads to compute the full column of the output section, since each column as $BM$ entries. Then, each block has $BN$ column to compute, so the total number of threads is 
    $$ \frac{BM}{TM} \times BN $$

13. Suppose that we instead have a kernel where each thread computes a $TM \times TN$ section of $C$. How many total loads from $A$, $B$, and $C$ does that thread have to make from (shared) memory? How many loads are made per computed result? Your answers should both be in terms of $M$, $K$, $N$, $TM$, and/or $TN$.

    Comparing your result to those from CQ 11, how does the number of entries loaded made per computed result change with this scenario? How does arithmetic intensity change?

    In order to calculate aa $TM \times TN$ section of $C$, we need to load $TM$ rows of $C$, each with $K$ elements. We need to also load $TN$ columns of $B$ with $K$ elements each and a $TM \times TN$ section of $C$. Therefore, the total memory loads is
    $$ TMK + TNK + \bigg(TM \times TN\bigg)=K\bigg(TM+TN\bigg)+\bigg(TM \times TN\bigg)$$
    however, since we are computing $TM \times TN$ entries, the memory loads per computed result is
    $$\frac{K}{TN}+\frac{K}{TM}+1$$

    The difference between the number of entries loaded made per computed result of the 1D-tiling and 2D-tiling approach is 
    $$K + \frac{K}{TM} + 1 - \frac{K}{TN}-\frac{K}{TM}-1 = K- \frac{K}{TN}$$
    so we can save by another factor of $K$.

    The total memory operations for a 
    $TM \times TN$ section of $C$ is $$K\bigg(TM+TN\bigg)+2\bigg(TM \times TN\bigg)$$ and the total number of arithmetic operations $TM \times TN \times K$ multiplications, $TM \times TN \times K$ additions, $2 \times TM \times TN$ for the scaling so the total is $$\bigg(TM \times TN\bigg)\bigg(2K + 1\bigg)$$
    therefore we can calculate the arithmetic intensity
    $$\frac{\bigg(TM \times TN\bigg)\bigg(2K + 1\bigg)}{K\bigg(TM+TN\bigg)+2\bigg(TM \times TN\bigg)}$$

    This comes out to around $2$ as well which shows that we have just also doubled the AI.


14. With this new configuration, how many threads should each thread block contain? Your answer should be in terms of $BM$, $BN$, $BK$, $TM$, and/or $TN$.

    Following similar logic from the 1D tiling, we know that each thread block is now computing a $BM \times BN$ rectangle of $C$. Each thread in the 1D tiling scenario was responsible for a sliver of a column, but now each thread is also responsble for a sliver of the row as well so each thread is responsible for $\frac{BM}{TM}$ rows, and $\frac{BN}{TN}$ rows which is expressed as
    $$\bigg(\frac{BM}{TM}, \frac{BN}{TN}\bigg)$$

## Part 2: 1D Convolution

1. What is the GPU memory bandwidth for the T4? What is the single precision compute bandwidth for the T4? A datasheet for the NVIDIA T4 GPU can be found [at this link](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf).

    The GPU memory bandwidth for a T4 is 300 GB/sec and the single precision compute bandwidth 8.1 TFLOPS. 

2. What is the theoretical arithmetic intensity for the T4, using the information above? Your answer should be given in FLOPs/byte.

   Given these stats on the T4, we can calculate the theoretical arithmetic intensity as
   $$\frac{8.1 \times 10^{12} \texttt{ FLOPS}}{320 \times 10^9 \texttt{ bytes}} = 25 FLOPs/byte$$

3. What is the total number of floating point operations required to compute the convolution?

   Since our kernel length is $3$ then for each element we need to do $3$ multiplications and $2$ additions to sum. Then there is one more addition to add the bias term so we get a total of $6$ operations per element. Then for every convolution we have a total of $B \times D \times L$ output elements. Therefore the number of floating point operations required to compute the convolution is 
   $$1 \times 8192 \times 8192 \times 6=402,653,184$$
   
4. What is the total number of bytes accessed (read + written) to compute the convolution? For this question, assume that the input is first read from memory, padded, and the padded input is written back to memory. As our filter has a length of 3, we will have to pad the original input along the $L$ dimension with 4 zeros, 2 on both sides. The padded input, filter, and bias are then read from memory, the convolution is computed, and the output is then written back to memory.

    First we read the original input which is $(1, 8192, 8192)$ and each element is 4 bytes so we read 
    $$1 \times 8192 \times 8192 \times 4=268,435,456 \texttt{ bytes}$$
    Then we write the padded output back out where the sequence length now increases by $2$
    $$1 \times 8192 \times 8194 \times 4=268,500,992 \texttt{ bytes}$$
    Then we read the padded input, filter, and bias
    $$1 \times 8192 \times 8194 \times 4=268,500,992 \texttt{ bytes}$$
    $$8192 \times 3 \times 4=98,304 \texttt{ bytes}$$
    $$ 8192 \times 4=32,768 \texttt{ bytes}$$
    Finally we write the output out which is the shape of the original input
    $$1 \times 8192 \times 8192 \times 4=268,435,456 \texttt{ bytes}$$
    We can then sum up all of the bytes to get
    $$\text{Total bytes}\approx1,074,003,968\texttt{ bytes}$$

5. Using your answers from the previous two questions, what is the arithmetic intensity of the operation?

   We can calculate the arithmetic intensity of the 1D convolution as
   $$\frac{402,653,184}{1,074,003,968}=0.3749$$

6. Is this operation compute or memory bound on a T4 GPU? Justify your answer.

    This operation is clearly memory-bound as it is less than the calculated theoretical AI for the T4 and we can see that we are doing far fewer operations per byte loaded.

7. Given your previous responses, why is having two separate kernels (instead of a single one): one for padding and another for computing the convolution a bad idea?

    Having two separate kernels where one is responsible for pdding and another the computing, it would mean we would need to separately read and write the input for each kernel which would add to the already memory-bound convolution. Instead, we want to find a way to do more computations on each memory load rather than increasing the number of them.

8. As mentioned in lecture and part 1, there a few simple principles that can help us achieve good performance (fusion, tiling, pipelining, caching/recomputation, etc.). What opportunities exist to apply (1) fusion and (2) tiling to a CUDA kernel that implements 1D depthwise convolution?

    The first and most obvious fusion opportunity is to pad the kernel on-the-fly and combine it with the same kernel for the convolution. Then as for tiling, we can load chunks of the input into shared memory and work over those or try to compute multiple outputs per thread like TM outputs per thread.

9. We want to expose enough parallel work to the GPU in order to achieve good performance. How will you parallelize the work in the algorithm (i.e. how will you split work along individual threads)?

   The way that I plan on dividing the work is so that each thread is responsible for one output position in the sequence which will be divided amongst the blocks. Each thread can handle a different position in the sequence, whereas each block is processing a different dimension, and then each batch is processed independently. 