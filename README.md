[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/T8a-qz2H)

# Assignment 2: Optimizing Transformers

## Part 1.1: Implementing Attention

1. Provide a brief (1-3 sentence) explanation for the “causal mask” in self-attention. What is its purpose, and how is it calculated?

The causal mask is used during training for models that use self-attention. This is because during training, the model has access to future tokens, but this is against the auto-regressive nature of the task. So, in order to properly make sure the model predicts the next token and doesn't just peek ahead, a casual mask is applied such that the attention scores for future tokens are "masked" so that they do not affect the final attention output for that token. This is calculated by using the attention score matirx and making the values in the upper right hand triangle (all future tokens) are very negative such that when they are added to the calculated attention scores, they still zero out.

2. To build intuition for the number of attention layers required to solve this associative recall task, determine when the model starts solving the task with 90+% accuracy as a function of the number of attention layers. You can control the number of such layers as an argument to `run.sh` (e.g. `./run.sh num_layers=4`). What is the minimum number of layers at which the transformer starts solving with 90+% accuracy?

   We need two attention layers to run this associative recall task with 90+% accuracy. This makes intuitive sense because the first attention layer is required to learn the token embedding of the previous token was. This will help us keep track of what tokens have appeared before. However, to make sure that each key-value pair only pays attention to its key or value, we need a second layer that learns the token representation for the proceeding token. This forward-backwards relationship builds the representations, while the second layer uses these representations to perform the actual recall operation when queried.

## Part 2.1: Analyzing Transformer Computations

### Part 2.1.1: Matrix multiplication

1. Derive an expression for the number of floating point operations required to perform the operation $C = AB$. Your answer should be in terms of $M$, $K$, and/or $N$.

   Lets assume A is a matrix of teh shape shape $M \times K$, B is a matrix of the shape $K \times N$. Therefore, we want C to be a matrix of the shape $M \times N$.

   For each element in $C$, $C[i,j]$, we need to compute $$A[i,k] \times B[k,j] \quad \quad i \in M \mid \mid j \in N$$Therefore, for each element we need to compute $K$ multiplication operations and $K-1$ additions to sum up the products. A total of $2K-1$ operation will be needed for each element in $C$.

   Now, we have a total of $M \times N$ operations in terms of just $M$, $K$, and $N$ is $$2MNK$$

2. Derive an expression for the total number of bytes read/written to memory in order to perform the operation $C = AB$. Assume, as well as for the rest of the assignment, that the operation can be done by reading each input matrix once and writing down the output matrix once. Your answer should be in terms of $M$, $K$, $N$, and/or $n$.

   In order to compute $C = AB$ we need to read matrix $A$, read matrix $B$ and write matrix $C$. Since $A$ is a matrix of the shape $M \times K$ and $B$ is of the shape $K \times N$. We represent the number of bytes for each element as $n$. Therefore the read costs can be expressed as $$A = n(M \times K)$$ $$B = n(K \times N)$$ Now the result of this multiplication needs to be written to matrix $C$ of the shape $M \times N$ so this cost is expressed as
   $$C = n(M \times N)$$
   so the total memory read and writes is expressed as
   $$\textrm{Total} = n (MK + KN + MN)

### Part 2.1.2: Input Projections

1. Derive an expression for the number of floating point operations required to perform the input projections and generate $Q$, $K$, and $V$. Your answer should be in terms of $B$, $N$, and/or $D$.

   We have an input matrix, $X$, of the shape $(B, N, D)$ where $B$ is batch size, $N$ is sequence length, and $D$ is the model embedding dimension. In order to calculate $Q$, $K$, and $V$, we need
   $$Q = X \times W_Q \quad \mid \quad K = X \times W_K \quad \mid \quad V = X \times W_V$$
   Now, each of these intermediary weight matrices are of the shape $(D \times D)$. Therefore, we can express the calculations for each $K, Q$ and $V$ using the above formula
   $$\textrm{Key || Queries || Value} = 2BND^2$$
   since we are multiplying a $(B, N, D)$ matrix with a $(D, D)$ matrix. Since we have $3$ matrices, $K, Q$ and $V$, we express the total as
   $$\textrm{Total FLOPs for Q,K,V}= 6BND^2$$

2. Derive an expression for the total number of bytes read/written to memory in order to perform the input projections. You should assume that any inputs for an operation are read newly _each time_ that operation is performed. Your answer should be in terms of $B$, $N$, $D$, and/or $n$.

   In order to perform the input projections for $K, Q$ and $V$ where we need to read the input everytime, we need to

   - Read the input $(B,N,D)$ on every iteration
   - Read the weight matrix $(D,D)$ either $W_K, W_Q$ or $W_V$
   - Write the output matrix $(B,N,D)$ either $K, Q$ or $V$

   For each matrix, $K, Q$ or $V$, we need
   $$\textrm{Bytes per K,Q,V} =n(2BND+D^2)$$

   again, where $n$ is the number of bytes per element. Now, again we have to do this for each $K, Q$ and $V$ so we get
   $$\textrm{Total Bytes for K,Q,V} =n(6BND+3D^2)$$

### Part 2.1.3: Scaled Dot-Product Attention

1. **Computing dot-product scores via $QK^T$**

   For this question, you may assume that $\sqrt{d_k}$ is provided and you do not need to read it from memory or compute it. You also do not need to consider the transpose operation in your answer (you can think of it as a (negligible) change to the underlying metadata about $K$).

   1. Derive an expression for the number of floating point operations required to perform the operation of $\frac{QK^T}{\sqrt{d_k}}$. Your answer should be in terms of $B$, $N$, and/or $D$.

      We have the shapes of $Q$ and $K$ as $(N, D)$ for $B$ batches which means that the operation $QK^T$ can be expressed as a batched multiplication of a $N \times D$ matrix by a $D \times N$ matrix $B$ times. Using the formula from above, we get

      $$\textrm{O}=2BN^2D$$

      Now we have to divide each element in the output matrix $O$, which has the shape $(B,N,N)$, by $\sqrt{d_k}$. This results in
      $$\textrm{Normalized O} = BN^2$$

      operations. Therefore, the total number of FLOPs to perform $\frac{QK^T}{\sqrt{d_k}}$ can be expressed as
      $$\textrm{Total FLOPs} = 2BN^2D + BN^2$$

      where the dominating term is just

      $$\textrm{Total FLOPs} = 2BN^2D $$

   2. How many bytes are read from/written to memory for the operation? (You may assume that $\frac{QK^T}{\sqrt{d_k}}$ is a fused operation: both the $QK^T$ matrix multiplication and the division by $\sqrt{d_k}$ happen before any result is written to memory.) Your answer should be in terms of $B$, $N$, $D$, and/or $n$.

      In order to compute ${QK^T}{\sqrt{d_k}}$, we have to read $Q$ and $K$ and write back the result ($d_k$ is provided). We know that $Q$ and $K$ have the shapes $(B, N, D)$ and each element is $n$ bytes so for each matrix
      $$\text{Bytes Read per Matrix} = n(BND)$$

      we do this two times for the key and query, so we get a total of
      $$\text{Total Bytes Read} = n(2BND)$$

      Now, we have to write out a matrix of the shape $(B,N,N)$ so the total write cost is

      $$\text{Total Bytes Written} = n(BN^2)$$

      So the total bytes read from/written to memory for the operation is

      $$\text{Total Bytes} = n(2BND + BN^2)$$

2. **Causal masking**

   Masking is used to enforce a causal relationship where the output at each step does not depend on future steps, thereby preserving the temporal order of the input data. Masking can be implemented in multiple ways. For this problem, you can assume that masking is performed by an elementwise check (as a single FLOP) if an entry should be masked (based on its position in $QK^T$), then reading from memory either the masking value ($-\infty$) or the entry in $QK^T$ at that location, and then storing this value in the transformed $QK^T$.

   1. What is the number of floating point operations required for the masking operation? Your answer should be in terms of $B$, $N$, and/or $D$.

      Each element in the output matrix of the operation $QK^T$, $O$, needs to be checked if it needs to be masked. Since $O$ has the shape $(B,N,N)$ and each elementwise check is a single FLOP, this is simply expressed as
      $$\textrm{FLOPs for Masking} = BN^2$$

   2. How many bytes are read from/written to memory for the masking operation? Your answer should be in terms of $B$, $N$, $D$, and/or $n$.

      In order to perform masking, we first need to read $O$ and then write back $O'$ which applies the masking. The shapes of $O$ and $O'$ are $(B, N, N)$ and each element is $n$ bytes. Therefore we can express the total bytes read/written to/from memory as
      $$\text{Bytes for Masking} = n(2BN^2)$$

3. **Softmax**

   The softmax function is used in machine learning, particularly in the field of deep learning, to convert a vector of real numbers into a probability distribution. Each output of the softmax function is between 0 and 1, and the sum of the outputs is equal to 1. The softmax function $S$ for a vector $z$ of length $K$ is defined as follows:
   $$S(z)_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}$$
   Denote the output of the masking stage as $A = \frac{QK^T}{\sqrt{d_k}}$

   1. What is the number of floating point operations required for the softmax operation, i.e. $\text{softmax}(A)$? You should assume that the summation in the denominator is calculated _once_ per the dimension softmax operates over. Your answer should be in terms of $B$, $N$, and/or $D$.

      In order to perform the softmax operation, $\text{softmax}(A)$, we need to exponentiate each term in $A$, but divide each element by the sum of its row. We know $A$ is an $(N,N)$ matrix. Therefore we have $N$ rows that are each $N$ long. For each row, each element needs to exponentiated,they need to be summed, and then divided by the sum. Exponentiation, summation, and division are all each 1 FLOP. For $N$ elements in a row, each element requires 3 FLOPs for a total of $3N$ FLOPs per row. Since there are $N$ rows, and $B$ batches, the total floating point operations required for the softmax operation are
      $$\text{FLOPs for Softmax} = 3BN^2$$

   2. How many bytes are read from/written to memory to perform the softmax operation, i.e. $\text{softmax}(A)$? Your answer should be in terms of $B$, $N$, $D$, and/or $n$.

      In order to perform the softmax operation, we need to read the attention matrix, $A$, which has the shape $(B,N, N)$. Once the softmax operation is applied, the dimensions of the matrix remain unchanged so a $(B,N,N)$ matrix needs to be written out. Again, from above, each element is represented by $n$ bytes so we can express the total bytes read from/written to memory for the softmax operation as
      $$ \text{Bytes for Softmax} = n(2BN^2)$$

4. **Computing the output**

   Now denote the output of the softmax stage as $A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$. The final stage of scaled dot-product attention can be written as:
   $$\text{Attention}(Q, K, V) = AV$$

   1. What is the number of floating point operations required for this step? Your answer should be in terms of $B$, $N$, and/or $D$.

      From above, we know that $A$ is the attention matrix in the shape $(B,N,N)$ and $V$ is a matrix of the shape $(B,N,D)$. From the formulas from above, when muliplying two matrices, we find that the total number of floating point operations for the final stage of scaled dot-product attention can be expressed as

      $$\text{Total FLOPs for AV}=2BN^2D$$

   2. How many bytes are read from/written to memory for this step? Your answer should be in terms of $B$, $N$, $D$, and/or $n$.

      In order to calculate $AV$, we first need to read the attention matrix of the shape $(B,N,N)$ and the value matrix of the shape $(B,N,D)$. The result will therefore be matrix of the shape $(B,N,D)$. Again, each element is represented by $n$ bytes so we can express the total bytes are read from/written to memory for the final stage of scaled dot-product attention as
      $$\text{Bytes for AV} = n(BN^2+2BND)$$

5. **Putting everything together**

   Now that we have computed the number of floating point operations and memory accessions required for each sub-operation in scaled dot-product attention, we can compute the following:

   1. What is the total number of floating point operations required for scaled-dot product attention? Your answer should be in terms of $B$, $N$, and/or $D$.

      The total number of operation required to perform scaled-dot product attention is the sum of each sub-operation:

      - Dot-product: $2BN^2D + BN^2 $
      - Masking: $BN^2$
      - Softmax: $3BN^2$
      - Attention-Value: $2BN^2D$

      The total is the sum of these operations so
      $$\text{FLOPs for SDPA} = 2BN^2D + BN^2+BN^2+3BN^2+2BN^2D$$
      $$\text{FLOPs for SDPA} = 4BN^2D +5BN^2$$

   2. What is the total amount of memory reads/writes in bytes? Your answer should be in terms of $B$, $N$, $D$, and/or $n$.

      Similar to above, the total amount of memory reads/writes in bytes required to perform scaled-dot product attention is the sum of each sub-operation expressed as

      - Dot-product: $n(2BND + BN^2) $
      - Masking: $n(2BN^2)$
      - Softmax: $n(2BN^2)$
      - Attention-Value: $n(BN^2+2BND)$

      The total is the sum of these operations so
      $$\text{Bytes for SDPA} = n(2BND + BN^2+2BN^2+2BN^2+BN^2+2BND)$$
      $$\text{Bytes for SDPA} = n(4BND +6BN^2)$$

### Part 2.1.4: Multi-Head Attention (MHA)

1. Derive an expression for the number of floating point operations required for MHA (do not include the input projections in your calculation). Your answer should be in terms of $B$, $N$, $D$, and/or $H$.

   From above, we know that that the total FLOPs for SDPA for a single attention head can be expressed as
   $$\text{FLOPs for SDPA} = 4BN^2D +5BN^2$$

   Now in MHA, we have $H$ attention heads, each head of attention only operates on a subsect of the embedding. This is calculated as $d_{\text{model}}=\frac{D}{H}$ Now, each attention head gets a smaller dimension, but we are doing it $H$ times so we can express this as

   $$\text{FLOPs for MHA} = H \times \bigg(4BN^2\bigg(\frac{D}{H}\bigg) +5BN^2\bigg)$$
   $$\text{FLOPs for MHA} = 4BN^2D +5HBN^2$$

2. Derive an expression for the total number of bytes read/written to memory in order to perform MHA (do not include the input projections in your calculation). Your answer should be in terms of $B$, $N$, $D$, $H$, and/or $n$.

   Similar to the above seps, we know the total number of bytes read/written to/from memory for SPDA. This is expressed as
   $$\text{Bytes for SDPA} = n(4BND +6BN^2)$$

   Now, for multi-headed attention, there are $H$ heads of attention, and each head is only operating on a $\frac{D}{H}$ subsect of the input embedding. Therefore, we can express the total bytes for MHA as

   $$\text{Bytes for MHA} = H \times n\bigg(4BN\bigg(\frac{D}{H}\bigg) +6BN^2\bigg)$$
   $$\text{Bytes for MHA} = n\bigg(4BND+6HBN^2\bigg)$$

3. How does the number of FLOPs for MHA scale with sequence length? (e.g. linearly, quadratically, cubically?)

   Looking at the expression $4BN^2D +5HBN^2$, we can see that the number of FLOPs scales quadratically $O(n^2)$ with sequence length. This is because each token needs to attend to all other tokens, creating a quadratic relationship.

4. How does the total memory access in bytes scale with sequence length? (e.g. linearly, quadratically, cubically?)

   Similar to the number of FLOPs, looking at the expression $n\bigg(4BND+6HBN^2\bigg)$, we can see that the number of bytes scales quadratically $O(n^2)$ with sequence length since $6HBN^2$ is the dominating term.

5. Given a batch size $B = 16$, model dimension $D = 4096$, sequence length $N = 16384$, number of attention heads $H = 16$, and using half-precision (FP16), i.e. using a datatype with a size of 2 bytes, is MHA compute or memory bound on an A100-80GB SXM? The datasheet for an A100 can be found [here](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf).

   The offline compute of the A100-80GB SXM is $312 \times 10^{12}$ FLOPs with a memory bandwidth of $2.039 \times 10^{12}$ bytes/s. In order to find out whether MHA is compute bound or memory-bound for this given example, we have to first calculate the machine's balance and compare it to the arithmetic intensity of MHA in this scenario. The ridge-point on the roofline model for MHA on an A100-80GB SXM determines when it transitions from memory-bound to compute-bound. The machine balance is calculated as

   $$\text{Machine Balance} = \frac{\text{Peak FLOPs}}{\text{Memory Bandwidth}}$$

   Now, we can plug in our offline measurements for the A100-80GB SXM as

   $$\text{MB}_{\text{A100-80GB}}=\frac{312 \times 10^{12} \text{ FLOPs}}{2.039 \times 10^{12} \text{ bytes/s}} = 153.02 \text{ FLOPS/byte}$$

   Now, we have to calculate the arithmetic intensity, which can be expressed as

   $$\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Bytes Accessed}}$$

   Applying this in the case MHA, we get

   $$\text{Arithmetic Intensity for MHA} = \frac{4BN^2D +5HBN^2}{n\bigg(4BND+6HBN^2\bigg)}$$

   plugging in for our known variables, we get

   $$\text{AI}_{\text{MHA}} = \frac{4(16)(16384^2)(4096) +5(16)(16)(16384^2)}{2\bigg(4(16)(16384)(4096)+6(16)(16)(16384^2)\bigg)}=84.865$$

   Since we can see that $$\text{AI}_{\text{MHA}} < \text{MB}_{\text{A100-80GB}}$$ $$84.865<153.02$$

   Therefore, since out arithmetic intensity is less than our machine balance, MHA is memory-bound.

### Part 2.1.5: Output Projections

1. Derive an expression for the number of floating point operations required to perform the output projection $XW$. Your answer should be in terms of $B$, $N$, and/or $D$.

   Now, in order to compose the final output of the attention layer, we need to multiply $X$, the concatenated output of the multi-head attention, with the output projection matrix, $W$. The dimensions of $X$ are therefore $(B,N,D)$ and $W$ is $(D,D)$. Following the formula from part 2.1.1, we get
   $$\text{FLOPS for Output Projection} = 2BND^2$$

2. Derive an expression for the total number of bytes read/written to memory in order to perform the output project $XW$. Your answer should be in terms of $B$, $N$, $D$, and/or $n$.

   Now, in order to calculate $XW$, we need to read $X$ and $W$, and then write the output. Fromt above, the shape of $X$ is $(B,N,D)$ and the shape of $W$ is $(D,D)$. Therefore, the output shape of $XV$ is $(B,N,D)$. Again, each element is represented by $n$ bytes. The total number of bytes read/written to memory in order to perform the output project $XW$ is
   $$\text{Bytes for Output Projection} = n(2BND+D^2)$$

### Part 2.1.6: Feed-Forward Networks (MLPs)

1. Derive an expression for the number of floating point operations for a forward pass of the MLP. Your answer should be in terms of $B$, $N$, and/or $D$.

   In the MLP layer for a forward pass, we compute $\hat{z}=W_1x+b_1$ and $\hat{y}=W_2\hat{z}+b_2$ where $x$ is the output of the multi-head attention block with the shape $(B,N,D)$. The hidden dimension is $4D$ is traditional MLP blocks which means the shape of $W_1$ is $(D, 4D)$ and $W_2$ is $(4D, D)$ The bias vectors are of the shape $(1,D)$ Using the above formula, we can write

   $$
   \text{FLOPs for }\hat{z}=8BND^2+4BND
   $$

   $$
   \text{FLOPs for }\hat{y}=8BND^2+BND
   $$

   $$
   \text{Total FLOPs for MLP}=16BND^2+5BND
   $$

2. Derive an expression for the total number of bytes read/written to memory for a single forward pass through the MLP. Assume that the operation is not fused, i.e. the intermediate result between the two projections is stored and then read. Your answer should be in terms of $B$, $N$, $D$, and/or $n$.

   In order to calculate the total number of bytes read/written to memory for a single forward pass through the MLP, we need to first read the input matrix $x$ of the shape $(B,N,D)$ and the first weight matrix, $W_1$, of the shape $(D, 4D)$, and the first bias vector of the shape $(1,4D)$. We first compute and write out the intermediary output, $\hat{z}$, of the shape $(B,N,4D)$. Then, we compute and write out the MLP output, $\hat{y}$ of the shape $(B,N,D)$ which requires us to read in the intermediary output, $\hat{z}$, the second weight matrix, $W_2$, of the shape $(4D,D)$, and the secodn bias term $b_2$ of the shape $(1,D)$. Again, each element is represented by $n$ bytes so we have a total of

   $$
   \text{Total Bytes for MLP} = n(BND+4D^2+4D+4BND+4BND+4D^2+D+BND)
   $$

   $$
   \text{Total Bytes for MLP} = n(10BND+8D^2+5D)
   $$

3. Using the same values from question 5 of the [part 2.1.4](#Part-214-Multi-Head-Attention-MHA), how does the arithmetic intensity of the MLP compare to that of MHA when run on a A100-80GB SXM?

   In order to calculate the arithmetic intensity of the MLP, we can use the formula from abouve

   $$\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Bytes Accessed}}$$

   Applying this in the case of the MLP, we get

   $$\text{Arithmetic Intensity for MLP} = \frac{16BND^2+5BND}{n(10BND+8D^2+5D)}$$

   plugging in for our known variables, we get

   $$\text{AI}_{\text{MLP}} = \frac{16(16)(16384)(4096^2)+5(16)(16384)(4096)}{2\bigg(10(16)(16384)(4096)+8(4096^2)+5(4096)\bigg)}=3236.586$$

   From above, we know that the machine balance on an A100-80GB SXM is $\text{MB}_{\text{A100-80GB}}=153.02$ and the arithmetic intensity for MHA is $\text{AI}_{\text{MHA}} =84.865$. We can see the arithmetic intensity of the MLP is much higher than MHA, and so much so that $AI_{\text{MLP}} > MB_{\text{{A100-80GB}}}$ meaning that the MLP is actually compute-bound where as MHA is memory-bound.

## Part 2.2: Analyzing Naive GPT-2 XL Inference Computations

GPT-2 XL has the following architectural details:

- `n_vocab` = 50257
- `n_embed_dim` = 1600
- `n_heads` = 25
- `n_layers` = 48
- `mlp_hidden_dim` = 6400

Some additional notes for this problem:

- We’ll refer to the batch size as B and the sequence length as N , just as before.
- We’ll ignore the embedding layers, layer normalization layers, skip connections, and language modeling head as they’re all relatively inconsequential compared to the core transformer loop.
- Assume all computations are done with 16-bit precision (i.e. using a datatype
  with a size of 2 bytes).
- We are focused on inference, so we’ll consider just the forward pass, as before

1. How many FLOPs and memory accesses are required to perform attention (performing input projections, then MHA, and then the output projection) on a $B \times N$ batch of tokens? (Use your results from the last part!)

   _Hint: your answer will take the form:_ \
   FLOPs = $C_1 \times BN + C_2 \times BN^2$ \
   MEM = $C_3 + C_4 \times BN + C_5 \times BN^2$

   Now, in order to calculate the total FLOPs and memory accesses required to perform attention, we have to sum up all of the component parts we calculated.

   - Input Projections
     - FLOPs: $6BND^2$
     - Bytes:$n(6BND+3D^2)$
   - MHA
     - FLOPs: $4BN^2D +5HBN^2$
     - Bytes: $n\left(4BND +6HBN^2\right)$
   - Output Projection:
     - FLOPs: $2BND^2$
     - Bytes: $n(2BND+D^2)$

   We ca sum these up to get
   $$\text{Total FLOPs for Attention}=6BND^2+4BN^2D +5HBN^2+2BND^2$$
   $$\text{Total FLOPs for Attention}=8BND^2+4BN^2D +5HBN^2$$

   $$\text{Total Bytes for Attention}=n\left(6BND+3D^2+4BND +6HBN^2+2BND+D^2\right)$$
   $$\text{Total Bytes for Attention}=n\left(12BND+4D^2 +6HBN^2\right)$$

2. How many FLOPs and memory accesses are is required to run an MLP on a $B \times N$ batch of tokens? (Use your answer from the last part!)

   From above, we found that

   $$
   \text{Total FLOPs for MLP}=16BND^2+5BND
   $$

   $$
   \text{Total Bytes for MLP} = n(10BND+8D^2+5D)
   $$

3. How many FLOPs and memory accesses are is therefore required to perform an inference using all layers of the full GPT-2 XL model?

   First, to calculate the full attention block + MLP FLOPs and memory access, we must combine the results from part 1 and 2. We find
   $$\text{Total FLOPs for Attention + MLP}=24BND^2+4BN^2D +5HBN^2+5BND$$
   $$\text{Total Bytes for Attention + MLP}=n\left(22BND+12D^2 +6HBN^2+5D\right)$$

   Now, in the GPT-2 XL model, there are 48 layers of consisting of an attention and MLP block. Therefore, we can express the total FLOPs for GPT2-XL by performing these operations 48 times as

   $$\text{Total FLOPs for GPT2-XL}=1152BND^2+192BN^2D +240HBN^2+240BND$$
   $$\text{Total Bytes for GPT2-XL}=n\left(1056BND+576D^2 +288HBN^2+240D\right)$$

4. We’ll assume that we are using an NVIDIA T4 GPU with a theoretical memory bandwidth of 300 GB/s and a theoretical compute bandwidth of 65 TFLOPs/s. Let’s look at a few scenarios to try to understand inference bottlenecks in practice. In each of the following cases: how many FLOPs and memory accesses are required for a forward pass of the full GPT-2 XL model? Is the GPU memory bound or compute bound?

   In order to determine if the GPU is memory of compute bound in each scenario, we need to calculate $T_{\text{compute}}$ and $T_{\text{memory}}$ which are calculated as
   $$T_{\text{compute}}=\frac{\text{FLOPs}}{\text{Compute Bandwidth}}$$
   $$T_{\text{memory}}=\frac{\text{Memory Accesses}}{\text{Memory Bandwidth}}$$
   If $T_{\text{compute}} > T_{\text{memory}}$, then the scenario is compute-bound, otherwise it is memory-bound. On an NVIDIA T4 GPU, the memory bandwidth is $3 \times 10^{11}$ bytes/s and compute bandwidth of $6.5 \times 10^{13}$ FLOPs/s. The GPT2-XL architecture has a model dimension of $D=1600$ and a total of $H=25$ attention heads. GPT2 also uses FP16 for the elements so $n=2$. From above, we know
   $$\text{Total FLOPs for GPT2-XL}=1152BND^2+192BN^2D +240HBN^2+240BND$$
   $$\text{Total Bytes for GPT2-XL}=n\left(1056BND+576D^2 +288HBN^2+240D\right)$$

   - B=1 and N=32

     Using the above formulas, we can plug in to find the FLOPs and memory accesses
     $$\text{FLOPs}_{(1,32)}=9.47 \times 10^{10}$$
     $$\text{Memory}_{(1,32)}=3.07 \times 10^{9}$$

     Now, we can compute $T_{\text{compute}}$ and $T_{\text{memory}}$.
     $$T_{\text{compute}}=\frac{9.47 \times 10^{10}}{6.5 \times 10^{13}}=0.00145 \ s$$
     $$T_{\text{memory}}=\frac{3.07 \times 10^{9}}{3 \times 10^{11}}= 0.01023 \ s$$

     Since $T_{\text{memory}} >T_{\text{compute}}$ we can see the GPU is memory bound in this situation.

   - B=1 and N=1024

     Same steps as above
     $$\text{FLOPs}_{(1,1024)}=3.35 \times 10^{12}$$
     $$\text{Memory}_{(1,1024)}=2.15 \times 10^{10}$$

     Now, we can compute $T_{\text{compute}}$ and $T_{\text{memory}}$.
     $$T_{\text{compute}}=\frac{3.35 \times 10^{12}}{6.5 \times 10^{13}}=0.0515 \ s$$
     $$T_{\text{memory}}=\frac{2.15 \times 10^{10}}{3 \times 10^{11}}= 0.0716 \ s$$

     Since $T_{\text{memory}} >T_{\text{compute}}$ we can see the GPU is memory bound in this sitatuion.

   - B=64 and N=32

     Same steps as above
     $$\text{FLOPs}_{(64,32)}=6.06 \times 10^{12}$$
     $$\text{Memory}_{(64,32)}=1.08 \times 10^{10}$$

     Now, we can compute $T_{\text{compute}}$ and $T_{\text{memory}}$.
     $$T_{\text{compute}}=\frac{6.06 \times 10^{12}}{6.5 \times 10^{13}}=0.0932 \ s$$
     $$T_{\text{memory}}=\frac{1.08 \times 10^{10}}{3 \times 10^{11}}= 0.0360 \ s$$

     Since $T_{\text{compute}} >T_{\text{memory}}$, the GPU is actually compute bound in this situation.

   - B=64 and N=1024

     Same steps as above
     $$\text{FLOPs}_{(64,1024)}=2.14 \times 10^{14}$$
     $$\text{Memory}_{(64,1024)}=1.19 \times 10^{12}$$

     Now, we can compute $T_{\text{compute}}$ and $T_{\text{memory}}$.
     $$T_{\text{compute}}=\frac{2.14 \times 10^{14}}{6.5 \times 10^{13}}=3.2923 \ s$$
     $$T_{\text{memory}}=\frac{1.19 \times 10^{12}}{3 \times 10^{11}}= 3.9667 \ s$$

     Since $T_{\text{memory}} >T_{\text{compute}}$, the GPU is again memory bound in this situation.

## Part 2.3: Analyzing GPT-2 XL Inference with KV Caching!

We’re now going to assume that we’ve already filled our KV cache with $B × (N − 1)$ tokens, and we want to decode B new tokens in parallel using that KV cache (e.g., decode 1 new token each for each item in the batch). Let’s do the same math and see what changes!

1. How big is the KV cache at each layer, in terms of $B$ and $N$? Assume it’s also stored in 16-bit precision.

   The KV cache is storing the progressively built K and V matrices for each batch($B$), layer($L$), token($N$), and head($H$). For each head, its cache occupancy will be $d_{\text{model}}=\frac{D}{H}$ and each element occupies $n$ bytes. Now, we can express this as

   $$
   \text{Size of KV Cache} = 2n(L \times B \times N \times H \times d_{\text{model}} )
   $$

   We have the factor $2$ as we store both $K$ and $V$. Now, applying this to the case of GPT2-XL, we have $n=2$, $L=48$, $H=25$, $d_{\text{model}}=64$. Now, plugging these factors in, we get

   $$
   \text{Size of KV Cache} = 4(48 \times B \times N \times 25 \times 64 ) = 307200BN \text{ bytes}
   $$

2. How many FLOPs and memory accessions are required to perform attention to decode this batch of $B$ tokens?

   - You should assume that the $KV$ cache is not in memory. Assume that when the keys and values outputted in the input projection step (part 2.1.2) are stored, they are stored in the $KV$ cache (i.e. they are not stored elsewhere for use in later steps). As such, assume that K and V are loaded from the $KV$ cache when they are needed (e.g. for $QK^T$ and matrix multiplication by $V$, respectively).

   _Hint: we only need to perform the attention calculations for this batch of tokens, and there's no masking required! Your answer will take the form:_ \
   FLOPs = $C_1 \times B + C_2 \times BN$ \
   MEM = $C_3 + C_4 \times B + C_5 \times BN$

   Similar to above, in order to calculate the total FLOPs and memory accesses required to perform attention, we have to sum up all of the component parts we calculated, but now with the KV cache.

   - Input Projections:
     Now with KV caching, we only caclulate the projections for the current token in the sequence, so $N=1$
     - FLOPs: $6BD^2$
     - Bytes:$n(6BD+3D^2)$
   - MHA
     Now, $Q$ has the shape $(B,H,1, \frac{D}{H})$, since and the cached $K$ and $V$ are of the shape $(B,H,N,\frac{D}{H})$. Now, we can recalculate the different parts of MHA across all heads as:

     - **SDPA ($QK^T$)**:
       - FLOPs: $2BND + BN$ (including the dividing by $d_{\text{model}}$)
       - Memory Access: $n(BD+BND+BHN)$
     - **Softmax**:
       Since Softmax operates on the attention output, we compute it across the $N$ dimension for current attention row
       - FLOPs: $3BHN$
       - Memory Access: $n(2BHN)$
     - **Attention Output**:
       Now the final step of attention is to multiply the attention output, $A$, which has the shape $(B, H, 1, N)$ and the cached values $V$.
       - FLOPs: $2BND$
       - Memory Access: $n(BHN+BND+BD)$

     This makes the total for MHA with KV caching
     $$\text{FLOPs for MHA with KV}=4BND+3BHN+BN$$
     $$\text{Bytes for MHA with KV}=n(2BD+2BND+2BHN)$$

   - Output Projection:

     Now the output, $X$, which is of the shape $(B, 1, D)$ by the projection matrix, $P$, which is of the shape $(D,D)$. This means the total FLOPs here are calculated as
     $$\text{FLOPs for Projection} = 2BD^2$$
     $$\text{Bytes for Projection} = n(2BD+D^2)$$

   This makes the total FLOPs for an attention block with KV caching expressed as (omitting the extra $BN$ term):
   $$\text{FLOPs for Attention with KV} = 8BD^2+4BND+3BHN$$
   $$\text{Bytes for Attention with KV} = n(10BD+4D^2+2BND+2BHN)$$

3. How many FLOPs and memory accessions are required to run an MLP on this batch of $B$ tokens?

   From above, we found that

   $$
   \text{Total FLOPs for MLP}=16BND^2+5BND
   $$

   $$
   \text{Total Bytes for MLP} = n(10BND+8D^2+5D)
   $$

   However, now we effectively have $N=1$, so we can simplify these to

   $$
   \text{Total FLOPs for MLP with KV}=16BD^2+5BD
   $$

   $$
   \text{Total Bytes for MLP with KV} = n(10BD+8D^2+5D)
   $$

4. How many FLOPs and memory accessions are required to decode this batch of $B$ tokens through the full GPT-2 XL model?

   As from above, the full decoding pass will be the combination of the MHA without casual masking + the MLP. Therefore, we can express to the total FLOPs and memory accessions for each layer as
   $$\text{Total FLOPs for KV Attn + MLP}=24BD^2+4BND+3BHN+5BD$$
   $$\text{Total Bytes for KV Attn + MLP}=n(20BD+12D^2+2BND+2BHN+5D)$$

   Now, we have $48$ layers of Attention + MLP in the GPT2-XL model. Therefore, we can express the total as

   $$\text{Total FLOPs for KV GPT2-XL}=1152BD^2+192BND+144BHN+240BD$$
   $$\text{Total Bytes for KV GPT2-XL}=n(960BD+576D^2+96BND+96BHN+240D)$$

5. Let’s look at the same cases we examined before. For each of these, how many FLOPs and memory accesses are now required for a forward pass with the KV cache in place? Are we memory or compute bound? And (roughly) how much faster would we expect case each to theoretically run on the T4 GPU, compared to your answers from the previous part?

   In order to determine if the GPU is memory of compute bound in each scenario, we need to calculate $T_{\text{compute}}$ and $T_{\text{memory}}$ which are calculated as
   $$T_{\text{compute}}=\frac{\text{FLOPs}}{\text{Compute Bandwidth}}$$
   $$T_{\text{memory}}=\frac{\text{Memory Accesses}}{\text{Memory Bandwidth}}$$
   If $T_{\text{compute}} > T_{\text{memory}}$, then the scenario is compute-bound, otherwise it is memory-bound. On an NVIDIA T4 GPU, the memory bandwidth is $3 \times 10^{11}$ bytes/s and compute bandwidth of $6.5 \times 10^{13}$ FLOPs/s. The GPT2-XL architecture has a model dimension of $D=1600$ and a total of $H=25$ attention heads. GPT2 also uses FP16 for the elements so $n=2$. From above, we know
   $$\text{Total FLOPs for KV GPT2-XL}=1152BD^2+192BND+144BHN+240BD$$
   $$\text{Total Bytes for KV GPT2-XL}=n(960BD+576D^2+96BND+96BHN+240D)$$

   - B=1 and N=32

     Using the above formulas, we can plug in to find the FLOPs and memory accesses
     $$\text{FLOPs}_{(1,32)}=2.96 \times 10^{9}$$
     $$\text{Memory}_{(1,32)}=2.96 \times 10^{9}$$

     Now, we can compute $T_{\text{compute}}$ and $T_{\text{memory}}$.
     $$T_{\text{compute}}=\frac{2.96 \times 10^{9}}{6.5 \times 10^{13}}=0.0000456 \ s$$
     $$T_{\text{memory}}=\frac{2.96 \times 10^{9}}{3 \times 10^{11}}= 0.00987 \ s$$

     Since $T_{\text{memory}} >T_{\text{compute}}$ we can see the GPU is memory bound in this situation.

   - B=1 and N=1024

     Same steps as above
     $$\text{FLOPs}_{(1,1024)}=3.27 \times 10^{9}$$
     $$\text{Memory}_{(1,1024)}=3.27 \times 10^{9}$$

     Now, we can compute $T_{\text{compute}}$ and $T_{\text{memory}}$.
     $$T_{\text{compute}}=\frac{3.27 \times 10^{9}}{6.5 \times 10^{13}}=0.0000503 \ s$$
     $$T_{\text{memory}}=\frac{3.27 \times 10^{9}}{3 \times 10^{11}}= 0.0109 \ s$$

     Since $T_{\text{memory}} >T_{\text{compute}}$ we can see the GPU is memory bound in this situation.

   - B=64 and N=32

     Same steps as above
     $$\text{FLOPs}_{(64,32)}=1.89 \times 10^{11}$$
     $$\text{Memory}_{(64,32)}=3.79 \times 10^{9}$$

     Now, we can compute $T_{\text{compute}}$ and $T_{\text{memory}}$.
     $$T_{\text{compute}}=\frac{1.89 \times 10^{11}}{6.5 \times 10^{13}}=0.0029077 \ s$$
     $$T_{\text{memory}}=\frac{3.79 \times 10^{9}}{3 \times 10^{11}}= 0.012633 \ s$$

     Since $T_{\text{memory}} >T_{\text{compute}}$, the GPU is still memory bound in this situation.

   - B=64 and N=1024

     Same steps as above
     $$\text{FLOPs}_{(64,1024)}=2.09 \times 10^{11}$$
     $$\text{Memory}_{(64,1024)}=2.36 \times 10^{10}$$

     Now, we can compute $T_{\text{compute}}$ and $T_{\text{memory}}$.
     $$T_{\text{compute}}=\frac{2.09 \times 10^{11}}{6.5 \times 10^{13}}=0.0032154 \ s$$
     $$T_{\text{memory}}=\frac{2.36 \times 10^{10}}{3 \times 10^{11}}= 0.078633 \ s$$

     Since $T_{\text{memory}} >T_{\text{compute}}$, the GPU is still memory bound in this situation.

6. How does the use of a KV cache affect the number of FLOPs and memory accesses, and why? How does it affect the effective arithmetic intensity of the forward pass? Does it make the operation more memory-bound or more compute-bound?

   We can calculate the arithmetic intensity as
   $$\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Memory Accession}}$$

   Now, for regular MHA + MLP, we get an arithmetic intensity of
   $$\text{AI}_{\text{regular}}=\frac{24BND^2+4BN^2D +5HBN^2+5BND}{n\left(22BND+12D^2 +6HBN^2+5D\right)}$$

   We will plug in the the known values for the GPT2-XL architecture for comparison

   $$\text{AI}_{\text{regular}} = \frac{BN\Bigl(61448000 + 6525N\Bigr)}{BN\Bigl(70400 + 300\,N\Bigr) + 61456000}.$$

   and for the MHA + MLP with KV caching, we get

   $$\text{AI}_{\text{KV Caching}}=\frac{24BD^2+4BND+3BHN+5BD}{n(20BD+12D^2+2BND+2BHN+5D)}$$

   We will plug in the the known values for the GPT2-XL architecture for comparison
   $$\text{AI}_{\text{KV Caching}} = \frac{B\Bigl(61448\,000 + 6475N\Bigr)}{B\Bigl(64000 + 6500N\Bigr) + 61456000}$$

   We can see that the numerator for $\text{AI}_{\text{regular}}$ is dependent upon both $B$ and $N$ which means for $N>1$, the numerator of $\text{AI}_{\text{regular}}$ grows faster than the numerator of $\text{AI}_{\text{KV Caching}}$ which is only dependent on $B$. We then see that as $N$ grows larger, the numerator of $\text{AI}_{\text{regular}}$ grows proportionally, whereas as the denominator grows much slower. On the other hand, the numerator and denominator of $\text{AI}_{\text{KV Caching}}$ both grow large with an increase in $N$. This means
   $$\text{AI}_{\text{regular}} > \text{AI}_{\text{KV Caching}}$$

   as $B$ and $N$ grow. This means that since KV caching decreases the arithmetic intensity of attention + MLP, this makes it more memory-bound and less compute bound.

## Part 2.4: Motivating Speculative Decoding

1. Let's decompose the time required to do a forward pass with the model into three components:

   - The time required to read model weights from global memory.
   - The time required to perform all other global memory accesses besides reading model weights (ignore the cost of reading the KV cache).
   - The time required to perform the all the computations (FLOPs) in the forward pass.

   In our inference setting (decoding with $B = 1$ and a KV cache on a T4 GPU), which of these three components is most significant? Justify your answer (use your results from earlier questions!).

   In order to do a forward pass with the model, we have our three components. As a reminder, the T4 has a theoretical memory bandwidth of 300 GB/s and theoretical compute bandwidth of 65 TFLOPs/s. Breaking these components down:

   1. Reading weights from memory

      Reading $W_k, W_q, W_v$ and the output projection matrix $P$, and the MLP weights. Going over the dimensions of each of these:

      - $W_k, W_q, W_v = (B,D,D)$
      - $P = (D,D)$
      - $MLP = (D, 4D) + (1,4D) \rightarrow (4D,D) + (1+D)$

      Therefore, the total weights loaded where $B=1$ is
      $$\text{Total Weights Loaded}=3D^2+D^2+8D^2+4D+D$$
      $$\text{Total Weights Loaded}=12D^2+5D$$

      We know $D=1600$ so we get the total number of weights per layer as
      $$\text{Total Weights Loaded}=12(1600)^2+5(1600)=30728000$$

      Now, we have $48$ layers, and each parameter requires $2$ bytes in order to be read so we get
      $$\text{Total Size of All Model Weights}=2\times 48(30728000)=2.95 \text{ GB}$$

      The total amount of time it takes to load the model weights is therefore
      $$\text{Time to load Wights}=\frac{2.95 \text{ GB}}{300 \text{GB/s}}=0.00983296 \text{ s}$$

   2. Time requires to perform all other global memory accesses

      All other memory accesses include reading the input, writing the query projection, writing and reading the attention scores, writing and reading the softmax, writing and reading the attention output, writing and reading the intermediate MLP output, and finally writing out the output of the MLP. Breaking this down, we get

      - Reading the Input:
        - $X=n(B,1,D)$
      - Writing and Reading the Query:
        - $Q=2n(B,1,D)$
      - Writing and Reading the Attention Scores:
        - $QK^T=2n(B,H,N)$
      - Writing and Reading the Softmax:
        - $\text{Softmax}(QK^T)=2n(B,H,N)$
      - Writing and Reading the Attention output:
        - $A=2n(B,1,D)$
      - Writing and Reading the MLP intermediate output:
        - $\hat{z}=2n(B,1,4D)$
      - Writing block output:
        - $\hat{y}=n(B,1,D)$

      This brings th total memory accesses for all other global memory to
      $$\text{Total Other Memory}=n(BD+ 2BD+ 2BHN+2BHN+2BD+8BD+BD)$$
      $$\text{Total Other Memory}=n(14BD+ 4BHN)$$

      Plugging in our known variables of $B=1$, $D=1600$, $n=2$, and $H=25$ with $48$ layers we get
      $$\text{Total Other Memory}=0.0021504+ 0.0000096N \text{ GB}$$

      Therefore, we can see for any sequence length less than $N< 307067$ results in less memory. Therefore, If we assume the max sequence length in GPT2-XL of $N=4096$, we get
      $$\text{Total Other Memory}=0.041472 \text{ GB}$$

      The total amount of time it takes for all other global memory accesses is therefore
      $$\text{Time to load Weights}=\frac{0.041472 \text{ GB}}{300 \text{GB/s}}=0.00013824 \text{ s}$$

   3. The total number of FLOPs required to perform all of the computations in the forward pass for all layers with KV caching we calculated as
      $$\text{Total FLOPs for KV GPT2-XL}=1152BD^2+192BND+144BHN+240BD$$

      Plugging in our known variables of $B=1$, $D=1600$, and $H=25$ we get

      $$\text{Total FLOPs for KV GPT2-XL}=2.95 \times 10^9 + 310800N$$

      Again, we will assume a sequence length of $N=4096$ and we find in TFLOPs

      $$\text{Total TFLOPs for KV GPT2-XL}=0.004223 \text{ TFLOPs}$$

      Now, in order to compute the to compute the total amount of time, we can divide this by the compute bandwidth
      $$\text{Time to Compute} = \frac{0.004223}{65}=0.0000645 \text{ s}$$

   Therefore, we can see the limiting factor here is the **time to load the weights** which takes the most amount of time.

2. Given the bottleneck that you identified, how does speculative decoding help improve latency?

   Speculative decoding helps alleviate the primary bottleneck in inference time which is weight loading by reducing the number of times the large model's weights must be loaded from memory. In speculative decoding, a smaller model is used to predict several tokens at once. The main larger model then verifies whether these tokens are correct using a single forward pass (relying on the autoregressive nature of transformers). If the smaller model's predictions are correct, multiple tokens are generated while loading the weights just once. This effectively amoritizes the weight-loading cost by the number of accepted tokens.

   For example, if 4 of the smaller model's tokens are all accepted, the per-token weight loading cost decreases from 0.00983s to approximately 0.00246s,which improves latency and at worst takes the time of the regular model.
