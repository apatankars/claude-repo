def saxpy_memory_accesses(N: int) -> int:
    """
    Determines the number of bytes of memory accesses (reads and stores)
    that the SAXPY CUDA kernel performs on 2 vectors with N elements.
    Note: assume the kernel uses single precision (FP32), i.e. uses
    elements with a size of 4 bytes.
    
    Args:
        N (int): the number of elements in the input vectors
    Returns:
        int: the number of memory accesses the SAXPY kernel performs
    """
    # TODO (Warm-up, Task 4): Implement!
    bytes_per_element = 4  
    return 3 * N * bytes_per_element

def saxpy_transferred(N: int) -> int:
    """
    Determines the number of bytes that the SAXPY CUDA kernel transfers
    to and from device memory per each single `cudaMemcpy` operation.
    Note: assume the kernel uses single precision (FP32), i.e. uses
    elements with a size of 4 bytes.
    
    Args:
        N (int): the number of elements in the input vectors
    Returns:
        int: the number of memory accesses the SAXPY kernel performs
    """
    # TODO (Warm-up, Task 4): Implement!
    return N * 4

def saxpy_flops(N: int) -> int:
    """
    Determines the number of FLOPs that the SAXPY CUDA kernel performs on 
    2 vectors with N elements.
    
    Args:
        N (int): the number of elements in the input vectors
    Returns:
        int: the number of FLOPs the SAXPY kernel performs
    """
    # TODO (Warm-up, Task 5): Implement!

    return 2 * N

def sgemm_memory_accesses(M: int, K: int, N: int) -> int:
    """
    Determines the number of bytes of memory accesses (reads and stores)
    that the SGEMM CUDA kernel performs on three input matrices of size M x K,
    K x N, and M x N.
    Note: assume the kernel uses single precision (FP32), i.e. uses
    elements with a size of 4 bytes.
    
    Args:
        M (int): the number of rows of the first and third input matrix
        K (int): the number of columns and rows of the first and second input
        matrices, respectively
        N (int): the number of columns of the second and third input matrix
    Returns:
        int: the number of memory accesses the SGEMM kernel performs
    """
    # TODO (Part 1.0): Implement!
    return 4 * ((M * K) + (K * N) + (2 * M * N))

def sgemm_flops(M: int, K: int, N: int) -> int:
    """
    Determines the number of FLOPs that the SAXPY CUDA kernel performs on 
    three input matrices of size M x K, K x N, and M x N.
    
    Args:
        M (int): the number of rows of the first and third input matrix
        K (int): the number of columns and rows of the first and second input
        matrices, respectively
        N (int): the number of columns of the second and third input matrix
    Returns:
        int: the number of memory accesses the SGEMM kernel performs
    """
    # TODO (Part 1.0): Implement!
    return (M * N) * ((2 * K) + 3)