import os, sys
import errno
import subprocess
import itertools
import time
import csv
import re
import functools
from io import StringIO
from collections.abc import Iterable, Callable

from query_config import is_cuda_available, get_compute_cap
from perf import saxpy_flops, saxpy_memory_accesses, saxpy_transferred
from perf import sgemm_flops, sgemm_memory_accesses

RED = "\x1B[0;31m"
GREEN = "\x1B[0;32m"
YELLOW = "\x1B[0;33m"
BLUE = "\x1B[0;34m"
BOLD = "\x1B[1m"
UNDERLINE = "\x1B[4m"
CLEAR = "\x1B[0m"
UP = "\x1B[%dA"

CUBLAS_STDOUT = "cublas_stdout"
KERNEL_STDOUT = "kernel_stdout"
NUM_SGEMM = 5 # number of SGEMM CUDA kernels (excluding sequential)

# -----------------------------------------------------------------------------------
# Helper utility functions

def silent_remove(filename: str):
    """
    Silently removes a file that may or may not exist.
    
    Args:
        filename (str): the path of the file to delete
    """
    try:
        os.remove(filename)
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise

# -----------------------------------------------------------------------------------
# Helper tester functions

def profile_cmd(cmd_args: list[str], use_nvprof: bool, op_keywords: list[str]) -> dict[str, float] | None:
    """
    Profiles a given CUDA executable for the runtime of specified kernel(s) and other
    CUDA operations.
    
    Args:
        cmd_args (list[str]): the command-line arguments for the command to be profiled
        use_nvprof (bool): whether to use `nvprof` or `nsys` for profiling
        op_keywords (list[str]): a list of regex patterns to match the name of an operation
        whose runtime is to be profiled
    Returns:
        dict[str, float] | None: on success, a mapping from each regex pattern to 
        the runtime in ms of the first matched operation (additionally containing 
        a mapping from `KERNEL_STDOUT` and `CUBLAS_STDOUT` to the executable's 
        printed kernel and cuBLAS runtime); on failure, None
    """
    keyword_to_runtime = {}
    if use_nvprof:
        OUT_CSV = "/tmp/runtime.csv"
        # use `nvprof` to profile, output info to CSV
        res = subprocess.run(
            ["nvprof", "--normalized-time-unit", "ms", "--csv", "--force-overwrite", "--log-file", OUT_CSV] + cmd_args,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        if not os.path.isfile(OUT_CSV): # correctness failed
            print("ERROR: `nvprof` did not output a log file")
            return None
        with open(OUT_CSV, errors="ignore") as f:
            output = f.read()
        if output.startswith("======== Error: Application returned non-zero code"):
            print(f"{RED}Correctness test failed!{CLEAR}")
            silent_remove(OUT_CSV)
            return None
        
        # parse output CSV information
        start_line = re.search(r'==\d+== Profiling result:', output)
        if not start_line:
            print("ERROR: Unable to parse profiler output.")
            print(output)
            silent_remove(OUT_CSV)
            return None
        reader = csv.DictReader(StringIO(output[start_line.end():].strip()))
        for row in reader:
            for keyword in op_keywords:
                if re.search(keyword, row['Name']):
                    keyword_to_runtime[keyword] = float(row['Avg'])
                    break
        silent_remove(OUT_CSV)
    else:
        PROFILE_OUT = "/tmp/profile.nsys-rep"
        SQLITE_OUT = "/tmp/profile.sqlite"
        OUT_PREFIX = "/tmp/out"
        # use `nsys profile` to profile
        res = subprocess.run(
            ["nsys", "profile", "--force-overwrite", "true", "--output", PROFILE_OUT] + cmd_args,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        if res.returncode != 0:
            print(f"{RED}Correctness test failed!{CLEAR}")
            silent_remove(PROFILE_OUT)
            return None
        if not os.path.isfile(PROFILE_OUT):
            print("ERROR: `nsys` profiler did not output a log file")
            silent_remove(PROFILE_OUT)
            return None

        # parse output of `nsys profile` into CSV using `nsys stats`
        subprocess.run(
            ["nsys", "stats", "--format", "csv", "--report", "cuda_gpu_kern_sum,cuda_gpu_mem_time_sum",
             "--force-export", "--sqlite", SQLITE_OUT, "--force-overwrite", 
             "--output", OUT_PREFIX, PROFILE_OUT],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        
        # open and parse CSV outputs
        KERN_OUT = OUT_PREFIX + "_cuda_gpu_kern_sum.csv"
        MEM_OUT = OUT_PREFIX + "_cuda_gpu_mem_time_sum.csv"
        OUTPUT_FILES = [PROFILE_OUT, SQLITE_OUT, KERN_OUT, MEM_OUT]
        if not os.path.isfile(KERN_OUT):
            print("ERROR: `nsys` stats analyzer did not output a kernel analysis log file")
            for filename in OUTPUT_FILES:
                silent_remove(filename)
            return None
        if not os.path.isfile(MEM_OUT):
            print("ERROR: `nsys` stats analyzer did not output a memory transfer analysis log file")
            for filename in OUTPUT_FILES:
                silent_remove(filename)
            return None
        with open(KERN_OUT, errors="ignore") as kern_f, open(MEM_OUT, errors="ignore") as mem_f:
            kernel_reader = csv.DictReader(kern_f)
            mem_reader = csv.DictReader(mem_f)
            for row in kernel_reader:
                for keyword in op_keywords:
                    if re.search(keyword, row['Name']):
                        keyword_to_runtime[keyword] = float(row['Avg (ns)']) * 1e-6
                        break
            for row in mem_reader:
                for keyword in op_keywords:
                    if re.search(keyword, row['Operation']):
                        keyword_to_runtime[keyword] = float(row['Avg (ns)']) * 1e-6
                        break

        for filename in OUTPUT_FILES:
            silent_remove(filename)
    
    # Scrape runtime from stdout
    cublas_match = re.search(r"cuBLAS ran in ([0-9.]+)ms", res.stdout)
    if cublas_match:
        keyword_to_runtime[CUBLAS_STDOUT] = float(cublas_match.group(1))
    kernel_match = re.search(r"Kernel \d ran in ([0-9.]+)ms", res.stdout)
    if kernel_match:
        keyword_to_runtime[KERNEL_STDOUT] = float(kernel_match.group(1))
    
    return keyword_to_runtime


CURR_TEST_NUM = 1

def run_test(test: list[str] | Callable[[], bool], test_name: str) -> bool:
    """
    Runs a given correctness test executable, determining if it passed or not 
    based on its exit code.
    
    Args:
        test (list[str]): either a test command and its arguments to run, or a 
        test function to call that returns whether or not the test passed
        test_name (str): the display name for the test
    Returns:
        bool: whether the test passed or not
    """
    is_cmd = isinstance(test, list)
    global CURR_TEST_NUM
    print(f"{CURR_TEST_NUM}. [ ...... ] {YELLOW}{test_name}{CLEAR}")
    if is_cmd:
        print(f"> {' '.join(test)}")
        res = subprocess.run(test, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        passed = res.returncode == 0
    else:
        passed = test()
    if passed:
        print(f"{UP % (2 if is_cmd else 1)}{CURR_TEST_NUM}. [ {GREEN}PASSED{CLEAR} ]", end=('\n\n' if is_cmd else '\n'))
    else:
        print(f"{UP % (2 if is_cmd else 1)}{CURR_TEST_NUM}. [ {RED}FAILED{CLEAR} ]", end=('\n\n' if is_cmd else '\n'))
    CURR_TEST_NUM += 1
    return passed

# -----------------------------------------------------------------------------------
# Test suites

def print_title(title: str):
    """
    Prints the title of a test suite.
    
    Args:
        title (str): the test suite's title
    """
    print(f"\n{BLUE}==={CLEAR} {YELLOW}{BOLD}{title}{CLEAR} {BLUE}==={CLEAR}")

def saxpy_test_suite() -> tuple[int, int]:
    """
    Runs the SAXPY CUDA kernel correctness test suite.
    
    Returns:
        tuple[int, int]: the tuple of tests that passed and total tests ran
    """
    print_title("SAXPY CORRECTNESS TESTS")
    total, correct = 0, 0
    for n in [1024, 4096, 1048576, 1, 1225, 10000000]:
        total += 1
        if run_test(["./saxpy", "-n", str(n)], f"SAXPY correctness with N={n}"):
            correct += 1
    return correct, total

def saxpy_perf_suite(use_nvprof: bool):
    """
    Runs the SAXPY performance test suite
    
    Args:
        use_nvprof (bool): whether to use nvprof or nsys for profiling
    """
    print_title("SAXPY PERFORMANCE")
    
    N = 10000000
    cmd = ["./saxpy", "-n", str(N)]
    print("> ", " ".join(cmd))
    profile_res = profile_cmd(cmd, use_nvprof, ["saxpy", "HtoD", "DtoH"])
    if not profile_res: # profiling or correctness failed, error printed
        return

    kernel, dtoh, htod = profile_res['saxpy'], profile_res['DtoH'], profile_res['HtoD']
    print(f"Kernel ran in {kernel:.3f}ms")
    print(f"Effective compute bandwidth: {saxpy_flops(N) / kernel * 1e-6:.3f} GFLOPs/s")
    print(f"Effective memory bandwidth: {saxpy_memory_accesses(N) / kernel * 1e-6:.3f} GB/s")
    print(f"Effective arithmetic intensity: {saxpy_flops(N) / saxpy_memory_accesses(N):.3f} FLOPs/B")
    print(f"Effective host-to-device memory bandwidth: {saxpy_transferred(N) / htod * 1e-6:.3f} GB/s")
    print(f"Effective device-to-host memory bandwidth: {saxpy_transferred(N) / dtoh * 1e-6:.3f} GB/s")


def sgemm_test_suite(sgemm_nums: Iterable[int], mkns: list[tuple[int, int, int]]) -> tuple[int, int]:
    """
    Runs the SGEMM CUDA kernel correctness test suite on the given M, K, and N sizes.
    
    Args:
        sgemm_nums (Iterable[int]): the SGEMM kernels to test
        mkns (list[tuple[int, int, int]]): a list of M, K, and N values to test
        correctness with
    Returns:
        tuple[int, int]: the tuple of tests that passed and total tests ran
    """
    print_title("SGEMM CORRECTNESS TESTS")
    total, correct = 0, 0
    for i, sgemm_num in enumerate(sgemm_nums):
        print(f"{os.linesep if i != 0 else ''}{UNDERLINE}Tests for Kernel {sgemm_num}{CLEAR}:")
        for M, K, N in mkns:
            if sgemm_num == 0:
                M //= 8; K //= 8; N //= 8
            if run_test(["./sgemm", str(sgemm_num), "-M", str(M), "-K", str(K), "-N", str(N)], 
                        f"SGEMM correctness with M={M}, K={K}, N={N}"):
                correct += 1
            total += 1
    return correct, total

def sgemm_perf_suite(sgemm_nums: Iterable[int], use_nvprof: bool, verbose: bool = False):
    """
    Runs the SGEMM CUDA kernel performance test suite.
    
    Args:
        sgemm_nums (Iterable[int]): the SGEMM kernels to test
        use_nvprof (bool): whether to use nvprof or nsys for profiling
        verbose (bool): whether to run tests in verbose mode; defaults to False
    """
    first = True
    M = K = N = 1024
    for sgemm_num in sgemm_nums:
        if first:
            print_title("SGEMM PERFORMANCE")
            first = False
        cmd = ["./sgemm", str(sgemm_num), "-M", str(M), "-K", str(K), "-N", str(N)]
        print("> ", " ".join(cmd))
        kern_iden = rf"sgemm{SGEMM_SUFFIXES[sgemm_num]}" if sgemm_num != 0 else KERNEL_STDOUT
        cublas_iden = rf"sgemm(?!{SGEMM_SUFFIXES[sgemm_num]})" if sgemm_num != 0 else CUBLAS_STDOUT
        profile_res = profile_cmd(cmd, use_nvprof, [kern_iden, cublas_iden])
        
        if profile_res: # profiling or correctness didn't fail
            kernel, cublas = profile_res[kern_iden], profile_res[cublas_iden]
            print(f"Kernel {sgemm_num} ran in {kernel:.3f}ms, "
                f"{BOLD}{UNDERLINE}{kernel/cublas:.2f}x cuBLAS's runtime{CLEAR}")
            if verbose:
                print(f"Effective compute bandwidth: {sgemm_flops(M, K, N) / kernel * 1e-6:.3f} GFLOPs/s")
                print(f"Effective memory bandwidth: {sgemm_memory_accesses(M, K, N) / kernel * 1e-6:.3f} GB/s")
                print(f"Effective arithmetic intensity: {sgemm_flops(M, K, N) / sgemm_memory_accesses(M, K, N):.3f} FLOPs/B")


def conv1d_test_suite(bldks: Iterable[tuple[int, int, int, int]]) -> tuple[int, int]:
    """
    Runs the 1D depthwise convolution CUDA kernel correctness test suite on 
    the given dimensions sizes.
    
    Args:
        bldks (Iterable[tuple[int, int, int, int]]): the values of the batch,
        length, depth, and filter length dimensions respectively to test
    Returns:
        tuple[int, int]: the tuple of tests that passed and total tests ran
    """
    print_title("CONV1D CORRECTNESS TESTS")
    total, correct = 0, 0
    for B, L, D, K in bldks:
        if run_test(["./conv1d", "-B", str(B), "-L", str(L), "-D", str(D), "-K", str(K)], 
                    f"1D convolution performance with B={B}, L={L}, D={D}, K={K}"):
            correct += 1
        total += 1
    return correct, total

def conv1d_perf_suite(bldks: Iterable[tuple[int, int, int, int]]):
    """
    Runs the 1D depthwise convolution kernel performance test suite on the given
    dimension sizes.
    
    Args:
        bldks (Iterable[tuple[int, int, int, int]]): the values of the batch,
        length, depth, and filter length dimensions respectively to test
    """
    print("\nImporting PyTorch...")
    import torch
    import torch.nn as nn
    from torch.utils.cpp_extension import load
    from prettytable import PrettyTable
    
    os.environ['TORCH_CUDA_ARCH_LIST'] = get_compute_cap()
    conv1d_cpp = load(
        name="conv1d_cpp",
        sources=["depth_conv1d/conv1d_pytorch.cu"],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-lineinfo", "--use_fast_math", "-std=c++17", "--threads", "4"],
        verbose=True
    )

    torch.manual_seed(1390)
    torch.set_default_device('cuda')
    torch.set_default_dtype(torch.float32)
    nbytes = 4

    results = PrettyTable()
    results.field_names = ["B", "L", "D", "K", "torch time (ms)", "cudatime (ms)", "speedup", "Effective bandwidth (GB/s)", "TFLOPS"]
    results.float_format = "0.3"
    
    REPEATS = 15
    def run_conv1d_test(B: int, L: int, D: int, K: int) -> bool:
        u = torch.randn([B, D, L])
        conv1d_torch = nn.Conv1d(
            in_channels=D,
            out_channels=D,
            kernel_size=K,
            groups=D,
            padding=K//2
        )  
        
        # Warm-up with torch
        y_torch = conv1d_torch(u)
        torch.cuda.synchronize()
        # Run and time torch
        start = time.time()
        for _ in range(REPEATS):
            y_torch = conv1d_torch(u)
        torch.cuda.synchronize()
        torch_time = (time.time() - start) * 1000 / REPEATS
        
        # Warm-up with CUDA
        cuda_weight = conv1d_torch.weight.squeeze().detach().clone().contiguous()
        cuda_bias = conv1d_torch.bias.detach().clone().contiguous()
        y_cuda = conv1d_cpp.forward(u, cuda_weight, cuda_bias)
        torch.cuda.synchronize()
        # Run and time CUDA
        start = time.time()
        for _ in range(REPEATS):
            y_cuda = conv1d_cpp.forward(u, cuda_weight, cuda_bias)
        torch.cuda.synchronize()
        cuda_time = (time.time() - start) * 1000 / REPEATS
        
        speedup = torch_time / cuda_time
        effective_bandwidth = (B * L * D * 2 + K * D) * nbytes / (cuda_time * 1e-3) / (2 ** 30)
        tera_flops = (B * L * D * 2 * K) / (cuda_time * 1e-3) / (2 ** 40)
        results.add_row([B, L, D, K, torch_time, cuda_time, speedup, effective_bandwidth, tera_flops])
        
        return torch.allclose(y_torch, y_cuda, atol=1e-2)
    
    print_title("CONV1D PERFORMANCE TESTS")
    all_passed = True
    for B, L, D, K in bldks:
        if not run_test(functools.partial(run_conv1d_test, B, L, D, K), f"1D convolution performance with B={B}, L={L}, D={D}, K={K}"):
            all_passed = False
    if all_passed:
        print("\n1D Convolution Performance:")
        print(results)


OPTIONS = {"all", "saxpy_correct", "saxpy_perf", "sgemm_all", "sgemm_perf", "sgemm0", "sgemm1", 
           "sgemm2", "sgemm3", "sgemm4", "sgemm5", "nondivisible", "conv1d_correct", "conv1d_perf"}
SGEMM_SUFFIXES = {
    1: "_naive", 
    2: "_global_coalescing",
    3: "_shared_mem_cache",
    4: "_1D_thread_tiling",
    5: "_2D_thread_tiling",
}

MKNs: list[tuple[int, int, int]] = [
    (512, 512, 512), (1024, 1024, 1024), (1024, 2048, 4096), (1024, 4096, 2048), 
    (2048, 1024, 4096), (2048, 4096, 1024), (4096, 1024, 2048), (4096, 2048, 1024)
]
NONDIV_MKNs: list[tuple[int, int, int]] = [
    (1025, 1025, 1025), (2047, 2047, 2047), (1000, 2025, 4071), (1000, 4071, 2025), 
    (2025, 1000, 4071), (2025, 4071, 1000), (4071, 1000, 2025), (4071, 2025, 1000)
]

BLDK_CORRECT: Iterable[tuple[int, int, int, int]] = itertools.product(
    [3], [1024, 2000, 4096], [768, 1000, 2048], [3, 5]
)
BLDK_PERF: Iterable[tuple[int, int, int, int]] = itertools.product(
    [1], [4096, 8192], [2048, 8192], [3]
)

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        args = {"all"}
    else:
        args = set(sys.argv[1:])
    
    if not is_cuda_available():
        print("ERROR: Could not find an installation of CUDA. Please ensure that you are running on a Hydra GPU node using the `interact` script.")
        sys.exit(1)
    use_nvprof = float(get_compute_cap()) < 7.0
    
    invalid = args.difference(OPTIONS)
    if invalid:
        if len(invalid) == 1:
            print(f"ERROR: Received invalid test option: {invalid.pop()}")
        else:
            print(f"ERROR: Received invalid test options: {list(invalid)}")
        sys.exit(1)
    
    if "saxpy_correct" in args or "all" in args:
        saxpy_test_suite()
    
    if "saxpy_perf" in args or "all" in args:
        saxpy_perf_suite(use_nvprof)
    
    # sgemm => run all tests for all kernels, no perf
    # sgemm# => run divisible tests for just those kernels, with perf displayed verbosely
    # sgemm_perf => run just perf tests for all, non-verbosely (overrides sgemm#'s perf display)
    # nondivisible => run nondivisible tests for all kernels, no perf
    spec_sgemms = sorted([int(arg[-1]) for arg in args if re.fullmatch(r'sgemm\d', arg)])
    sgemms = []
    if "all" in args or "sgemm_all" in args:
        sgemms = range(NUM_SGEMM+1)
        mkns = MKNs + NONDIV_MKNs
    else:
        mkns = []
        if len(spec_sgemms) > 0:
            sgemms = spec_sgemms
            mkns += MKNs
        if "nondivisible" in args:
            sgemms = range(NUM_SGEMM+1)
            mkns += NONDIV_MKNs
    if len(sgemms) > 0:
        sgemm_test_suite(sgemms, mkns)
    
    run_all_perf = "all" in args or "sgemm_perf" in args
    perf_sgemms = range(NUM_SGEMM+1) if run_all_perf else spec_sgemms
    sgemm_perf_suite(perf_sgemms, use_nvprof, not run_all_perf)
    
    if "all" in args or "conv1d_correct" in args:
        conv1d_test_suite(BLDK_CORRECT)
    if "all" in args or "conv1d_perf" in args:
        conv1d_perf_suite(BLDK_PERF)