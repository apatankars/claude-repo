import sys
import subprocess

def is_cuda_available() -> bool:
    """
    Determines if CUDA is available on the given machine.
    
    Returns:
        bool: whether or not CUDA is available on the given machine
    """
    try:
        return subprocess.run(
            ['nvidia-smi'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        ).returncode == 0
    except:
        return False

def get_compute_cap() -> str:
    """
    Get the compute capability of the GPU node we are running on.
    
    Returns:
        str: the GPU's compute capability
    """
    return subprocess.getoutput('nvidia-smi --id=$CUDA_VISIBLE_DEVICES '
                                '--query-gpu=compute_cap --format=csv,noheader')

main_config = {
    '2.0': {
        "Compute Capability": "2.0",
        "Threads Per Warp": 32,
        "Warps Per Multiprocessor": 48,
        "Threads Per Multiprocessor": 1536,
        "Thread Blocks Per Multiprocessor": 8,
        "Shared Memory Per Multiprocessor (bytes)": 49152,
        "Number of 32-bit Registers per Multiprocessor": 32768,
        "Register Allocation Unit Size": 64,
        "Register Allocation Granularity": "warp",
        "Max Registers Per Thread": 63,
        "Max Registers Per Block": 32768,
        "Shared Memory Allocation Unit Size (bytes)": 128,
        "Warp Allocation Granularity (for register allocation)": 2,
        "Max Thread Block Size": 1024
    },
    '2.1': {
        "Compute Capability": "2.1",
        "Threads Per Warp": 32,
        "Warps Per Multiprocessor": 48,
        "Threads Per Multiprocessor": 1536,
        "Thread Blocks Per Multiprocessor": 8,
        "Shared Memory Per Multiprocessor (bytes)": 49152,
        "Number of 32-bit Registers per Multiprocessor": 32768,
        "Register Allocation Unit Size": 64,
        "Register Allocation Granularity": "warp",
        "Max Registers Per Thread": 63,
        "Max Registers Per Block": 32768,
        "Shared Memory Allocation Unit Size (bytes)": 128,
        "Warp Allocation Granularity (for register allocation)": 2,
        "Max Thread Block Size": 1024
    },
    '3.0': {
        "Compute Capability": "3.0",
        "Threads Per Warp": 32,
        "Warps Per Multiprocessor": 64,
        "Threads Per Multiprocessor": 2048,
        "Thread Blocks Per Multiprocessor": 16,
        "Shared Memory Per Multiprocessor (bytes)": 49152,
        "Number of 32-bit Registers per Multiprocessor": 65536,
        "Register Allocation Unit Size": 256,
        "Register Allocation Granularity": "warp",
        "Max Registers Per Thread": 63,
        "Max Registers Per Block": 65536,
        "Shared Memory Allocation Unit Size (bytes)": 256,
        "Warp Allocation Granularity (for register allocation)": 4,
        "Max Thread Block Size": 1024
    },
    "3.2": {
        "Compute Capability": "3.2",
        "Threads Per Warp": 32,
        "Warps Per Multiprocessor": 64,
        "Threads Per Multiprocessor": 2048,
        "Thread Blocks Per Multiprocessor": 16,
        "Shared Memory Per Multiprocessor (bytes)": 49152,
        "Number of 32-bit Registers per Multiprocessor": 65536,
        "Register Allocation Unit Size": 256,
        "Register Allocation Granularity": "warp",
        "Max Registers Per Thread": 255,
        "Max Registers Per Block": 65536,
        "Shared Memory Allocation Unit Size (bytes)": 256,
        "Warp Allocation Granularity (for register allocation)": 4,
        "Max Thread Block Size": 1024
    },
    "3.5": {
        "Compute Capability": "3.5",
        "Threads Per Warp": 32,
        "Warps Per Multiprocessor": 64,
        "Threads Per Multiprocessor": 2048,
        "Thread Blocks Per Multiprocessor": 16,
        "Shared Memory Per Multiprocessor (bytes)": 49152,
        "Number of 32-bit Registers per Multiprocessor": 65536,
        "Register Allocation Unit Size": 256,
        "Register Allocation Granularity": "warp",
        "Max Registers Per Thread": 255,
        "Max Registers Per Block": 65536,
        "Shared Memory Allocation Unit Size (bytes)": 256,
        "Warp Allocation Granularity (for register allocation)": 4,
        "Max Thread Block Size": 1024
    },
    "3.7": {
        "Compute Capability": "3.7",
        "Threads Per Warp": 32,
        "Warps Per Multiprocessor": 64,
        "Threads Per Multiprocessor": 2048,
        "Thread Blocks Per Multiprocessor": 16,
        "Shared Memory Per Multiprocessor (bytes)": 114688,
        "Number of 32-bit Registers per Multiprocessor": 131072,
        "Register Allocation Unit Size": 256,
        "Register Allocation Granularity": "warp",
        "Max Registers Per Thread": 255,
        "Max Registers Per Block": 65536,
        "Shared Memory Allocation Unit Size (bytes)": 256,
        "Warp Allocation Granularity (for register allocation)": 4,
        "Max Thread Block Size": 1024
    },
    "5.0": {
        "Compute Capability": "5.0",
        "Threads Per Warp": 32,
        "Warps Per Multiprocessor": 64,
        "Threads Per Multiprocessor": 2048,
        "Thread Blocks Per Multiprocessor": 32,
        "Shared Memory Per Multiprocessor (bytes)": 65536,
        "Number of 32-bit Registers per Multiprocessor": 65536,
        "Register Allocation Unit Size": 256,
        "Register Allocation Granularity": "warp",
        "Max Registers Per Thread": 255,
        "Max Registers Per Block": 65536,
        "Shared Memory Allocation Unit Size (bytes)": 256,
        "Warp Allocation Granularity (for register allocation)": 4,
        "Max Thread Block Size": 1024
    },
    "5.2": {
        "Compute Capability": "5.2",
        "Threads Per Warp": 32,
        "Warps Per Multiprocessor": 64,
        "Threads Per Multiprocessor": 2048,
        "Thread Blocks Per Multiprocessor": 32,
        "Shared Memory Per Multiprocessor (bytes)": 98304,
        "Number of 32-bit Registers per Multiprocessor": 65536,
        "Register Allocation Unit Size": 256,
        "Register Allocation Granularity": "warp",
        "Max Registers Per Thread": 255,
        "Max Registers Per Block": 32768,
        "Shared Memory Allocation Unit Size (bytes)": 256,
        "Warp Allocation Granularity (for register allocation)": 4,
        "Max Thread Block Size": 1024
    },
    "5.3": {
        "Compute Capability": "5.3",
        "Threads Per Warp": 32,
        "Warps Per Multiprocessor": 64,
        "Threads Per Multiprocessor": 2048,
        "Thread Blocks Per Multiprocessor": 32,
        "Shared Memory Per Multiprocessor (bytes)": 65536,
        "Number of 32-bit Registers per Multiprocessor": 65536,
        "Register Allocation Unit Size": 256,
        "Register Allocation Granularity": "warp",
        "Max Registers Per Thread": 255,
        "Max Registers Per Block": 32768,
        "Shared Memory Allocation Unit Size (bytes)": 256,
        "Warp Allocation Granularity (for register allocation)": 4,
        "Max Thread Block Size": 1024
    },
    "6.0": {
        "Compute Capability": "6.0",
        "Threads Per Warp": 32,
        "Warps Per Multiprocessor": 64,
        "Threads Per Multiprocessor": 2048,
        "Thread Blocks Per Multiprocessor": 32,
        "Shared Memory Per Multiprocessor (bytes)": 65536,
        "Number of 32-bit Registers per Multiprocessor": 65536,
        "Register Allocation Unit Size": 256,
        "Register Allocation Granularity": "warp",
        "Max Registers Per Thread": 255,
        "Max Registers Per Block": 65536,
        "Shared Memory Allocation Unit Size (bytes)": 256,
        "Warp Allocation Granularity (for register allocation)": 2,
        "Max Thread Block Size": 1024
    },
    "6.1": {
        "Compute Capability": "6.1",
        "Threads Per Warp": 32,
        "Warps Per Multiprocessor": 64,
        "Threads Per Multiprocessor": 2048,
        "Thread Blocks Per Multiprocessor": 32,
        "Shared Memory Per Multiprocessor (bytes)": 98304,
        "Number of 32-bit Registers per Multiprocessor": 65536,
        "Register Allocation Unit Size": 256,
        "Register Allocation Granularity": "warp",
        "Max Registers Per Thread": 255,
        "Max Registers Per Block": 65536,
        "Shared Memory Allocation Unit Size (bytes)": 256,
        "Warp Allocation Granularity (for register allocation)": 4,
        "Max Thread Block Size": 1024
    },
    "6.2": {
        "Compute Capability": "6.2",
        "Threads Per Warp": 32,
        "Warps Per Multiprocessor": 64,
        "Threads Per Multiprocessor": 2048,
        "Thread Blocks Per Multiprocessor": 32,
        "Shared Memory Per Multiprocessor (bytes)": 65536,
        "Number of 32-bit Registers per Multiprocessor": 65536,
        "Register Allocation Unit Size": 256,
        "Register Allocation Granularity": "warp",
        "Max Registers Per Thread": 255,
        "Max Registers Per Block": 65536,
        "Shared Memory Allocation Unit Size (bytes)": 256,
        "Warp Allocation Granularity (for register allocation)": 4,
        "Max Thread Block Size": 1024
    },
    "7.0": {
        "Compute Capability": "7.0",
        "Threads Per Warp": 32,
        "Warps Per Multiprocessor": 64,
        "Threads Per Multiprocessor": 2048,
        "Thread Blocks Per Multiprocessor": 32,
        "Shared Memory Per Multiprocessor (bytes)": 98304,
        "Number of 32-bit Registers per Multiprocessor": 65536,
        "Register Allocation Unit Size": 256,
        "Register Allocation Granularity": "warp",
        "Max Registers Per Thread": 255,
        "Max Registers Per Block": 65536,
        "Shared Memory Allocation Unit Size (bytes)": 256,
        "Warp Allocation Granularity (for register allocation)": 4,
        "Max Thread Block Size": 1024
    },
    "7.5": {
        "Compute Capability": "7.5",
        "Threads Per Warp": 32,
        "Warps Per Multiprocessor": 32,
        "Threads Per Multiprocessor": 1024,
        "Thread Blocks Per Multiprocessor": 16,
        "Shared Memory Per Multiprocessor (bytes)": 65536,
        "Number of 32-bit Registers per Multiprocessor": 65536,
        "Register Allocation Unit Size": 256,
        "Register Allocation Granularity": "warp",
        "Max Registers Per Thread": 255,
        "Max Registers Per Block": 65536,
        "Shared Memory Allocation Unit Size (bytes)": 256,
        "Warp Allocation Granularity (for register allocation)": 4,
        "Max Thread Block Size": 1024
    },
    "8.0": {
        "Compute Capability": "8.0",
        "Threads Per Warp": 32,
        "Warps Per Multiprocessor": 64,
        "Threads Per Multiprocessor": 2048,
        "Thread Blocks Per Multiprocessor": 32,
        "Shared Memory Per Multiprocessor (bytes)": 167936,
        "Number of 32-bit Registers per Multiprocessor": 65536,
        "Register Allocation Unit Size": 256,
        "Register Allocation Granularity": "warp",
        "Max Registers Per Thread": 255,
        "Max Registers Per Block": 65536,
        "Shared Memory Allocation Unit Size (bytes)": 128,
        "Warp Allocation Granularity (for register allocation)": 4,
        "Max Thread Block Size": 1024
    },
    "8.6": {
        "Compute Capability": "8.6",
        "Threads Per Warp": 32,
        "Warps Per Multiprocessor": 48,
        "Threads Per Multiprocessor": 1536,
        "Thread Blocks Per Multiprocessor": 16,
        "Shared Memory Per Multiprocessor (bytes)": 102400,
        "Number of 32-bit Registers per Multiprocessor": 65536,
        "Register Allocation Unit Size": 256,
        "Register Allocation Granularity": "warp",
        "Max Registers Per Thread": 255,
        "Max Registers Per Block": 65536,
        "Shared Memory Allocation Unit Size (bytes)": 128,
        "Warp Allocation Granularity (for register allocation)": 4,
        "Max Thread Block Size": 1024
    }
}

if __name__ == "__main__":
    if not is_cuda_available():
        print("ERROR: Could not find an installation of CUDA. Please ensure that you are running on a Hydra GPU node using the `interact` script.")
        sys.exit(1)
    
    cc = get_compute_cap()
    if cc not in main_config:
        print(f"Compute capability {cc} not found.")
        sys.exit(1)

    print(f"CUDA Configuration for Compute Capability {cc}:")
    print("=" * 40)
    for key, value in main_config[cc].items():
        print(f"{key}: {value}")