#!/bin/bash

# Print usage function
usage() {
    echo "Usage: $0 (kvcache | speculative) [model=<model>]"
    exit 1
}

# Check if the first argument is valid
if [[ "$1" != "kvcache" && "$1" != "speculative" ]]; then
    echo "Invalid argument: $1"
    echo "First argument must be 'kvcache' or 'speculative'."
    usage
fi

MODE="$1"
MODEL="gpt2-large"
GPU="--gres=gpu:1"

shift
while [[ $# -gt 0 ]]; do
    case "$1" in
        model=*)
            MODEL="${1#*=}"
            ;;
        *)
            echo "Error: Unknown option $1"
            usage
            ;;
    esac
    shift
done

if [[ "$MODEL" == "gpt2-large" ]]; then
    GPU="--gres=gpu:geforce_gtx_2080_ti:1"
elif [[ "$MODEL" == "gpt2-xl" ]]; then
    GPU="--gres=gpu:1 --constraint=titan_rtx|rtx_a6000|l40"
fi

# Check if python or python3 is available
if command -v python3 &>/dev/null; then
    PYTHON_CMD=python3
elif command -v python &>/dev/null; then
    PYTHON_CMD=python
else
    echo "Python is not installed."
    exit 1
fi

# Get execution command
case "$MODE" in
    kvcache)
        # For evaluating the KV cache
        EXEC_CMD="$PYTHON_CMD sample.py --init_from=$MODEL --start=\"humaneval.jsonl\" --batch_size=8 --max_new_tokens=128 --prompt_length=128 --num_samples=1 --num_warmup=1"
        ;;
    speculative)
        # For evaluating speculative decoding
        EXEC_CMD="$PYTHON_CMD sample.py --init_from=$MODEL --start=\"humaneval.jsonl\" --batch_size=1 --max_new_tokens=256 --prompt_length=192 --num_samples=1 --num_warmup=1"
        ;;
esac

# Make sure we are in the same directory as the script
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
cd $SCRIPT_DIR

VENV_PATH="../venv/bin/activate"
if [ -f "$VENV_PATH" ]; then
    echo "Activating virtual environment..."
    source "$VENV_PATH"
else
    echo "Error: could not find virtual environment at $VENV_PATH"
    exit 1
fi

echo "> $EXEC_CMD"
unset LD_LIBRARY_PATH # changes made for project 3 interferes with Python virtual env searching for CUDA dependencies
srun -u --ntasks=1 --partition=gpus $GPU --time=00:10:00 $EXEC_CMD