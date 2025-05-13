#!/bin/bash

dataset=lm_synthetic 
epochs=50 
lr=0.0005
train_batch=32
num_layers=4  # Default value for num_layers

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        num_layers=*) num_layers="${1#*=}"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
done

# Check if python or python3 is available
if command -v python3 &>/dev/null; then
    PYTHON_CMD=python3
elif command -v python &>/dev/null; then
    PYTHON_CMD=python
else
    echo "Python is not installed."
    exit 1
fi

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

EXEC_CMD="$PYTHON_CMD src/experiments.py --task ${dataset} --epochs ${epochs} --learning_rate ${lr} --train_batch ${train_batch} --layers ${num_layers}"
echo "> $EXEC_CMD"
unset LD_LIBRARY_PATH # changes made for project 3 interferes with Python virtual env searching for CUDA dependencies
srun -u --ntasks=1 --partition=gpus --gres=gpu:1 --time=00:10:00 $EXEC_CMD