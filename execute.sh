#!/usr/bin/env bash

# Path to the `.py` file you want to run
PYTHON_SCRIPT_PATH="/home/hlcv_team030/im2latex/"
# Path to the Python binary of the conda environment
CONDA_PYTHON_BINARY_PATH="/home/hlcv_team030/miniconda3/envs/hlcv-ss23/bin/python"
# Path to torchrun binary
TORCHRUN_PATH="/home/hlcv_team030/miniconda3/envs/hlcv-ss23/bin/torchrun"

cd $PYTHON_SCRIPT_PATH
$TORCHRUN_PATH --nproc_per_node=5 "$@"