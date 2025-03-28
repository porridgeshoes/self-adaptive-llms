# !/bin/bash

# Task Selection
TASK="gsm8k" # Available options: mbpp2, gsm8k, ai2_arc, cls

# Training Setting
NUM_ITERS=1


# This script needs 2 gpus
CUDA_VISIBLE_DEVICES=0,1,2,3 python svd_reinforce_hydra_v0.py \
    base_model@_global_=qwq32b \
    task@_global_=$TASK \
    mode@_global_=training \
    num_iters=$NUM_ITERS
