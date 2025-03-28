# !/bin/bash

# Task Selection
TASK="gsm8k" # Available options: mbpp2, gsm8k, ai2_arc, cls

# Training Setting
NUM_ITERS=1


/root/miniconda3/envs/t2/bin/accelerate launch --config_file accelerate_configs/zero3.nami.yaml svd_reinforce_hydra.py \
    base_model@_global_=qwq32b \
    task@_global_=$TASK \
    mode@_global_=training \
    num_iters=$NUM_ITERS
