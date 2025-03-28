# !/bin/bash

# 明确设置IP地址变量，避免使用0.0.0.0
export VLLM_HOST_IP=127.0.0.1
export HOST_IP=127.0.0.1
export MASTER_ADDR=127.0.0.1

# 修改CUDA设备顺序 - 把GPU 0放在首位，确保vllm能优先使用它
export CUDA_VISIBLE_DEVICES=1,2,3,0

# Task Selection
TASK="gsm8k" # Available options: mbpp2, gsm8k, ai2_arc, cls

# Training Setting
NUM_ITERS=1

/root/miniconda3/envs/t2/bin/accelerate launch --config_file accelerate_configs/zero3.nami.yaml svd_reinforce_hydra_v0.py \
    base_model@_global_=qwq32b \
    task@_global_=$TASK \
    mode@_global_=training \
    num_iters=$NUM_ITERS
