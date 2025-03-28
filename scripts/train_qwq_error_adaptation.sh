#!/bin/bash

# 指定错误案例文件路径（如果不指定，将使用代码中的示例案例）
# export ERROR_CASES_FILE="/path/to/your/error_cases.json"

# 训练设置
NUM_ITERS=200
BATCH_SIZE=32  # 根据GPU内存调整

# 这个脚本至少需要2个GPU（可根据QwQ-32B的资源需求调整）
CUDA_VISIBLE_DEVICES=0,1,2,3 python svd_reinforce_hydra.py \
    base_model@_global_=qwq32b \
    task@_global_=error_case \
    mode@_global_=training \
    num_iters=$NUM_ITERS \
    batch_size=$BATCH_SIZE \
    exp_suffix="error_adaptation" 