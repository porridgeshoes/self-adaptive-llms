#!/bin/bash

# 指定错误案例文件路径（如果不指定，将使用代码中的示例案例）
# export ERROR_CASES_FILE="/path/to/your/error_cases.json"

# 指定专家模型路径（使用训练后生成的专家模型）
EXPERT_PATH="results/qwq32b/reinforce-error_adaptation/learnable_params.pt"

# 这个脚本需要至少2个GPU（可根据QwQ-32B的资源需求调整）
CUDA_VISIBLE_DEVICES=0,1,2,3 python svd_reinforce_hydra.py \
    base_model@_global_=qwq32b \
    task@_global_=error_case \
    mode@_global_=evaluation \
    test_only=true \
    prompt_based_eval=true \
    experts_path_dict="{\"error_case\": \"$EXPERT_PATH\"}" 