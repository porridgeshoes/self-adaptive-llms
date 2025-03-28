#!/bin/bash

# 确保脚本在出错时退出
set -e

# 清理旧进程
echo "【清理】终止可能的残留Python进程..."
pkill -9 python || true
sleep 2

# 显示GPU状态
echo "【初始】显示GPU状态"
nvidia-smi

# 设置网络环境变量
export VLLM_HOST_IP=127.0.0.1
export HOST_IP=127.0.0.1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29501

# 禁用可能导致通信问题的NCCL功能
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_DEBUG=INFO
export NCCL_NVLINK_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_ASYNC_ERROR_HANDLING=1

# PyTorch分布式设置
export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_NCCL_BLOCKING_WAIT=1
export PYTORCH_NO_CUDA_MEMORY_CACHING=1
export PYTORCH_NO_NVML=1

# CUDA相关设置
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_MODULE_LOADING=LAZY

# 避免死锁
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# 确保所有GPU可见，但会通过GPU隔离工具控制使用
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 设置环境变量指示使用GPU隔离
export USE_GPU_ISOLATION=1
export VLLM_GPU_IDS=0
export TRAINING_GPU_IDS=1,2,3

# 设置任务和迭代次数
TASK="gsm8k" # Available options: mbpp2, gsm8k, ai2_arc, cls
NUM_ITERS=1

echo "【启动】开始训练，使用GPU隔离工具..."
/root/miniconda3/envs/t2/bin/accelerate launch \
    --config_file accelerate_configs/zero3.nami.yaml \
    svd_reinforce_hydra_dynamic.py \
    base_model@_global_=qwq32b \
    task@_global_=$TASK \
    mode@_global_=training \
    num_iters=$NUM_ITERS 