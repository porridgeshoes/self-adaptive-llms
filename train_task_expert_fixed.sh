#!/bin/bash

# 显示当前环境状态
echo "【初始状态】显示GPU状态"
nvidia-smi

# 清理可能占用端口的进程
echo "【清理】运行网络修复工具..."
python scripts/network_fix.py

# 设置明确的IP地址和网络参数
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

# PyTorch分布式设置
export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
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

# 设置CUDA设备顺序 - 把GPU 0放在首位明确优先使用
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Task Selection
TASK="gsm8k" # Available options: mbpp2, gsm8k, ai2_arc, cls

# Training Setting
NUM_ITERS=1

echo "【环境】环境变量设置完成，准备启动训练..."

# 先确保所有GPU可用
nvidia-smi

echo "【启动】开始训练..."
/root/miniconda3/envs/t2/bin/accelerate launch --config_file accelerate_configs/zero3.nami.yaml svd_reinforce_hydra_v0.py \
    base_model@_global_=qwq32b \
    task@_global_=$TASK \
    mode@_global_=training \
    num_iters=$NUM_ITERS 