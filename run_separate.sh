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

# 设置模型和任务
MODEL_PATH="/data1/nanzhiwang/python_work/open-r1/models/DeepSeek-R1-Distill-Qwen-7B"
TASK="gsm8k"
PORT=8000
NUM_ITERS=1

# 启动VLLM服务（后台运行）
echo "【启动】在GPU 0上启动VLLM服务..."
CUDA_VISIBLE_DEVICES=0 python scripts/vllm_server.py \
    --model ${MODEL_PATH} \
    --port ${PORT} \
    --max_model_len 1024 \
    --gpu_memory_utilization 0.8 &

VLLM_PID=$!
echo "【服务】VLLM服务启动，PID: ${VLLM_PID}"

# 等待VLLM服务启动
echo "【等待】等待VLLM服务就绪..."
sleep 30

# 设置环境变量指定使用GPU 1,2,3进行训练
export CUDA_VISIBLE_DEVICES=1,2,3

# 设置其他环境变量
export VLLM_HOST_IP=127.0.0.1
export HOST_IP=127.0.0.1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29501

# PyTorch分布式设置
export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_NCCL_BLOCKING_WAIT=1
export PYTORCH_NO_CUDA_MEMORY_CACHING=1
export PYTORCH_NO_NVML=1

# 禁用可能导致通信问题的NCCL功能
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_DEBUG=INFO
export NCCL_NVLINK_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_ASYNC_ERROR_HANDLING=1

# CUDA相关设置
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_MODULE_LOADING=LAZY

# 避免死锁
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# 设置环境变量使代码使用HTTP客户端而不是直接加载VLLM
export USE_VLLM_CLIENT=1
export VLLM_SERVER_PORT=${PORT}

echo "【启动】在GPU 1,2,3上启动训练..."
/root/miniconda3/envs/t2/bin/accelerate launch \
    --config_file accelerate_configs/zero3.nami.yaml \
    svd_reinforce_hydra_v0.py \
    base_model@_global_=qwq32b \
    task@_global_=$TASK \
    mode@_global_=training \
    num_iters=$NUM_ITERS

# 训练结束后清理VLLM进程
echo "【清理】训练完成，终止VLLM服务 (PID: ${VLLM_PID})..."
kill ${VLLM_PID} || true 