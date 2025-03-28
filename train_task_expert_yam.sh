# !/bin/bash

# nvcc确实导致通信失败的问题解决：可能会略微降低多GPU通信效率，但仍然可以使用多GPU
export NCCL_P2P_DISABLE=1 #禁用GPU之间的点对点直接通信，但仍允许通过主机内存进行通信
export NCCL_IB_DISABLE=1  #禁用InfiniBand通信，转而使用以太网
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO    #开启NCCL调试信息

# 指定网络接口和Socket通信线程数
export NCCL_SOCKET_IFNAME=eth0 #指定网络接口
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 #限制内存块分配大小
export NCCL_SOCKET_NTHREADS=8 #设置socket通信线程数

# 解决nvmlDeviceGetNvLinkRemoteDeviceType相关问题
export NCCL_NVLINK_DISABLE=1  #完全禁用NVLink通信，但不影响其他通信方式
export TORCH_DISTRIBUTED_DEBUG=INFO  #PyTorch分布式调试信息
export NCCL_SHM_DISABLE=1  #禁用共享内存传输，使用更安全但可能稍慢的通信方式
export NCCL_CUMEM_ENABLE=0  #禁用CUDA统一内存

# 添加以下变量解决nvmlDeviceGetNvLinkRemoteDeviceType错误
export PYTORCH_NO_CUDA_MEMORY_CACHING=1
export CUDA_MODULE_LOADING=LAZY
export PYTORCH_JIT_USE_NNC_NOT_NVFUSER=1
export TORCH_USE_CUDA_DSA=0
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_SHOW_CPP_STACKTRACES=1

# 添加解决VLLM IP地址问题的环境变量
export VLLM_HOST_IP=127.0.0.1
export HOST_IP=127.0.0.1

# 添加避免死锁的环境变量
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_ASYNC_ERROR_HANDLING=1

# 修改PyTorch分布式初始化方式
# 添加环境变量来禁用PyTorch对NVML的使用
export PYTORCH_NO_NVML=1

# export CUDA_LAUNCH_BLOCKING=1 #默认情况下，CUDA操作是异步的（非阻塞），设置该变量后强制使其变为同步（阻塞）执行；
# 确保每个CUDA操作完成后才执行下一个操作，使错误能够在发生点被立即捕获
# 易于错误定位，但是影响效率。

# Task Selection
TASK="gsm8k" # Available options: mbpp2, gsm8k, ai2_arc, cls

# Training Setting
NUM_ITERS=1


# This script needs 2 gpus
# CUDA_VISIBLE_DEVICES=0,1,2,3 python svd_reinforce_hydra.py \
#     base_model@_global_=qwq32b \
#     task@_global_=$TASK \
#     mode@_global_=training \
#     num_iters=$NUM_ITERS 

/root/miniconda3/envs/t2/bin/accelerate launch --config_file accelerate_configs/zero3.nami.yaml svd_reinforce_hydra.py \
    base_model@_global_=qwq32b \
    task@_global_=$TASK \
    mode@_global_=training \
    num_iters=$NUM_ITERS
