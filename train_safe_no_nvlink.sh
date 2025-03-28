#!/bin/bash

# 禁用所有与NVLink相关的功能
export NCCL_P2P_DISABLE=1 #禁用GPU之间的点对点直接通信
export NCCL_IB_DISABLE=1  #禁用InfiniBand通信
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO    #开启NCCL调试信息
export NCCL_SOCKET_IFNAME=eth0 #指定网络接口
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 #限制内存块分配大小
export NCCL_SOCKET_NTHREADS=8 #设置socket通信线程数

# 解决nvmlDeviceGetNvLinkRemoteDeviceType相关问题
export NCCL_NVLINK_DISABLE=1  #完全禁用NVLink通信
export TORCH_DISTRIBUTED_DEBUG=INFO  #PyTorch分布式调试信息
export NCCL_SHM_DISABLE=1  #禁用共享内存传输
export NCCL_CUMEM_ENABLE=0  #禁用CUDA统一内存

# 添加额外的环境变量解决NVML问题
export PYTORCH_NO_NVML=1
export PYTORCH_NO_CUDA_MEMORY_CACHING=1
export CUDA_MODULE_LOADING=LAZY
export PYTORCH_JIT_USE_NNC_NOT_NVFUSER=1
export TORCH_USE_CUDA_DSA=0
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_SHOW_CPP_STACKTRACES=1

# 修改代码添加补丁
cat > patch_svd_gpu_fix.py << 'EOL'
import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.abspath('.'))

# 读取原始文件
with open('svd_reinforce_hydra.py', 'r') as f:
    content = f.read()

# 导入我们的安全函数
additional_imports = '''
from scripts.nvlink_patch import safe_svd, safe_forward, safe_load_base_params
'''

# 替换原始torch.svd调用
content = content.replace('U, S, V = torch.svd(v)', 'U, S, V = safe_svd(v)')

# 替换forward调用
content = content.replace('forward(policy, model, base_params, decomposed_params, learnable_params)', 
                         'safe_forward(policy, model, base_params, decomposed_params, learnable_params)')

# 替换load_base_params调用
content = content.replace('load_base_params(model=model, base_params=base_params)',
                         'safe_load_base_params(model=model, base_params=base_params)')

# 插入导入语句
import_pos = content.find('import hydra')
if import_pos != -1:
    content = content[:import_pos] + additional_imports + content[import_pos:]

# 写入修补后的文件
with open('svd_reinforce_hydra_patched.py', 'w') as f:
    f.write(content)

print("成功应用补丁，生成了svd_reinforce_hydra_patched.py文件")
EOL

# 应用补丁
python patch_svd_gpu_fix.py

# Task Selection
TASK="gsm8k" # Available options: mbpp2, gsm8k, ai2_arc, cls

# Training Setting
NUM_ITERS=1

# 使用补丁后的脚本运行
CUDA_VISIBLE_DEVICES=0,1,2,3 python svd_reinforce_hydra_patched.py \
    base_model@_global_=qwq32b \
    task@_global_=$TASK \
    mode@_global_=training \
    num_iters=$NUM_ITERS 