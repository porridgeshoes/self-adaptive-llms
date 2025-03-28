#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NVLink补丁文件 - 用于替代torch.svd以避免NVLink相关错误
"""

import os
import torch
import torch.nn.functional as F
import warnings

# 检查是否需要禁用NVML
os.environ["PYTORCH_NO_NVML"] = "1"

# 安全的SVD实现，避免使用可能触发NVLink/NVML错误的CUDA操作
def safe_svd(tensor):
    """
    安全的SVD分解，避免使用可能引起NVLink或NVML相关错误的CUDA操作
    
    如果在GPU上运行时出错，将尝试在CPU上运行
    """
    try:
        # 首先尝试在原始设备上运行
        return torch.svd(tensor)
    except RuntimeError as e:
        if "nvmlDeviceGetNvLinkRemoteDeviceType" in str(e) or "CUDA" in str(e):
            warnings.warn("在GPU上执行SVD时遇到NVML/CUDA错误，尝试在CPU上执行")
            # 将张量移动到CPU，在CPU上执行SVD，然后将结果移回原始设备
            device = tensor.device
            cpu_tensor = tensor.cpu()
            U, S, V = torch.svd(cpu_tensor)
            return U.to(device), S.to(device), V.to(device)
        else:
            # 如果是其他错误，继续传播
            raise

# 安全的前向计算函数，避免NVLink相关问题
def safe_forward(policy, model, base_params, decomposed_params, learnable_params):
    """
    替代原始forward函数，添加错误处理
    """
    try:
        new_params = {}
        for k in base_params:
            if "mlp" in k:
                try:
                    # 尝试使用policy.get_mask生成新参数
                    mm = policy.get_mask(learnable_params[k])
                    u_key = f"{k}.U"
                    s_key = f"{k}.S"
                    v_key = f"{k}.V"
                    
                    # 检查确保所有必要的组件都存在
                    if u_key not in decomposed_params or s_key not in decomposed_params or v_key not in decomposed_params:
                        print(f"警告: 缺少参数 {k} 的SVD组件，使用原始参数")
                        new_params[k] = base_params[k]
                        continue
                    
                    # 构建新参数
                    U = decomposed_params[u_key]
                    S = decomposed_params[s_key]
                    V = decomposed_params[v_key]
                    
                    # 使用CPU计算以避免CUDA错误
                    device = U.device
                    U_cpu = U.cpu()
                    S_cpu = S.cpu()
                    V_cpu = V.cpu()
                    mm_cpu = mm.cpu()
                    
                    # 在CPU上执行矩阵乘法
                    diag_S = torch.diag_embed(S_cpu * mm_cpu)
                    result = U_cpu @ diag_S @ V_cpu.T
                    
                    # 计算缩放因子
                    scale = S_cpu.sum() / (S_cpu * mm_cpu).sum()
                    
                    # 应用缩放并移回原始设备
                    new_params[k] = (result * scale).to(device)
                    
                    # 更新模型参数
                    model.get_parameter(k).copy_(new_params[k])
                except Exception as e:
                    print(f"在处理参数 {k} 时发生错误: {e}")
                    # 如果发生错误，使用原始参数
                    new_params[k] = base_params[k]
                    model.get_parameter(k).copy_(new_params[k])
            else:
                new_params[k] = base_params[k]
        return new_params
    except Exception as e:
        print(f"在forward过程中发生未处理的错误: {e}")
        # 返回原始参数
        return base_params

# 安全的参数加载函数
def safe_load_base_params(model, base_params):
    """
    安全地加载基本参数，处理可能的设备错误
    """
    for k in base_params:
        if "mlp" in k:
            try:
                # 尝试将参数复制到CUDA设备
                model.get_parameter(k).copy_(base_params[k].cuda())
            except RuntimeError as e:
                if "CUDA" in str(e):
                    print(f"将参数 {k} 移动到CUDA时发生错误，尝试使用CPU")
                    # 尝试使用CPU作为中介
                    cpu_param = base_params[k].cpu()
                    model.get_parameter(k).copy_(cpu_param)
                else:
                    raise 