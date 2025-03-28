#!/usr/bin/env python3
"""
VLLM与训练进程GPU隔离工具

此模块提供GPU隔离功能，允许VLLM和训练进程共存于同一Python进程中，
但使用不同的GPU资源，避免资源冲突导致的超时问题。
"""
import os
import torch
import gc
from contextlib import contextmanager

class GPUIsolationManager:
    """GPU隔离管理器，用于确保VLLM和训练进程使用不同的GPU资源"""
    
    def __init__(self, vllm_gpu_ids=[0], training_gpu_ids=[1,2,3]):
        """初始化GPU隔离管理器
        
        Args:
            vllm_gpu_ids: 用于VLLM推理的GPU ID列表
            training_gpu_ids: 用于训练的GPU ID列表
        """
        self.vllm_gpu_ids = vllm_gpu_ids
        self.training_gpu_ids = training_gpu_ids
        self.original_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        
        # 记录所有设备的原始状态
        self.all_gpu_ids = list(set(vllm_gpu_ids + training_gpu_ids))
        self.all_gpu_ids.sort()
        
        # 确保vllm_gpu_ids和training_gpu_ids没有重叠
        overlap = set(vllm_gpu_ids).intersection(set(training_gpu_ids))
        if overlap:
            raise ValueError(f"VLLM和训练进程不能共享相同的GPU: {overlap}")
    
    @contextmanager
    def vllm_context(self):
        """创建VLLM专用的GPU上下文"""
        try:
            self._isolate_for_vllm()
            yield
        finally:
            self._restore_environment()
    
    @contextmanager
    def training_context(self):
        """创建训练专用的GPU上下文"""
        try:
            self._isolate_for_training()
            yield
        finally:
            self._restore_environment()
    
    def _isolate_for_vllm(self):
        """隔离GPU资源用于VLLM"""
        # 释放所有GPU内存
        self._clear_gpu_memory()
        
        # 设置环境变量，只允许VLLM看到指定的GPU
        visible_devices = ','.join(map(str, self.vllm_gpu_ids))
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
        
        # 设置当前设备
        if len(self.vllm_gpu_ids) > 0:
            torch.cuda.set_device(0)  # 在新的环境变量下，第一个设备的索引是0
    
    def _isolate_for_training(self):
        """隔离GPU资源用于训练"""
        # 释放所有GPU内存
        self._clear_gpu_memory()
        
        # 设置环境变量，只允许训练进程看到指定的GPU
        visible_devices = ','.join(map(str, self.training_gpu_ids))
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
        
        # 设置当前设备
        if len(self.training_gpu_ids) > 0:
            torch.cuda.set_device(0)  # 在新的环境变量下，第一个设备的索引是0
    
    def _restore_environment(self):
        """恢复原始环境设置"""
        # 释放所有GPU内存
        self._clear_gpu_memory()
        
        # 恢复原始的CUDA_VISIBLE_DEVICES
        if self.original_visible_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.original_visible_devices
        else:
            # 如果原始值不存在，则删除该环境变量
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    
    def _clear_gpu_memory(self):
        """清理所有GPU内存"""
        torch.cuda.empty_cache()
        gc.collect()

    def get_vllm(self, model_id, **kwargs):
        """在VLLM专用GPU上加载VLLM模型"""
        import vllm
        
        with self.vllm_context():
            # 在VLLM专用GPU上下文中加载模型
            print(f"在GPU {self.vllm_gpu_ids}上加载VLLM模型: {model_id}")
            model = vllm.LLM(
                model_id,
                tensor_parallel_size=len(self.vllm_gpu_ids),  # 使用指定的GPU数量
                **kwargs
            )
            print("VLLM模型加载完成")
            
            # 获取并冻结模型参数，以便后续可以修改
            print("获取VLLM底层模型参数")
            underlying_model = model.llm_engine.model_executor.driver_worker.model_runner.model
            
            # 记录参数信息，用于后续更新
            param_info = {}
            for name, param in underlying_model.named_parameters():
                param_info[name] = {
                    'shape': param.shape,
                    'dtype': param.dtype,
                    'device': param.device,
                    'requires_grad': param.requires_grad
                }
                param.requires_grad = False  # 冻结参数
            
            return model, underlying_model, param_info

    def update_vllm_params(self, underlying_model, new_params_dict):
        """更新VLLM模型参数
        
        Args:
            underlying_model: VLLM的底层模型对象
            new_params_dict: 包含新参数的字典 {param_name: new_tensor}
        """
        with self.vllm_context():
            print(f"在GPU {self.vllm_gpu_ids}上更新VLLM模型参数")
            # 临时解除冻结
            for name, param in underlying_model.named_parameters():
                if name in new_params_dict:
                    param.requires_grad = True
                    # 确保新参数与原参数形状匹配
                    assert param.shape == new_params_dict[name].shape, \
                        f"参数形状不匹配: {name}, 原始: {param.shape}, 新: {new_params_dict[name].shape}"
                    # 更新参数
                    param.data.copy_(new_params_dict[name].data.to(param.device))
                    param.requires_grad = False
            
            print("VLLM模型参数更新完成")
            # 清理GPU内存
            torch.cuda.empty_cache()
            gc.collect()

# 示例用法
if __name__ == "__main__":
    # 初始化GPU隔离管理器
    manager = GPUIsolationManager(vllm_gpu_ids=[0], training_gpu_ids=[1,2,3])
    
    # 加载VLLM模型
    model_id = "/path/to/model"
    with manager.vllm_context():
        # 这里的代码将只能看到vllm_gpu_ids指定的GPU
        import vllm
        model = vllm.LLM(model_id, tensor_parallel_size=1)
        print("VLLM模型加载完成")
    
    # 切换到训练上下文
    with manager.training_context():
        # 这里的代码将只能看到training_gpu_ids指定的GPU
        import torch
        model = torch.nn.Linear(10, 10).cuda()  # 会自动放在第一个可见的GPU上
        print("训练模型创建完成") 