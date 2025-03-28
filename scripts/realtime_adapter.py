#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
QwQ-32B实时适应脚本
该脚本用于在推理时实时调整QwQ-32B模型，根据用户的输入和历史记录选择合适的专家向量
"""

import os
import json
import argparse
import numpy as np
from typing import List, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import hydra
from omegaconf import OmegaConf, DictConfig

from utils import load_base_params, apply_svd_params
from policy import WeightedCombination
from base_model import QwQ32B


def parse_args():
    parser = argparse.ArgumentParser(description="QwQ-32B实时适应")
    parser.add_argument(
        "--model_id", 
        type=str, 
        default="01-ai/Yi-VL-34B",  # 使用QwQ-32B的ID
        help="模型ID"
    )
    parser.add_argument(
        "--experts_dir", 
        type=str, 
        default="results/qwq32b/",
        help="专家模型目录，包含多个专家模型的权重"
    )
    parser.add_argument(
        "--decomposed_params_file", 
        type=str, 
        default="qwq32b_decomposed_params.pt",
        help="模型分解参数文件"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda:0",
        help="设备"
    )
    parser.add_argument(
        "--interactive", 
        action="store_true",
        help="是否进入交互模式"
    )
    return parser.parse_args()


def find_expert_paths(experts_dir: str) -> List[str]:
    """
    查找所有专家模型路径
    """
    expert_paths = []
    for root, dirs, files in os.walk(experts_dir):
        for file in files:
            if file == "learnable_params.pt":
                expert_paths.append(os.path.join(root, file))
    
    return expert_paths


def select_experts(query: str, expert_paths: List[str], history: List[Dict[str, str]]) -> List[float]:
    """
    根据查询和历史记录选择专家权重
    这个函数可以替换为更复杂的方法，比如基于语义相似度的选择
    
    Args:
        query: 用户查询
        expert_paths: 专家模型路径列表
        history: 历史对话记录
    
    Returns:
        专家权重列表
    """
    # 简单的启发式方法：对所有专家使用均匀权重
    weights = [1.0 / len(expert_paths) for _ in expert_paths]
    
    # 这里可以添加更复杂的专家选择逻辑
    # 例如，检测查询中特定关键词，增加相关专家的权重
    
    return weights


def load_model_and_experts(
    model_id: str,
    decomposed_params_file: str,
    expert_paths: List[str],
    device: str
):
    """
    加载模型和专家
    """
    print(f"正在加载模型：{model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    
    # 加载分解参数
    print(f"正在加载分解参数：{decomposed_params_file}")
    if os.path.exists(decomposed_params_file):
        decomposed_params = torch.load(decomposed_params_file)
    else:
        # 如果没有预先计算的分解参数，可以实时计算
        # 但这需要更多的计算资源和时间
        print("警告：找不到预计算的分解参数，将使用原始模型")
        decomposed_params = None
    
    # 加载基础参数
    base_params = {}
    for n, p in model.named_parameters():
        if "mlp" in n:
            base_params[n] = p.data.to(device)
    
    # 创建策略
    print(f"正在加载专家模型，共 {len(expert_paths)} 个专家")
    policy = WeightedCombination(
        base_params=base_params,
        decomposed_params=decomposed_params,
        base_policy_cfg=None,
        params_paths=expert_paths,
        gpu=device,
        norm_coeffs=None,
        per_layer=False,
        init_values=None
    )
    
    return model, tokenizer, policy


def apply_expert_weights(model, policy, expert_weights: List[float]):
    """
    应用专家权重到模型
    """
    with torch.no_grad():
        policy.set_expert_weights(expert_weights)
        learnable_params = policy.get_learnable_params()
        
        # 应用SVD参数
        for n, p in model.named_parameters():
            if n in learnable_params:
                if hasattr(policy, "decomposed_params") and policy.decomposed_params is not None:
                    apply_svd_params(model, n, learnable_params[n], policy.decomposed_params)
                else:
                    # 如果没有分解参数，直接应用
                    p.data.mul_(learnable_params[n])
                    
    return model


def generate_response(
    model, 
    tokenizer, 
    query: str, 
    history: List[Dict[str, str]] = None,
    max_new_tokens: int = 512
):
    """
    生成模型回答
    """
    if history is None:
        history = []
    
    messages = [
        {"role": "system", "content": "你是一个智能助手，请根据问题提供准确、有帮助的回答。"}
    ]
    
    for item in history:
        messages.append({"role": "user", "content": item["user"]})
        messages.append({"role": "assistant", "content": item["assistant"]})
    
    messages.append({"role": "user", "content": query})
    
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response


def main():
    args = parse_args()
    
    # 查找专家模型
    expert_paths = find_expert_paths(args.experts_dir)
    if not expert_paths:
        print(f"错误：在 {args.experts_dir} 中找不到专家模型")
        return
    
    print(f"找到 {len(expert_paths)} 个专家模型：")
    for i, path in enumerate(expert_paths):
        print(f"  {i+1}. {path}")
    
    # 加载模型和专家
    model, tokenizer, policy = load_model_and_experts(
        args.model_id,
        args.decomposed_params_file,
        expert_paths,
        args.device
    )
    
    if args.interactive:
        # 交互模式
        history = []
        
        print("\n============ QwQ-32B 实时适应模式 ============")
        print("输入 'exit' 或 'quit' 退出\n")
        
        while True:
            query = input("\n用户: ")
            if query.lower() in ["exit", "quit"]:
                break
            
            # 选择专家权重
            expert_weights = select_experts(query, expert_paths, history)
            
            # 应用专家权重
            model = apply_expert_weights(model, policy, expert_weights)
            
            # 生成回答
            response = generate_response(model, tokenizer, query, history)
            
            print(f"\nQwQ-32B: {response}")
            
            # 更新历史
            history.append({"user": query, "assistant": response})
    else:
        # 非交互模式
        query = "请介绍一下量子计算的基本原理"
        
        # 选择专家权重
        expert_weights = select_experts(query, expert_paths, [])
        
        # 应用专家权重
        model = apply_expert_weights(model, policy, expert_weights)
        
        # 生成回答
        response = generate_response(model, tokenizer, query)
        
        print(f"查询: {query}")
        print(f"回答: {response}")


if __name__ == "__main__":
    main() 