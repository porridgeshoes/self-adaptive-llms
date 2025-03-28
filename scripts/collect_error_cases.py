#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
误判漏判案例收集脚本
此脚本用于收集QwQ-32B模型的误判漏判案例，将其保存为适合训练用的JSON格式
"""

import os
import json
import argparse
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="收集LLM误判漏判案例")
    parser.add_argument(
        "--model_id", 
        type=str, 
        default="01-ai/Yi-VL-34B",  # 使用QwQ-32B的ID
        help="模型ID"
    )
    parser.add_argument(
        "--data_file", 
        type=str, 
        required=True,
        help="包含问题和期望答案的数据文件，JSON格式"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="error_cases.json",
        help="输出的误判漏判案例文件"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=1,
        help="批处理大小"
    )
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=512,
        help="生成的最大token数"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda:0",
        help="设备"
    )
    return parser.parse_args()


def load_model_and_tokenizer(model_id: str, device: str):
    """加载模型和分词器"""
    print(f"正在加载模型：{model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    return model, tokenizer


def generate_responses(
    model, 
    tokenizer, 
    questions: List[str], 
    max_new_tokens: int = 512,
    batch_size: int = 1
):
    """生成模型回答"""
    responses = []
    
    for i in range(0, len(questions), batch_size):
        batch_questions = questions[i:i + batch_size]
        batch_responses = []
        
        for question in batch_questions:
            messages = [
                {"role": "system", "content": "你是一个智能助手，请根据问题提供准确、有帮助的回答。"},
                {"role": "user", "content": question}
            ]
            
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
                    do_sample=False
                )
            
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            batch_responses.append(response)
        
        responses.extend(batch_responses)
        print(f"已处理 {min(i + batch_size, len(questions))}/{len(questions)} 个问题")
    
    return responses


def evaluate_responses(
    questions: List[str], 
    expected_answers: List[str], 
    model_answers: List[str],
    error_types: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """评估模型回答，识别误判和漏判案例"""
    error_cases = []
    
    for i, (question, expected, model_answer) in enumerate(zip(questions, expected_answers, model_answers)):
        # 可以根据具体任务类型定制评估逻辑
        is_correct = expected.lower() in model_answer.lower()
        
        if not is_correct:
            error_type = error_types[i] if error_types and i < len(error_types) else "misclassification"
            
            error_case = {
                "question": question,
                "expected_answer": expected,
                "model_answer": model_answer,
                "error_type": error_type
            }
            
            error_cases.append(error_case)
    
    return error_cases


def main():
    args = parse_args()
    
    # 加载数据集
    with open(args.data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取问题和期望答案
    questions = [item["question"] for item in data]
    expected_answers = [item["expected_answer"] for item in data]
    error_types = [item.get("error_type", "misclassification") for item in data] if "error_type" in data[0] else None
    
    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(args.model_id, args.device)
    
    # 生成模型回答
    model_answers = generate_responses(
        model, 
        tokenizer, 
        questions, 
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size
    )
    
    # 评估回答，识别误判漏判案例
    error_cases = evaluate_responses(
        questions, 
        expected_answers, 
        model_answers,
        error_types
    )
    
    # 保存误判漏判案例
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(error_cases, f, ensure_ascii=False, indent=2)
    
    print(f"共发现 {len(error_cases)} 个误判漏判案例，已保存到 {args.output_file}")


if __name__ == "__main__":
    main() 