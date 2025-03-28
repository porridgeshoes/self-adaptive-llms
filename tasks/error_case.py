import os
import json
import re
from dataclasses import dataclass
from typing import Iterable, List, Dict, Tuple, Optional, Any

import torch
import datasets
import fishfarm
from fishfarm.models.vllm_model import VLLMModel
from fishfarm.tasks.base import TaskResult
from fishfarm.tasks.evalplus import load_dataset

from .base import Task, get_download_dir


def mean(iterable: Iterable[float]) -> float:
    total, count = 0.0, 0
    for x in iterable:
        total += x
        count += 1
    return total / max(count, 1)


@dataclass
class ErrorCaseSample:
    """误判或漏判的样本类"""
    question: str
    expected_answer: str
    model_answer: Optional[str] = None
    error_type: str = "misclassification"  # 可以是 "misclassification" 或 "missed"
    metadata: Dict[str, Any] = None


class ErrorCaseTask(fishfarm.tasks.base.Task):
    """基于误判漏判案例的评估任务"""
    def __init__(
        self,
        samples,
        context_messages,
    ):
        self.samples = list(samples)
        self.context_messages = context_messages

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    def evaluate(
        self,
        model,
        sample_ids,
    ):
        if sample_ids is None:
            sample_ids = range(len(self.samples))
        samples = [self.samples[sample_id] for sample_id in sample_ids]

        requests = []
        for sample in samples:
            messages = self.context_messages.copy()
            messages.append({"role": "user", "content": sample.question})
            requests.append(
                {
                    "model_id": model.model_id,
                    "messages": messages,
                }
            )

        responses = model.generate_async_json(
            requests, 
            max_tokens=512, 
            temperature=0.0,
            top_p=0.9
        )

        correct = []
        for i, response in enumerate(responses):
            sample = samples[i]
            model_response = response.output["choices"][0]["message"]["content"]
            
            # 根据任务类型判断是否正确
            # 此处需根据具体任务进行定制
            is_correct = self._is_answer_correct(model_response, sample.expected_answer)
            correct.append(int(is_correct))

        return TaskResult(correct=correct, responses=responses)
    
    def _is_answer_correct(self, model_response, expected_answer):
        """
        判断模型回答是否与预期回答一致
        可以根据具体任务类型进行定制，这里提供简单的字符串匹配
        """
        return expected_answer.lower() in model_response.lower()


class ErrorCaseAdaptationTask(Task):
    """用于适应误判漏判案例的任务类"""
    def __init__(self, error_cases_file=None):
        """
        error_cases_file: 包含误判漏判样本的JSON文件
        格式应为:
        [
            {
                "question": "问题文本",
                "expected_answer": "期望答案",
                "model_answer": "模型产生的错误答案",
                "error_type": "misclassification或missed",
                "metadata": {}  # 可选的元数据
            },
            ...
        ]
        """
        self.error_cases_file = error_cases_file
        self.samples = []
        self.has_training_split = True
        self.has_transfer_split = False
        
        if error_cases_file and os.path.exists(error_cases_file):
            with open(error_cases_file, 'r', encoding='utf-8') as f:
                error_cases = json.load(f)
                for case in error_cases:
                    self.samples.append(ErrorCaseSample(
                        question=case["question"],
                        expected_answer=case["expected_answer"],
                        model_answer=case.get("model_answer", None),
                        error_type=case.get("error_type", "misclassification"),
                        metadata=case.get("metadata", {})
                    ))
        
        # 如果没有提供文件或文件为空，创建一些示例样本
        if not self.samples:
            self._create_sample_error_cases()
            
        self.train_samples, self.test_samples = self.split_samples(self.samples)
    
    def _create_sample_error_cases(self):
        """创建示例误判漏判案例用于测试"""
        sample_cases = [
            {
                "question": "这款产品的质量如何？",
                "expected_answer": "非常好",
                "model_answer": "一般",
                "error_type": "misclassification"
            },
            {
                "question": "请解释量子计算的基本原理",
                "expected_answer": "量子计算利用量子叠加态和量子纠缠...",
                "model_answer": None,
                "error_type": "missed"
            }
        ]
        
        for case in sample_cases:
            self.samples.append(ErrorCaseSample(
                question=case["question"],
                expected_answer=case["expected_answer"],
                model_answer=case.get("model_answer"),
                error_type=case["error_type"]
            ))
    
    def split_samples(self, samples):
        """将样本分为训练集和测试集"""
        split_idx = int(len(samples) * 0.8)
        return samples[:split_idx], samples[split_idx:]
    
    def get_train_data(self, max_samples=400):
        """获取训练数据"""
        return self.train_samples[:max_samples]
    
    def build_samples(self):
        """构建样本用于评估"""
        return self.samples
    
    def get_rewards(self, res):
        """计算奖励值"""
        rewards = []
        for output, sample in zip(res["outputs"], res["samples"]):
            # 通过简单的字符串匹配来判断是否正确
            expected = sample.expected_answer.lower()
            model_output = output.lower()
            
            # 检查模型输出是否包含预期答案
            is_correct = expected in model_output
            
            # 根据错误类型调整奖励
            if sample.error_type == "misclassification":
                # 对于误判，如果现在正确了，给予高奖励
                reward = 1.0 if is_correct else -0.5
            else:  # "missed"
                # 对于漏判，如果能够正确回答，给予高奖励
                reward = 1.0 if is_correct else -0.5
                
            rewards.append(reward)
        
        return rewards
    
    def get_evaluator(self) -> Tuple:
        """获取评估器"""
        context_messages = [
            {"role": "system", "content": "你是一个智能助手，请根据问题提供准确、有帮助的回答。"}
        ]
        
        # 创建评估任务
        eval_task = ErrorCaseTask(
            samples=self.test_samples,
            context_messages=context_messages,
        )
        
        return eval_task, 0, len(self.test_samples)
    
    def get_prompt(self, tokenizer, samples, ix, model_id):
        """生成提示"""
        sample = samples[ix]
        messages = [
            {"role": "system", "content": "你是一个智能助手，请根据问题提供准确、有帮助的回答。"},
            {"role": "user", "content": sample.question}
        ]
        
        if "meta-llama" in model_id:
            from .base import LLAMA3_COT
            fmt = LLAMA3_COT
        else:
            from .base import CODE_PROMPT
            fmt = CODE_PROMPT

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return prompt, sample
    
    def get_vllm_model(self, model_id) -> VLLMModel:
        """获取VLLM模型"""
        return VLLMModel(
            model_id=model_id,
            tensor_parallel_size=1,
            trust_remote_code=True,
        ) 