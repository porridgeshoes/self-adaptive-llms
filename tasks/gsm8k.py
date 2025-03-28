from typing import Tuple

import fishfarm
import vllm
from datasets import load_dataset
from fishfarm.models.vllm_model import VLLMModel
from fishfarm.tasks.language_restricted_math import (
    LanguageRestrictedMathTask, MathSample, extract_answer_number)

from .base import Task, get_download_dir


class Gsm8kTask(Task):
    def __init__(
        self,
    ):
        self.cache_dir="/data1/yammyjiang/new_space/light_things/LLM/self-adaptive-llms/download"
        self.model_to_template = {
            "meta-llama/Meta-Llama-3-8B-Instruct": (
                "{% set loop_messages = messages %}"
                "{% for message in loop_messages %}"
                "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>"
                "\n\n'+ message['content'] | trim + '<|eot_id|>' %}"
                "{% if loop.index0 == 0 %}{% set content = bos_token + content %}"
                "{% endif %}"
                "{{ content }}"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
                "{% endif %}"
            ),
            "mistralai/Mistral-7B-Instruct-v0.3": None,
            # 为自定义模型增加一个prompt
            "/data1/nanzhiwang/python_work/open-r1/models/DeepSeek-R1-Distill-Qwen-7B": (
                "{% set loop_messages = messages %}"
                "{% for message in loop_messages %}"
                "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>"
                "\n\n'+ message['content'] | trim + '<|eot_id|>' %}"
                "{% if loop.index0 == 0 %}{% set content = bos_token + content %}"
                "{% endif %}"
                "{{ content }}"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
                "{% endif %}"
            ),
        }
        self.system_msg = (
            "Below is an instruction that describes a task."
            " Write a response that appropriately completes the request.\n\n"
        )

        self.target_metric_train = "acc"
        self.target_metric_valid = self.target_metric_train
        self.target_metric_test = self.target_metric_train
        self.target_metric_transfer = self.target_metric_train
        self.has_transfer_split = False
        self.has_training_split = True

    def get_train_data(
        self,
    ):
        # 修复数据下载的问题。
        # train_data = load_dataset("gsm8k", "main", split="train")
        train_data = load_dataset("gsm8k", "main", split="train", cache_dir=self.cache_dir)

        train_size = len(train_data)
        train_ix = range(0, train_size, 2)
        valid_ix = range(1, train_size, 2)
        return train_data, train_ix, valid_ix

    def get_rewards(self, res):
        rewards = [1.0 if x["correct"] else -1.0 for x in res.sample_details]
        return rewards

    def get_evaluator(self) -> Tuple:
        res = []
        for split in ["train", "test"]:
            # dataset = load_dataset("gsm8k", "main", split=split)
            dataset = load_dataset("gsm8k", "main", split=split, cache_dir=self.cache_dir)
            samples = []
            for sample in dataset:
                answer = sample["answer"]
                answer = extract_answer_number(answer)
                answer = int(answer) if answer is not None else None
                samples.append(
                    MathSample(
                        problem=sample["question"],
                        answer=answer,
                    )
                )
            res.append(
                LanguageRestrictedMathTask(
                    samples=samples,
                    context_messages=[
                        fishfarm.Message("system", self.system_msg),
                    ],
                    languages=[],
                )
            )
        return tuple(res)

    def get_prompt(self, tokenizer, samples, ix, model_id):
        chat_template = self.model_to_template[model_id]
        context_msg = {"role": "system", "content": self.system_msg}
        user_msg = {"role": "user", "content": samples["question"][ix]}
        prompt = tokenizer.apply_chat_template(
            conversation=[context_msg, user_msg],
            chat_template=chat_template,
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt

    def get_vllm_model(self, model_id) -> VLLMModel:
        """Get a VLLM model for inference."""
        print("【debug】设置 VLLM 使用 cuda:0 设备，tensor_parallel_size=1")
        
        try:
            import time
            import os
            import torch
            
            # 设置PyTorch当前设备为CUDA 0，帮助vllm正确选择设备
            # 并设置环境变量确保vllm能找到正确的设备
            torch.cuda.set_device(0)
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            start_time = time.time()
            
            # 降低内存使用，提高稳定性
            model = vllm.LLM(
                model_id,
                max_model_len=1024,
                gpu_memory_utilization=0.6,  # 降低内存使用率
                enforce_eager=True,
                dtype="bfloat16",
                download_dir=get_download_dir(),
                tensor_parallel_size=1,  # 明确只使用1个GPU
                # trust_remote_code=True,
            )
            
            # 恢复原始GPU可见性设置
            os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,0"
            
            load_time = time.time() - start_time
            print(f"【debug】VLLM 模型加载完成，耗时: {load_time:.2f} 秒")
            
            chat_template = self.model_to_template[model_id]
            print("【debug】获取 chat_template 成功")

            # This may change with vLLM versions.
            print("【debug】开始获取底层模型并冻结参数")
            m = model.llm_engine.model_executor.driver_worker.model_runner.model
            for _, param in m.named_parameters():
                param.requires_grad = False # 参数冻结
            print("【debug】参数冻结完成")

            # VLLMModel用于对 vLLM 模型进行进一步的抽象和管理。
            print("【debug】开始创建 VLLMModel 对象")
            vllm_model = VLLMModel(
                model,
                sampling_params=vllm.SamplingParams(
                    temperature=0, 
                    top_p=1, 
                    max_tokens=512,
                    stop=["Instruction:", "Instruction", "Response:", "Response"], 
                    repetition_penalty=1.0, 
                ),
                chat_template=chat_template, 
            )
            print("【debug】VLLMModel 对象创建成功，返回模型")
            return vllm_model
            
        except Exception as e:
            import traceback
            print(f"【error】加载 VLLM 模型时出错: {str(e)}")
            print(f"【error】详细错误信息: {traceback.format_exc()}")
            raise
