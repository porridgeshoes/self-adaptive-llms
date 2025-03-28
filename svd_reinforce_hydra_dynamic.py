#!/usr/bin/env python3
"""
动态更新VLLM参数的训练脚本 (基于svd_reinforce_hydra_v0.py)

此版本使用GPU隔离工具，允许VLLM和训练进程在同一Python进程中运行，
但使用不同的GPU资源，避免资源冲突，并支持动态更新VLLM参数。
"""
import gc
import json
import os
import sys
from datetime import datetime
from typing import Dict
import time

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

# 添加scripts目录到路径以导入GPU隔离工具
script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
if script_dir not in sys.path:
    sys.path.append(script_dir)

# 导入GPU隔离管理器
from vllm_gpu_isolation import GPUIsolationManager

from base_model import BaseModel
from logging_utils import Metrics, get_mean_std_max_min_dict
from optim_modules import OptimizationAlgorithm
from policy import Policy
from tasks import Task
from utils import (eval_model, eval_model_experts_prompt_based, forward,
                   load_hf_params_to_vllm)


def wandb_init(cfg, run_name: str, group_name: str, log_dir: str):
    import wandb

    config_dict = OmegaConf.to_container(
        cfg,
        resolve=True,
        throw_on_missing=False,
    )
    config_dict["log_dir"] = log_dir
    config_dict["wandb_run_name"] = run_name
    config_dict["wandb_group_name"] = group_name

    # wandb has a 128-size character limit on the group name
    wandb.init(
        project=cfg.wandb_project,
        group=group_name[:127],
        name=run_name[:127],
        config=config_dict,
    )
    return wandb


def create_gpu_isolation_manager():
    """创建GPU隔离管理器"""
    # 检查是否开启了GPU隔离
    use_isolation = os.environ.get("USE_GPU_ISOLATION", "0") == "1"
    if not use_isolation:
        return None
    
    # 解析GPU ID配置
    vllm_gpu_ids = [int(x) for x in os.environ.get("VLLM_GPU_IDS", "0").split(",")]
    training_gpu_ids = [int(x) for x in os.environ.get("TRAINING_GPU_IDS", "1,2,3").split(",")]
    
    print(f"创建GPU隔离管理器: VLLM GPUs={vllm_gpu_ids}, 训练 GPUs={training_gpu_ids}")
    return GPUIsolationManager(vllm_gpu_ids=vllm_gpu_ids, training_gpu_ids=training_gpu_ids)


@hydra.main(version_base=None, config_path="cfgs", config_name="config")
def main(cfg):
    """Main function."""

    # 创建GPU隔离管理器
    gpu_isolation_manager = create_gpu_isolation_manager()

    # (无效)为了让程序从执行的缓存目录下读取数据集
    hf_home = cfg.hf_home
    os.environ["HF_HOME"] = hf_home

    # 配置分解后的svd参数在哪个文件夹下保存。默认是运行目录，导致老被删。
    param_folder_path = cfg.param_folder_path

    num_iters = cfg.num_iters
    test_interval = cfg.test_interval

    batch_size = cfg.batch_size
    seed = cfg.seed
    policy_name = cfg.policy_name
    test_only = cfg.test_only
    save_legacy_params = cfg.save_legacy_params
    exp_name = cfg.exp_name
    run_name = cfg.run_name

    task_name = cfg.task_name

    load_ckpt = cfg.load_ckpt
    use_lora = cfg.use_lora
    prompt_based_eval = cfg.prompt_based_eval
    experts_path_dict = cfg.experts_path_dict

    resuming_from_ckpt = False
    if load_ckpt is not None:
        if load_ckpt == "scratch" or load_ckpt == "base":
            resuming_from_ckpt = False
        else:
            resuming_from_ckpt = True

    # Create task
    task_loader: Task = hydra.utils.instantiate(cfg.task_loader)

    base_model: BaseModel = hydra.utils.instantiate(cfg.base_model)

    model_id = base_model.get_model_id()
    decomposed_param_file = base_model.get_param_file(param_folder_path=param_folder_path)

    extract_svd = cfg.extract_svd or (not os.path.exists(decomposed_param_file))

    has_training_split = task_loader.has_training_split
    has_transfer_split = task_loader.has_transfer_split

    if not has_training_split:
        assert test_only, "Cannot train on a task with no training split"

    if exp_name is None:
        exp_name = "temp"

    metrics_to_log = Metrics()

    # Create log dir.
    if run_name is None:
        now = datetime.now()
        run_name = now.strftime("%Y%m%d-%H%M%S")
    if test_only and (not resuming_from_ckpt):
        log_dir = f"{cfg.out_dir}/{task_name}/{cfg.base_model_name}_base"
        group_name = cfg.base_model_name
    else:
        log_dir = f"{cfg.out_dir}/{task_name}/{policy_name}/{exp_name}/{run_name}"
        group_name = cfg.wandb_group_name
    os.makedirs(log_dir, exist_ok=True)

    print("【info】 开始获取vllm模型")
    os.system("nvidia-smi")
    
    # 在使用vllm前清理GPU内存
    torch.cuda.empty_cache()
    gc.collect()
    print("【info】GPU内存已清理，准备加载vllm模型")

    # 使用GPU隔离管理器加载VLLM模型
    if gpu_isolation_manager:
        print("【info】使用GPU隔离管理器加载VLLM模型")
        vllm_model, underlying_vllm_model, param_info = None, None, None
        try:
            # 加载VLLM模型
            vllm_kwargs = {
                "max_model_len": 1024,
                "gpu_memory_utilization": 0.8,
                "enforce_eager": True,
                "dtype": "bfloat16",
                "download_dir": task_loader.get_download_dir(),
            }
            vllm_model, underlying_vllm_model, param_info = gpu_isolation_manager.get_vllm(
                model_id=model_id, **vllm_kwargs
            )
            
            # 创建VLLMModel包装器
            from tasks.base import VLLMModel
            import vllm
            
            chat_template = task_loader.model_to_template[model_id]
            vllm_model = VLLMModel(
                vllm_model,
                sampling_params=vllm.SamplingParams(
                    temperature=0, 
                    top_p=1, 
                    max_tokens=512,
                    stop=["Instruction:", "Instruction", "Response:", "Response"], 
                    repetition_penalty=1.0, 
                ),
                chat_template=chat_template, 
            )
            print("【info】 获取vllm模型成功")
        except Exception as e:
            import traceback
            print(f"【error】 加载vllm模型失败: {str(e)}")
            traceback.print_exc()
            # 继续执行，尝试不使用vllm进行训练
            vllm_model = None
    else:
        # 常规方式加载VLLM模型
        try:
            # 调用原始方法加载VLLM模型
            vllm_model = task_loader.get_vllm_model(model_id=model_id)
            underlying_vllm_model = vllm_model.model.llm_engine.model_executor.driver_worker.model_runner.model
            param_info = None
            print("【info】 获取vllm模型成功")
        except Exception as e:
            print(f"【error】 加载vllm模型失败: {str(e)}")
            import traceback
            traceback.print_exc()
            # 继续执行，尝试不使用vllm进行训练
            vllm_model = None
            underlying_vllm_model = None
    
    # 显示GPU状态
    os.system("nvidia-smi")
    
    # 如果使用GPU隔离，切换到训练上下文
    if gpu_isolation_manager:
        print("【info】切换到训练GPU上下文")
        gpu_isolation_manager._isolate_for_training()
    
    train_eval, *test_evals = task_loader.get_evaluator()
    if task_loader.has_transfer_split:
        test_eval, transfer_eval = test_evals
    else:
        test_eval = test_evals[0]

    train_data, train_ix, valid_ix = task_loader.get_train_data()
    
    # 设置训练设备，确保在训练GPU上
    if gpu_isolation_manager:
        if len(gpu_isolation_manager.training_gpu_ids) > 0:
            gpu = torch.device(f"cuda:{gpu_isolation_manager.training_gpu_ids[0]}")
        else:
            gpu = torch.device("cpu")
    else:
        gpu = torch.device("cuda:1")  # 原始设置
    
    np_random = np.random.RandomState(seed)

    # cpu + float32 for initial SVD decomposition
    if extract_svd:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="cpu", torch_dtype=torch.float32
        )
    else:
        # Load model and tokenizer.
        # 使用当前训练设备
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map=gpu, torch_dtype=torch.bfloat16
        )

    print("【info】 结束AutoModelForCausalLM")
    os.system("nvidia-smi")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_params = model.state_dict()

    original_model_params = {
        k: v.clone().detach().cpu() for k, v in base_params.items() if "mlp" in k
    }

    # Load decomposed parameters.
    if not os.path.exists(decomposed_param_file):
        print("Decomposed params not found. Decomposing...")
        decomposed_params = {}
        for k, v in base_params.items():
            print(f"k的名称:{k}, v的shape: {v.shape}")
            if "norm" not in k:
                # print(k)
                if v.dim() < 2:
                    print(f"Skipping {k} (shape: {v.shape}): requires at least 2D for SVD.")
                    continue  # 跳过当前参数，继续下一个循环
                U, S, V = torch.svd(v)
                print(f"U的shape: {U.shape}, S的shape: {S.shape}, V的shape: {V.shape}")
                decomposed_params[f"{k}.U"] = U
                decomposed_params[f"{k}.S"] = S
                decomposed_params[f"{k}.V"] = V
        torch.save(decomposed_params, decomposed_param_file)
        print("successfully decomposed model - returning")
        return
    elif extract_svd:
        print(f"ERROR: SVD file already exists at {decomposed_param_file}")
    else:
        print("Decomposed params found. Loading...")
        assert not extract_svd
        decomposed_params = torch.load(decomposed_param_file)

    print("【info】 开始v.to(torch.bfloat16).to(gpu)")
    for k, v in decomposed_params.items():
        decomposed_params[k] = v.to(torch.bfloat16).to(gpu)

    print("【info】 结束v.to(torch.bfloat16).to(gpu)")
    os.system("nvidia-smi")
    if cfg.wandb_log:
        wandb = wandb_init(
            cfg=cfg, group_name=group_name, run_name=run_name, log_dir=log_dir
        )

    policy: Policy = hydra.utils.instantiate(
        cfg.shakeoff_policy,
        base_params=base_params,
        decomposed_params=decomposed_params,
        gpu=gpu,
    )

    print("【info】 开始DOptimizationAlgorithm")
    optimization_algorithm: OptimizationAlgorithm = hydra.utils.instantiate(
        cfg.optimization_algorithm,
        policy=policy,
        gpu=gpu,
    )

    # Load model and tokenizer and params.
    if prompt_based_eval:
        validation_result, vllm_valid_result = eval_model_experts_prompt_based(
            train_data,
            train_eval,
            valid_ix,
            tokenizer,
            model=model,
            vllm_model=vllm_model,
            experts_path_dict=experts_path_dict,
            model_id=model_id,
        )
    else:
        validation_result, vllm_valid_result = eval_model(
            train_data,
            train_eval,
            valid_ix,
            tokenizer,
            model=model,
            vllm_model=vllm_model,
            model_id=model_id,
        )
    print(f">> Validation result: {validation_result}")
    # TODO unify this
    # but note this will break previous logs
    base_acc = torch.mean(validation_result["acc"])
    metrics_to_log.update(
        {
            "val_acc": base_acc,
            "val_acc_std": torch.std(validation_result["acc"]),
            "val_acc_correct": validation_result["correct"],
            "val_acc_top_acc": validation_result["acc"].max(),
            "val_acc_min_acc": validation_result["acc"].min(),
        }
    )
    if vllm_valid_result is not None and len(vllm_valid_result) > 0:
        vllm_acc = torch.mean(vllm_valid_result["acc"])
        metrics_to_log.update(
            {
                "vllm_val_acc": vllm_acc,
                "vllm_val_acc_std": torch.std(vllm_valid_result["acc"]),
                "vllm_val_acc_correct": vllm_valid_result["correct"],
            }
        )
    if save_legacy_params:
        torch.save(validation_result, f"{log_dir}/base_result.pt")

    if test_only:
        if prompt_based_eval:
            test_result, vllm_test_result = eval_model_experts_prompt_based(
                train_data,
                test_eval,
                range(train_data.num_rows),
                tokenizer,
                model=model,
                vllm_model=vllm_model,
                experts_path_dict=experts_path_dict,
                model_id=model_id,
            )
        else:
            test_result, vllm_test_result = eval_model(
                train_data,
                test_eval,
                range(train_data.num_rows),
                tokenizer,
                model=model,
                vllm_model=vllm_model,
                model_id=model_id,
            )
        print(f">> Test result: {test_result}")
        # TODO unify this
        metrics_to_log.update(
            {
                "test_acc": torch.mean(test_result["acc"]),
                "test_acc_std": torch.std(test_result["acc"]),
                "test_acc_correct": test_result["correct"],
            }
        )
        if vllm_test_result is not None and len(vllm_test_result) > 0:
            metrics_to_log.update(
                {
                    "vllm_test_acc": torch.mean(vllm_test_result["acc"]),
                    "vllm_test_acc_std": torch.std(vllm_test_result["acc"]),
                    "vllm_test_acc_correct": vllm_test_result["correct"],
                }
            )
        if cfg.wandb_log:
            wandb.log(metrics_to_log.to_dict())

        if has_transfer_split:
            transfer_result, vllm_transfer_result = eval_model(
                train_data,
                transfer_eval,
                range(train_data.num_rows),
                tokenizer,
                model=model,
                vllm_model=vllm_model,
                model_id=model_id,
            )
            print(f">> Transfer result: {transfer_result}")
            metrics_to_log.update(
                {
                    "transfer_acc": torch.mean(transfer_result["acc"]),
                    "transfer_acc_std": torch.std(transfer_result["acc"]),
                    "transfer_acc_correct": transfer_result["correct"],
                }
            )
            if vllm_transfer_result is not None and len(vllm_transfer_result) > 0:
                metrics_to_log.update(
                    {
                        "vllm_transfer_acc": torch.mean(vllm_transfer_result["acc"]),
                        "vllm_transfer_acc_std": torch.std(vllm_transfer_result["acc"]),
                        "vllm_transfer_acc_correct": vllm_transfer_result["correct"],
                    }
                )
            if cfg.wandb_log:
                wandb.log(metrics_to_log.to_dict())
        return

    # reset policy
    if load_ckpt is not None and load_ckpt != "scratch" and load_ckpt != "base":
        print(f"Loading policy from {load_ckpt}")
        policy.load(load_ckpt)
    else:
        policy.reset()

    best_val_acc = base_acc
    best_val_epoch = -1
    # Learning
    for it in range(num_iters):
        print(f"Iteration {it}")
        metrics_this_iter = Metrics()

        # Update all parameters at once.
        new_params = policy()

        # Train the model.
        optimization_algorithm.pre_epoch(it)

        # 准备训练样本
        train_size = len(train_ix)
        example_indices = np_random.permutation(train_size)

        # 批量处理训练样本
        batch_start_indices = list(range(0, train_size, batch_size))
        for batch_idx, start_idx in enumerate(batch_start_indices):
            batch_example_indices = example_indices[start_idx : start_idx + batch_size]
            batch_train_indices = [train_ix[i] for i in batch_example_indices]

            batch_prompts = [
                task_loader.get_prompt(tokenizer, train_data, idx, model_id)
                for idx in batch_train_indices
            ]

            # Forward pass with policy parameters.
            fw_metrics = forward(
                new_params=new_params,
                prompts=batch_prompts,
                model=model,
                tokenizer=tokenizer,
                batch_index=batch_idx,
                train_batch_size=batch_size,
            )

            rewards = task_loader.get_rewards(train_eval.evaluate_generation(fw_metrics["completions"]))
            optimization_algorithm.update(rewards=torch.tensor(rewards, device=gpu))

        policy.optimizer_step(optimization_algorithm)

        # 如果使用GPU隔离且VLLM模型已加载，更新VLLM参数
        if gpu_isolation_manager and vllm_model and underlying_vllm_model:
            # 将策略生成的参数传递给VLLM模型
            print("【info】更新VLLM模型参数")
            vllm_params_to_update = {}
            
            # 将分解后的SVD参数重组为原始参数
            for key in original_model_params:
                if key in policy.param_dict:
                    vllm_params_to_update[key] = policy.param_dict[key]
            
            # 更新VLLM参数
            gpu_isolation_manager.update_vllm_params(underlying_vllm_model, vllm_params_to_update)
            print("【info】VLLM模型参数更新完成")

        # Evaluate on validation set.
        metrics_this_iter.update(
            {
                "iter": it,
                "reward": torch.mean(torch.tensor(rewards)),
                "reward_std": torch.std(torch.tensor(rewards)),
            }
        )

        if (it + 1) % test_interval == 0 or it == num_iters - 1:
            # Evaluate validation accuracy.
            if prompt_based_eval:
                validation_result, vllm_valid_result = eval_model_experts_prompt_based(
                    train_data,
                    train_eval,
                    valid_ix,
                    tokenizer,
                    model=model,
                    vllm_model=vllm_model,
                    overwrite_param_dict=new_params,
                    experts_path_dict=experts_path_dict,
                    model_id=model_id,
                )
            else:
                validation_result, vllm_valid_result = eval_model(
                    train_data,
                    train_eval,
                    valid_ix,
                    tokenizer,
                    model=model,
                    vllm_model=vllm_model,
                    overwrite_param_dict=new_params,
                    model_id=model_id,
                )
            val_acc = torch.mean(validation_result["acc"])
            print(f">> [Val] Iteration {it} result: {validation_result}")
            metrics_this_iter.update(
                {
                    "val_acc": val_acc,
                    "val_acc_std": torch.std(validation_result["acc"]),
                    "val_acc_correct": validation_result["correct"],
                }
            )
            if vllm_valid_result is not None and len(vllm_valid_result) > 0:
                metrics_this_iter.update(
                    {
                        "vllm_val_acc": torch.mean(vllm_valid_result["acc"]),
                        "vllm_val_acc_std": torch.std(vllm_valid_result["acc"]),
                        "vllm_val_acc_correct": vllm_valid_result["correct"],
                    }
                )
            if save_legacy_params:
                torch.save(validation_result, f"{log_dir}/validation_result_{it}.pt")

            # Save the best model.
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_epoch = it
                print(f">> [Best] New best model at iteration {it} with val_acc {val_acc}")
                policy.save(f"{log_dir}/best.pt")

            # Get test accuracy.
            if prompt_based_eval:
                test_result, vllm_test_result = eval_model_experts_prompt_based(
                    train_data,
                    test_eval,
                    range(train_data.num_rows),
                    tokenizer,
                    model=model,
                    vllm_model=vllm_model,
                    overwrite_param_dict=new_params,
                    experts_path_dict=experts_path_dict,
                    model_id=model_id,
                )
            else:
                test_result, vllm_test_result = eval_model(
                    train_data,
                    test_eval,
                    range(train_data.num_rows),
                    tokenizer,
                    model=model,
                    vllm_model=vllm_model,
                    overwrite_param_dict=new_params,
                    model_id=model_id,
                )
            print(f">> [Test] Iteration {it} result: {test_result}")
            metrics_this_iter.update(
                {
                    "test_acc": torch.mean(test_result["acc"]),
                    "test_acc_std": torch.std(test_result["acc"]),
                    "test_acc_correct": test_result["correct"],
                }
            )
            if vllm_test_result is not None and len(vllm_test_result) > 0:
                metrics_this_iter.update(
                    {
                        "vllm_test_acc": torch.mean(vllm_test_result["acc"]),
                        "vllm_test_acc_std": torch.std(vllm_test_result["acc"]),
                        "vllm_test_acc_correct": vllm_test_result["correct"],
                    }
                )

            # Update metrics with best val acc infomation.
            metrics_this_iter.update(
                {
                    "best_val_acc": best_val_acc,
                    "best_val_epoch": best_val_epoch,
                }
            )

            # Log the metrics.
            metrics_to_log.update(metrics_this_iter)
            if cfg.wandb_log:
                wandb.log(metrics_to_log.to_dict())

        # Save the model after each iteration.
        policy.save(f"{log_dir}/model_{it}.pt")

    # 清理资源
    if gpu_isolation_manager:
        gpu_isolation_manager._restore_environment()


if __name__ == "__main__":
    main() 