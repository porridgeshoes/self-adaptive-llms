# QwQ-32B 大模型误判漏判自适应框架

基于Transformer²自适应LLM框架实现的QwQ-32B误判漏判实时调整方案。该方案允许大模型在发现误判漏判案例后，通过轻量级参数调整实现快速适应，无需对整个模型进行完整的微调。

## 项目背景

传统的大模型微调方法存在以下问题：
- 计算资源消耗大
- 适应性差，每次遇到新问题都需要重新训练
- 容易过拟合，影响模型在其他任务上的表现

本项目基于Transformer²框架（来自[Sakana AI](https://sakana.ai/transformer-squared)），实现了一种轻量级的自适应方法，通过仅调整模型权重矩阵的奇异值分量，使QwQ-32B模型能够针对误判漏判案例进行实时调整。

## 项目结构

```
self-adaptive-llms/
├── base_model/                  # 基础模型定义
│   ├── __init__.py              # 模型导入
│   ├── base.py                  # 基础模型抽象类
│   ├── llama3instruct.py        # Llama3模型定义
│   ├── mistral03instruct.py     # Mistral模型定义
│   └── qwq32b.py                # 新增：QwQ-32B模型定义
│
├── cfgs/                        # 配置文件
│   ├── base_model/              # 模型配置
│   │   └── qwq32b.yaml          # 新增：QwQ-32B模型配置
│   ├── task/                    # 任务配置
│   │   └── error_case.yaml      # 新增：误判漏判任务配置
│   └── config.yaml              # 主配置文件
│
├── policy/                      # 策略模块
│   ├── __init__.py
│   ├── base.py                  # 基础策略类
│   └── weighted_combination.py  # 修改：添加了set_expert_weights方法
│
├── scripts/                     # 脚本目录
│   ├── collect_error_cases.py   # 新增：误判漏判案例收集脚本
│   ├── eval_qwq_error_adaptation.sh  # 新增：QwQ评估脚本
│   ├── realtime_adapter.py      # 新增：实时适应脚本
│   └── train_qwq_error_adaptation.sh # 新增：QwQ训练脚本
│
├── tasks/                       # 任务模块
│   ├── __init__.py              # 修改：导入新任务
│   ├── base.py                  # 基础任务类
│   └── error_case.py            # 新增：误判漏判任务类
│
├── utils.py                     # 修改：添加apply_svd_params函数
└── ...                          # 其他文件
```

## 主要改动说明

本项目对原始的Transformer²框架进行了以下改动，以支持QwQ-32B模型的误判漏判实时适应：

### 1. 添加QwQ-32B模型支持

- 创建了`base_model/qwq32b.py`文件，定义了QwQ-32B模型类
- 更新了`base_model/__init__.py`，导入新的模型类
- 创建了`cfgs/base_model/qwq32b.yaml`配置文件

### 2. 实现误判漏判任务

- 创建了`tasks/error_case.py`，实现了误判漏判适应任务类
- 定义了`ErrorCaseSample`类来表示误判/漏判样本
- 实现了`ErrorCaseTask`和`ErrorCaseAdaptationTask`两个类
- 更新了`tasks/__init__.py`，导入新的任务类
- 创建了`cfgs/task/error_case.yaml`任务配置文件

### 3. 增强策略模块

- 修改了`policy/weighted_combination.py`，添加了`set_expert_weights`方法
- 此方法允许在推理时动态调整专家权重，实现实时适应

### 4. 添加SVD参数应用功能

- 在`utils.py`中添加了`apply_svd_params`函数
- 该函数实现了将SVD分解后的参数应用到模型的功能

### 5. 创建工作流脚本

- 创建了误判漏判案例收集脚本`scripts/collect_error_cases.py`
- 创建了QwQ-32B训练脚本`scripts/train_qwq_error_adaptation.sh`
- 创建了QwQ-32B评估脚本`scripts/eval_qwq_error_adaptation.sh`
- 创建了实时适应脚本`scripts/realtime_adapter.py`

## 使用方法

### 环境准备

```bash
conda create -n t2 python=3.11 -y
conda activate t2
pip install --upgrade pip
pip install -r requirements.txt

# 安装任务评估器
cd evaluation/fishfarm
pip install -e .
```

### 步骤1：收集误判漏判案例

1. 准备一个包含问题和期望答案的JSON文件，格式如下：
```json
[
    {
        "question": "问题文本",
        "expected_answer": "期望答案"
    },
    ...
]
```

2. 使用收集脚本识别模型的误判漏判案例：
```bash
python scripts/collect_error_cases.py --data_file your_data.json --output_file error_cases.json
```

### 步骤2：训练误判漏判专家模型

1. 设置环境变量指向收集的误判漏判案例文件：
```bash
export ERROR_CASES_FILE="/path/to/error_cases.json"
```

2. 运行训练脚本：
```bash
bash scripts/train_qwq_error_adaptation.sh
```

训练会在`results/qwq32b/reinforce-error_adaptation/`目录下生成专家模型。

### 步骤3：评估专家模型

评估模型在误判漏判案例上的表现：
```bash
bash scripts/eval_qwq_error_adaptation.sh
```

### 步骤4：实时使用

使用实时适应脚本进行交互式使用：
```bash
python scripts/realtime_adapter.py --interactive
```

或在非交互模式下使用：
```bash
python scripts/realtime_adapter.py
```

## 核心原理

### 1. 选择性参数调整

该系统不调整整个模型的权重，而是通过奇异值分解(SVD)方法，仅调整模型权重矩阵的奇异值分量，实现轻量级的模型调整。这种方法有几个优势：

- 参数量大幅减少，训练更高效
- 避免了过拟合，保持了模型的泛化能力
- 允许实时动态调整，适应不同的输入

### 2. 专家模型组合

系统训练了针对不同误判漏判案例的"专家"模型，在推理时通过加权组合这些专家的权重，实现对不同类型错误的适应：

```
最终参数 = w₁·专家₁ + w₂·专家₂ + ... + wₙ·专家ₙ
```

其中，权重 w₁, w₂, ..., wₙ 是根据输入动态确定的。

### 3. 强化学习训练

系统使用REINFORCE算法训练专家模型，通过奖励机制鼓励模型正确回答之前的误判漏判案例：

- 对于误判案例：如果修正了错误，给予高奖励
- 对于漏判案例：如果能够给出答案，给予高奖励

## 技术要点

1. **SVD分解与重建**: 通过对权重矩阵进行奇异值分解，仅调整奇异值而保持奇异向量不变，实现高效的参数调整。

2. **自适应专家混合**: 在推理时动态选择和组合专家模型权重，使用`policy.set_expert_weights`方法实现。

3. **实时错误矫正**: 通过实时检测和应用适当的专家模型，系统能够即时纠正模型的误判漏判情况。

## 局限性与未来改进

1. **改进专家选择**：目前的专家选择策略比较简单，可以引入更复杂的算法，例如基于语义相似度或错误模式分析的选择机制。

2. **增强SVD计算**：对于新模型，需要预先计算SVD分解，这一步可能耗时较长。可以探索增量SVD方法，或使用近似方法加速计算。

3. **支持更多模型**：当前框架已适配QwQ-32B模型，未来可扩展支持更多开源和闭源模型。

## 参考

- Transformer²: Self-adaptive LLMs (论文：[arxiv.org/abs/2501.06252](https://arxiv.org/abs/2501.06252))
- Sakana AI (博客：[sakana.ai/transformer-squared](https://sakana.ai/transformer-squared)) 