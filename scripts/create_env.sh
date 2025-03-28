# !/bin/bash

# 基于nanzhi 1340镜像继续构建
conda create --name t2 --clone torch
conda activate t2
cd /data1/yammyjiang/new_space/light_things/LLM/self-adaptive-llms
pip install --upgrade pip

# 删除requirements.txt里的evalplus，以及torch、torchvision、transformers、datasets、vllm再安装。
pip install -r requirements.txt

# 安装本地下载的 evalplus
pip install /data1/yammyjiang/new_space/heavy_things/self-adaptive-llms/download/evalplus-0.3.0.dev16.zip

# 安装项目自创的fishfarm
cd evaluation/fishfarm
pip install -e .
