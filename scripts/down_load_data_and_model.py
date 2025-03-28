
from datasets import load_dataset
import os

# os.environ["HF_HUB_OFFLINE"] = "1"
# 注意datasets的版本要一致。

# 保存路径
cache_dir="/data1/yammyjiang/new_space/light_things/LLM/self-adaptive-llms/download"
train_data = load_dataset("gsm8k", "main", split="train", cache_dir=cache_dir)

# os.environ["HF_HOME"] = "/data1/yammyjiang/new_space/light_things/LLM/self-adaptive-llms/download"
# train_data = load_dataset("gsm8k", "main", split="train")


