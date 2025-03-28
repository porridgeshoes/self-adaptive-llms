import os

from .base import BaseModel


class QwQ32B(BaseModel):
    def __init__(self):
        # self.model_id = "01-ai/Yi-VL-34B" # 使用QwQ-32B的HuggingFace ID
        self.model_id = "/data1/nanzhiwang/python_work/open-r1/models/DeepSeek-R1-Distill-Qwen-7B"
        self.dec_param_file_n = "qwq32b_decomposed_params.pt"

    def get_model_id(self):
        return self.model_id

    def get_model_name(self):
        return self.model_id.split("/")[1]

    def get_param_file(self, param_folder_path=""):
        return os.path.join(param_folder_path, self.dec_param_file_n) 