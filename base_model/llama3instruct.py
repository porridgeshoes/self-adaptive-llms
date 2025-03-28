import os

from .base import BaseModel
"""
类的作用：
模型标识: 定义了 QwQ-32B 模型在系统中的标识和相关属性
参数文件管理: 指定了 QwQ-32B 模型的参数分解文件存储路径
模型接口: 为框架提供了统一的接口，使系统能够获取模型ID、名称和参数文件路径
在 Transformer² 框架中，这个类充当了"适配器"的角色，使框架能够识别并使用 QwQ-32B 模型。通过这个类，系统可以知道如何加载模型、在哪里存储和读取分解后的参数文件，以及如何在日志和输出中引用这个模型。
"""

class Llama3Instruct8B(BaseModel):
    def __init__(self):
        self.model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        self.dec_param_file_n = "llama3_decomposed_params.pt"

    def get_model_id(self):
        return self.model_id

    def get_model_name(self):
        return self.model_id.split("/")[1]

    def get_param_file(self, param_folder_path=""):
        return os.path.join(param_folder_path, self.dec_param_file_n)
