import torch
import torch.nn as nn

def get_soft_mask(n, fraction):
    indices = torch.linspace(0, n - 1, n, dtype=torch.bfloat16) + 1
    scaled_indices = indices.to(fraction.device) - fraction * n
    result = torch.clamp(scaled_indices, 0, 1)
    return 1.0 - result

class Policy(nn.Module):
    def __init__(self, base_params, gpu, init_val, max_mult=1, **kwargs):
        # Create learnable parameters.
        super().__init__()
        self.learnable_params = {}
        self.num_params = 0
        self.max_mult = max_mult
        for k, v in base_params.items():
            # each param initialized with small gaussian noise
            if "mlp" in k:
                # 这个就是需要核心要学习的参数，对应着每个奇异值的缩放系数。
                self.learnable_params[k] = torch.nn.Parameter(
                    data=(
                        torch.randn(
                            min(v.shape),
                            device=gpu,
                            dtype=torch.bfloat16,
                        )
                        * 0.01
                        + init_val
                    ),
                    requires_grad=True,
                )
                # .numel() 计算这个张量包含多少个元素（即单个参数的数量）
                
                self.num_params += self.learnable_params[k].numel()
        print(f"#params={self.num_params}") # qwen7b被蒸馏模型一共#params=301056
        self.learnable_params_list = list(self.learnable_params.values())
        self.trainable_params = self.learnable_params_list
        self.learnable_params_module_list = nn.ParameterList(self.learnable_params_list)

    def get_learnable_params(self, detach=False):
        return self.learnable_params

    def set_trainable_params_values(self, new_values):
        with torch.no_grad():
            for p, v in zip(self.trainable_params, new_values):
                p.data.copy_(v)

    def get_mask(self, p):
        '''
        限制掩码(mask)的取值范围：在使用SVD分解调整大型语言模型参数时，max_mult控制掩码值的上限
        控制参数适应的强度：较大的max_mult允许更剧烈的参数修改，较小的值则使修改更保守
        '''
        return torch.sigmoid(p).to(torch.bfloat16) * self.max_mult

    def record_state(self, metrics_to_log):
        pass
