import torch
import torch.nn as nn

class AddNoise(nn.Module):
    def __init__(self, mean=0, std=1.0) -> None:
        super().__init__()
        self.mean = mean
        self.std = std
    
    def forward(self, A):
        """_summary_

        Args:
            A (_type_): Tensor
        """
        #p_mask = torch.bernoulli(torch.ones(A.shape[0]) * self.p).to(A.device)
        normal_mask = torch.normal(self.mean, self.std, size=A.shape).to(A.device)
        return A + normal_mask