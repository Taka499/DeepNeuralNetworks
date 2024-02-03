import torch
import torch.nn as nn

class RandomApply(nn.Module):
    def __init__(self, transforms, p=0.5) -> None:
        super().__init__()
        self.transforms = transforms
        self.p = p

    def forward(self, X):
        if self.p < torch.rand(1):
            return X
        out = X
        for t in self.transforms:
            out = t(out)
        return out