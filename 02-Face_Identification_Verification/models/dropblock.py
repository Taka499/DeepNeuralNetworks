import torch
import torch.nn as nn
import torch.nn.functional as F

class DropBlock(nn.Module):
    """_summary_
    Implemented following https://arxiv.org/pdf/1810.12890.pdf
    """
    def __init__(self, block_size: int, keep_prob: float) -> None:
        super().__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob
    
    def forward(self, A) -> torch.Tensor:
        """
        Args:
            A (torch.Tensor): (N, C, H, W)
        """
        # Expecting input tensor to be the shape of (N, C, H, W)
        assert A.dim() == 4, "Expecting input tensor to be the shape of (N, C, H, W)"

        if not self.training or self.keep_prob == 1.0:
            return A
        gamma = self._calc_gamma(A.shape)
        #mask_sampled = (torch.rand(A.shape) < gamma).float()
        mask_sampled = torch.bernoulli(torch.ones_like(A) * gamma)
        mask_sampled = mask_sampled.to(A.device)
        mask = 1 - F.max_pool2d(input=mask_sampled, 
                                kernel_size=self.block_size, 
                                stride=1, 
                                padding=self.block_size//2)
        Z = A * mask * mask.numel() / mask.sum()

        assert Z.shape == A.shape

        return Z

    def _calc_gamma(self, shape) -> float:
        return ((1 - self.keep_prob) / self.block_size**2) \
            * (shape[2] / (shape[2] - self.block_size + 1)) \
            * (shape[3] / (shape[3] - self.block_size + 1))