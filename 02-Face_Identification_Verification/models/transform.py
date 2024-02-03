import numpy as np
import cv2
import torch
import torch.nn as nn

class ToContourContrast(nn.Module):
    def __init__(self, threshold=250, maxval=255) -> None:
        super().__init__()
        self.threshold = threshold
        self.maxval = maxval
    
    def forward(self, image: torch.Tensor):
        x = image.cpu().detach().numpy()
        grayscale = cv2.cvtColor(np.uint8(x), cv2.COLOR_BAYER_BG2GRAY)
        _, binary = cv2.threshold(grayscale, self.threshold, self.maxval, cv2.THRESH_BINARY_INV)
        return torch.from_numpy(binary)
