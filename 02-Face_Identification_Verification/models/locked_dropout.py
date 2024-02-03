import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class LockedDropout(nn.Module):
    def __init__(self, p=0.5) -> None:
        super().__init__()
        self.p = p
    
    def forward(self, x: torch.Tensor):
        if not self.training:
            return x
        if not self.p:
            return x
        is_packed = isinstance(x, torch.nn.utils.rnn.PackedSequence)
        if is_packed:
            xx, x_lengths = pad_packed_sequence(x, batch_first=True)
            xx = xx.permute((1, 0, 2))
        else:
            xx = x.clone().permute((1, 0, 2))
        mask = xx.new_empty((1, xx.shape[1], xx.shape[2]), requires_grad=False)
        mask = mask.bernoulli_(1 - self.p)
        mask = mask.div_(1 - self.p) # normalize
        mask = mask.expand_as(xx)
        torch.cuda.empty_cache()
        if is_packed:
            return pack_padded_sequence((xx * mask).permute(1, 0, 2), x_lengths.cpu(), batch_first=True, enforce_sorted=False)
        else:
            return (xx * mask).permute(1, 0, 2)
