import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class EmbeddingWithDropout(nn.Module):
    def __init__(self, embed: nn.Embedding, dropout=0.2) -> None:
        super(EmbeddingWithDropout, self).__init__()
        self.embed = embed
        self.dropout = dropout
    
    def forward(self, x: torch.Tensor):
        if self.dropout:
            mask = self.embed.weight.data.new().resize_((self.embed.weight.size(0), 1)).bernoulli_(1 - self.dropout).expand_as(self.embed.weight) / (1 - self.dropout)
            mask = torch.autograd.Variable(mask)
            masked_embed_weight = mask * self.embed.weight
        else:
            masked_embed_weight = self.embed.weight

        padding_idx = self.embed.padding_idx
        if padding_idx is None:
            padding_idx = -1
        X = F.embedding(x, masked_embed_weight, padding_idx=padding_idx, 
                        max_norm=self.embed.max_norm, norm_type=self.embed.norm_type, 
                        scale_grad_by_freq=self.embed.scale_grad_by_freq, 
                        sparse=self.embed.sparse)
        return X
