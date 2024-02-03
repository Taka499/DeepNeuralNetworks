import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from networks.LAS_pre_train_2 import Encoder

class ListenerPreTrainer(nn.Module):
    def __init__(self, input_dim, encoder_hidden_dim, key_value_size=128, dropout=None) -> None:
        super().__init__()
        
        self.encoder = Encoder(input_dim, encoder_hidden_dim, key_value_size=key_value_size, dropout=dropout)
        self.dilateCNN1 = nn.ConvTranspose1d(in_channels=encoder_hidden_dim*2, out_channels=encoder_hidden_dim*2, kernel_size=2, stride=2)
        self.dilateCNN2 = nn.ConvTranspose1d(in_channels=encoder_hidden_dim*2, out_channels=encoder_hidden_dim, kernel_size=2, stride=2)
        self.dilateCNN3 = nn.ConvTranspose1d(in_channels=encoder_hidden_dim, out_channels=input_dim, kernel_size=2, stride=2)
    
    def forward(self, x, x_len):
        out, out_lengths = self.encoder(x, x_len, pre_train=True)
        # out size is now (B, T, C)
        out = out.permute(0, 2, 1)
        out = self.dilateCNN1(out)
        out = self.dilateCNN2(out)
        out = self.dilateCNN3(out)
        out = out.permute(0, 2, 1)
        
        return out