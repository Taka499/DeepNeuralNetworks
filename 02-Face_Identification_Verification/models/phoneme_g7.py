import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class Permute(nn.Module):
    def __init__(self, order: tuple) -> None:
        super().__init__()
        self.order = order
    
    def forward(self, X):
        return torch.permute(X, self.order)


class PhonemeNetworkMedium(nn.Module):

    def __init__(self, architecture: dict): # You can add any extra arguments as you wish
        super(PhonemeNetworkMedium, self).__init__()

        # construct embedding layers
        in_feature = 15
        
        self.embedding_param = architecture["embedding"]
        embedding_layers = []
        for i in range(len(self.embedding_param)):
            out_channels, kernel_size, stride, dropout = self.embedding_param[i]

            if stride >= 1:
                embedding_layers.append(nn.Sequential(
                    nn.Conv1d(
                        in_channels=in_feature,
                        out_channels=out_channels,
                        kernel_size=kernel_size, 
                        stride=stride,
                        padding=kernel_size//2,
                        bias=False
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.GELU(),
                ))
            else:
                embedding_layers.append(nn.Sequential(
                    nn.ConvTranspose1d(
                        in_channels=in_feature,
                        out_channels=out_channels,
                        kernel_size=kernel_size, 
                        stride=int(1 / stride),
                        padding=kernel_size//2,
                        bias=False
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.GELU(),
                ))
            if dropout:
                embedding_layers.append(nn.Dropout(dropout))
            in_feature = out_channels

        self.embedding = nn.Sequential(*embedding_layers)
        
        # construct LSTM layers
        self.lstm = nn.LSTM(**architecture["lstm"])
        # Use nn.LSTM() Make sure that you give in the proper arguments as given in https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html

        # construct classification layers
        D = 2 if architecture["lstm"]["bidirectional"] else 1
        num_in = D * architecture["lstm"]["hidden_size"]
        classification_layers = []
        for num_out in architecture["classification"]:
            classification_layers.append(nn.Sequential(
                nn.Linear(in_features=num_in, out_features=num_out), 
                # Permute((0, 2, 1)), 
                # nn.BatchNorm1d(num_out), 
                # Permute((0, 2, 1)),
            ))
            classification_layers.append(nn.GELU())
            if architecture["cls_dropout"] > 0:
                classification_layers.append(nn.Dropout(p=architecture["cls_dropout"]))
            num_in = num_out
        if architecture["cls_dropout"] > 0:
            self.classification = nn.Sequential(*classification_layers[:-2])
        else:
            self.classification = nn.Sequential(*classification_layers[:-1])
        self.logsoftmax = nn.LogSoftmax(dim=2)

        #self._initialize_weights()

    def forward(self, x, lengths_x):

        out = self.embedding(torch.permute(x, (0, 2, 1)))

        new_lengths_x = lengths_x
        for _, k, s, _ in self.embedding_param:
            if s >= 1:
                new_lengths_x = torch.div((new_lengths_x - k + 2 * (k//2)), s, rounding_mode='floor')
            else:
                new_lengths_x = (new_lengths_x - 1) * int(1/s)  - 2 * (k//2) + 1

        # x is returned from the dataloader. So it is assumed to be padded with the help of the collate_fn
        packed_input = pack_padded_sequence(torch.permute(out, (0, 2, 1)), new_lengths_x, batch_first=True, enforce_sorted=False)
        torch.cuda.empty_cache()

        out, (out2, out3) = self.lstm(packed_input)
        # As you may see from the LSTM docs, LSTM returns 3 vectors. Which one do you need to pass to the next function?
        out, lengths = pad_packed_sequence(out, batch_first=True)
        torch.cuda.empty_cache()
        
        out = self.classification(out)
        out = self.logsoftmax(out)
        out = torch.permute(out, (1, 0, 2)) # permute for CTC Loss input requirement (T, B, C)

        return out, new_lengths_x # TODO: Need to return 2 variables
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
            if isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.kaiming_normal_(param)
