import torch
import torch.nn as nn
import torchvision
from .dropblock import DropBlock
import math

def depth_wise(in_channels: int, out_channels: int, kernel_size=7, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=out_channels, bias=False), 
        nn.BatchNorm2d(out_channels), 
    )
    # Use batchnorm for this assignment instead of layer norm
    # Since using batchnorm, following convolution is not using Linear

def point_wise1(in_channels: int, out_channels: int):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0), 
        nn.GELU()
    )

def point_wise2(in_channels: int, out_channels: int):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0), 
    )

class Block(nn.Module):
    def __init__(self,
                 num_channels: int,
                 stride: int, 
                 expand_ratio: int,
                 stochastic_depth_prob: float=0.0):
        super().__init__()
        
        hidden_dim = num_channels * expand_ratio

        self.conv1 = depth_wise(in_channels=num_channels, out_channels=num_channels, kernel_size=7, stride=stride)
        self.conv2 = point_wise1(in_channels=num_channels, out_channels=hidden_dim)
        self.conv3 = point_wise2(in_channels=hidden_dim, out_channels=num_channels)
        self.stochastic_depth = torchvision.ops.StochasticDepth(p=stochastic_depth_prob, mode="batch")


    def forward(self, X):
        input = X
        out = self.conv1(X)
        out = self.conv2(out)
        out = self.conv3(out)

        return self.stochastic_depth(out) + input

class ConvNeXt(nn.Module):
    """
    Implemented following https://arxiv.org/pdf/2201.03545.pdf
    with stochastic depth https://arxiv.org/pdf/1603.09382.pdf
    """
    def __init__(self, stochastic_depth_prob=None, stage_cfgs=None, input_width=224, num_classes=7000):
        super().__init__()
        self.num_classes = num_classes

        """
            model configurations
        """
        self.stage_cfgs = stage_cfgs
        if self.stage_cfgs is None:
            # if stages are not specified, use ConvNeXt-T as default
            self.stage_cfgs = [
                # out_channels, # blocks
                [ 96, 3], 
                [192, 3], 
                [384, 9], 
                [768, 3], 
            ]
            total_blocks = 18
        else:
            total_blocks = 0
            for stage in self.stage_cfgs:
                total_blocks += stage[1]

        if stochastic_depth_prob is None:
            # not using linear decay rule. Instead specify for each stage. 
            self.stochastic_depth_prob = [0, 0, 0.1, 0.2]
        else:
            self.stochastic_depth_prob = stochastic_depth_prob
        
        """
            building the model architecture
        """
        # constructing the downsample layers
        stem = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=4, stride=4, padding=0), 
            nn.BatchNorm2d(96), 
        )
        self.downsample_layers = []
        self.downsample_layers.append(stem)
        for i in range(len(self.stage_cfgs)-1):
            self.downsample_layers.append(nn.Sequential(
                nn.BatchNorm2d(self.stage_cfgs[i][0]), 
                nn.Conv2d(in_channels=self.stage_cfgs[i][0], 
                          out_channels=self.stage_cfgs[i+1][0], 
                          kernel_size=2, 
                          stride=2, 
                          padding=0), 
            ))

        # calculate the dimension of input
        dim = (input_width - 4) // 4 + 1 # downsample by stem
        for _ in range(len(self.stage_cfgs)-1):
            dim = (dim - 2) // 2 + 1

        # constructing the residual blocks
        layers = []
        for curr_stage, downsample, sd_prob in zip(self.stage_cfgs, self.downsample_layers, self.stochastic_depth_prob):
            num_channels, num_blocks = curr_stage
            layers.append(downsample) # downsample layer

            for block_idx in range(num_blocks):
                layers.append(Block(
                    num_channels=num_channels, 
                    stride=1, 
                    expand_ratio=4, 
                    stochastic_depth_prob=sd_prob
                ))
        self.layers = nn.Sequential(*layers)

        # Average Pooling layer and flattening the features
        self.final_global = nn.Sequential(
            nn.AvgPool2d(dim), 
            nn.BatchNorm2d(self.stage_cfgs[-1][0]), 
            nn.Flatten(), 
        )

        # Final cls layer
        self.cls_layer = nn.Sequential( 
            nn.Dropout(0.1), 
            nn.Linear(in_features=self.stage_cfgs[-1][0], 
                      out_features=num_classes)
        )

        self._initialize_weights()
    
    def forward(self, X, return_feats=False):
        out = self.layers(X)
        feats = self.final_global(out)
        out = self.cls_layer(feats)

        if return_feats:
            return feats, out
        else:
            return out
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
