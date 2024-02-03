import torch
import torch.nn as nn
import torchvision

def feature_mixing(in_channels: int, out_channels: int):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False), 
        nn.BatchNorm2d(out_channels), 
        nn.PReLU(), 
    )

def spatial_mixing(in_channels: int, out_channels: int, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False), 
        nn.BatchNorm2d(out_channels), 
        nn.PReLU(), 
    )

def bottleneck_channels(in_channels: int, out_channels: int, kernel_size=1):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False), 
        nn.BatchNorm2d(out_channels), 
    )

def shortcut(in_channels: int, out_channels: int, stride: int, do_identity: bool):
    if do_identity:
        return nn.Identity()
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False), 
            nn.BatchNorm2d(out_channels), 
        )

class BasicBlock(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 stride: int, 
                 stochastic_depth_prob: float=0.0):
        super().__init__()

        if stride == 1 and in_channels == out_channels:
            do_identity = True
        else:
            do_identity = False

        self.shortcut = shortcut(in_channels=in_channels, out_channels=out_channels, stride=stride, do_identity=do_identity)
        self.spatial_mixing = spatial_mixing(in_channels, out_channels, stride)
        self.bottleneck_channels = bottleneck_channels(out_channels, out_channels, 3)
        self.stochastic_depth = torchvision.ops.StochasticDepth(p=stochastic_depth_prob, mode="batch")

        self.activation = nn.PReLU()

    def forward(self, X):
        out = self.spatial_mixing(X)
        out = self.bottleneck_channels(out)
        out = self.stochastic_depth(out) + self.shortcut(X)

        return self.activation(out)

class BottleneckBlock(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 hidden_channels: int, 
                 stride: int, 
                 stochastic_depth_prob: float=0.0):
        super().__init__()

        if stride == 1 and in_channels == out_channels:
            do_identity = True
        else:
            do_identity = False

        self.shortcut = shortcut(in_channels=in_channels, out_channels=out_channels, stride=stride, do_identity=do_identity)
        self.feature_mixing = feature_mixing(in_channels, hidden_channels)
        self.spatial_mixing = spatial_mixing(hidden_channels, hidden_channels, stride)
        self.bottleneck_channels = bottleneck_channels(hidden_channels, out_channels)
        self.stochastic_depth = torchvision.ops.StochasticDepth(p=stochastic_depth_prob, mode="batch")

        self.activation = nn.PReLU()

    def forward(self, X):
        out = self.feature_mixing(X)
        out = self.spatial_mixing(out)
        out = self.bottleneck_channels(out)
        out = self.stochastic_depth(out) + self.shortcut(X)

        return self.activation(out)

class ResNet50(nn.Module):
    def __init__(self, stochastic_depth_prob=None, input_width=224, num_classes=7000):
        super().__init__()
        self.num_classes = num_classes

        self.stage_cfgs = [
            # hidden_channels, out_channels, # blocks, stride of first block
            # stride is 1 because there will be max pool with stride 2 before it
            [64, 256, 3, 1], 
            [128, 512, 4, 2], 
            [256, 1024, 6, 2], 
            [512, 2048, 3, 2],
        ]
        total_num_blocks = 0
        for stage in self.stage_cfgs:
            total_num_blocks += stage[2]

        if stochastic_depth_prob is None:
            # using simple linear decay rule
            self.stochastic_depth_prob = 0.5
        else:
            self.stochastic_depth_prob = stochastic_depth_prob

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3), 
            nn.BatchNorm2d(64), 
        )
        dim = (input_width - 7 + 2*3) // 2 + 1

        in_channels = 64
        out_channels = None

        layers = []
        # 3x3 max pool with stride 2
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        dim = (dim - 3 + 2) // 2 + 1
        curr_block = 1
        for curr_stage in self.stage_cfgs:
            hidden_channels, num_channels, num_blocks, stride = curr_stage

            for block_idx in range(num_blocks):
                sd_prob = (curr_block)/total_num_blocks*(self.stochastic_depth_prob)

                out_channels = num_channels
                layers.append(BottleneckBlock(
                    in_channels=in_channels, 
                    out_channels=out_channels, 
                    hidden_channels=hidden_channels, 
                    stride=stride if block_idx == 0 else 1,
                    stochastic_depth_prob=sd_prob
                ))
                in_channels = out_channels
                curr_block += 1
            dim = (dim - 3 + 2) // stride + 1
        
        layers.append(nn.AvgPool2d(dim))
        layers.append(nn.Flatten())
        self.layers = nn.Sequential(*layers)

        self.cls_layer = nn.Sequential(
            nn.Linear(in_features=out_channels, out_features=num_classes)
        )

        self._initialize_weights()
    
    def forward(self, X, return_feats=False):
        out = self.stem(X)
        feats = self.layers(out)
        out = self.cls_layer(feats)

        if return_feats:
            return feats
        else:
            return out
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

class ResNet(nn.Module):
    def __init__(self, stochastic_depth_prob=None, stage_cfgs=None, input_width=224, num_classes=7000):
        super().__init__()
        self.num_classes = num_classes

        self.stage_cfgs = stage_cfgs
        if self.stage_cfgs is None:
            # if stages are not specified, use ResNet34 as default
            self.stage_cfgs = [
                # hidden_channels, out_channels, # blocks, stride of first block
                # if hidden_channels is None, use the basic block
                # stride is 1 because there will be max pool with stride 2 before it
                [None, 64, 3, 1], 
                [None, 128, 4, 2], 
                [None, 256, 6, 2], 
                [None, 512, 3, 2], 
            ]
        total_num_blocks = 0
        for stage in self.stage_cfgs:
            total_num_blocks += stage[2]

        if stochastic_depth_prob is None:
            # using simple linear decay rule
            self.stochastic_depth_prob = 0.5
        else:
            self.stochastic_depth_prob = stochastic_depth_prob
            
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3), 
            nn.BatchNorm2d(64)
        )
        dim = (input_width - 7 + 2*3) // 2 + 1

        in_channels = 64
        out_channels = None

        layers = []
        # 3x3 max pool with stride 2
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        dim = (dim - 3 + 2) // 2 + 1
        curr_block = 1
        for curr_stage in self.stage_cfgs:
            hidden_channels, num_channels, num_blocks, stride = curr_stage

            for block_idx in range(num_blocks):
                sd_prob = (curr_block)/total_num_blocks*(self.stochastic_depth_prob)


                out_channels = num_channels
                if hidden_channels is None:
                    layers.append(BasicBlock(
                        in_channels=in_channels, 
                        out_channels=out_channels, 
                        stride=stride if block_idx == 0 else 1,
                        stochastic_depth_prob=sd_prob
                    ))
                else:
                    layers.append(BottleneckBlock(
                        in_channels=in_channels, 
                        out_channels=out_channels, 
                        hidden_channels=hidden_channels, 
                        stride=stride if block_idx == 0 else 1,
                        stochastic_depth_prob=sd_prob
                    ))
                in_channels = out_channels
            dim = (dim - 3 + 2) // stride + 1
        
        layers.append(nn.AvgPool2d(dim))
        layers.append(nn.Flatten())
        self.layers = nn.Sequential(*layers)

        self.cls_layer = nn.Sequential(
            nn.Linear(in_features=out_channels, out_features=num_classes)
        )

        self._initialize_weights()
    
    def forward(self, X, return_feats=False):
        out = self.stem(X)
        feats = self.layers(out)
        out = self.cls_layer(feats)

        if return_feats:
            return feats
        else:
            return out
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

class ResNetGray(nn.Module):
    def __init__(self, stochastic_depth_prob=None, stage_cfgs=None, input_width=224, num_classes=7000):
        super().__init__()
        self.num_classes = num_classes

        self.stage_cfgs = stage_cfgs
        if self.stage_cfgs is None:
            # if stages are not specified, use ResNet34 as default
            self.stage_cfgs = [
                # hidden_channels, out_channels, # blocks, stride of first block
                # if hidden_channels is None, use the basic block
                # stride is 1 because there will be max pool with stride 2 before it
                [None, 64, 3, 1], 
                [None, 128, 4, 2], 
                [None, 256, 6, 2], 
                [None, 512, 3, 2], 
            ]
        total_num_blocks = 0
        for stage in self.stage_cfgs:
            total_num_blocks += stage[2]

        if stochastic_depth_prob is None:
            # using simple linear decay rule
            self.stochastic_depth_prob = 0.5
        else:
            self.stochastic_depth_prob = stochastic_depth_prob
            
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3), 
            nn.BatchNorm2d(64)
        )
        dim = (input_width - 7 + 2*3) // 2 + 1

        in_channels = 64
        out_channels = None

        layers = []
        # 3x3 max pool with stride 2
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        dim = (dim - 3 + 2) // 2 + 1
        curr_block = 1
        for curr_stage in self.stage_cfgs:
            hidden_channels, num_channels, num_blocks, stride = curr_stage

            for block_idx in range(num_blocks):
                sd_prob = (curr_block)/total_num_blocks*(self.stochastic_depth_prob)


                out_channels = num_channels
                if hidden_channels is None:
                    layers.append(BasicBlock(
                        in_channels=in_channels, 
                        out_channels=out_channels, 
                        stride=stride if block_idx == 0 else 1,
                        stochastic_depth_prob=sd_prob
                    ))
                else:
                    layers.append(BottleneckBlock(
                        in_channels=in_channels, 
                        out_channels=out_channels, 
                        hidden_channels=hidden_channels, 
                        stride=stride if block_idx == 0 else 1,
                        stochastic_depth_prob=sd_prob
                    ))
                in_channels = out_channels
            dim = (dim - 3 + 2) // stride + 1
        
        layers.append(nn.AvgPool2d(dim))
        layers.append(nn.Flatten())
        self.layers = nn.Sequential(*layers)

        self.cls_layer = nn.Sequential(
            nn.Linear(in_features=out_channels, out_features=num_classes)
        )

        self._initialize_weights()
    
    def forward(self, X, return_feats=False):
        out = self.stem(X)
        feats = self.layers(out)
        out = self.cls_layer(feats)

        if return_feats:
            return feats
        else:
            return out
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
