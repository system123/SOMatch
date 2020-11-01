import torch

import torch.nn.functional as F
import torchvision.models as models
import numpy as np

from torch import nn
from torchvision import models

from utils.modules import *

class CorrelationFeatureNet(nn.Module):
    def __init__(self, column_depth=1024, normalize=False, return_hypercol=False, no_relu=False, attention=False, return_attn=True, attn_act="sigmoid"):
        super().__init__()
        self.normalize = normalize
        self.column_depth = column_depth
        self.return_hypercol = return_hypercol
        self.no_relu = no_relu
        self.attention = attention
        self.return_attn = return_attn

        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=1, padding=0),       #0      
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.05, affine=True),  #1
            nn.ReLU(),                                                  #2
            nn.MaxPool2d(3, 2),                                         #3
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=0),      #4    
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.05, affine=True),  #5
            nn.ReLU(),                                                  #6
            nn.MaxPool2d(3, 2),                                         #7
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=0),     #8     
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True), #9
            nn.ReLU(),                                                  #10
            nn.MaxPool2d(3, 2),                                         #11
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0),    #12     
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True), #13
            nn.ReLU(),                                                  #14
            nn.MaxPool2d(3, 2)                                          #15
        )
        self.build_from = [3, 7, 11, 15]

        squish = [nn.Conv2d(352, 256, kernel_size=1, stride=1, padding=0),            
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0),            
            nn.ReLU(),
            nn.Conv2d(512, column_depth, kernel_size=1, stride=1, padding=0)]
            
        if not self.no_relu:
            squish.append(nn.ReLU())

        self.squash = nn.Sequential(*squish)

        if self.attention:
            # self.ch_attn = ChannelAttention(self.column_depth)
            self.sp_attn = SpatialAttention(kernel_size=3, activation=attn_act)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.weight.shape[1])
                nn.init.uniform_(m.weight, -init_range, init_range)

    def _build_hypercolumn(self, ft_maps):
        # Get the output size we want
        size = ft_maps[0].shape[2:]

        stack = []

        for ft in ft_maps:
            stack.append( F.interpolate(ft, size=size, mode='bilinear', align_corners=True) )

        # Stack the tensors in the channel dimension
        return torch.cat(stack, dim=1)

    # Convolve two tensors together
    def correlation_map(self, tensor_a, tensor_b):
        b, c, h, w = tensor_a.shape
        _, _, h1, w1 = tensor_b.shape
        h1 = h - h1 + 1
        w1 = w - w1 + 1
        tensor_a = tensor_a.view(1, b*c, h, w)
        heatmap = F.conv2d(tensor_a, tensor_b, groups=b).view(b, 1, h1, w1)

        return heatmap

    def forward(self, x):
        y = x

        outputs = []
        for i, block in enumerate(self.stem):
            y = block(y)
            if i in self.build_from:
                outputs.append(y)

        hypercol = self._build_hypercolumn(outputs)

        hypercol_reduced = self.squash(hypercol)

        if self.attention:
            # hypercol_attn = self.ch_attn(hypercol_reduced) * hypercol_reduced
            spatial_attn = self.sp_attn(hypercol_reduced)
            hypercol_reduced = spatial_attn * hypercol_reduced

        if self.normalize:
            hypercol_reduced = F.normalize(hypercol_reduced, p=2, dim=1)

        if self.return_hypercol:
            return hypercol_reduced, hypercol
        elif self.attention and self.return_attn:
            return hypercol_reduced, spatial_attn
        else:
            return hypercol_reduced
