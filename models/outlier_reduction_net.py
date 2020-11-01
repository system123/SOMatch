import torch
import torch.nn.functional as F

from torch import nn
from torchvision import models
import math

class ORN(nn.Module):
    def __init__(self, classes=1, padding=False):
        super().__init__()
        self.classes = classes

        if padding:
            pad = [3, 2, 2, 1]
        else:
            pad = [0, 0, 0, 0]

        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=1, padding=pad[0]),       #0      
            nn.InstanceNorm2d(32, eps=1e-05, momentum=0.05, affine=True),  #1
            nn.ReLU(),                                                  #2
            nn.MaxPool2d(3, 2),                                         #3b
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=pad[1]),      #4    
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.05, affine=True),  #5
            nn.ReLU(),                                                  #6
            nn.MaxPool2d(3, 2),                                         #7c
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=pad[2]),     #8     
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True), #9
            nn.ReLU(),                                                  #10
            nn.MaxPool2d(3, 2),                                         #11d
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=pad[3]),    #12     
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True), #13
            nn.ReLU()                                                   #14
        )

        self.head = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, self.classes, kernel_size=1, stride=1, padding=0)
        )
        
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.weight.shape[1])
                nn.init.uniform_(m.weight, -init_range, init_range)

    def rescale_heatmap(self, heatmap, shape=(129, 129), pad=True, norm=False):
        if norm:
            heatmap = (heatmap - heatmap.min())/(heatmap.max() - heatmap.min())
        
        scaled_heatmap = F.interpolate(heatmap, size=shape, align_corners=True, mode='bilinear')

        if pad:
            pad_x = shape[1]//2
            pad_y = shape[0]//2
            scaled_heatmap = F.pad(scaled_heatmap, (pad_y, pad_y-1, pad_x, pad_x-1), "constant", heatmap.min())
        
        return scaled_heatmap

    def forward(self, x, pool=True):
        y = self.stem(x)
        y = self.head(y)

        if pool:
            y = F.adaptive_avg_pool2d(y, 1)
            y = y.view(y.shape[0], -1)

        return y
