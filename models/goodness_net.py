import math
import torch

import torch.nn.functional as F
import numpy as np

from torch import nn
from skimage.feature import corner_peaks, peak_local_max
from torchvision import models

class VGGBasedGoodnessNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.pooling = kwargs["pooling"] if "pooling" in kwargs else "max"

        self.leg_a = self._make_siamese_leg()

        self.conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.fc1 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.dropout = nn.Dropout2d(p=0.5)
        self.fc2 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)

    def fine_tune(self):
        for param in self.leg_a.parameters():
            param.requires_grad = False

    def _make_siamese_leg(self):
        def _make_vgg_block(layers=[], input_depth=1, depth=32, kernel_size=3, padding=True, max_pool=True):
            pad = kernel_size//2 if padding else 0

            layers.append( nn.Conv2d(input_depth, depth, kernel_size=kernel_size, padding=pad) )
            layers.append( nn.ReLU() )
            layers.append( nn.BatchNorm2d(depth) )
            layers.append( nn.Conv2d(depth, depth, kernel_size=kernel_size, padding=pad) )
            layers.append( nn.ReLU() )
            layers.append( nn.BatchNorm2d(depth) )
            if max_pool:
                layers.append( nn.MaxPool2d(2) )

            return layers

        layers = []
        layers = _make_vgg_block(layers=layers, input_depth=1, depth=32, kernel_size=3, padding=True, max_pool=True)
        layers = _make_vgg_block(layers=layers, input_depth=32, depth=64, kernel_size=3, padding=True, max_pool=True)
        layers = _make_vgg_block(layers=layers, input_depth=64, depth=128, kernel_size=3, padding=True, max_pool=True)
        layers = _make_vgg_block(layers=layers, input_depth=128, depth=128, kernel_size=3, padding=True, max_pool=False)

        layers.append( nn.Dropout2d(p=0.25) )

        return nn.Sequential(*layers)
  
    def spatial_softnms(self, heatmap, soft_local_max_size=3):
        b = heatmap.size(0)
        pad = soft_local_max_size//2

        heatmap = torch.sigmoid(heatmap)

        max_per_sample = torch.max(heatmap.view(b, -1), dim=1)[0]
        
        exp = torch.exp(heatmap / max_per_sample.view(b, 1, 1, 1))

        sum_exp = (
            soft_local_max_size ** 2 *
            F.avg_pool2d(
                F.pad(exp, [pad] * 4, mode='constant', value=1.),
                soft_local_max_size, stride=1
            )
        )
        local_max_score = exp / sum_exp

        return local_max_score

    # Fuse two goodness maps together
    def fuse_goodness(self, gdo, gds, reduce="max", pool=False):
        if pool:
            gdo_p = F.avg_pool2d(gdo, 4, stride=1, padding=2)
            gds_p = F.avg_pool2d(gds, 4, stride=1, padding=2)
            
            gdo_p = F.interpolate(gdo_p, size=gdo.shape[2:], align_corners=True, mode='bilinear')
            gds_p = F.interpolate(gds_p, size=gds.shape[2:], align_corners=True, mode='bilinear')
        
        if reduce == "mean":
            return (gdo_p + gds_p)/2
        elif reduce == "min":
            return torch.min(gdo_p, gds_p)
        elif reduce == "max":
            return torch.max(gdo_p, gds_p)

    def extract_good_points(self, heatmap, input_shape, exclude_border=64, nms_k=5, peak_k=5):
        hm = F.pad(heatmap, (2,2,2,2), "constant", heatmap.min())
        hm = F.interpolate(hm, size=input_shape, align_corners=None, mode='bilinear')
        
        # Get rid of the edge effects
        nms = self.spatial_softnms(hm, nms_k)[:1,:1,nms_k:-nms_k,nms_k:-nms_k]
        # Pad the edge back in
        nms = F.pad(nms, (nms_k, nms_k, nms_k, nms_k), "constant", nms.min())
        nms = (nms - nms.min())/(nms.max() - nms.min())
        nms = hm*nms
        
        nms_np = nms.cpu().numpy()[0,0,]
        
        pnts = peak_local_max(nms_np, footprint=np.full((peak_k,peak_k),True), exclude_border=exclude_border, 
                            indices=True, threshold_abs=0.9)
        fltrd = []
        hm_pad = np.zeros_like(nms_np)
        
        for p in pnts:
            if np.any(p < exclude_border) or np.any(p > (np.array(input_shape) - exclude_border)):
                continue
            fltrd.append(p)
            hm_pad[p[0],p[1]] = 1
            
        return np.array(fltrd), nms, hm_pad

    def forward(self, x1, pool=True):
        x1 = self.leg_a(x1)

        fts = self.conv1(x1)
        fts = self.bn1( F.relu(fts) )
        fts = self.conv2(fts)
        fts = self.bn2( F.relu(fts) )

        fts = self.fc1(fts)
        fts = self.dropout(fts)
        fts = F.relu(fts)
        fts = self.fc2(fts)

        if pool:
            if self.pooling == "max":
                fts = F.adaptive_max_pool2d(fts, 1)
            else: 
                fts = F.adaptive_avg_pool2d(fts, 1)     

            fts = fts.view(fts.size(0), -1)

        return fts