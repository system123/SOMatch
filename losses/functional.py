import torch
from torch import nn
import torch.nn.functional as F

EPS = 1e-4

def mse_loss_weighted(input, target, reduction="mean", pos_weight=None):
    negs = input[target == 0]
    pos = input[target == 1]

    loss = torch.zeros_like(target, device=target.device)
    loss[target == 1] = pos_weight*F.mse_loss(pos, target[target == 1], reduction="none")
    loss[target == 0] = F.mse_loss(negs, target[target == 0], reduction="none")

    if reduction == "mean":
        loss = loss.mean()
        # loss /= (pos_weight*target[target == 1].size().numel() + target[target == 0].size().numel())
    elif reduction == "sum":
        loss = loss.sum()

    return loss
