import math
import torch
from torch.optim import Optimizer

class Smith1CycleLR:
    def __init__(self, optimizer, total_iter=1000, max_lr=1e-2, min_lr=1e-3, anneal_pct=10, anneal_lr_div=100, max_momentum=0.95, min_momentum=0.85, momentum_name='momentum'):

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(type(optimizer).__name__))

        self.optimizer = optimizer

        self.momentum_name = momentum_name

        self.lr_fcn = self.smith_1cycle(total_iter, max_lr, min_lr, anneal_pct, anneal_lr_div)
        self.mom_fcn = self.smith_1cycle(total_iter, min_momentum, max_momentum, anneal_pct, 1)

        self.last_epoch = -1

        self.step(self.last_epoch + 1)

    def smith_1cycle(self, total_iter=1000, max_lr=1e-2, min_lr=1e-3, anneal_pct=10, anneal_div=100):
        anneal_iter = math.ceil(total_iter * anneal_pct/100.0)
        total_iter -= anneal_iter

        step_size = float(total_iter//2)

        piecewise_linear = lambda y0, y1, x, dx: (y1 - y0) * float(x)/dx + y0
        triangle = lambda y0, y1, x, dx: (y1 - y0) * (1 - abs(float(x)/dx - 1)) + y0

        def next(itr):
            if itr <= total_iter:
                lr = triangle(min_lr, max_lr, itr, step_size)
            else:
                lr = piecewise_linear(min_lr, min_lr/anneal_div, itr - total_iter, anneal_iter)
            
            return lr

        return next

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def get_lr(self):
        return self.lr_fcn(self.last_epoch)

    def get_momentum(self):
        return self.mom_fcn(self.last_epoch)

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            
        self.last_epoch = epoch
        lr = self.get_lr()
        momentum = self.get_momentum()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

            if self.momentum_name in param_group:
                param_group[self.momentum_name] = momentum