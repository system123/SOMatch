import math
import numpy as np

def custom_scheduler(factor=0.1, patience=10, min_lr=1e-5):
	return lambda i: 1e-3

def smith_1cycle(total_iter=1000, max_lr=1e-2, min_lr=1e-3, anneal_pct=10, anneal_div=100):
    anneal_iter = math.ceil(total_iter * anneal_pct/100.0)
    total_iter -= anneal_iter

    step_size = float(total_iter//2)

    piecewise_linear = lambda y0, y1, x, dx: (y1 - y0) * abs(float(x)/dx) + y0
    triangle = lambda y0, y1, x, dx: (y1 - y0) * (1 - abs(float(x)/dx - 1)) + y0

    def next(itr):
        if itr <= total_iter:
            lr = triangle(min_lr, max_lr, itr, step_size)
        elif itr - total_iter <= anneal_iter:
            lr = piecewise_linear(min_lr, min_lr/anneal_div, itr - total_iter, anneal_iter)
        else:
            lr = min_lr/anneal_div
        
        return lr

    return next

def lr_finder(min_lr=1e-6, max_lr=1, lr_multiplier=1.1, iter_div=1):
    max_itr = int(np.log(max_lr/min_lr)/np.log(lr_multiplier))

    def next(itr):
        i = min(itr//iter_div, max_itr)
        lr = min_lr * (lr_multiplier**i)
        return lr

    return next

def cyclical_lr(step_sz=2000, min_lr=0.001, max_lr=1, mode='triangular', scale_func=None, scale_md='cycles', gamma=1.):
    """implements a cyclical learning rate policy (CLR).
    Notes: the learning rate of optimizer should be 1

    Parameters:
    ----------
    mode : str, optional
        one of {triangular, triangular2, exp_range}.
    scale_md : str, optional
        {'cycles', 'iterations'}.
    gamma : float, optional
        constant in 'exp_range' scaling function: gamma**(cycle iterations)

    Examples:
    --------
    >>> # the learning rate of optimizer should be 1
    >>> optimizer = optim.SGD(model.parameters(), lr=1.)
    >>> step_size = 2*len(train_loader)
    >>> clr = cyclical_lr(step_size, min_lr=0.001, max_lr=0.005)
    >>> scheduler = lr_scheduler.LambdaLR(optimizer, [clr])
    >>> # some other operations
    >>> scheduler.step()
    >>> optimizer.step()
    """
    if scale_func == None:
        if mode == 'triangular':
            scale_fn = lambda x: 1.
            scale_mode = 'cycles'
        elif mode == 'triangular2':
            scale_fn = lambda x: 1 / (2.**(x - 1))
            scale_mode = 'cycles'
        elif mode == 'exp_range':
            scale_fn = lambda x: gamma**(x)
            scale_mode = 'iterations'
        else:
            raise ValueError(f'The {mode} is not valid value!')
    else:
        scale_fn = scale_func
        scale_mode = scale_md

    lr_lambda = lambda iters: min_lr + (max_lr - min_lr) * rel_val(iters, step_sz, scale_mode)

    def rel_val(iteration, stepsize, mode):
        cycle = math.floor(1 + iteration / (2 * stepsize))
        x = abs(iteration / stepsize - 2 * cycle + 1)
        if mode == 'cycles':
            return max(0, (1 - x)) * scale_fn(cycle)
        elif mode == 'iterations':
            return max(0, (1 - x)) * scale_fn(iteration)
        else:
            raise ValueError(f'The {scale_mode} is not valid value!')

    return lr_lambda

# if __name__=="__main__":
#     fn = smith_1cycle(total_iter=100, max_lr=1e-2, min_lr=1e-3, anneal_pct=50, anneal_div=100)
#     lrs = [fn(i) for i in range(0, 105)]
#     print(lrs)