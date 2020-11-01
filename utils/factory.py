from .helpers import *
from torch.utils.data import DataLoader, SubsetRandomSampler
import ignite
import torch.nn.functional as F
import torch.optim.lr_scheduler

def create_dataset(config):
    dataset = get_module('./datasets', config.type)
    return dataset(config) or None

def build_model(config):
    ident = list(config.keys())[0]
    model = get_module('./models', ident)
    return model(**config[ident].toDict()) or None

def get_optimizer(params, config):
    ident = list(config.keys())[0]

    optim = None
    try:
        optim = get_module('./optimizers', ident)
    except:
        pass
    
    if optim is None:
        optim = str_to_class('torch.optim', ident)
        
    return optim(params, **config[ident].toDict()) or None

def get_data_loader(dset, config, indices=None):
    sampler = None

    if indices is not None:
        sampler = SubsetRandomSampler(dset)
        config.shuffle = False

    if getattr(dset, "get_batch_sampler", None):
        batch_sampler = dset.get_batch_sampler(config.batch_size)
        return DataLoader(dset, num_workers=config.workers, batch_sampler=batch_sampler)

    return DataLoader(dset, batch_size=config.batch_size, shuffle=config.shuffle, num_workers=config.workers, sampler=sampler)

def get_trainer(model, optimizer, loss_fn, exp_logger, config):
    trainer = get_module('./trainer', config.trainer)
    return trainer(model, optimizer, loss_fn, exp_logger, config) or None

def get_experiment_logger(log_dir, config):
    ident = list(config.keys())[0]
    logger = get_module('./logger', ident)
    config = config[ident].toDict()
    config["log_dir"] = log_dir
    return logger(**config) or None

def get_metric(name):
    metric = get_if_implemented(ignite.metrics, name)

    if metric is None:
        try:
            metric = get_module('./metrics', name)
        except:
            pass

    if metric is None:
        loss_fcn = get_loss(name)
        assert loss_fcn, "No loss function {} was found for use as a metric".format(name)
        metric = ignite.metrics.Loss(loss_fcn)
    else:
        metric = metric()

    return metric or None

def get_loss(loss_fn):
    loss = get_if_implemented(F, loss_fn)
    if loss is None:
        loss = get_function('losses.functional', loss_fn)
    return loss

def get_lr_scheduler(optimizer, config):
    name = list(config.keys())[0]
    args = config[name].toDict().copy()
    scheme = args["scheme"]
    args = copy_and_delete(args, 'scheme')

    lr_scheduler = get_if_implemented(torch.optim.lr_scheduler, name)

    if lr_scheduler is None:
        try:
            lr_scheduler = get_module('./schedulers', name)
        except:
            pass

    if lr_scheduler is None:
        fcn = get_function('schedulers.functional', name)
        assert fcn, "No functional implementation of {} was found".format(name)
        fcn_wrapper = fcn(**args)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [fcn_wrapper])
    else:
        lr_scheduler = lr_scheduler(optimizer, **args)

    return lr_scheduler, scheme
