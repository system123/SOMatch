import importlib
import pkgutil
import inspect
import os
import shutil
import re

def validate_config(config):
    # assert config.device in ["cpu", "cuda"], "Invalid compute device was specified. Only 'cpu' and 'cuda' are supported."
    return True

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def freeze_net(model, filter=None, freeze=True):
    for p in model.parameters():
        p.requires_grad = not freeze

def load_file_list(base_dir, path):
        img_list = []
        for line in open(path, 'r'):
            if base_dir:
                img_list.append(os.path.join(base_dir, line.strip()))
            else:
                img_list.append(line.strip())
        return(img_list)

def empty_folder(path):
    if os.path.exists(path):
        for the_file in os.listdir(path):
            file_path = os.path.join(path, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                pass

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

def extract_numbers(x):
    r = re.compile('(\d+(?:\.\d+)?)')
    l = r.split(x)
    return [float(y) for y in l if is_float(y)]

def copy_and_delete(d, key):
    copy = d.copy()
    del copy[key]
    return(copy)

def get_modules(path):
    modules = {}

    for loader, name, is_pkg in pkgutil.walk_packages(path):
        module = loader.find_module(name).load_module(name)
        for name, value in inspect.getmembers(module):
            # Only import classes we defined
            if inspect.isclass(value) is False or value.__module__ is not module.__name__:
                continue

            modules[name] = value

    return modules

def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [ param_group['lr'] ]
    return lr

def get_function(module, fcn):
    try:
        fn = str_to_class(module, fcn)
    except:
        fn = None
    return fn

def get_module(path, name):
    modules = get_modules([path])
    assert name in modules.keys(), "Could not find module {}".format(name)
    return modules[name]

def __classname_to_modulename(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def str_to_class(module_name, class_name):
    try:
        module_ = importlib.import_module(module_name)
        try:
            class_ = getattr(module_, class_name)
        except AttributeError:
            logging.error('Class does not exist')
    except ImportError:
        logging.error('Module does not exist')
    return class_ or None

def check_if_implemented(obj, fcn):
    op = getattr(obj, fcn, None)
    return callable(op)

def get_if_implemented(obj, fcn):
    op = getattr(obj, fcn, None)
    if not callable(op):
        op = None
    return op

def accuracy(output, target):
    """Computes the accuracy for multiple binary predictions"""
    pred = output >= 0.5
    truth = target >= 0.5
    acc = pred.eq(truth).sum() / target.numel()
    return acc


class BinaryClassificationMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.acc = 0
        self.pre = 0
        self.rec = 0
        self.f1 = 0

    def update(self, output, target):
        pred = output >= 0.5
        truth = target >= 0.5
        self.tp += pred.mul(truth).sum(0).float()
        self.tn += (1 - pred).mul(1 - truth).sum(0).float()
        self.fp += pred.mul(1 - truth).sum(0).float()
        self.fn += (1 - pred).mul(truth).sum(0).float()
        self.acc = (self.tp + self.tn).sum() / (self.tp + self.tn + self.fp + self.fn).sum()
        self.pre = self.tp / (self.tp + self.fp)
        self.rec = self.tp / (self.tp + self.fn)
        self.f1 = (2.0 * self.tp) / (2.0 * self.tp + self.fp + self.fn)
        # self.avg_pre = torch.nanmean(self.pre)
        # self.avg_rec = nanmean(self.rec)
        # self.avg_f1 = nanmean(self.f1)