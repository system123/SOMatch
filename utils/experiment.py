from dotmap import DotMap
from glob import glob
from numbers import Number
import os
import json

from utils.helpers import extract_numbers

def number_ordering(x):
    n = extract_numbers(x)
    return n[-1] if len(n) > 0 else 0

class Experiment(DotMap):
    def __init__(self, name, desc="", result_dir="./results", data={}):
        super(Experiment, self).__init__(data)

        self.name = name
        self.desc = desc
        self.result_dir = result_dir
        self.result_path = os.path.join(self.result_dir, self.name)

    def save(self):
        os.makedirs(self.result_path, exist_ok=True)
        with open(os.path.join(self.result_path, "config.json"), "w") as f:
            f.write(json.dumps(self.toDict(), indent=4, sort_keys=False))

    def load(self):
        try:
            with open(os.path.join(self.result_path, "config.json")) as f:
                data = json.load(f)
            super(Experiment, self).__init__(data)
            return self
        except:
            return None

    def exists(self):
        return os.path.exists(self.result_path)

    # def get_checkpoint_path(self):
    #     model_path = None

    #     if 'checkpoint' in self and len(self.checkpoint) > 0:
    #         _, ext = os.path.splitext(self.checkpoint)

    #         if ext == ".pth":
    #             model_path = self.checkpoint if os.path.exists(self.checkpoint) else None
    #             path = os.path.join(self.result_path, self.checkpoint) if model_path is None else model_path
    #             model_path = path if os.path.exists(path) else None
    #         elif self.checkpoint == "best":
    #             cpts = glob(os.path.join(self.result_path, 'best_checkpoint_model*.pth'))
    #             cpts = sorted(cpts, key=number_ordering)
    #             if len(cpts) > 0:
    #                 model_path = cpts[-1]
    #         elif self.checkpoint.isdigit():
    #             path = os.path.join(self.result_path, 'checkpoint_model_{}.pth'.format(self.resume_from))
    #             model_path = path if os.path.exists(path) else None

    #     optim_path = model_path.replace('model','optim') if model_path else None
    #     return (model_path, optim_path)

    def get_checkpoints(self, path=None, tag="best"):
        checkpoints = {}
        last_epoch = 0

        path = self.result_path if path is None else path
        filelist = glob(os.path.join(path, "*{}*.pth".format(tag)))

        # Clean the filenames so we can use split() to extract parts of them
        filelist = [os.path.splitext(fname)[0] for fname in filelist]
        epochs = set([p for fname in filelist for p in os.path.basename(fname).split('_') if p.isdigit()])

        # Find the largest epoch 
        for e in epochs:
            if e.isdigit() and int(e) > last_epoch:
                last_epoch = int(e)

        # Ensure we only select the chosen epoch
        filelist = [fname for fname in filelist if str(last_epoch) in fname]

        for fname in filelist:
            parts = os.path.basename(fname).split('_')
            name = parts[len(parts)//2] # Middle element is always the name
            checkpoints[name] = fname + ".pth" # Add back the file extension

        return checkpoints, last_epoch

    @staticmethod
    def load_from_path(path, overloads=None, suffix=None):
        with open(path) as f:
            data = json.load(f)

        if overloads:
            data.update(overloads)

        data['name'] = data["name"] if suffix is None else "{}_{}".format(data['name'], suffix)

        return Experiment(data['name'], data['desc'], data['result_dir'], data)


    @staticmethod
    def load_by_name(name, conf_dir="./config"):
        exp = Experiment(name, result_dir=conf_dir).load()
        return(exp)
