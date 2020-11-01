import torch

from torchvision import transforms
from torch.utils.data import Dataset
from functools import partial
from skimage.io import imread
from glob import glob
from skimage import exposure, img_as_float, util
from utils.augmentation import Augmentation, cropCenter, toGrayscale, cropCorner, cutout

import numpy as np
import pandas as pd
import os

AUG_PROBS = {
    "fliplr": 0.5,
    "flipud": 0.5,
    "scale": 0.1,
    "scale_px": (1.1, 1.1),
    "translate": 0,
    "translate_perc": (0.0, 0.0),
    "rotate": 0,
    "rotate_angle": (-5, 5),
    "contrast": 0.7,
    "dropout": 0.8
}

class CSVUADataset(Dataset):
    def __init__(self, config):
        super()

        self.domain = config.domain if isinstance(config.domain, str) else "opt_crop"
        self.balance = config.balance if isinstance(config.balance, bool) else False
        self.thresh_loss = config.thresh_loss if 'thresh_loss' in config else [0, 12]
        self.thresh_l2 = config.thresh_l2 if 'thresh_l2' in config else [1, 2.5]
        self.named = config.named if isinstance(config.named, bool) else False
        self.normed = config.normed if isinstance(config.normed, bool) else True

        self.base_dir = config.base_dir
        self.df = pd.read_csv(os.path.join(self.base_dir, config.csv))

        dataset_name = os.path.splitext(os.path.basename(config.csv))[0].rsplit("_", 1)[1]
        self.img_dir = os.path.join(self.base_dir, dataset_name)

        func = []

        if config.augment:
            # If it is true like then just use the default augmentation parameters - this keeps things backwards compatible
            if config.augment is True or len(config.augment) == 0:
                config.augment = AUG_PROBS.copy()

            self.augmentor = Augmentation(probs=config.augment)
        else:
            self.augmentor = None

        func.append(transforms.ToTensor())
        self.transforms = transforms.Compose(func)

        self._label_and_prune(self.thresh_l2[0], self.thresh_loss[0], self.thresh_l2[1], self.thresh_loss[1])

    def _label_and_prune(self, l2_pos=1, loss_pos=2.2, l2_neg=2.5, loss_neg=1.2):
        self.df["label"] = np.nan
        # Label positive samples
        self.df.loc[(self.df.l2 <= l2_pos) & (self.df.nlog_match_loss >= loss_pos), "label"] = 1
        self.df.loc[(self.df.l2 >= l2_neg) & (self.df.nlog_match_loss <= loss_neg), "label"] = 0

        # Remove all unlabeled points
        self.df.dropna(axis=0, inplace=True)

        if self.balance:
            limit = min( sum(self.df["label"] == 0), sum(self.df["label"] == 1) )
            limited_df = self.df.groupby("label").apply( lambda x: x.sample(n=limit) )
            limited_df.reset_index(drop=True, inplace=True)
            self.df = limited_df.sample(frac=1).reset_index(drop=True)
 
    def _get_filepath(self, row, img="sar"):
        return f"{self.img_dir}/['{row.city}']_['{row.wkt}']_{img}.npy"

    def _load_image(self, row, domain=None):
        data = np.load(self._get_filepath(row, img=domain))[0,]
        # Put in HxWxC format so data augmentation works
        return np.ascontiguousarray(data.transpose((1,2,0)))

    def normalize(self, img):
        return (img - img.min())/(img.ptp() + 1e-6)

    def _get_raw_triplet(self, row, crop=False):
        suffix = "_crop" if crop else ""
        opt = (self.transforms(self._load_image(row, f"opt{suffix}")).numpy().transpose((1,2,0))*255).astype(np.uint8)
        sar = (self.normalize(self.transforms(self._load_image(row, f"sar{suffix}")).numpy().transpose((1,2,0)))*255).astype(np.uint8)
        y = np.ones_like(sar) * row.label
        return sar, opt, y, {"sar": f"{row.city}_{row.name}_sar.png", "opt": f"{row.city}_{row.name}_opt.png", "label": row.label}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        x = self._load_image(row, self.domain)

        name = {"WKT": row.wkt, "city": row.city}

        if self.augmentor:
            self.augmentor.refresh_random_state()
            x = self.augmentor(x)

        if "sar" in self.domain and self.normed:
            x = self.normalize(x)

        if "hm" in self.domain and self.normed:
            x = self.normalize(x)

        x = self.transforms(x.copy()).float()

        y = np.array([row.label])

        if self.named:
            return x, y, name
        else:
            return x, y
   
