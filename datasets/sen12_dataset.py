from torchvision import transforms
from torch.utils.data import Dataset
import torch
import operator

import torch.nn.functional as F

from skimage.io import imread
from skimage import exposure, img_as_float
from utils.augmentation import Augmentation, cropCenter, toGrayscale
from utils.visualisation_helpers import plot_side_by_side
from utils.helpers import load_file_list

from torch.utils.data import SubsetRandomSampler
from samplers.round_robin_batch_sampler import RoundRobinBatchSampler
from itertools import chain, repeat
from utils.basic_cache import BasicCache
from tools.dfc_sen12ms_dataset import DFCSEN12MSDataset, S1Bands, S2Bands


import numpy as np
import pandas as pd
import os

AUG_PROBS = {
    "fliplr": 0.4,
    "flipud": 0,
    "scale": 0,
    "scale_px": (1.0, 1.0),
    "translate": 0,
    "translate_perc": (0.0, 0.0),
    "rotate": 0,
    "rotate_angle": (-5, 5)
}

def null_norm(x):
    return x

class SEN12Dataset(Dataset):
    def __init__(self, config):
        super(SEN12Dataset, self).__init__()

        self.crop_size = config.crop if isinstance(config.crop, (int, float)) else None
        self.named = config.named if isinstance(config.named, bool) else False
        self.hist_norm = config.hist_norm if isinstance(config.hist_norm, bool) else True

        func = []

        if config.augment:
            # If it is true like then just use the default augmentation parameters - this keeps things backwards compatible
            if config.augment is True or len(config.augment) == 0:
                config.augment = AUG_PROBS.copy()

            self.augmentor = Augmentation(probs=config.augment)
            func.append(self.augmentor)
        else:
            self.augmentor = None

        func.append(transforms.Lambda(lambda img: self._preprocess(img, self.crop_size)))
        func.append(transforms.ToTensor())

        self.transforms = transforms.Compose(func)

        if "sar" in config.normalize:
            self.sar_norm = transforms.Normalize(mean=[config.normalize.sar[0]], std=[config.normalize.sar[1]])
        else:
            self.sar_norm = null_norm

        if "opt" in config.normalize:
            self.opt_norm = transforms.Normalize(mean=[config.normalize.opt[0]], std=[config.normalize.opt[1]])
        else:
            self.opt_norm = null_norm

        self.cache_dir = config.cache_dir if isinstance(config.cache_dir, str) else None
        self.cache_size = config.cache_size if isinstance(config.cache_size, (int, float)) else 0

        if self.cache_dir is not None:
            self.cache = BasicCache(self.cache_dir, size=self.cache_size, scheme="fill", clear=False, overwrite=False)
        else:
            self.cache = None
        
        self.sar = load_file_list(config.base_dir, config.data_path_supervised[0])
        self.opt = load_file_list(config.base_dir, config.data_path_supervised[1])
        self.labels = np.loadtxt(config.data_path_labels)

        self.limit_supervised = config.limit_supervised if isinstance(config.limit_supervised, int) else -1
        if self.limit_supervised > 0 and self.limit_supervised < len(self.sar[0]):
            idxs = range(self.limit_supervised)
            self.sar[0] = [self.sar[0][i] for i in idxs]
            self.opt[0] = [self.opt[0][i] for i in idxs]
            self.labels = self.labels[idxs]

        self.noise = config.noise if isinstance(config.noise, bool) else False

        self._get_scenes(seasons=["winter"])

    def _get_scenes(self, seasons=["winter"]):
        scenes = []
        for s, o, l in zip(self.sar, self.opt, self.labels):
            if l == 0:
                continue

            scenes.append({
                "sar_path": s, 
                "opt_path": o,
                "scene": os.path.splitext(os.path.basename(s))[0].rsplit("_", 1)[0]
                })
        
        self.df = pd.DataFrame.from_dict(scenes)
        self.df = self.df.sort_values("scene").reset_index()
 
    def _preprocess(self, x, crop=None, stack=False):
        x = toGrayscale(x)

        if crop:
            x = cropCenter(x, (crop, crop))
            
        return(x)

    def __len__(self):
        # For every patch there are actually 128*128 
        return np.sum(df.groupby("scene").sar.nunique().values**2)

    # TODO: Add hard negative mining as a third dataset option.
    def _load_and_label(self, index):

        img_sar = img_as_float(imread(self.sar[0][index], as_gray=True, plugin="pil"))
        img_opt = img_as_float(imread(self.opt[0][index], as_gray=True, plugin="pil"))

        # Rescale the image to be between 0 and 1 - otherwise normalisation won't work later
        if self.hist_norm:
            img_sar = exposure.rescale_intensity(img_sar, out_range=(0, 1), in_range='dtype')
            img_opt = exposure.rescale_intensity(img_opt, out_range=(0, 1), in_range='dtype')
        
        if len(img_sar.shape) < 3:
            img_sar = np.expand_dims(img_sar, axis=2)

        if len(img_opt.shape) < 3:
            img_opt = np.expand_dims(img_opt, axis=2)

        name_sar = os.path.basename(self.sar[0][index])
        name_opt = os.path.basename(self.opt[0][index])

        y = self.labels[index].astype(np.float)

        return img_sar, img_opt, y, {"name_a": name_sar, "name_b": name_opt}

    def __getitem__(self, index):
        # Fix the random state so we get the same transformations
        if self.augmentor:
            self.augmentor.refresh_random_state()

        img_sar, img_opt, y, names = self._load_and_label(index)

        img_sar = self.sar_norm( self.transforms(img_sar).float() )
        img_opt = self.opt_norm( self.transforms(img_opt).float() )

        if self.noise:
            img_sar = img_sar + 0.01*torch.randn_like(img_sar) + img_sar.mean()
            img_opt = img_opt + 0.01*torch.randn_like(img_opt) + img_opt.mean()

        if self.named:
            return (img_sar, img_opt), y, names
        else:
            return (img_sar, img_opt), y

    # def get_batch_sampler(self, batch_size):    
    #     super_idxs = range(len(self.opt[0]), len(self.opt[0]) + len(self.opt[1]))
    #     unsuper_idxs = range(0, len(self.opt[0]))

    #     super_idxs = list(chain.from_iterable( repeat( tuple(super_idxs), self.n) ))

    #     self.unsupervised_sampler = SubsetRandomSampler( unsuper_idxs )
    #     self.supervised_sampler = SubsetRandomSampler( super_idxs )
    #     self.batch_sampler = RoundRobinBatchSampler([self.unsupervised_sampler, self.supervised_sampler], batch_size=batch_size)

    #     return self.batch_sampler

def cropCenterT(img, bounding, shift=(0,0,0,0)):
    imshape = [x+y*2 for x,y in zip(img.shape, shift)]
    bounding = list(bounding)
    start = tuple(map(lambda a, da: a//2-da//2, imshape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]

def get_gt_heatmap(shift_x=0, shift_y=0, w=64, h=64, device="cpu", full_size=False):
    x = int(w//2 + shift_x)
    y = int(h//2 + shift_y)
    hm = np.zeros((h, w))
    hm[y, x] = 1
    gt_hm = torch.Tensor(hm[np.newaxis, np.newaxis, :, :]).to(device)
    gt_hm = cropCenterT(gt_hm, (1,1,h,w))
    if not full_size:
        gt_hm = F.max_pool2d(gt_hm, 3, stride=2)
        gt_hm = F.interpolate(gt_hm, size=(h//2+1, w//2+1), align_corners=False, mode='bilinear')
        gt_hm.div_(gt_hm.max())
    return gt_hm.numpy()[0,]

class SEN12DatasetHeatmap(Dataset):
    SPLITS = {
        "train": {"summer": (0, 0.5), "spring": (0, 1), "autumn": (0, 1)},
        "val": {"summer": (0.5, 1)},
        "test": {"winter": (0, 1)}
    }


    def __init__(self, config):
        super().__init__()

        self.crop_size = config.crop if isinstance(config.crop, (int, float)) else None
        self.crop_size_a = config.crop_a if isinstance(config.crop_a, (int, float)) else None
        self.crop_size_b = config.crop_b if isinstance(config.crop_b, (int, float)) else None
        self.named = config.named if isinstance(config.named, bool) else False
        self.return_all = config.return_all if isinstance(config.return_all, bool) else False
        self.stretch_contrast = config.stretch_contrast if isinstance(config.stretch_contrast, bool) else False
        self.full_size = config.full_size if isinstance(config.full_size, bool) else False

        self.cache_dir = config.cache_dir if isinstance(config.cache_dir, str) else None
        self.cache_size = config.cache_size if isinstance(config.cache_size, (int, float)) else 0

        if self.cache_dir is not None:
            self.cache = BasicCache(self.cache_dir, size=self.cache_size, scheme="fill", clear=False, overwrite=False)
        else:
            self.cache = None

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

        self.base_dir = config.base_dir
        # Only read the corresponding patches
        self.files = np.unique(load_file_list(None, os.path.join(config.base_dir, config.filelist)))
        self.dataset = self._make_dataset()

    def _make_dataset(self):
        dataset = []
        
        for f in self.files:
            f = os.path.basename(f)
            sar = os.path.join(self.base_dir, "sar", f)
            opt = os.path.join(self.base_dir, "opt", f)
            _, season, scene, patch = os.path.splitext(f)[0].split("_")

            dataset.append( (sar, opt, season, scene, patch) )

        return dataset
        

    #     self.sen12 = DFCSEN12MSDataset(config.base_dir)
    #     self.split = config.split if isinstance(config.split, str) else "train"
    #     self.dataset = self._get_split(self.split)

    # def _get_split(self, split):
    #     dataset = []

    #     for season, ratio in self.SPLITS[split].items():
    #         scene_ids = self.sen12.get_scene_ids(season)
    #         start, end = int(ratio[0]*len(scene_ids)), int(ratio[1]*len(scene_ids))

    #         scene_ids = scene_ids[start:end]
    #         for scene_id in scene_ids:
    #             patch_ids = get_patch_ids(season, scene_id)
    #             items = [(season, scene_id, patch_id) for patch_id in patch_ids]
    #             dataset.extend(items)

    #     return dataset

    def get_gt_heatmap(self, shift_x=0, shift_y=0, w=64, h=64, sigma=1):
        x = int(w//2 + shift_x)
        y = int(h//2 + shift_y)

        hm = np.zeros((h, w))
        hm[y, x] = 1

        return hm[np.newaxis, :, :]

    def __len__(self):
        return len(self.dataset)

    def _cache_key(self, index):
        _, _, season, scene, patch = self.dataset[index]
        return f"{season}{scene}{patch}"

    def _try_cache(self, index):
        if self.cache is not None:
            # Try get data for the point
            data = self.cache[self._cache_key(index)]

            if data is not None:
                # Hack as we have a dict in a 0-d numpy array
                data["INFO"] = data["INFO"].item()

                return data

    def _load_and_label(self, index):
        data = self._try_cache(index)

        if not data:
            data = {"SAR": None, "OPT": None, "INFO": None}

            sar, opt, season, scene, patch = self.dataset[index]
            # data["SAR"], data["OPT"], _ = self.sen12.get_s1_s2_pair(season, scene, patch, s1_bands=S1Bands.VV, s2_bands=S2Bands.RGB)

            data["SAR"] = img_as_float(imread(sar, as_gray=True, plugin="pil"))
            data["OPT"] = img_as_float(imread(opt, as_gray=True, plugin="pil"))
            data["INFO"] = {"season": season, "scene": scene, "patch": patch}

            if self.cache is not None:
                self.cache[self._cache_key(index)] = data

        return data["SAR"], data["OPT"], data["INFO"]

    def __getitem__(self, index):
        if self.augmentor:
            self.augmentor.refresh_random_state()

        img_sar, img_opt, img_info = self._load_and_label(index)

        if self.augmentor is not None:
            img_sar = self.augmentor(img_sar)
            img_opt = self.augmentor(img_opt)

        assert self.crop_size_a <= img_sar.shape[1], "The input image is too small to crop"
        assert self.crop_size_b <= img_opt.shape[1], "The input image is too small to crop"

        if self.full_size:
            fa_sz = self.crop_size_a
            fb_sz = self.crop_size_b
        else:
            fa_sz = (self.crop_size_a - 6)//2 - 1
            fb_sz = (self.crop_size_b - 6)//2 - 1

        hm_size = np.abs(fa_sz - fb_sz) + 1

        # We already in the center, so we can only shift by half of the radius (thus / 4)
        max_shift = min(fa_sz//4, fb_sz//4)
        shift_x = (2*np.random.randint(2) - 1)*(np.random.randint(max_shift) + 1)
        shift_y = (2*np.random.randint(2) - 1)*(np.random.randint(max_shift) + 1)

        if self.crop_size_a > self.crop_size_b:
            if img_sar.shape[1] - self.crop_size_a > 0:
                # Also ensure we don't shift the keypoint out of the search region
                max_shift = min((fa_sz - fb_sz)//4, max_shift)
                max_shift_x = min((fa_sz - fb_sz)//4 - shift_x//2, max_shift)
                max_shift_y = min((fa_sz - fb_sz)//4 - shift_y//2, max_shift)
                shift_x_s = (2*np.random.randint(2) - 1)*(np.random.randint(max_shift))
                shift_y_s = (2*np.random.randint(2) - 1)*(np.random.randint(max_shift))
            else:
                shift_x_s = 0
                shift_y_s = 0

            search_img = np.ascontiguousarray(cropCenter(img_sar, (self.crop_size_a, self.crop_size_a), (shift_x_s, shift_y_s)))
            template_img = np.ascontiguousarray(cropCenter(img_opt, (self.crop_size_a, self.crop_size_a), (shift_x_s, shift_y_s)))
            search_hard = np.ascontiguousarray(cropCenter(img_sar, (self.crop_size_b, self.crop_size_b), (shift_x, shift_y)))
            template_hard = np.ascontiguousarray(cropCenter(img_opt, (self.crop_size_b, self.crop_size_b), (shift_x, shift_y)))

            if self.stretch_contrast:
                search_img = (search_img - search_img.min())/(search_img.ptp())

            search_img = self.transforms(search_img).float()
            template_img = self.transforms(template_img).float()
            search_hard = self.transforms(search_hard).float()
            template_hard = self.transforms(template_hard).float()
        else:
            if img_opt.shape[1] - self.crop_size_b > 0:
                # Also ensure we don't shift the keypoint out of the search region
                max_shift_x = min((fb_sz - fa_sz)//4 - shift_x//2, max_shift)
                max_shift_y = min((fb_sz - fa_sz)//4 - shift_y//2, max_shift)
                shift_x_s = (2*np.random.randint(2) - 1)*(np.random.randint(max_shift_x))
                shift_y_s = (2*np.random.randint(2) - 1)*(np.random.randint(max_shift_y))                
            else:
                shift_x_s = 0
                shift_y_s = 0

            search_img = cropCenter(img_opt, (self.crop_size_b, self.crop_size_b), (shift_x_s, shift_y_s))
            template_img = cropCenter(img_sar, (self.crop_size_b, self.crop_size_b), (shift_x_s, shift_y_s))
            search_hard = cropCenter(img_opt, (self.crop_size_a, self.crop_size_a), (shift_x, shift_y))
            template_hard = cropCenter(img_sar, (self.crop_size_a, self.crop_size_a), (shift_x, shift_y))

            if self.stretch_contrast:
                template_img = (template_img - template_img.min())/(template_img.ptp())

            search_img =  self.transforms(search_img).float()
            template_img = self.transforms(template_img).float()
            search_hard = self.transforms(search_hard).float()
            template_hard = self.transforms(template_hard).float()

        # print(f"a: {shift_x}  b: {shift_y}  hm: {shift_x_s} ca: {shift_y_s}")
        # This is dependant on the Model!!!!!!!!!! We should move this there
        # print("WARNING THIS IS DEPENDANT ON THE MODEL")
        shift_x = shift_x - shift_x_s
        shift_y = shift_y - shift_y_s
        
        if self.full_size:
            scale = 1
        else:
            scale = ((1 - 6/search_img.shape[1])*(3/2))

        # y_p = self.get_gt_heatmap(w=hm_size, h=hm_size, sigma=None)
        y_hn = self.get_gt_heatmap(shift_x=shift_x//scale, shift_y=shift_y//scale, w=hm_size, h=hm_size, sigma=None)
        # y_hn = get_gt_heatmap(shift_x, shift_y, hm_size, hm_size, full_size=False)
        # print(f"HEATMAP: {y_hn.shape}  {shift_x}  {shift_y}")

        # y_p = y_p/y_p.max()
        y_hn = y_hn/y_hn.max()

        # y = np.array([0, shift_x, shift_y], dtype=np.float32)

        if self.return_all:
            imgs = (search_img, template_img, template_hard, search_hard)
        else:
            imgs = (search_img, template_img, template_hard)

        if self.named:
            cx = int(hm_size//2 + shift_x//scale)
            cy = int(hm_size//2 + shift_y//scale)
            img_info.update({
                    "p_match": (cx, cy),
                    "shift": (shift_x, shift_y)
                    })
  
            return imgs, y_hn, img_info
        else:
            return imgs, y_hn