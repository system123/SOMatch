from torchvision import transforms
from torch.utils.data import Dataset
from functools import partial
import torch
import operator
from skimage.io import imread
from glob import glob
from skimage import exposure, img_as_float, util
from utils.augmentation import Augmentation, cropCenter, toGrayscale, cropCorner, cutout
from utils.helpers import load_file_list
from tools.urban_atlas_helpers import UrbanAtlas
from torch.utils.data import SubsetRandomSampler
from samplers.round_robin_batch_sampler import RoundRobinBatchSampler
from itertools import chain, repeat, accumulate
from utils.basic_cache import BasicCache

import numpy as np
import os

AUG_PROBS = {
    "fliplr": 0.5,
    "flipud": 0.5,
    "scale": 0,
    "scale_px": (1.0, 1.0),
    "translate": 0,
    "translate_perc": (0.0, 0.0),
    "rotate": 0,
    "rotate_angle": (-5, 5)
}

def null_norm(x):
    return x

class UrbanAtlasDataset(Dataset):
    def __init__(self, config):
        super()

        self.crop_size = config.crop if isinstance(config.crop, (int, float)) else None
        # Non-symetric cropping
        self.crop_size_a = config.crop_a if isinstance(config.crop_a, (int, float)) else None
        self.crop_size_b = config.crop_b if isinstance(config.crop_b, (int, float)) else None
        self.named = config.named if isinstance(config.named, bool) else False
        self.stretch_contrast = config.stretch_contrast if isinstance(config.stretch_contrast, bool) else False
        self.return_all = config.return_all if isinstance(config.return_all, bool) else False
        # self.toDb = config.toDb if isinstance(config.toDb, bool) else False
        self.noise = config.noise if isinstance(config.noise, bool) else False
        self.zca = config.zca if isinstance(config.zca, bool) else False
        self.single_domain = config.single_domain if isinstance(config.single_domain, bool) else False
        self.full_size = config.full_size if isinstance(config.full_size, bool) else False
        self.shift_range = config.shift_range if isinstance(config.shift_range, (list, tuple)) else [5, 15]

        # If cache is specified then we will save the patches to the local disk somewhere to prevent needing to reload them all the time
        self.cache_dir = config.cache_dir if isinstance(config.cache_dir, str) else None
        self.cache_size = config.cache_size if isinstance(config.cache_size, (int, float)) else 0

        if self.cache_dir is not None:
            self.cache = BasicCache(self.cache_dir, size=self.cache_size, scheme="fill", clear=False, overwrite=False)
        else:
            self.cache = None

        # Load the Urban Atlas Dataset and windows
        optdir = os.path.join(config.base_dir, "PRISM")
        sardir = os.path.join(config.base_dir, "TSX")
        self.ua = UrbanAtlas(optdir, sardir, cities=config.cities, crs="EPSG:3035", load_geometry=True, workers=config.workers)
        self.windows = self.ua.get_windows(self.ua.geometry, reduce=True, load_existing=True)
        self.lut = list(accumulate([len(df) for df in self.ua.geometry.values()]))

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

        if "sar" in config.normalize:
            self.sar_norm = transforms.Normalize(mean=[config.normalize.sar[0]], std=[config.normalize.sar[1]])
        else:
            self.sar_norm = null_norm

        if "opt" in config.normalize:
            self.opt_norm = transforms.Normalize(mean=[config.normalize.opt[0]], std=[config.normalize.opt[1]])
        else:
            self.opt_norm = null_norm

        self.perc_supervised = config.perc_supervised/100 if isinstance(config.perc_supervised, int) else 1
        self.ratios = self._randomly_assign_supervised(self.perc_supervised)

        print(f"UrbanAtlas Dataset created with {list(self.windows.keys())} cities covering {self.lut} points")
 
    def _randomly_assign_supervised(self, percent):
        ratios = {}
        for city in self.windows.keys():
            self.ua.geometry[city]["super"] = np.random.rand(len(self.ua.geometry[city])) <= percent

            # Calculate the ratio between the supervised and unsupervised datasets
            n_s = self.ua.geometry[city]["super"].sum()
            if n_s < len(self.ua.geometry[city]):
                n = max(n_s, len(self.ua.geometry[city]) - n_s)/min(n_s, len(self.ua.geometry[city]) - n_s)
            else:
                n = 1

            # Set which points are 'supervised' points
            self.windows[city] = self.windows[city].merge(self.ua.geometry[city].loc[:, ["wkt", "super"]], left_on=["wkt"], right_on=["wkt"], how="left")
            ratios[city] = n

        return ratios

    def __len__(self):
        # Count the unique number of points, not windows - as a single point can have multiple windows
        return self.lut[-1]

    # Return a point from a dataframe
    def _index_to_point(self, idx):
        for i, cnt in enumerate(self.lut):
            if idx < cnt:
                # Move the offset to the start of the dataframe
                offset = self.lut[i - 1] if i > 0 else 0
                # Get the city the index is in and then query the point in that city
                return list(self.ua.geometry.keys())[i], list(self.ua.geometry.values())[i].iloc[idx - offset]

    def _get_indexes(self):
        super_idxs = []
        unsuper_idxs = []

        for i, city in enumerate(self.ua.geometry.keys()):
            offset = 0 if i == 0 else self.lut[i-1]
            # Get the indices of the supervised points
            super_idxs.extend( np.argwhere(self.ua.geometry[city].super).flatten() + offset )
            unsuper_idxs.extend( np.argwhere(self.ua.geometry[city].super == False).flatten() + offset )

        return super_idxs, unsuper_idxs

    def _normalize_scale(self, x, in_range=(0, 255)):
        return (x.clip(*in_range) - in_range[0])/(in_range[1] - in_range[0])

    def _toDb(self, x, scale=True):
        x = 10*np.ma.log10(x.astype(np.float32))
        # 10*np.log10(2**16) == 48
        if scale:
            mu = x.mean()
            std = x.std()
            # x = self._normalize_scale(x, (mu-3*std, mu+3*std))
            # Approximate 3 sigma scaling
            x = self._normalize_scale(x, (10, 30))

        return x #if not scale else self._normalize_scale(x, in_range=(1, 48))

    def _get_group_patches(self, grp, opt_transform=None, sar_transform=None):
        N = len(grp)
        imgs = {"OPT": [], "SAR": []}
        idxs = {"OPT": [], "SAR": []}

        for idx, item in grp.iterrows():
            transform = sar_transform if item.sensor == "SAR" else opt_transform
            im = self.ua.get_patch(item, masked=True, transform=transform)

            if im is not None:
                im = im[0,:,:].data
                imgs[item.sensor].append(im)
                idxs[item.sensor].append(idx)
                
        return imgs, idxs

    def _try_cache(self, index):
        if self.cache is not None:
            df_city, point = self._index_to_point(index)
            # Try get data for the point
            data = self.cache[point.wkt]

            if data is not None:
                # Overwrite the values for info as we not actually sure about them any more
                group = self.windows[df_city].groupby("wkt").get_group(point.wkt)
                # Hack as we have a dict in a 0-d numpy array
                data["INFO"] = data["INFO"].item()
                data["INFO"]["supervised"] = group.super.values[0]
                data["INFO"]["SAR"] = None
                data["INFO"]["OPT"] = None
                data["INFO"]["WKT"] = point.wkt

                return data

    def _load_and_label(self, index, drop=False):
        data = self._try_cache(index)

        # If we can't load the required patch, at least make sure we cache the fallback under the original key so we don't loop on this again
        cache_key = self._index_to_point(index)[1].wkt

        if not data:
            data = {"SAR": None, "OPT": None, "Y": None, "INFO": None}

        # HACK: This might cause long loading times if there are too many invalid patches.
        while (data["OPT"] is None) and (data["SAR"] is None):
            # Load a point from the UrbanAtlas dataset, so we can extract the windows
            df_city, point = self._index_to_point(index)

            # Get the corresponding windows to the point we extracted
            group = self.windows[df_city].groupby("wkt").get_group(point.wkt)

            imgs, idxs = self._get_group_patches(group, opt_transform=partial(self._normalize_scale, in_range=(0,255)), sar_transform=self._toDb)

            if len(idxs["OPT"]) > 0 and len(idxs["SAR"]) > 0:
                data["OPT"] = imgs["OPT"][0]
                data["SAR"] = imgs["SAR"][0]
                data["INFO"] = {"SAR": idxs["SAR"][0], "OPT": idxs["OPT"][0], "city": df_city, "supervised": group.super.values[0], "WKT": cache_key}
                # Urban Atlas is always corresponding on a patch level. Use sub patches for negatives
                data["Y"] = np.ones(1)

                # Try and cache the data point
                if self.cache is not None:
                    self.cache[cache_key] = data
            else:
                if drop:
                    break

                # HACK: As the dataset isn't fully cleaned yet we chose another sample if the requested one is an invalid patch
                index += 1

        if len(data["SAR"].shape) < 3:
            data["SAR"] = np.expand_dims(data["SAR"], axis=2)

        if len(data["OPT"].shape) < 3:
            data["OPT"] = np.expand_dims(data["OPT"], axis=2)

        return data["SAR"], data["OPT"], data["Y"], data["INFO"]

    def __getitem__(self, index):
        raise NotImplementedError("You cannot call the parent class, please use a specific child class implementation")
  
    def get_batch_sampler(self, batch_size):
        super_idxs, unsuper_idxs = self._get_indexes()
        # Get the average resampling factor (not as accurate but good enough)
        n = np.round( np.mean( [c for c in self.ratios.values()] ), 0).astype(np.int)

        # Expand the smaller dataset
        if len(super_idxs) < len(unsuper_idxs):
            super_idxs = list(chain.from_iterable( repeat( tuple(super_idxs), n) ))
        elif len(super_idxs) > len(unsuper_idxs):
            unsuper_idxs = list(chain.from_iterable( repeat( tuple(unsuper_idxs), n) ))

        samplers = []
        if len(unsuper_idxs) > 0:
            samplers.append( SubsetRandomSampler( unsuper_idxs ) )
        
        if len(super_idxs) > 0:
            samplers.append( SubsetRandomSampler( super_idxs ) )
  
        return RoundRobinBatchSampler(samplers, batch_size=batch_size)

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

# Returns Triplets of patches - but in the form needed for training a siamese network
# i.e Either a positive SAR patch and both pos and neg optical or visa versa
# The other triplet wrapper does triplets of SAR and triplet of Optical
class UrbanAtlasDatasetSiameseTriplet(UrbanAtlasDataset):
    def multivariate_gaussian(self, pos, mu, Sigma):
        n = mu.shape[0]
        Sigma_det = np.linalg.det(Sigma)
        Sigma_inv = np.linalg.inv(Sigma)
        N = np.sqrt((2*np.pi)**n * Sigma_det)
        # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
        # way across all the input variables.
        fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)
        return np.exp(-fac / 2) / N

    def get_gt_heatmap(self, shift_x=0, shift_y=0, w=64, h=64, sigma=1):
        x = int(w//2 + shift_x)
        y = int(h//2 + shift_y)

        if sigma is None:
            hm = np.zeros((h, w))
            hm[y, x] = 1
        else:
            X = np.linspace(0, w-1, w)
            Y = np.linspace(0, h-1, h)
            X, Y = np.meshgrid(X, Y)

            mu = np.array([x, y])
            Sigma = np.array([[sigma , 0], [0,  sigma]])
            pos = np.empty(X.shape + (2,))
            pos[:, :, 0] = X
            pos[:, :, 1] = Y
            hm = self.multivariate_gaussian(pos, mu, Sigma)

        return hm[np.newaxis, :, :]

    def __getitem__(self, index):
        # Fix the random state so we get the same transformations
        if self.augmentor:
            self.augmentor.refresh_random_state()

        img_sar, img_opt, y, img_info = self._load_and_label(index)

        # Augment the images if we want to
        if self.augmentor is not None:
            img_sar = self.augmentor(img_sar)
            img_opt = self.augmentor(img_opt)

        if self.single_domain:
            img_sar = img_opt
            self.sar_norm = self.opt_norm

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
        max_shift = min(min(fa_sz//4, fb_sz//4), (hm_size-1)//4)
        shift_x = (2*np.random.randint(2) - 1)*(np.random.randint(max_shift) + 1)
        shift_y = (2*np.random.randint(2) - 1)*(np.random.randint(max_shift) + 1)

        if self.crop_size_a > self.crop_size_b:
            if img_sar.shape[1] - self.crop_size_a > 0:
                # Also ensure we don't shift the keypoint out of the search region
                max_shift = min((fa_sz - fb_sz)//4, max_shift)
                max_shift_x = min((fa_sz - fb_sz)//4 - shift_x//2, max_shift)
                max_shift_y = min((fa_sz - fb_sz)//4 - shift_y//2, max_shift)
                shift_x_s = (2*np.random.randint(2) - 1)*(np.random.randint(max_shift_x))
                shift_y_s = (2*np.random.randint(2) - 1)*(np.random.randint(max_shift_y))
            else:
                shift_x_s = 0
                shift_y_s = 0

            search_img = np.ascontiguousarray(cropCenter(img_sar, (self.crop_size_a, self.crop_size_a), (shift_x_s, shift_y_s)))
            template_img = np.ascontiguousarray(cropCenter(img_opt, (self.crop_size_a, self.crop_size_a), (shift_x_s, shift_y_s)))
            search_hard = np.ascontiguousarray(cropCenter(img_sar, (self.crop_size_b, self.crop_size_b), (shift_x, shift_y)))
            template_hard = np.ascontiguousarray(cropCenter(img_opt, (self.crop_size_b, self.crop_size_b), (shift_x, shift_y)))

            if self.stretch_contrast:
                search_img = (search_img - search_img.min())/(search_img.ptp())

            search_img = self.sar_norm( self.transforms(search_img).float() )
            template_img = self.opt_norm( self.transforms(template_img).float() )
            search_hard = self.sar_norm( self.transforms(search_hard).float() )
            template_hard = self.opt_norm( self.transforms(template_hard).float() )
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

            search_img = self.opt_norm( self.transforms(search_img).float() )
            template_img = self.sar_norm( self.transforms(template_img).float() )
            search_hard = self.opt_norm( self.transforms(search_hard).float() )
            template_hard = self.sar_norm( self.transforms(template_hard).float() )

        # print(f"a: {shift_x}  b: {shift_y}  hm: {shift_x_s} ca: {shift_y_s}")
        # This is dependant on the Model!
        # print("WARNING THIS IS DEPENDANT ON THE MODEL")
        shift_x = shift_x - shift_x_s
        shift_y = shift_y - shift_y_s
        if self.full_size:
            scale = 1
        else:
            scale = ((1 - 6/search_img.shape[1])*(3/2))
            
        y_hn = self.get_gt_heatmap(shift_x=shift_x//scale, shift_y=shift_y//scale, w=hm_size, h=hm_size, sigma=None)

        y_hn = y_hn/y_hn.max()

        if self.return_all:
            imgs = (search_img, template_img, template_hard, search_hard)
        else:
            imgs = (search_img, template_img, template_hard)

        if self.named:
            cx = int(hm_size//2 + shift_x//scale)
            cy = int(hm_size//2 + shift_y//scale)
            named = {"WKT": img_info["WKT"], 
                    "city": img_info["city"], 
                    "supervised":int(img_info["supervised"]),
                    "p_match": (cx, cy),
                    "shift": (shift_x, shift_y)}
            return imgs, y_hn, named
        else:
            return imgs, y_hn
