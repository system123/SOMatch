from imgaug import augmenters as iaa
from skimage.color import rgb2gray
import operator
import numpy as np

DEFAULT_PROBS = {
    "fliplr": 0.5,
    "flipud": 0.3,
    "scale": 0.1,
    "scale_px": (0.98, 1.02),
    "translate": 0.15,
    "translate_perc": (-0.05, 0.05),
    "rotate": 0.2,
    "rotate_angle": (-5, 5),
    "contrast": 0,
    "dropout": 0
}

import torch
import numpy as np

def cutout(img, n_holes=1, length=8):
    """
    Args:
        img (Tensor): Tensor image of size (C, H, W).
    Returns:
        Tensor: Image with n_holes of dimension length x length cut out of it.
    """
    h = img.size(1)
    w = img.size(2)

    mask = np.ones((h, w), np.float32)

    for n in range(n_holes):
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - length // 2, 0, h)
        y2 = np.clip(y + length // 2, 0, h)
        x1 = np.clip(x - length // 2, 0, w)
        x2 = np.clip(x + length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.

    mask = torch.from_numpy(mask)
    mask = mask.expand_as(img)
    img = img * mask

    return img

def cropCenter(img, bounding, shift=(0,0)):
    imshape = [x+y*2 for x,y in zip(img.shape, shift)]
    bounding = list(bounding)
    imshape.reverse()
    bounding.reverse()
    start = tuple(map(lambda a, da: a//2-da//2, imshape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]

def cropCorner(img, bounding, corner="tl"):
    # Corner in (y, x) coordinates
    if corner == "tl":
        start = (0, 0)
    elif corner == "bl":
        start = (img.shape[1]-bounding[1], 0)
    elif corner == "br":
        start = (img.shape[1]-bounding[1], img.shape[0]-bounding[0])
    else:
        start = (0, img.shape[0]-bounding[0])
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]

def toGrayscale(img):
    if len(img.shape) >= 3 and img.shape[-1] == 3:
        img = rgb2gray(img)

    if len(img.shape) < 3:
        img = np.expand_dims(img, axis=2)

    return img

class Augmentation:

    def __init__(self, probs=DEFAULT_PROBS):
        trans = []
        if "fliplr" in probs:
            trans.append(iaa.Fliplr(probs['fliplr']))

        if "flipud" in probs:
            trans.append(iaa.Fliplr(probs['flipud']))
        
        if "scale" in probs:
            trans.append(iaa.Sometimes(probs["scale"], iaa.Affine(scale={"x": probs['scale_px'], "y": probs['scale_px']})))

        if "translate" in probs:
            trans.append(iaa.Sometimes(probs["translate"], iaa.Affine(translate_percent={"x": probs['translate_perc'], "y": probs['translate_perc']})))

        if "rotate" in probs:
            trans.append(iaa.Sometimes(probs["rotate"], iaa.Affine(rotate=probs["rotate_angle"])))

        if "contrast" in probs:
            # trans.append(iaa.ContrastNormalization((0.9, 1.5), per_channel=probs["contrast"]))
            trans.append(iaa.Multiply((0.7, 1.3), per_channel=probs["contrast"]))

        if "dropout" in probs:
            trans.append(iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.2), per_channel=probs["dropout"]))

        self.seq = iaa.Sequential(trans)
        self.transformer = self.seq

    def __call__(self, imgs):
        if isinstance(imgs, list):
            imgs = [self.transformer.augment_images(img) for img in imgs]
        else:
            imgs = self.transformer.augment_image(imgs)

        return imgs

    def refresh_random_state(self):
        self.transformer = self.seq.to_deterministic()
        return self.transformer
