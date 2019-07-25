# Source: https://github.com/pytorch/vision/blob/master/references/segmentation/transforms.py

import numpy as np
from PIL import Image
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class Resize(object):
    def __init__(self, resize_shape):
        self.resize_shape = resize_shape

    def __call__(self, image, target):
        image = F.resize(image, self.resize_shape)
        target = F.resize(target, self.resize_shape, interpolation=Image.NEAREST)
        return image, target


class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size)
        target = F.resize(target, size, interpolation=Image.NEAREST)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.asarray(target), dtype=torch.int64)
        return image, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class ConvertCityscapesIds(object):
    """Converts Cityscape masks - adds an ignore index"""

    def __init__(self, ignore_index):
        self.ignore_index = ignore_index

        self.id_to_trainid = {-1: ignore_index, 0: ignore_index, 1: ignore_index, 2: ignore_index, 3: ignore_index,
                              4: ignore_index, 5: ignore_index, 6: ignore_index, 7: 0, 8: 1, 9: ignore_index,
                              10: ignore_index, 11: 2, 12: 3, 13: 4, 14: ignore_index, 15: ignore_index, 16: ignore_index,
                              17: 5, 18: ignore_index, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_index, 30: ignore_index, 31: 16, 32: 17, 33: 18}

    def __call__(self, image, target):

        target_copy = target.clone()

        for k, v in self.id_to_trainid.items():
            target_copy[target == k] = v

        return image, target_copy
