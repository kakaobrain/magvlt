from collections.abc import Callable

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from magvlt.datamodules.utils import convert_image_to_rgb
import random


class DalleAugmentation(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.size = size

    def forward(self, img: int):
        if hasattr(F, '_get_image_size'):
            w, h = F._get_image_size(img)
        else:
            w, h = F.get_image_size(img)
        s_min = min(w, h)

        off_h = torch.randint(low=3 * (h - s_min) // 8,
                              high=max(3 * (h - s_min) // 8 + 1, 5 * (h - s_min) // 8),
                              size=(1,)).item()
        off_w = torch.randint(low=3 * (w - s_min) // 8,
                              high=max(3 * (w - s_min) // 8 + 1, 5 * (w - s_min) // 8),
                              size=(1,)).item()

        img = F.crop(img, top=off_h, left=off_w, height=s_min, width=s_min)

        t_max = max(min(s_min, round(9 / 8 * self.size)), self.size)
        t = torch.randint(low=self.size, high=t_max + 1, size=(1,)).item()
        img = F.resize(img, [t, t])
        return img


class DalleTransform(Callable):
    splits = {"train", "val", "val_mix"}

    def __init__(self, cfg, split: str):
        assert split in self.splits, f"{split} is not in {self.splits}"
        self._resolution = cfg.dataset.transform.hparams.resolution
        self.split = split
        if split == 'train':
            self._transforms = transforms.Compose(
                [
                    convert_image_to_rgb,
                    DalleAugmentation(size=self._resolution),
                    transforms.RandomCrop(size=(self._resolution, self._resolution)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]
            )
        else:
            self._transforms = transforms.Compose(
                [
                    convert_image_to_rgb,
                    transforms.Resize(size=self._resolution),
                    transforms.CenterCrop(size=self._resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]
            )

        if 'mix' in split:
            self._transforms = None
            from PIL import Image
            self.corgi_sample = Image.open('/data/private/IT2IT_maskgit/it2it/datamodules/corgi.jpeg')
            self._horizontal_transforms = transforms.Compose(
                [
                    convert_image_to_rgb,
                    transforms.Resize(size=(self._resolution, self._resolution // 2)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]
            )
            self._vertical_transforms = transforms.Compose(
                [
                    convert_image_to_rgb,
                    transforms.Resize(size=(self._resolution // 2, self._resolution)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]
            )

    def __call__(self, sample):
        if not 'mix' in self.split:
            return self._transforms(sample)

        if random.random() > 0.5:
            cat_dim = 1
            sample_first = sample
            sample_second = self.corgi_sample
        else:
            cat_dim = 2
            sample_first = self.corgi_sample
            sample_second = sample

        if cat_dim == 1:
            sample_first = self._vertical_transforms(sample_first)
            sample_second = self._vertical_transforms(sample_second)
        else:
            sample_first = self._horizontal_transforms(sample_first)
            sample_second = self._horizontal_transforms(sample_second)

        return torch.cat([sample_first, sample_second], dim=cat_dim)

        # sample_a = self._transforms_a(sample)
        # sample_bg = self._transforms_b(self.corgi_sample)
        # return torch.where(sample_a[0] == -1., sample_bg, sample_a)


class HalfTransform(Callable):
    splits = {"train", "val"}

    def __init__(self, resolution):
        self._resolution = resolution
        self._horizontal_transforms = transforms.Compose(
            [
                convert_image_to_rgb,
                transforms.Resize(size=(self._resolution, self._resolution//2)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )
        self._vertical_transforms = transforms.Compose(
            [
                convert_image_to_rgb,
                transforms.Resize(size=(self._resolution//2, self._resolution)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

    def __call__(self, sample):
        return self._horizontal_transforms(sample), self._vertical_transforms(sample)