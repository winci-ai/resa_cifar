# Copyright (c) Winci.
# Licensed under the Apache License, Version 2.0 (the "License");

import random
from PIL import Image, ImageFilter, ImageOps
import torchvision.transforms as transforms
import torch
import numpy as np
import torchvision.datasets as datasets

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
    
class Solarization:
    """Solarization as a callable object."""

    def __call__(self, img: Image) -> Image:
        """Applies solarization to an input image.
        Args:
            img (Image): an image in the PIL.Image format.
        Returns:
            Image: solarized image.
        """
        return ImageOps.solarize(img)

class Equalization:
    def __call__(self, img: Image) -> Image:
        return ImageOps.equalize(img)

class BaseTransform:
    """Adds callable base class to implement different transformation pipelines."""

    def __call__(self, x: Image) -> torch.Tensor:
        return self.transform(x)


class MultiVisionDataset(datasets.VisionDataset):
    def __init__(
        self,
        data_path,
        args,
        dataset_type='cifar10',  # CIFAR10 or CIFAR100 or STL10
        download=True,
        return_index=False,
    ):
        # Load the appropriate dataset
        if dataset_type == 'cifar10':
            self.dataset = datasets.CIFAR10(data_path, train=True, download=download)
            self.mean, self.std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        elif dataset_type == 'cifar100':
            self.dataset = datasets.CIFAR100(data_path, train=True, download=download)
            self.mean, self.std = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
        elif dataset_type == 'stl10':
            self.dataset = datasets.STL10(data_path, split="train+unlabeled", download=download)
            self.mean, self.std = (0.43, 0.42, 0.39), (0.27, 0.26, 0.27)
        else:
            raise ValueError("Invalid dataset type.")
        
        self.dataset_type = dataset_type

        # Limit the dataset size if specified
        if args.size_dataset >= 0:
            self.dataset.data = self.dataset.data[:args.size_dataset]
            self.dataset.targets = self.dataset.targets[:args.size_dataset]

        weak_transform = transforms.Compose([
            transforms.RandomResizedCrop(args.crops_size[0], scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])

        self.return_index = return_index

        trans = [weak_transform]

        for i in range(len(args.crops_nmb)):
            trans.extend([VisionTransform(crop_size=args.crops_size[i], solarization_prob=args.solarization_prob[i], 
                                        mean=self.mean, std=self.std) ] * args.crops_nmb[i])    

        self.trans = trans

    def __getitem__(self, index):
        # Fetch the image and target from the dataset
        image = self.dataset.data[index]
        
        if self.dataset_type == 'stl10':
            image = Image.fromarray(np.transpose(image, (1, 2, 0)))
        else:
            image = Image.fromarray(image)

        # Apply the transformations
        crops = list(map(lambda trans: trans(image), self.trans))

        if self.return_index:
            return index, crops
        return crops
    
    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.dataset.data)

class VisionTransform(BaseTransform):
    def __init__(
        self,
        brightness: float = 0.4,
        contrast: float = 0.4,
        saturation: float = 0.2,
        hue: float = 0.1,
        color_jitter_prob: float = 0.8,
        gray_scale_prob: float = 0.2,
        horizontal_flip_prob: float = 0.5,
        gaussian_prob: float = 0.0,
        solarization_prob: float = 0.0,
        min_scale: float = 0.2,
        max_scale: float = 1.0,
        crop_size: int = 32,
        mean: tuple = (0.4914, 0.4822, 0.4465),
        std: tuple = (0.2470, 0.2435, 0.2616),
    ):
        super().__init__()

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (crop_size, crop_size),
                    scale=(min_scale, max_scale),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness, contrast, saturation, hue)],
                    p=color_jitter_prob,
                ),
                transforms.RandomGrayscale(p=gray_scale_prob),
                transforms.RandomApply([GaussianBlur()], p=gaussian_prob),
                transforms.RandomApply([Solarization()], p=solarization_prob),
                transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
