#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   ishtos_datasets.py
@Time    :   2022/07/04 14:12:45
@Author  :   ishtos
@Version :   1.0
@License :   (C)Copyright 2022 ishtos
"""

import os

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from ishtos_transforms import get_transforms
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image
from tqdm import tqdm


class MyDataset(Dataset):
    def __init__(self, config, df, transforms=None, phase="train"):
        self.config = config
        self.image_paths = df["image_path"].values
        if phase in ["train", "valid"]:
            self.targets = df[config.dataset.target].values
        self.transforms = transforms
        self.store_train = phase == "train" and config.dataset.store_train
        self.store_valid = phase == "valid" and config.dataset.store_valid
        self.phase = phase
        self.len = len(self.image_paths)

        if self.store_train or self.store_valid:
            self.images = [
                self.load_image(image_path, config)
                for image_path in tqdm(self.image_paths)
            ]

    def __getitem__(self, index):
        if self.store_train or self.store_valid:
            image = self.images[index]
        else:
            image = self.load_image(self.image_paths[index], self.config)

        if self.transforms:
            if self.config.dataset.cv2_or_pil == "cv2":
                image = self.transforms(image=image)["image"]
            elif self.config.dataset.cv2_or_pil == "pil":
                image = self.transforms(image)

        if self.phase in ["train", "valid"]:
            return image, torch.tensor(self.targets[index], dtype=torch.long)
        else:
            return image

    def __len__(self):
        return self.len

    def load_image(self, image_path, config):
        assert os.path.isfile(image_path), f"{image_path} doesn't exist."
        if config.dataset.cv2_or_pil == "cv2":
            image = self.load_image_cv2(image_path, config)
        elif config.dataset.cv2_or_pil == "pil":
            image = self.load_image_pil(image_path, config)
        return image

    def load_image_cv2(self, image_path, config):
        image = cv2.imread(image_path)
        if config.dataset.grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = image[:, :, np.newaxis]
            image = np.repeat(image, 3, 2)
        else:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if config.dataset.gradcam:
            image = cv2.resize(
                image,
                (
                    config.transforms.albumentations.Resize.params.height,
                    config.transforms.albumentations.Resize.params.width,
                ),
            )

        return image

    def load_image_pil(self, image_path, config):
        if config.dataset.grayscale:
            image = read_image(image_path, ImageReadMode.GRAY)
            image = image[:, :, np.newaxis]
            image = np.repeat(image, 3, 2)
        else:
            image = read_image(image_path, ImageReadMode.RGB)

        image = TF.resize(image, **config.transforms.torchvision.resize.params)

        return image


# --------------------------------------------------
# getter
# --------------------------------------------------
def get_dataset(config, df, phase, apply_transforms=True):
    transforms = get_transforms(config, phase) if apply_transforms else None
    return MyDataset(config, df, transforms, phase)


if __name__ == "__main__":
    import pandas as pd
    from utils.loader import load_config

    config = load_config("config.yaml")
    df = pd.DataFrame(columns=["image_path", config.dataset.target])
    dataset = get_dataset(config, df, "train", False)

    assert isinstance(dataset, Dataset)
