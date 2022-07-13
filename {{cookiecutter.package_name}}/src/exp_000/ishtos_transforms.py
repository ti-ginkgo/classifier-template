#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   ishtos_transforms.py
@Time    :   2022/07/04 14:14:15
@Author  :   ishtos
@Version :   1.0
@License :   (C)Copyright 2022 ishtos
"""

import albumentations as A
import torch
import torchvision.transforms as T
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms_Av1(config):
    augmentations = [A.Resize(**config.transforms.albumentations.Resize.params)]
    if config.transforms.pretrained:
        augmentations.append(A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    else:
        augmentations.append(A.Normalize(mean=0, std=1))
    augmentations.append(ToTensorV2())
    return A.Compose(augmentations)


def get_valid_transforms_Av1(config):
    augmentations = [A.Resize(**config.transforms.albumentations.Resize.params)]
    if config.transforms.pretrained:
        augmentations.append(A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    else:
        augmentations.append(A.Normalize(mean=0, std=1))
    augmentations.append(ToTensorV2())
    return A.Compose(augmentations)


def get_train_transforms_Tv1(config):
    augmentations = [
        T.ConvertImageDtype(torch.float32),
    ]
    if config.transforms.pretrained:
        augmentations.append(T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    else:
        augmentations.append(T.Normalize(mean=0, std=1))

    return T.Compose(augmentations)


def get_train_transforms_Tv2(config):
    augmentations = [
        T.RandAugment(**config.transforms.RandAugment.params),
        T.ConvertImageDtype(torch.float32),
    ]
    if config.transforms.pretrained:
        augmentations.append(T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    else:
        augmentations.append(T.Normalize(mean=0, std=1))

    return T.Compose(augmentations)


def get_valid_transforms_Tv1(config):
    augmentations = [
        T.ConvertImageDtype(torch.float32),
    ]
    if config.transforms.pretrained:
        augmentations.append(T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    else:
        augmentations.append(T.Normalize(mean=0, std=1))

    return T.Compose(augmentations)


# --------------------------------------------------
# getter
# --------------------------------------------------
def get_transforms(config, phase):
    if phase == "train":
        return eval(f"get_train_transforms_{config.transforms.train_version}")(config)
    elif phase in ["valid", "test"]:
        return eval(f"get_valid_transforms_{config.transforms.valid_version}")(config)
    else:
        raise ValueError(f"Not supported transforms phase: {phase}.")


if __name__ == "__main__":
    from utils.loader import load_config

    config = load_config("config.yaml")

    config.transforms.train_version = "Av1"
    config.transforms.valid_version = "Av1"

    train_transform = get_transforms(config, "train")
    valid_transform = get_transforms(config, "valid")

    assert isinstance(train_transform, A.Compose)
    assert isinstance(valid_transform, A.Compose)

    config.transforms.train_version = "Tv1"
    config.transforms.valid_version = "Tv1"

    train_transform = get_transforms(config, "train")
    valid_transform = get_transforms(config, "valid")

    assert isinstance(train_transform, T.Compose)
    assert isinstance(valid_transform, T.Compose)
