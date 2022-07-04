#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   ishtos_runner.py
@Time    :   2022/07/04 14:13:55
@Author  :   ishtos
@Version :   1.0
@License :   (C)Copyright 2022 ishtos
"""


import os
from abc import abstractmethod
from collections import OrderedDict
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
from ishtos_datasets import get_dataset
from ishtos_models import get_model
from ishtos_transforms import get_transforms
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch.utils.data import DataLoader
from tqdm import tqdm


class Runner:
    def __init__(self, config, df, ckpt="loss"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.df = df
        self.models = self.load_models(ckpt)

    def load_model(self, fold, ckpt):
        model = get_model(self.config)
        state_dict = OrderedDict()
        ckpt_path = os.path.join(
            self.config.general.save_dir, "checkpoints", ckpt, f"fold-{fold}.ckpt"
        )
        for k, v in torch.load(ckpt_path)["state_dict"].items():
            name = k.replace("model.", "", 1)
            state_dict[name] = v
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        return model

    def load_models(self, ckpt):
        self.ckpt = ckpt
        models = []
        for fold in range(self.config.preprocess.fold.n_splits):
            model = self.load_model(fold, ckpt)
            models.append(model)

        return models

    def load_dataloader(self, df, phase, apply_transforms=True):
        dataset = get_dataset(self.config, df, phase, apply_transforms)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.dataset.loader.batch_size,
            num_workers=self.config.dataset.loader.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=False,
        )

        return dataloader

    @abstractmethod
    def run(self):
        pass

    def predict(self, model, dataloader):
        outputs = []
        with torch.inference_mode():
            for images, _ in tqdm(dataloader):
                logits = model(images.to(self.device)).squeeze(1)
                preds = logits.softmax(dim=1).detach().cpu().numpy()
                outputs.append(preds)

        return np.concatenate(outputs)

    def save(self, outputs, fname):
        df = self.df.copy()
        for i in range(self.config.model.params.num_classes):
            df[f"class_{i}"] = outputs[:, i]
        df.to_csv(
            os.path.join(self.config.general.save_dir, fname),
            index=False,
        )


class Validator(Runner):
    def run(self):
        oofs = np.zeros((len(self.df), self.config.model.params.num_classes))
        for fold in range(self.config.preprocess.fold.n_splits):
            valid_df = self.df[self.df["fold"] == fold].reset_index(drop=True)
            model = self.models[fold]
            dataloader = self.load_dataloader(valid_df, "valid")
            oofs[valid_df.index] = self.predict(model, dataloader)

        self.save(oofs, f"oof_{self.ckpt}.csv")

    def load_cam(self, model, target_layers, reshape_transform=None):
        cam = GradCAMPlusPlus(
            model=model,
            target_layers=target_layers,
            use_cuda=True,
            reshape_transform=reshape_transform,
        )
        return cam

    def get_target_layers(self, model_name, model):
        if model_name == "convnext":
            return [model.model.layers[-1].blocks[-1].norm1]
        elif model_name == "efficientnet":
            return [model.model.blocks[-1][-1].bn1]
        elif model_name == "resnet":
            return [model.model.layer4[-1]]
        elif model_name == "swin":
            return [model.model.layers[-1].blocks[-1].norm1]
        else:
            raise ValueError(f"Not supported model: {model_name}.")

    def get_reshape_transform(self, model_name):
        if model_name == "convnext":
            if "224" in self.config.model.params.base_model:
                fn = partial(reshape_transform, height=7, width=7)
            elif "384" in self.config.model.params.base_model:
                fn = partial(reshape_transform, height=12, width=12)
            return fn
        elif model_name == "efficientnet":
            return None
        elif model_name == "resnet":
            return None
        elif model_name == "swin":
            if "224" in self.config.model.params.base_model:
                fn = partial(reshape_transform, height=7, width=7)
            elif "384" in self.config.model.params.base_model:
                fn = partial(reshape_transform, height=12, width=12)
            return fn
        else:
            raise ValueError(f"Not supported model: {model_name}.")

    def run_cam(self):
        self.config.dataset.gradcam = True
        for fold in range(self.config.preprocess.fold.n_splits):
            valid_df = self.df[self.df["fold"] == fold].reset_index(drop=True)
            model = self.models[fold]
            dataloader = self.load_dataloader(valid_df, "valid", False)
            cam = self.load_cam(
                model,
                target_layers=self.get_target_layers(self.config.model.name, model),
                reshape_transform=self.get_reshape_transform(self.config.model.name),
            )
            transforms = get_transforms(self.config, "valid")
            original_images, grayscale_cams, preds, labels = self.inference_cam(
                model, dataloader, transforms, cam
            )
            self.save_cam(original_images, grayscale_cams, preds, labels, fold)

    def inference_cam(self, model, dataloader, transforms, cam):
        original_images, targets = iter(dataloader).next()
        images = torch.stack(
            [transforms(image=image.numpy())["image"] for image in original_images]
        )
        logits = model(images.to(self.device)).squeeze(1)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        labels = targets.detach().cpu().numpy()
        grayscale_cams = cam(input_tensor=images, targets=None, eigen_smooth=True)
        original_images = original_images.detach().cpu().numpy() / 255.0
        return original_images, grayscale_cams, preds, labels

    def save_cam(self, original_images, grayscale_cams, preds, labels, fold):
        batch_size = self.config.dataset.loader.batch_size
        fig, axes = plt.subplots(
            batch_size // 4, 4, figsize=(32, 32), sharex=True, sharey=True
        )
        for i, (image, grayscale_cam, pred, label) in enumerate(
            zip(original_images, grayscale_cams, preds, labels)
        ):
            visualization = show_cam_on_image(image, grayscale_cam)
            ax = axes[i // 4, i % 4]
            ax.set_title(f"pred: {pred:.1f}, label: {label}")
            ax.imshow(visualization)
        fig.savefig(
            os.path.join(self.config.general.save_dir, f"cam_{self.ckpt}_{fold}.png")
        )


def reshape_transform(tensor, height, width):
    result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.permute(0, 3, 1, 2)
    return result


class Tester(Runner):
    def run(self):
        inferences = np.zeros((len(self.df), self.config.model.params.num_classes))
        for fold in range(self.config.preprocess.fold.n_splits):
            model = self.models[fold]
            dataloader = self.load_dataloader(self.df, "test")
            inferences += self.predict(model, dataloader)
        inferences = inferences / self.config.preprocess.fold.n_splits

        self.save(inferences, f"inference_{self.ckpt}.csv")
