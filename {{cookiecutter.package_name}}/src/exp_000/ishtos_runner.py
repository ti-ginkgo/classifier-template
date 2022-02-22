import os
from abc import abstractmethod
from collections import OrderedDict

import matplotlib as plt
import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from tqdm import tqdm

from ishtos_datasets import get_dataset
from ishtos_models import get_model
from ishtos_transforms import get_transforms


class Runner:
    def __init__(self, config_name="config.yaml", ckpt="loss", batch_size=32):
        self.config = None
        self.df = None

        self.init(config_name, ckpt, batch_size)

    def init(self, config_name, ckpt, batch_size):
        self.__load_config(config_name, batch_size)
        self.__load_df()
        self.__load_models(ckpt)

    def __load_config(self, config_name, batch_size):
        with initialize(config_path=".", job_name="config"):
            config = compose(config_name=config_name)
        config.dataset.loader.batch_size = batch_size
        config.dataset.store_valid = False

        self.config = config

    @abstractmethod
    def __load_df(self):
        pass

    def __load_model(self, fold, ckpt):
        model = get_model(self.config.model)
        state_dict = OrderedDict()
        ckpt_path = os.path.join(
            self.config.general.exp_dir, "checkpoints", ckpt, f"fold-{fold}.ckpt"
        )
        for k, v in torch.load(ckpt_path)["state_dict"].items():
            name = k.replace("model.", "", 1)
            state_dict[name] = v
        model.load_state_dict(state_dict)
        model.to("cuda")
        model.eval()

        return model

    def __load_models(self, ckpt):
        self.ckpt = ckpt
        models = []
        for fold in range(self.config.train.n_splits):
            model = self.__load_model(fold, ckpt)
            models.append(model)

        self.models = models

    def __load_dataloader(self, df, phase, apply_transforms=True):
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

    def __inference(self, model, dataloader):
        inferences = []
        with torch.inference_mode():
            for images in tqdm(dataloader):
                logits = model(images.to("cuda")).squeeze(1)
                preds = torch.sigmoid(logits).cpu().numpy()
                inferences.append(preds)

        return np.concatenate(inferences)


class Validator(Runner):
    def __init__(self):
        super(Validator, self).__init__()

    def load_df(self):
        df = pd.read_csv(
            os.path.join(self.config.dataset.base_dir, self.config.dataset.train_df)
        )

        skf = StratifiedKFold(
            n_splits=self.config.train.n_splits,
            shuffle=True,
            random_state=self.config.general.seed,
        )
        for n, (_, val_index) in enumerate(
            skf.split(df, df[self.config.dataset.target].values)
        ):
            df.loc[val_index, "fold"] = int(n)

        self.df = df

    def run_oof(self):
        inferences = np.zeros((len(self.df), self.config.model.params.num_classes))
        for fold in range(self.config.train.n_splits):
            valid_df = self.df[self.df["fold"] == fold].reset_index(drop=True)
            model = self.models[fold]
            dataloader = self.__load_dataloader(self.config, valid_df, "valid")
            inferences[valid_df.index] = self.__inference(model, dataloader)

        self.inferences = inferences
        self.save_oof()

    def save_oof(self):
        df = self.df.copy()
        df["oof"] = self.inferences
        df.to_csv(
            os.path.join(self.config.general.exp_dir, f"oof_{self.ckpt}.csv"),
            index=False,
        )

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
        elif model_name == "swin":
            return [model.model.layers[-1].blocks[-1].norm1]
        else:
            raise ValueError(f"Not supported model: {model_name}.")

    def get_reshape_transform(self, model_name):
        if model_name == "convnext":
            return reshape_transform
        elif model_name == "efficientnet":
            return None
        elif model_name == "swin":
            return reshape_transform
        else:
            raise ValueError(f"Not supported model: {model_name}.")

    def run_cam(self):
        self.config.dataset.gradcam = True
        for fold in range(self.config.train.n_splits):
            valid_df = self.df[self.df["fold"] == fold].reset_index(drop=True)
            model = self.models[fold]
            dataloader = self.__load_dataloader(self.config, valid_df, "valid", False)
            cam = self.load_cam(
                model,
                target_layers=self.get_target_layers(self.config.model.name, model),
                reshape_transform=self.get_reshape_transform(self.config.model.name),
            )
            transforms = get_transforms(self.config, "valid")
            results = self.inference_cam(model, dataloader, transforms, cam)
            self.save_cam(results, fold)

    def inference_cam(self, model, dataloader, transforms, cam):
        original_images, targets = iter(dataloader).next()
        images = torch.stack(
            [transforms(image=image.numpy())["image"] for image in original_images]
        )
        logits = model(images.to("cuda")).squeeze(1)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        labels = targets.detach().cpu().numpy()
        grayscale_cam = cam(
            input_tensor=images, target_category=None, eigen_smooth=True
        )
        original_images = original_images.detach().cpu().numpy() / 255.0
        return original_images, grayscale_cam, preds, labels

    def save_cam(self, results, fold):
        fig = plt.figure(figsize=(8, 8))
        batch_size = self.config.dataset.loader.batch_size
        for i, (image, grayscale_cam, pred, label) in enumerate(results):
            plt.subplot(batch_size, 4, i + 1)
            visualization = show_cam_on_image(image, grayscale_cam)
            plt.imshow(visualization)
            plt.title(f"pred: {pred:.1f}, label: {label}")
            plt.axis("off")
        fig.savefig(
            os.path.join(self.config.general.exp_dir, f"cam_{self.ckpt}_{fold}.png")
        )


def reshape_transform(tensor, height=12, width=12):
    result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.permute(0, 3, 1, 2)
    return result


class Tester(Runner):
    def __init__(self):
        super(Tester, self).__init__()

    def load_df(self):
        df = pd.read_csv(
            os.path.join(self.config.dataset.base_dir, self.config.dataset.test_df)
        )

        self.df = df

    def run_inference(self):
        inferences = np.zeros((len(self.df), self.config.model.params.num_classes))
        for fold in range(self.config.train.n_split):
            model = self.models[fold]
            dataloader = self.__load_dataloader(self.config, self.df, "test")
            inferences += self.__inference(model, dataloader)
        inferences = inferences / self.config.train.n_split

        self.inferences = inferences
        self.save_inference()

    def save_inference(self):
        df = self.df.copy()
        df.loc[:, self.config.dataset.target] = self.inferences
        df.to_csv(
            os.path.join(
                self.config.general.exp_dir, f"inferences_{self.ckpt}.csv", index=False
            )
        )
