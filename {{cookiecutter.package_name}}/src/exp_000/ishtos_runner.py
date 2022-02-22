import os
from abc import abstractmethod
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from tqdm import tqdm

from ishtos_datasets import get_dataset
from ishtos_models import get_model


class Runner:
    def __init__(self, config_name="config.yaml", batch_size=32, gradcam=False):
        self.config = None
        self.df = None

        self.init(config_name, batch_size, gradcam)

    def init(self, config_name, batch_size, gradcam):
        self.load_config(config_name, batch_size, gradcam)
        self.load_df()

    def load_config(self, config_name, batch_size, gradcam):
        with initialize(config_path=".", job_name="config"):
            config = compose(config_name=config_name)
        config.dataset.loader.batch_size = batch_size
        config.dataset.gradcam = gradcam
        config.dataset.store_valid = False

        self.config = config

    @abstractmethod
    def load_df(self):
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
        models = []
        for fold in range(self.config.train.n_splits):
            model = self.__load_model(fold, ckpt)
            models.append(model)

        self.models = models

    def __load_dataloader(self, df, phase, apply_transforms):
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

    def oof(self, ckpt):
        self.ckpt = ckpt
        inferences = np.zeros((len(self.df), self.config.model.params.num_classes))
        models = self.__load_models(ckpt)
        for fold in range(self.config.train.n_splits):
            valid_df = self.df[self.df["fold"] == fold]
            model = models[fold]
            dataloader = self.__load_dataloader(self.config, valid_df)
            inferences[valid_df.index] = self.__inference(model, dataloader)

        return inferences

    def save(self):
        self.df.to_csv(
            os.path.join(self.config.general.exp_dir, f"{self.ckpt}_oof.csv"),
            index=False,
        )

    def gradcam(self):
        pass


class Tester(Runner):
    def __init__(self):
        super(Tester, self).__init__()

    def load_df(self):
        df = pd.read_csv(
            os.path.join(self.config.dataset.base_dir, self.config.dataset.test_df)
        )

        self.df = df

    def inference(self, ckpt):
        self.ckpt = ckpt
        inferences = np.zeros((len(self.df), self.config.model.params.num_classes))
        models = self.__load_models(ckpt)
        for fold in range(self.config.train.n_split):
            model = models[fold]
            dataloader = self.__load_dataloader(self.config, self.df)
            inferences += self.__inference(model, dataloader)
        inferences = inferences / self.config.train.n_split

        self.inferences = inferences

    def save(self):
        df = self.df.copy()
        df.loc[:, self.config.dataset.target] = self.inferences
        df.to_csv(
            os.path.join(
                self.config.general.exp_dir, f"{self.ckpt}_inferences.csv", index=False
            )
        )
