import argparse
import os
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from tqdm import tqdm

from ishtos_datasets import get_dataset
from ishtos_models import get_model

warnings.filterwarnings("ignore")


def load_df(config):
    df = pd.read_csv(os.path.join(config.dataset.base_dir, config.dataset.test_df))
    return df


def load_models(config, ckpt):
    models = []
    for fold in range(config.train.n_splits):
        model = get_model(config.model)
        state_dict = OrderedDict()
        for k, v in torch.load(
            os.path.join(
                config.general.exp_dir, "checkpoints", ckpt, f"fold-{fold}.ckpt"
            )
        )["state_dict"].items():
            name = k.replace("model.", "", 1)
            state_dict[name] = v
        model.load_state_dict(state_dict)
        model.to("cuda")
        models.append(model)
    return models


def load_dataloader(config, df):
    dataset = get_dataset(config, df, phase="test")
    dataloader = DataLoader(
        dataset,
        batch_size=config.dataset.loader.batch_size,
        num_workers=config.dataset.loader.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )
    return dataloader


def inference(model, dataloader):
    inferences = []
    with torch.inference_mode():
        for images in tqdm(dataloader):
            logits = model(images.to("cuda")).squeeze(1)
            preds = torch.softmax(logits, dim=1).cpu().numpy()
            inferences.append(preds)
    return np.concatenate(inferences)


def main(args):
    with initialize(
        config_path=os.path.join("."),
        job_name="config",
    ):
        config = compose(config_name="config.yaml")
    seed_everything(config.general.seed)
    config.dataset.loader.batch_size = 32

    df = load_df(config)
    inferences = np.zeros((len(df), config.model.params.num_classes))
    models = load_models(config, args.ckpt)
    for fold in range(config.train.n_splits):
        model = models[fold]
        dataloader = load_dataloader(config, df)
        inferences += inference(model, dataloader)
    inferences /= 5

    df.loc[:, config.dataset.target] = inferences
    df.drop("image_path", axis=1, inplace=True)
    df.to_csv(
        os.path.join(config.general.exp_dir, f"{args.ckpt}_inference.csv"), index=False
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="loss")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
