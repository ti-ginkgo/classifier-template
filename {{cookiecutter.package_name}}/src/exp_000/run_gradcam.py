import argparse
import os
import warnings
from collections import OrderedDict

import matplotlib.pyplot as plt
import pandas as pd
import torch
from hydra import compose, initialize
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_lightning import seed_everything
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from tqdm import tqdm

from ishtos_datasets import get_dataset
from ishtos_models import get_model
from ishtos_transforms import get_transforms

warnings.filterwarnings("ignore")


def load_config(args):
    with initialize(
        config_path=os.path.join("."),
        job_name="config",
    ):
        config = compose(config_name=args.config)
    config.dataset.loader.batch_size = args.batch_size
    config.dataset.store_valid = False
    config.dataset.gradcam = True
    return config


def load_df(config):
    df = pd.read_csv(os.path.join(config.dataset.base_dir, config.dataset.train_df))

    skf = StratifiedKFold(
        n_splits=config.train.n_splits,
        shuffle=True,
        random_state=config.general.seed,
    )
    for n, (_, val_index) in enumerate(skf.split(df, df[config.dataset.target].values)):
        df.loc[val_index, "fold"] = int(n)
    return df


def load_model(config, fold, ckpt):
    model = get_model(config.model)
    state_dict = OrderedDict()
    for k, v in torch.load(
        os.path.join(config.general.exp_dir, "checkpoints", ckpt, f"fold-{fold}.ckpt")
    )["state_dict"].items():
        name = k.replace("model.", "", 1)
        state_dict[name] = v
    model.load_state_dict(state_dict)
    model.to("cuda")
    model.eval()
    return model


def load_dataloader(config, df, phase):
    dataset = get_dataset(config, df, phase=phase, apply_transforms=False)
    dataloader = DataLoader(
        dataset,
        batch_size=config.dataset.loader.batch_size,
        num_workers=config.dataset.loader.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=False,
    )
    return dataloader


def reshape_transform(tensor, height=12, width=12):
    result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.permute(0, 3, 1, 2)
    return result


def load_cam(model, target_layers, reshape_transform=None):
    cam = GradCAMPlusPlus(
        model=model,
        target_layers=target_layers,
        use_cuda=True,
        reshape_transform=reshape_transform,
    )
    return cam


def inference_cam(model, dataloader, transforms, cam):
    original_images, targets = iter(dataloader).next()
    images = torch.stack(
        [transforms(image=image.numpy())["image"] for image in original_images]
    )
    logits = model(images.to("cuda")).squeeze(1)
    preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
    labels = targets.detach().cpu().numpy()
    grayscale_cam = cam(input_tensor=images, target_category=None, eigen_smooth=True)
    original_images = original_images.detach().cpu().numpy() / 255.0
    return original_images, grayscale_cam, preds, labels


def save_cam(images, grayscale_cams, preds, labels, save_dir, fold, ckpt):
    fig = plt.figure(figsize=(16, 16))
    for i, (image, grayscale_cam, pred, label) in enumerate(
        zip(images, grayscale_cams, preds, labels)
    ):
        plt.subplot(4, 4, i + 1)
        visualization = show_cam_on_image(image, grayscale_cam)
        plt.imshow(visualization)
        plt.title(f"pred: {pred:.1f}, label: {label}")
        plt.axis("off")
    fig.savefig(os.path.join(save_dir, f"gradcam_{ckpt}_{fold}.png"))


def main(args):
    config = load_config(args)
    seed_everything(config.general.seed)

    df = load_df(config)
    for fold in tqdm(range(config.train.n_splits)):
        valid_df = df[df["fold"] == fold].reset_index(drop=True)
        model = load_model(config, fold, args.ckpt)
        dataloader = load_dataloader(config, valid_df, "valid")
        cam = load_cam(
            model,
            target_layers=[model.model.blocks[-1][-1].bn1],
            reshape_transform=None,
        )
        transforms = get_transforms(config, "valid")
        images, grayscale_cams, preds, labels = inference_cam(
            model, dataloader, transforms, cam
        )
        save_cam(
            images,
            grayscale_cams,
            preds,
            labels,
            f"{config.general.exp_dir}",
            fold,
            args.ckpt,
        )
        torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--ckpt", type=str, default="loss")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
