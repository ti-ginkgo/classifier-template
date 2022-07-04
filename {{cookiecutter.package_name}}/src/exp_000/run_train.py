#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   run_train.py
@Time    :   2022/07/04 14:15:20
@Author  :   ishtos
@Version :   1.0
@License :   (C)Copyright 2022 ishtos
"""

import argparse
import os
import warnings

import torch
import wandb
from ishtos_lightning_data_module import MyLightningDataModule
from ishtos_lightning_module import MyLightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger
from utils.loader import load_config

warnings.filterwarnings("ignore")


def get_callbacks(config, fold):
    callbacks = []
    if config.callback.early_stopping.enable:
        early_stopping = EarlyStopping(**config.callback.early_stopping.params)
        callbacks.append(early_stopping)

    if config.callback.lr_monitor.enable:
        lr_monitor = LearningRateMonitor(**config.callback.lr_monitor.params)
        callbacks.append(lr_monitor)

    if config.callback.model_loss_checkpoint.enable:
        config.callback.model_loss_checkpoint.params.filename = (
            f"{config.callback.model_loss_checkpoint.params.filename}-{fold}"
        )
        model_loss_checkpoint = ModelCheckpoint(
            **config.callback.model_loss_checkpoint.params
        )
        callbacks.append(model_loss_checkpoint)

    if config.callback.model_score_checkpoint.enable:
        config.callback.model_score_checkpoint.params.filename = (
            f"{config.callback.model_score_checkpoint.params.filename}-{fold}"
        )
        model_score_checkpoint = ModelCheckpoint(
            **config.callback.model_score_checkpoint.params
        )
        callbacks.append(model_score_checkpoint)

    return callbacks


def get_loggers(config, fold):
    loggers = []
    if config.logger.csv.enable:
        config.logger.csv.params.version = f"{config.logger.csv.params.version}-{fold}"
        csv_logger = CSVLogger(**config.logger.csv.params)
        loggers.append(csv_logger)

    if config.logger.tensorboard.enable:
        config.logger.tensorboard.params.version = (
            f"{config.logger.tensorboard.params.version}-{fold}"
        )
        tensorboard_logger = TensorBoardLogger(**config.logger.tensorboard.params)
        loggers.append(tensorboard_logger)

    if config.logger.wandb.enable:
        config.logger.wandb.params.name = f"{config.logger.wandb.params.name}-{fold}"
        wandb_logger = WandbLogger(**config.logger.wandb.params, config=config)
        loggers.append(wandb_logger)

    return loggers


def update_config_for_debug(config):
    config.trainer.params.fast_dev_run = 1
    config.trainer.params.limit_train_batches = 0.1
    config.trainer.params.limit_val_batches = 0.1
    config.trainer.params.max_epochs = 1
    config.trainer.params.num_sanity_val_steps = 1
    config.dataset.store_train = False
    config.dataset.store_valid = False

    return config


def main(args):
    fold = args.fold
    torch.autograd.set_detect_anomaly(True)
    config = load_config(args.config_name)

    os.makedirs(config.environment.save_dir, exist_ok=True)
    seed_everything(config.environment.seed)

    if not config.trainer.debug:
        callbacks = get_callbacks(config, fold)
        loggers = get_loggers(config, fold)
    else:
        config = update_config_for_debug(config)
        callbacks = None
        loggers = False

    trainer = Trainer(
        **config.trainer.params,
        callbacks=callbacks,
        logger=loggers,
    )

    datamodule = MyLightningDataModule(config, fold)
    datamodule.setup(None)
    # len_train_dataloader = datamodule.len_dataloader("train")  # TODO: this is overhead
    model = MyLightningModule(config, fold)

    if not config.trainer.debug:
        for logger in loggers:
            if isinstance(logger, WandbLogger):
                logger.watch(model)

    trainer.fit(model, datamodule=datamodule)

    if not config.trainer.debug:
        for logger in loggers:
            if isinstance(logger, WandbLogger):
                wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, default="config.yaml")
    parser.add_argument("--fold", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
