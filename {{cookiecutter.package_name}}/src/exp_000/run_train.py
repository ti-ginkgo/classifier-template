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


def get_loggers(config, fold):
    loggers = []
    if config.logger.csv.enable:
        config.logger.csv.params.version = (
            f"valid-{config.logger.csv.params.version}-{fold}"
        )
        csv_logger = CSVLogger(**config.logger.csv.params)
        loggers.append(csv_logger)

    if config.logger.wandb.enable:
        config.logger.wandb.params.name = (
            f"valid-{config.logger.wandb.params.name}-{fold}"
        )
        wandb_logger = WandbLogger(**config.logger.wandb.params, config=config)
        loggers.append(wandb_logger)

    if config.logger.tensorboard.enable:
        tensorboard_logger = TensorBoardLogger(**config.logger.tensorboard.params)
        loggers.append(tensorboard_logger)

    return loggers


def get_callbacks(config, fold):
    callbacks = []
    if config.callback.early_stopping.enable:
        early_stopping = EarlyStopping(
            monitor=config.callback.early_stopping.monitor,
            patience=config.callback.early_stopping.patience,
            verbose=config.callback.early_stopping.verbose,
            mode=config.callback.early_stopping.mode,
            strict=config.callback.early_stopping.strict,
            check_finite=config.callback.early_stopping.check_finite,
            check_on_train_epoch_end=config.callback.early_stopping.check_on_train_epoch_end,
        )
        callbacks.append(early_stopping)

    if config.callback.model_loss_checkpoint.enable:
        model_loss_checkpoint = ModelCheckpoint(
            dirpath=os.path.join(
                config.general.exp_dir, config.callback.model_loss_checkpoint.dirpath
            ),
            filename=f"{config.callback.model_loss_checkpoint.filename}-{fold}",
            monitor=config.callback.model_loss_checkpoint.monitor,
            verbose=config.callback.model_loss_checkpoint.verbose,
            save_last=config.callback.model_loss_checkpoint.save_last,
            save_top_k=config.callback.model_loss_checkpoint.save_top_k,
            mode=config.callback.model_loss_checkpoint.mode,
            save_weights_only=config.callback.model_loss_checkpoint.save_weights_only,
        )
        callbacks.append(model_loss_checkpoint)

    if config.callback.model_score_checkpoint.enable:
        model_score_checkpoint = ModelCheckpoint(
            dirpath=os.path.join(
                config.general.exp_dir, config.callback.model_score_checkpoint.dirpath
            ),
            filename=f"{config.callback.model_score_checkpoint.filename}-{fold}",
            monitor=config.callback.model_score_checkpoint.monitor,
            verbose=config.callback.model_score_checkpoint.verbose,
            save_last=config.callback.model_score_checkpoint.save_last,
            save_top_k=config.callback.model_score_checkpoint.save_top_k,
            mode=config.callback.model_score_checkpoint.mode,
            save_weights_only=config.callback.model_score_checkpoint.save_weights_only,
        )
        callbacks.append(model_score_checkpoint)

    if config.callback.lr_monitor.enable:
        lr_monitor = LearningRateMonitor(
            log_momentum=config.callback.lr_monitor.log_momentum
        )
        callbacks.append(lr_monitor)

    return callbacks


def main(args):
    fold = args.fold
    torch.autograd.set_detect_anomaly(True)
    config = load_config(args.config_name)

    os.makedirs(config.general.exp_dir, exist_ok=True)
    seed_everything(config.general.seed)

    loggers = get_loggers(config, fold)
    callbacks = get_callbacks(config, fold)

    trainer = Trainer(
        accumulate_grad_batches=config.trainer.accumulate_grad_batches,
        amp_backend=config.trainer.amp_backend,
        benchmark=config.trainer.benchmark,
        callbacks=callbacks,
        deterministic=config.trainer.deteministic,
        fast_dev_run=1 if config.general.debug else False,
        gpus=config.trainer.gpus,
        gradient_clip_val=config.trainer.gradient_clip_val,
        gradient_clip_algorithm=config.trainer.gradient_clip_algorithm,
        limit_train_batches=0.1 if config.general.debug else 1.0,
        limit_val_batches=0.1 if config.general.debug else 1.0,
        logger=loggers,
        max_epochs=1 if config.general.debug else config.trainer.max_epochs,
        num_sanity_val_steps=1 if config.general.debug else 0,
        precision=config.trainer.precision,
        resume_from_checkpoint=eval(config.trainer.resume_from_checkpoint),
        stochastic_weight_avg=config.trainer.stochastic_weight_avg,
    )

    datamodule = MyLightningDataModule(config, fold)
    datamodule.setup(None)
    len_train_dataloader = datamodule.len_dataloader("train")  # TODO: this is overhead
    model = MyLightningModule(config, fold, len_train_dataloader)

    for logger in loggers:
        if isinstance(logger, WandbLogger):
            logger.watch(model)

    trainer.fit(model, datamodule=datamodule)

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
