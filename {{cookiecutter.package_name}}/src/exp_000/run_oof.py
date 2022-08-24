#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   run_oof.py
@Time    :   2022/07/04 14:15:02
@Author  :   ishtos
@Version :   1.0
@License :   (C)Copyright 2022 ishtos
"""

import argparse
import warnings

import pandas as pd
from ishtos_runner import Validator
from utils.loader import load_config

warnings.filterwarnings("ignore")


def main(args):
    config = load_config(args.config_name)
    config.dataset.loader.batch_size = args.batch_size
    df = pd.read_csv(config.dataset.train_csv)

    validator = Validator(config=config, df=df, ckpt=args.ckpt)
    validator.run()

    if args.cam:
        validator.run_cam()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, default="config.yml")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--ckpt", type=str, default="loss")
    parser.add_argument("--cam", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
