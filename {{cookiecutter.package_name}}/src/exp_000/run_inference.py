#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   run_inference.py
@Time    :   2022/07/04 14:14:47
@Author  :   ishtos
@Version :   1.0
@License :   (C)Copyright 2022 ishtos
"""

import argparse
import warnings

import pandas as pd

from ishtos_runner import Tester
from utils.loader import load_config

warnings.filterwarnings("ignore")


def main(args):
    config = load_config(args.config_name)
    config.dataset.loader.batch_size = args.batch_size
    df = pd.read_csv(config.dataset.train_csv)

    tester = Tester(config=config, df=df, ckpt=args.ckpt)
    tester.run()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, default="config.yml")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--ckpt", type=str, default="loss")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
