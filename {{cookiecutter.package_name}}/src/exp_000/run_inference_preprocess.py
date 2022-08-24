#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   run_inference_preprocess.py
@Time    :   2022/07/04 14:14:35
@Author  :   ishtos
@Version :   1.0
@License :   (C)Copyright 2022 ishtos
"""

import argparse
import os

import pandas as pd
from utils.loader import load_config


def preprocess(df, config):
    df["image_path"] = df[config.dataset.id].apply(
        lambda x: os.path.join(
            config.preprocess.base_dir, config.preprocess.test_image_dir, x
        )
    )

    return df


def main(args):
    config = load_config(args.config_name)

    df = pd.read_csv(
        os.path.join(config.preprocess.base_dir, config.preprocess.test_csv)
    )
    df = preprocess(df, config)
    df.to_csv(config.dataset.test_csv, index=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, default="config.yml")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
