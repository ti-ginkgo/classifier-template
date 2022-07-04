#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   ishtos_metrics.py
@Time    :   2022/07/04 14:13:22
@Author  :   ishtos
@Version :   1.0
@License :   (C)Copyright 2022 ishtos
"""

import torchmetrics


# --------------------------------------------------
# getter
# --------------------------------------------------
def get_metrics(config):
    metrics = []
    metric_names = config.metric.names
    for metric_name in metric_names:
        if metric_name == "Accuracy":
            metrics.append(
                (metric_name, torchmetrics.Accuracy(**config.metric.Accuracy.params))
            )
        elif metric_name == "AUROC":
            metrics.append(
                (metric_name, torchmetrics.AUROC(**config.metric.AUROC.params))
            )
        elif metric_name == "CohenKappa":
            metrics.append(
                (
                    metric_name,
                    torchmetrics.CohenKappa(**config.metric.CohenKappa.params),
                )
            )
        elif metric_name == "MeanAbsoluteError":
            metrics.append((metric_name, torchmetrics.MeanAbsoluteError()))
        elif metric_name == "MeanAbsolutePercentageError":
            metrics.append((metric_name, torchmetrics.MeanAbsolutePercentageError()))
        elif metric_name == "MeanSquaredError":
            metrics.append(
                (
                    metric_name,
                    torchmetrics.MeanSquaredError(
                        **config.metric.MeanSquaredError.params
                    ),
                )
            )
        elif metric_name == "MeanSquaredLogError":
            metrics.append((metric_name, torchmetrics.MeanSquaredLogError()))
        else:
            raise ValueError(f"Not supported metric: {metric_name}.")
    return metrics


if __name__ == "__main__":
    import torch.nn as nn
    from utils.loader import load_config

    config = load_config("config.yaml")

    metrics = get_metrics(config)

    assert all(isinstance(metric[0], str) for metric in metrics)
    assert all(isinstance(metric[1], nn.Module) for metric in metrics)
