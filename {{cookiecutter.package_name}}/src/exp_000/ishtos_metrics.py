import torchmetrics


# --------------------------------------------------
# getter
# --------------------------------------------------
def get_metric(config):
    metric_name = config.name
    if metric_name == "AUROC":
        return torchmetrics.AUROC(num_classes=config.AUROC.params.num_classes)
    elif metric_name == "MeanAbsoluteError":
        return torchmetrics.MeanAbsoluteError()
    elif metric_name == "MeanAbsolutePercentageError":
        return torchmetrics.MeanAbsolutePercentageError()
    elif metric_name == "MeanSquaredError":
        return torchmetrics.MeanSquaredError(
            squared=config.MeanSquaredError.params.squared
        )
    elif metric_name == "MeanSquaredLogError":
        return torchmetrics.MeanSquaredLogError()
    else:
        raise ValueError(f"Not supported metric: {metric_name}.")
