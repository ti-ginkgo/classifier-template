import argparse
import os

import pandas as pd
from hydra import compose, initialize
from sklearn.model_selection import GroupKFold, StratifiedKFold


def split_folds(df, config):
    df["fold"] = -1

    fold_name = config.dataset.fold.name
    if fold_name == "GroupKFold":
        gkf = GroupKFold(n_splits=config.dataset.fold.n_splits)
        split_iter = gkf.split(
            df,
            y=df[config.dataset.target].values,
            groups=df[config.dataset.fold.group].values,
        )
    elif fold_name == "StratifiedKFold":
        skf = StratifiedKFold(
            n_splits=config.dataset.fold.n_splits,
            shuffle=True,
            random_state=config.general.seed,
        )
        split_iter = skf.split(df, y=df[config.dataset.target].values)
    else:
        raise ValueError(f"Not supported fold: {fold_name}.")

    for fold, (_, valid_idx) in enumerate(split_iter):
        df.loc[valid_idx] = fold

    return df


def main(args):
    with initialize(config_path="configs", job_name="config"):
        config = compose(config_name=args.config_name)

    csv_path = os.path.join(config.dataset.base_dir, config.dataset.train_df)
    df = pd.read_csv(csv_path)
    df = split_folds(df, config)
    df.to_csv(csv_path, index=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
