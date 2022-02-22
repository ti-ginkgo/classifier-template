import argparse
import warnings

from ishtos_runner import Tester

warnings.filterwarnings("ignore")


def main(args):
    tester = Tester(config_name=args.config_name, batch_size=args.batch_size)
    tester.inference(args.ckpt)
    tester.save()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, default="config.yaml")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--ckpt", type=str, default="loss")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
