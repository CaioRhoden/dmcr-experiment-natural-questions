import polars as pl
import argparse
from utils.get_random_nq_dataset import get_random_nq_dataset


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--n_samples", "-n", type=int, required=True)
    argparser.add_argument("--save_path", "-s", type=str, required=True)
    args = argparser.parse_args()

    n_samples = args.n_samples
    save_path = args.save_path

    get_random_nq_dataset(root_path="../..", n_samples=n_samples, save_path=save_path)