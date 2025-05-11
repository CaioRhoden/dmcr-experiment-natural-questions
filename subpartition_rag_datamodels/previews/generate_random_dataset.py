import polars as pl
import random

seed = 42
# NumPy
pl.set_random_seed(seed)
import argparse

def get_random_test_dataset(n_samples: int, save_path: str) -> None:

    """
    Sample a random subset of the NQ Open test dataset.

    Args:
        n_samples: The number of examples to sample.
        save_path: The path to save the sampled dataset.

    Returns:
        None
    """
    NQ_PATH = "../../data/nq_open_gold/processed/test.feather"

    nq_test = pl.read_ipc(NQ_PATH).sample(n=n_samples, seed=seed)
    nq_test.write_ipc(save_path)


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--n_samples", "-n", type=int, required=True)
    argparser.add_argument("--save_path", "-s", type=str, required=True)
    args = argparser.parse_args()

    n_samples = args.n_samples
    save_path = args.save_path

    get_random_test_dataset(n_samples=n_samples, save_path=save_path)