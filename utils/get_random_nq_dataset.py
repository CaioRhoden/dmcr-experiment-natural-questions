
import polars as pl


def get_random_nq_dataset(root_path: str, n_samples: int, save_path: str, partition="test", seed=42) -> None:

    """
    Sample a random subset of the NQ Open test dataset.

    Args:
        n_samples: The number of examples to sample.
        save_path: The path to save the sampled dataset.

    Returns:
        None
    """
    NQ_PATH = f"{root_path}/data/nq_open_gold/processed/{partition}.feather"

    nq_test = pl.read_ipc(NQ_PATH).sample(n=n_samples, seed=seed)
    nq_test.write_ipc(save_path)
