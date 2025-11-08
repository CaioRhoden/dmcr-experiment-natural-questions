import polars as pl
from utils.set_random_seed import set_random_seed
import random
import os


def get_random_nq_dataset(
    root_path: str, n_samples: int, save_path: str, partition="test", seed=42
) -> None:
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
    nq_test.write_ipc(save_path, compression="zstd")


def create_n_datasets(
    root_path: str,
    n_datasets: int,
    n_samples: int,
    partition="test",
    seed: int = 42,
    path_prefix: str = "experiment_",
    save_file_name: str = "questions.feather",
) -> None:
    
    """
    Create multiple random subsets of the Natural Questions dataset and save them to disk.

    This function generates `n_datasets` distinct random seeds (using `seed` as the
    initializer), creates a directory for each seed under `root_path` named
    "{path_prefix}_{seed}", and calls `get_random_nq_dataset` to produce and save
    a dataset of `n_samples` examples for the specified `partition`. Each dataset
    is written to a file named `save_file_name` inside its directory.

    Parameters
    ----------
    root_path : str
        Base directory under which per-dataset subdirectories will be created.
    n_datasets : int
        Number of distinct datasets to generate.
    n_samples : int
        Number of samples/questions to include in each generated dataset.
    partition : str, optional
        Dataset partition to sample from (e.g., "train", "validation", "test").
        Defaults to "test".
    seed : int, optional
        Seed used to initialize the random number generator that selects the
        per-dataset seeds. Using the same `seed` yields reproducible selection of
        the per-dataset seeds (default: 42).
    path_prefix : str, optional
        Prefix used for each generated dataset directory name. The final directory
        name is "{path_prefix}_{random_seed}" (default: "experiment_").
    save_file_name : str, optional
        File name used when saving each dataset inside its directory (default:
        "questions.feather").

    Returns
    -------
    None
        The function writes files to disk and does not return a value.

    Side effects
    ------------
    - Creates directories under the path of calling script
    - Calls `get_random_nq_dataset` for every generated seed; that function is
      expected to sample data and write the resulting file at the provided path.
    - May overwrite existing files if the same directory/filename already exist.

    Errors
    ------
    Exceptions raised by `os.makedirs`, `get_random_nq_dataset`, or the random
    module will propagate.
    Example
    -------
    create_n_datasets("/data/nq", n_datasets=5, n_samples=1000,
                      partition="validation", seed=2021,
                      path_prefix="exp", save_file_name="questions.feather")
    """

    assert(n_datasets > 0), "n_datasets must be a positive integer."
    assert(n_samples > 0), "n_samples must be a positive integer."

    set_random_seed(seed)
    random_seeds = random.sample(range(0, 100), n_datasets)
    for i in range(n_datasets):
        save_path_dir = f"{path_prefix}_{random_seeds[i]}"
        if not os.path.exists(save_path_dir):
            os.makedirs(save_path_dir)
        save_path = f"{save_path_dir}/{save_file_name}"
        get_random_nq_dataset(
            root_path, n_samples, save_path, partition=partition, seed=random_seeds[i]
        )
