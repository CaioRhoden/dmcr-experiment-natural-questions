##############################
# Step 1: Get the required data by the specified seed
# Step 2: Iterate over the test data to collect the perplexity values and save them as non_normalized_collections
##############################

from dataclasses import dataclass, field
from locale import normalize
import os
import accelerate
from pandas import DataFrame
import tyro
from utils.calculate_perplexity import calculate_batch_perplexity
from utils.set_random_seed import set_random_seed
from utils.generate_context import get_batch_context
from pathlib import Path
import polars as pl
import numpy as np
from numpy.typing import NDArray
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
import torch
import h5py
import json

set_random_seed(42)  # Set a fixed seed for reproducibility
root = Path(__file__).parent.parent.parent


@dataclass
class NormalizationParametersConfig:
    """
    Configuration class for saving perplexity collections.
    """

    seed_idx: int = 7270
    """Random index seed for reproducibility."""

    saving_prefix: str = "normalized_perplexity_collections"
    """Prefix for the saved collections."""

    target_prefix: str = "non_normalized_perplexity_collections"
    """Prefix for the target collections."""


def nomalize_perplexity(
    seed: int,
    target_prefix: str,
    saving_prefix: str,
):
    """
    This function receives a path for colletion within this folder indicated by 'collections/{seed}' where the files beggining with the target prefix will be extracted.
    With this collections extracted for each "test_idx" the perplexity must be normalized thorugh min-max normalization.
    A new collection must be saved usig the saving_prefix and with the updated values for the column "evaluation"
    """

    collections = []
    for file in os.listdir(f"collections/{seed}"):
        if file.startswith(target_prefix):
            collections.append(pl.read_ipc(f"collections/{seed}/{file}"))

    df = pl.concat(collections)

    _grouped_min_max = df.group_by("test_idx").agg(
        [
            pl.col("evaluation").min().alias("min"),
            pl.col("evaluation").max().alias("max"),
        ]
    )

    normalized_df = (
        df.join(_grouped_min_max, on="test_idx", how="left")
        .with_columns(
            ((pl.col("evaluation") - pl.col("min"))
            / (pl.col("max") - pl.col("min"))).alias("normalized_evaluation")
        )
        .drop(["min", "max", "evaluation"])
        .rename({"normalized_evaluation": "evaluation"})
    )

    normalized_df.write_ipc(f"collections/{seed}/{saving_prefix}.feather")


if __name__ == "__main__":
    args = tyro.cli(NormalizationParametersConfig)

    nomalize_perplexity(
        seed=args.seed_idx,
        target_prefix=args.target_prefix,
        saving_prefix=args.saving_prefix,
    )
