##############################
# Step 1: Get the required data by the specified seed
# Step 2: Iterate over the test data to collect the perplexity values and save them as non_normalized_collections
##############################

from dataclasses import dataclass, field
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
class ParametersConfig:
    """
    Configuration class for saving perplexity collections.
    """

    seed: int = 7270
    """Random index seed for reproducibility."""
    
    saving_prefix: str = "non_normalized_perplexity_collections"
    """Prefix for the saved collections."""

    model_configs: dict[str, float|int] = field(default_factory=lambda: {
            "max_length": 2048,
        })

    start_idx: int = 0
    """Starting index for perplexity calculation."""
    
    end_idx: int = 50
    """Ending index for perplexity calculation."""

    len_collection: int = 2000
    """Length of the collection for perplexity calculation."""



class PerplexityCollections:

    def __init__(self, seed: int) -> None:

        self.seed = seed

    def create_perplexity_collections(
            self,
            type: str,
            saving_prefix: str,
            start_idx: int,
            end_idx: int,
            len_collection: int
    ):
        ### Load data
        collection = self.load_collection(type, self.seed)
        wiki = self.load_wiki()
        questions = self.load_questions()
        collections_dataset = self.load_collections_dataset(type, self.seed)
        retrievals = self.load_retrievals(self.seed)

        ### Setup model and tokenizer
        accelerator = Accelerator()
        model_path = f"{root}/models/llms/Llama-3.2-3B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token for Llama models
        model = AutoModelForCausalLM.from_pretrained(model_path,  device_map={"": accelerator.process_index}, torch_dtype=torch.bfloat16,)


        perplexity_list = []
        for idx in range(start_idx, end_idx):
            print("Calculating perplexity for collection", idx)
            contexts = get_batch_context(collections_dataset, retrievals, wiki, idx, list(range(len_collection)))
            perplexity = calculate_batch_perplexity(
                texts=[questions[idx].select("question").item() for _ in range(len_collection)],
                model=model,
                tokenizer=tokenizer,
                contexts=contexts,
                device=accelerator.device
            )

            perplexity_list.append(
                collection
                .filter(pl.col("test_idx") == idx)
                .sort("collection_idx")
                .select("collection_idx", "test_idx", "input")
                .with_columns("collection_idx", pl.lit(self.create_binary_collection(list(collections_dataset[idx]))))
                .with_columns(pl.Series("evaluation", perplexity.cpu().numpy()).cast(pl.Float32))
            )
        
        print("Saving perplexity collections...")
        pl.concat(perplexity_list).write_ipc(f"{root}/experiments/perplexity_proxy_groundtruth/collections/{self.seed}/{saving_prefix}_{start_idx}_{end_idx-1}.feather")



    def load_retrievals(self, seed: int) -> dict:
        with open(f"retrieval/{seed}/rag_retrieval_indexes.json", "r") as f:
            return json.load(f)

    def load_collection(self, type: str, seed: int) -> pl.DataFrame:
        return pl.read_ipc(f"collections/{seed}/{type}.feather")
    
    def load_wiki(self) -> pl.DataFrame:
        return pl.read_ipc(f"{root}/data/wiki_dump2018_nq_open/processed/wiki.feather")

    def load_questions(self) -> pl.DataFrame:
        return pl.read_ipc(f"{root}/data/nq_open_gold/processed/test.feather")

    def load_collections_dataset(self, type, seed) -> NDArray:
        for file in os.listdir(f"collections_dataset/{seed}"):
            if file.endswith(".h5") and file.startswith(f"{type}_collection."):
                with h5py.File(f"collections_dataset/{seed}/{file}", "r") as f:
                    return f[f"{type}_collection"][()]
            
    def create_binary_collection(self, indices):
        result = [0] * 100
        for idx in indices:
            if 0 <= idx < 100:
                result[idx] = 1

        return result


if __name__ == "__main__":

    args = tyro.cli(ParametersConfig)

    perplexity_collections = PerplexityCollections(seed=args.seed)
    perplexity_collections.create_perplexity_collections(
        type="train",
        saving_prefix=args.saving_prefix,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        len_collection=args.len_collection
    )