import argparse
import json
import polars as pl
from utils.set_random_seed import set_random_seed
import argparse
set_random_seed(42)


def add_random_indexes(experiment="original", n=120):
    """Add random wiki indexes to existing retrieval index lists.

    Reads an indices JSON file and a wiki Feather file, samples random wiki row
    indexes, and extends each index list until it contains 120 unique entries.

    Side-effects:
      - Reads: retrieval/rag_retrieval_indexes.json
      - Reads: ../../data/wiki_dump2018_nq_open/processed/wiki.feather
      - Writes: retrieval/random_indeces_insertion.json

    Notes:
      - Deterministic sampling using seed 42.
      - No return value; updates are written to disk.
    """

    indices_path = f"{experiment}/retrieval/rag_retrieval_indexes.json"
    wiki_path = "../../data/wiki_dump2018_nq_open/processed/wiki.feather"

    indices = json.load(open(indices_path, "r"))
    new_indices = indices.copy()
    wiki = pl.read_ipc(wiki_path).with_row_index("index")

    for key in indices.keys():
        random_indexes = (
            wiki.sample(n=(n - len(indices[key])+10), with_replacement=False, seed=42)
            .select(pl.col("index"))
            .to_series()
            .to_list()
        )
        current_values = indices[key]
        n_i = 0
        while len(current_values) < n:
            if random_indexes[n_i] not in current_values:
                current_values.append(random_indexes[n_i])
            n_i += 1
        new_indices[key] = current_values

    json.dump(new_indices, open(f"{experiment}/retrieval/random_indeces_insertion.json", "w"), indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="original")
    parser.add_argument("--n", type=int, default=120)
    args = parser.parse_args()
    add_random_indexes(args.experiment, args.n)