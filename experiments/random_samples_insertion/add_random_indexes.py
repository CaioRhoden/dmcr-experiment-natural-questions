import json
import polars as pl
from utils.set_random_seed import set_random_seed
set_random_seed(42)


def add_random_indexes():

    """
    This function takes in the RAG retrieval indexes and adds 90 random wiki page indexes to each question.
    The output is saved as a new json file in the retrieval folder.
    The function is used to create a new retrieval index for the random sample insertion experiment.
    """
    indices_path = "retrieval/rag_retrieval_indexes.json"
    wiki_path = "../../data/wiki_dump2018_nq_open/processed/wiki.feather"

    indices = json.load(open(indices_path, "r"))
    new_indices = indices.copy()
    wiki = pl.read_ipc(wiki_path).with_row_index("index")
    for key in indices.keys():
        random_indexes = wiki.sample(n=30, with_replacement=False, seed=42).select(pl.col("index")).to_series().to_list()
        current_values = indices[key]
        n_i = 0
        while len(current_values) < 120:
            if random_indexes[n_i] not in current_values:
                current_values.append(random_indexes[n_i])
            n_i += 1
        new_indices[key] = current_values

    json.dump(new_indices, open("retrieval/random_indeces_insertion.json", "w"), indent=4)



if __name__ == "__main__":
    add_random_indexes()