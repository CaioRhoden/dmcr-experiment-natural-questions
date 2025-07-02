from functools import partial
import multiprocessing
import polars as pl
import numpy as np
from numpy.typing import NDArray


def get_context(
    collection_dataset: NDArray[np.int64],
    retrievals: dict,
    wiki: pl.DataFrame, 
    test_idx: int, 
    collection_idx: int
) -> str:
    """
    Retrieves the context for a given collection_idx and test_idx from a wiki
    dataframe and a retrievals dictionary.

    Args:
        collection_dataset (dict): A dictionary of collection indices to document indices.
        retrievals (dict): A dictionary of test indices to document indices.
        collection_idx (int): The index of the collection to retrieve.
        test_idx (int): The index of the test to retrieve.
        wiki (pl.DataFrame): A polars dataframe containing the wiki data.

    Returns:
        str: The context string.
    """
    count = 1
    context = ""
    for collection_idx in collection_dataset[collection_idx]:
        idx = retrievals[str(test_idx)][collection_idx]
        title = wiki[idx]["title"].to_numpy().flatten()[0]
        text = wiki[idx]["text"].to_numpy().flatten()[0]
        context += f"Document[{count}](Title: {title}){text}\n\n"

        count += 1

    return context


def get_batch_context(
    collection_dataset: NDArray[np.int64],
    retrievals: dict,
    wiki: pl.DataFrame,
    test_idx: int,
    collection_idx: list[int],
) -> list[str]:
    """
    Retrieves a list of contexts for a given collection_idx and test_idx from a wiki
    """

    get_context_with_args = partial(
        get_context, collection_dataset, retrievals, wiki, test_idx
    )

    contexts = list(map(get_context_with_args, collection_idx))


    return contexts
