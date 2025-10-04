import os
import polars as pl

def verify_collections_size(collections_dir, expected_size, collection_name):
    """
    Verify after a polars concat in all feather files starting with collection_name if the number of rows is as expected.

    Args:
        collections_dir (str): Path to the directory containing collection CSV files.
        expected_size (int): Expected number of rows in each CSV file.
        collection_name (str): Name of the collection for logging purposes.
    """
    # Get a list of all feather files in the collections directory
    collections_df = [
        pl.read_ipc(f, memory_map=False) for f in os.listdir(collections_dir)
        if f.endswith(".feather") and f.startswith(collection_name)
    ]

    # Check the number of rows in each feather file
    df = pl.concat(collections_df)
    if len(df) != expected_size:
        raise ValueError(f"Warning: concatenated DataFrame has {len(df)} rows, expected {expected_size} rows.")