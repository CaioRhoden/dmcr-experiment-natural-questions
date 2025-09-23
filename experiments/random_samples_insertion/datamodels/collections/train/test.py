
import polars as pl
import os

collections_arr: list[str] = [f for f in os.listdir(".") if f.endswith(".feather")]
dfs = []
for filename in collections_arr:
    c = pl.read_ipc(filename)
    dfs.append(c)

for id in range(len(dfs)):
    dfs[id] = dfs[id].with_columns((pl.arange(0, pl.count()) % 500).alias("test_idx"))

for id in range(len(dfs)):
    dfs[id].write_ipc(collections_arr[id])