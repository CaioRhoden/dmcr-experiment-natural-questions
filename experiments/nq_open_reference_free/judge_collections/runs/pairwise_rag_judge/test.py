import polars as pl
import os

files = os.listdir(".")
dfs = [pl.read_ipc(f) for f in files if f.endswith(".feather")]
df = pl.concat(dfs)

df.write_ipc("concatenated.feather")