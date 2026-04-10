import os
from re import M

import polars as pl
from pathlib import Path


def voting_to_binary(df: pl.DataFrame) -> pl.DataFrame:
    """Convert 'evaluation' column to binary (1 if > 0, else 0)."""
    return df.with_columns((pl.col("evaluation").is_in([0.1, 0.066667])).cast(pl.Int32).alias("evaluation"))

def to_binary(df: pl.DataFrame) -> pl.DataFrame:
    """Convert 'evaluation' column to binary (1 if > 0, else 0)."""
    return df.with_columns((pl.col("evaluation") > 0).cast(pl.Int32).alias("evaluation"))




def read_ipc(path: Path) -> pl.DataFrame:
    return pl.read_ipc(path, memory_map=False)


def write_ipc(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_ipc(path, compression="zstd")


def _read_feather_files(file_paths) -> list[pl.DataFrame]:
    """Read multiple feather files, skipping failed reads."""
    dfs = []
    for f in file_paths:
        try:
            dfs.append(read_ipc(f))
        except Exception:
            print(f"Failed to read {f}")
    return dfs


def unify_and_process_runs(runs_dir: Path, output_dir: Path, pattern=None, processing=True) -> None:
    """Unify checkpoints per split for each experiment and save binary collections."""
    pattern = pattern or "None"
    

    collections_path = runs_dir / "datamodels" / "collections"
    
    for split in ["train", "test"]:
        split_path = collections_path / split
        if not split_path.exists():
            continue
        
        # Get filtered feather files matching pattern
        feather_files = [
            f for f in sorted(split_path.glob("*.feather"))
            if pattern in f.name
        ]
        if not feather_files:
            continue

        
        # Read, unify, and save
        dfs = _read_feather_files(feather_files)
        if not dfs:
            continue
        
        
        if processing:
            if pattern == "Voting":
                unified = voting_to_binary(pl.concat(dfs, how="vertical"))
            else:
                unified = to_binary(pl.concat(dfs, how="vertical"))

        else:
            unified = pl.concat(dfs, how="vertical")

        os.makedirs(output_dir, exist_ok=True)

        out_path = output_dir / f"{split}.feather"
        write_ipc(unified, out_path)



def main() -> None:

    # unify_and_process_runs(
    #     runs_dir=Path("runs"),
    #     output_dir=Path("binary_collections/f1_binary_collection"),
    #     pattern="f1_collection",
    # )

    # unify_and_process_runs(
    #     runs_dir=Path("runs"),
    #     output_dir=Path("binary_collections/f1_collection"),
    #     pattern="f1_collection",
    #     processing=False,
    # )

    # unify_and_process_runs(
    #     runs_dir=Path("runs"),
    #     output_dir=Path("binary_collections/em_collection"),
    #     pattern="em_collection",
    # )



if __name__ == "__main__":
    main()


