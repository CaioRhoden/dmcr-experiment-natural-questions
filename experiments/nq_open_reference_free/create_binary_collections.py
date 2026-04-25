import os
from re import M

import polars as pl
from pathlib import Path


def voting_to_binary(df: pl.DataFrame) -> pl.DataFrame:
    """Convert 'evaluation' column to binary (1 if > 0, else 0)."""
    return df.with_columns((pl.col("evaluation").is_in([1.0, 0.666667])).cast(pl.Int32).alias("evaluation"))

def to_binary(df: pl.DataFrame) -> pl.DataFrame:
    """Convert 'evaluation' column to binary (1 if > 0, else 0)."""
    return df.with_columns((pl.col("evaluation") > 0).cast(pl.Int32).alias("evaluation"))




def read_ipc(path: Path) -> pl.DataFrame:
    return pl.read_ipc(path, memory_map=False)





def _read_feather_files(file_paths) -> list[pl.DataFrame]:
    """Read multiple feather files, skipping failed reads."""
    dfs = []
    for f in file_paths:
        try:
            dfs.append(read_ipc(f))
        except Exception:
            print(f"Failed to read {f}")
    return dfs


def unify_and_process_runs(target_dir: Path, output_dir: Path, processing=False) -> None:
    """Unify checkpoints per split for each experiment and save binary collections."""
    
        
    # Get filtered feather files matching pattern
    feather_files = [
        f for f in sorted(target_dir.glob("*.feather"))
    ]
    
    # Read, unify, and save
    dfs = _read_feather_files(feather_files)
    
    
    if processing:
        unified = voting_to_binary(pl.concat(dfs, how="vertical"))

    else:
        unified = pl.concat(dfs, how="vertical")

    os.makedirs(output_dir, exist_ok=True)

    out_path = "train.feather"
    unified.write_ipc(output_dir / out_path)



def main() -> None:

    # unify_and_process_runs(
    #     target_dir=Path("judge_collections/runs/naive"),
    #     output_dir=Path("binary_collections/naive"),
    # )

    # unify_and_process_runs(
    #     target_dir=Path("judge_collections/runs/pairwise_rag_judge"),
    #     output_dir=Path("binary_collections/pairwise_rag"),
    # )

    # unify_and_process_runs(
    #     target_dir=Path("judge_collections/runs/pairwise_zeroshot_judge"),
    #     output_dir=Path("binary_collections/pairwise_zeroshot"),
    # )

    unify_and_process_runs(
        target_dir=Path("judge_collections/runs/voting_naive"),
        output_dir=Path("binary_collections/voting_naive"),
        processing=True,
    )   


if __name__ == "__main__":
    main()


