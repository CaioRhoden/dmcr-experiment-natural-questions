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


def unify_and_process_runs(runs_dir: Path, output_dir: Path, pattern=None) -> None:
    """Unify checkpoints per split for each experiment and save binary collections."""
    pattern = pattern or "None"
    
    for exp_folder in runs_dir.iterdir():
        if not exp_folder.is_dir():
            continue
        
        collections_path = exp_folder / "datamodels" / "collections"
        if not collections_path.exists():
            continue
        
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

            if pattern == "Voting":
                unified_bin = voting_to_binary(pl.concat(dfs, how="vertical"))
            else:
                unified_bin = to_binary(pl.concat(dfs, how="vertical"))
            print(f"DEBUG: Len unified binary for {exp_folder.name} {split}: {len(unified_bin)} for pattern '{pattern}'")
            out_path = output_dir / exp_folder.name / f"{split}.feather"
            write_ipc(unified_bin, out_path)


def main() -> None:

    unify_and_process_runs(
        runs_dir=Path("runs"),
        output_dir=Path("binary_collections/binary_judge"),
        pattern="-BinaryJudge-"
    )
    unify_and_process_runs(
        runs_dir=Path("runs"),
        output_dir=Path("binary_collections/rougel"),
        pattern="Rouge-L"
    )

    unify_and_process_runs(
        runs_dir=Path("runs"),
        output_dir=Path("binary_collections/voting"),
        pattern="Voting"
    )



if __name__ == "__main__":
    main()


