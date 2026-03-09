import polars as pl
from pathlib import Path


def to_binary(df: pl.DataFrame) -> pl.DataFrame:
    """Convert 'evaluation' column to binary (1 if > 0, else 0)."""
    return df.with_columns((pl.col("evaluation").is_in([0.1, 0.666667])).cast(pl.Int32).alias("evaluation"))


def read_ipc(path: Path) -> pl.DataFrame:
    return pl.read_ipc(path)


def write_ipc(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_ipc(path, compression="zstd")


def process_rougel_groundtruth(input_dir: Path, output_dir: Path) -> None:
    """Process groundtruth collections per experiment, saving binary 'evaluation'."""
    for exp_folder in input_dir.iterdir():
        if not exp_folder.is_dir():
            continue
        for feather_file in exp_folder.glob("*.feather"):
            df = read_ipc(feather_file)
            df_bin = to_binary(df)
            out_path = output_dir / exp_folder.name / feather_file.name
            write_ipc(df_bin, out_path)


def unify_and_process_runs(runs_dir: Path, output_dir: Path, pattern=None) -> None:
    """Unify checkpoints per split for each experiment and save binary collections."""
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
            feather_files = sorted(split_path.glob("*.feather"))
            if pattern is None:
                pattern = "None"
            feather_files = [f for f in feather_files if pattern in f.name]
            if not feather_files:
                continue
            
            dfs = []
            for f in feather_files:
                try:
                    df = read_ipc(f)
                    dfs.append(df)
                except Exception:
                    print(f"Failed to read {f}")
                    continue
            if not dfs:
                continue
            unified = pl.concat(dfs, how="vertical")
            unified_bin = to_binary(unified)
            out_path = output_dir / exp_folder.name / f"{split}.feather"
            write_ipc(unified_bin, out_path)


def main() -> None:
    # Groundtruth
    # process_rougel_groundtruth(
    #     input_dir=Path("rougel_groundtruth"),
    #     output_dir=Path("binary_collections/groundtruth"),
    # )
    # # Runs (judge)
    # unify_and_process_runs(
    #     runs_dir=Path("runs"),
    #     output_dir=Path("binary_collections/judge"),
    # )

    unify_and_process_runs(
        runs_dir=Path("runs"),
        output_dir=Path("binary_collections/alt1"),
        pattern="ALT1"
    )
    # unify_and_process_runs(
    #     runs_dir=Path("runs"),
    #     output_dir=Path("binary_collections/alt2"),
    #     pattern="ALT2"
    # )

    # unify_and_process_runs(
    #     runs_dir=Path("runs"),
    #     output_dir=Path("binary_collections/"),
    #     pattern="Voting"
    # )



if __name__ == "__main__":
    main()


