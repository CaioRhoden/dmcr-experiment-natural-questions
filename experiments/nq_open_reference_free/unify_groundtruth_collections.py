import os
from re import M

import polars as pl
from pathlib import Path



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


def unify_and_process_runs(target_dir: Path, output_path: Path, suffix: str) -> None:
    """Unify checkpoints per split for each experiment and save binary collections."""
    
    
    assert target_dir.is_dir(), f"Target directory {target_dir} does not exist or is not a directory."
    # Get filtered feather files matching pattern
    feather_files = [
        f for f in sorted(target_dir.glob("*.feather")) if suffix in f.stem
    ]
    
    # Read, unify, and save
    dfs = _read_feather_files(feather_files)
    
    
    unified = pl.concat(dfs, how="vertical")

    print(f"Unified size: {unified.shape}")

    os.makedirs(output_path.parent, exist_ok=True)

    unified.write_ipc(output_path)


def main() -> None:

    # unify_and_process_runs(
    #     target_dir=Path("../nq_open_with_reference/runs/datamodels/collections/train"),
    #     output_path=Path("reference_collections/llama_rougel.feather"),
    #     suffix="default_collection",
    # )

    unify_and_process_runs(
        target_dir=Path("../nq_open_with_reference/runs/datamodels/collections/train"),
        output_path=Path("reference_collections/llama_em.feather"),
        suffix="em_collection",
    )

    unify_and_process_runs(
        target_dir=Path("../nq_open_with_reference/runs/datamodels/collections/train"),
        output_path=Path("reference_collections/llama_f1.feather"),
        suffix="f1_collection",
    )


if __name__ == "__main__":
    main()


