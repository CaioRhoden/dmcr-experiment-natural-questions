import os
import logging
from pathlib import Path
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import polars as pl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def to_binary(df: pl.DataFrame) -> pl.DataFrame:
    """Convert 'evaluation' column to binary (1 if > 0, else 0)."""
    return df.with_columns((pl.col("evaluation") > 0).cast(pl.Int32).alias("evaluation"))

def voting_to_binary(df: pl.DataFrame) -> pl.DataFrame:
    """Convert 'evaluation' column to a binary consensys, if two or more judge for the same 'collection_id' and 'question_id' vote for 1, then the final label is 1, otherwise 0."""
    clone_agg =  df.group_by(["collection_idx", "test_idx"]).agg(
        pl.col("evaluation").sum().cast(pl.Int32).alias("evaluation")
    )
    return (
            df
            .drop("evaluation")
            .join(clone_agg, on=["collection_idx", "test_idx"], how="left")
            .with_columns((pl.col("evaluation") >= 2).cast(pl.Int32).alias("evaluation"))
            .unique()
    )


def read_ipc(path: Path) -> pl.DataFrame:
    """Read feather file using IPC format."""
    return pl.read_ipc(path, memory_map=False)



def _read_feather_files(file_paths: List[Path]) -> list[pl.DataFrame]:
    """Read multiple feather files, skipping failed reads."""
    dfs = []
    for f in file_paths:
        try:
            dfs.append(read_ipc(f))
        except Exception as e:
            logger.warning(f"Failed to read {f}: {e}")
    return dfs


def get_judge_types(model_dir: Path) -> List[Path]:
    """Discover all judge_type subdirectories under a model directory.
    
    Args:
        model_dir: Path to model directory (e.g., judge_collections/llama)
        
    Returns:
        List of Path objects for each judge_type subdirectory, sorted
    """
    if not model_dir.exists():
        return []
    
    judge_types = [
        d for d in sorted(model_dir.iterdir())
        if d.is_dir() and d.name != 'test'
    ]
    return judge_types


def process_judge_type(source_dir: Path, output_path: Path, split_name: str = "train") -> None:
    """Combine all feather files in a judge_type directory and apply binary transformation.
    
    Args:
        source_dir: Directory containing batch_*.feather files
        output_path: Path where to save the output train.feather or test.feather
        split_name: Either "train" or "test" for logging purposes
    """
    if not source_dir.exists():
        logger.warning(f"Source directory does not exist: {source_dir}")
        return
    
    # Get all feather files (both batch_XXXX.feather and batch_XXXX_YYYY.feather patterns)
    feather_files = sorted(source_dir.glob("*.feather"))
    
    if not feather_files:
        logger.warning(f"No feather files found in {source_dir}")
        return
    
    logger.info(f"Processing {split_name} data: found {len(feather_files)} files in {source_dir}")
    
    # Create output directory before any write operations
    os.makedirs(output_path.parent, exist_ok=True)
    
    # Read and concatenate all feather files
    dfs = _read_feather_files(feather_files)
    
    if not dfs:
        logger.error(f"Failed to read any feather files from {source_dir}")
        return
    
    # Concatenate and apply binary transformation
    unified = pl.concat(dfs, how="vertical")
    processed = to_binary(unified)

    if "voting" in source_dir.name.lower():
        processed_binary = voting_to_binary(unified)
        processed_binary.write_ipc(output_path, compression="zstd")
        processed.write_ipc(output_path.with_name(output_path.stem + "_raw.feather"), compression="zstd")
        return
    processed.write_ipc(output_path, compression="zstd")
    
    logger.info(f"Saved {split_name} data: {len(unified)} rows to {output_path}")


def process_single_model(
    model_name: str,
    judge_collections_root: Optional[Path] = None,
    processed_collections_root: Optional[Path] = None,
    pattern: str | None = None
) -> None:
    """Process a specific model's judge collections.
    
    Args:
        model_name: Name of the model to process ('llama', 'llama_default', 'qwen', 'qwen_default')
        judge_collections_root: Root directory containing model subdirectories (default: judge_collections)
        processed_collections_root: Root directory where to save the processed collections (default: processed_collections)
        pattern: Optional pattern to filter judge types
    """
    if judge_collections_root is None:
        judge_collections_root = Path("judge_collections")
    if processed_collections_root is None:
        processed_collections_root = Path("processed_collections")
    
    judge_collections_root = Path(judge_collections_root)
    processed_collections_root = Path(processed_collections_root)
    
    # Verify model directory exists
    model_dir = judge_collections_root / model_name
    if not model_dir.exists():
        logger.error(f"Model directory does not exist: {model_dir}")
        logger.info(f"Available models: {sorted([d.name for d in judge_collections_root.iterdir() if d.is_dir()])}")
        return
    
    logger.info(f"Processing model: {model_name}")
    logger.info(f"Source: {model_dir}")
    logger.info(f"Output: {processed_collections_root}")
    
    # Get all judge_type subdirectories for this model
    judge_types = get_judge_types(model_dir)
    
    if not judge_types:
        logger.warning(f"No judge_type subdirectories found for model {model_name}")
        return
    
    logger.info(f"Found {len(judge_types)} judge types: {[d.name for d in judge_types]}")
    
    # Process each judge type
    for judge_type_dir in judge_types:
        if pattern and pattern not in judge_type_dir.name:
            logger.debug(f"Skipping {judge_type_dir.name} as it does not match pattern '{pattern}'")
            continue

        
        judge_type_name = judge_type_dir.name
        logger.info(f"\nProcessing judge type: {judge_type_name}")
        
        # Process training data (batch files)
        output_dir = processed_collections_root / model_name / judge_type_name
        output_train_path = output_dir / "train.feather"

        process_judge_type(judge_type_dir, output_train_path, split_name="train")
        
        # Process test data if it exists (test data is in model_dir/test/{judge_type_name})
        test_dir = model_dir / "test" / judge_type_name
        if test_dir.exists():
            output_test_path = output_dir / "test.feather"
            process_judge_type(test_dir, output_test_path, split_name="test")
        else:
            logger.debug(f"  No test subdirectory found for {judge_type_name}")
    
    logger.info(f"\nFinished processing model: {model_name}")




def main() -> None:
    """Main entry point. Process a specific model's judge collections."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Transform judge generations into unified binary collections for a specific model."
    )
    parser.add_argument(
        "model",
        type=str,
        choices=["llama", "llama_default", "qwen", "qwen_default", "llama_8b_instruction", "llama_8b_extraction"],
        help="Model to process (llama, llama_default, qwen, qwen_default)"
    )
    parser.add_argument(
        "--judge-collections-root",
        type=Path,
        default=Path("judge_collections"),
        help="Root directory containing model subdirectories (default: judge_collections)"
    )
    parser.add_argument(
        "--processed-collections-root",
        type=Path,
        default=Path("processed_collections"),
        help="Output root directory for processed collections (default: processed_collections)"
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Optional pattern to filter judge types (e.g., 'Voting' to only process judge types containing 'Voting')"
    )
    
    args = parser.parse_args()
    
    process_single_model(
        model_name=args.model,
        judge_collections_root=args.judge_collections_root,
        processed_collections_root=args.processed_collections_root,
        pattern=args.pattern
    )


if __name__ == "__main__":
    main()



