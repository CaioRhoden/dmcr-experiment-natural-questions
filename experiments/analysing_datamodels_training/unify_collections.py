### The goal of this script is to get the train and test collections of the "instruction_1" for each experiment from
### "debbuging_rag_recall_vllm_vs_hfpipeline/validating_datamodels" and unify them into a single collection for train and test per experiment and per evaluator (Rouge-l and Judge)""

import polars as pl
import os
from pathlib import Path
from typing import List

def get_experiment_collections(base_path: Path, experiment: str, split: str, evaluator: str) -> List[Path]:
    """
    Get all collection files for a specific experiment, split, and evaluator.
    
    Args:
        base_path: Base path to validating_datamodels folder
        experiment: Experiment name (e.g., 'experiment_1')
        split: 'train' or 'test'
        evaluator: 'Judge' or 'Rouge-L'
    
    Returns:
        List of paths to collection files
    """
    collections_path = f"{base_path}/{experiment}/instruction_1/datamodels/collections/{split}"

    
    # Find all feather files matching the pattern
    pattern = f"instruction-1_experiment-{experiment.split('_')[1]}_evaluator-{evaluator}_*.feather"
    if evaluator == "multi":
        pattern = f"evaluator-Judge_multi_*.feather"
    collection_files = sorted(Path(collections_path).glob(pattern))
    
    return collection_files

def unify_collections_for_experiment(base_path: Path, experiment: str, split: str, evaluator: str, output_dir: Path):
    """
    Unify all collections for a specific experiment, split, and evaluator.
    
    Args:
        base_path: Base path to validating_datamodels folder
        experiment: Experiment name
        split: 'train' or 'test'
        evaluator: 'Judge' or 'Rouge-L'
        output_dir: Directory to save unified collections
    """
    collection_files = get_experiment_collections(base_path, experiment, split, evaluator)
    if not collection_files:
        print(f"No collections found for {experiment} - {split} - {evaluator}")
        return
    
    print(f"\nUnifying {len(collection_files)} collections for {experiment} - {split} - {evaluator}")
    
    # Read and concatenate all collections
    dfs = []
    for file_path in collection_files:
        print(f"  Reading: {file_path.name}")
        df = pl.read_ipc(file_path)
        dfs.append(df)
    
    # Concatenate all dataframes
    unified_df = pl.concat(dfs, how="vertical")
    
    # Create output filename
    output_filename = f"{split}.feather"
    if evaluator == "Rouge-L":
        output_dir = f"{output_dir}/groundtruth/{experiment}"
    elif evaluator == "multi":
        output_dir = f"{output_dir}/multi/{experiment}"
    else:
        output_dir = f"{output_dir}/judge/{experiment}"
    output_path = f"{output_dir}/{output_filename}"
    
    # Save unified collection
    unified_df.write_ipc(output_path)
    print(f"  Saved: {output_path} (shape: {unified_df.shape})")

def unify_collections():
    """
    Main function to unify all collections for all experiments.
    """
    # Define paths
    validating_datamodels_path = "../debugging_rag_recall_vllm_vs_hfpipeline/validating_datamodels"
    
    # Create output directory
    output_dir = "collections"
    
    # Define experiments and evaluators
    experiments = ["experiment_1", "experiment_4", "experiment_54", "experiment_61", "experiment_73"]
    splits = ["train", "test"]
    evaluators = ["multi"]
    
    print(f"Base path: {validating_datamodels_path}")
    print(f"Output directory: {output_dir}")
    print(f"\nProcessing {len(experiments)} experiments with {len(splits)} splits and {len(evaluators)} evaluators")
    print("=" * 80)
    
    # Process each combination
    for experiment in experiments:
        print(f"\n{'=' * 80}")
        print(f"Processing {experiment}")
        print(f"{'=' * 80}")
        
        for split in splits:
            for evaluator in evaluators:
                unify_collections_for_experiment(
                    base_path=validating_datamodels_path,
                    experiment=experiment,
                    split=split,
                    evaluator=evaluator,
                    output_dir=output_dir
                )
    
    print(f"\n{'=' * 80}")
    print("Done! All collections unified.")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

if __name__ == "__main__":
    unify_collections()