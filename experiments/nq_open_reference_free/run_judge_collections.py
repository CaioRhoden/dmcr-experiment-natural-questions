from dataclasses import dataclass
import os
from typing import Literal
import time
import numpy as np
from utils.judges import PromptJudge, PairwiseJudge, ContextJudge
from utils.judges.services import get_generations, get_wiki_context
from dmcr.models import GenericVLLMBatch
import polars as pl
import json
import tyro
from pathlib import Path
import wandb
root = Path(__file__).parent.parent.parent


JDUGES = {
    "PromptJudge": PromptJudge,
    "PairwiseJudge": PairwiseJudge,
    "ContextJudge": ContextJudge,
}

@dataclass
class JudgeCollectionsConfig:
    '''
    Configuration class for the Judge Collections experiment.
    '''

    judge_type: Literal["PromptJudge", "PairwiseJudge"] = "PromptJudge"
    '''Type of judge to use. Options: "PromptJudge", "PairwiseJudge"'''

    prompt_instruction: Literal["naive_judge", "recall_naive_judge", "pairwise_judge", "recall_pairwise_judge"] = "naive_judge"
    '''Prompt instructions to use for the judge. Options: "naive_judge", "recall_naive_judge", "pairwise_judge", "recall_pairwise_judge" from the "judge_prompts.json'''

    model_path: str = "models/Llama-3.2-3B-Instruct"
    '''Path to the language model to be used as a judge'''

    pairwise_rag: bool = False
    '''Whether to use RAG-based retrieval for the pairwise judge. Only applicable if judge_type is "PairwiseJudge".'''

    saving_dir: str = "judge_collections"
    '''Path to save the results. Batch files are saved as batch_{collection_start:04d}_{collection_end:04d}.feather.'''

    regex_pattern: str =  r'\[(\d+)\]'
    '''Regex pattern to extract scores from model output.'''

    mode: str = "train"
    '''Whether to run on "train" or "test" pre-collections.'''

    batch_size: int = 250
    '''Number of collections to process per batch file. Each batch file will contain predictions from batch_size collections.'''

    n_generations: int = 1
    '''Number of generations by judge'''

    start_collection_idx: int = 0
    '''Starting collection index to process (inclusive). Collections are ordered by collection_idx.'''

    end_collection_idx: int | None = None
    '''Ending collection index to process (inclusive). If None, processes all collections from start_collection_idx to the end.'''
    
    runs_path: str = f"runs"
    
def _get_zero_shot_generations(runs_path: str) -> list[str]:
    '''
    Helper function to get zero-shot generations for the PromptJudge.
    '''
    generations_path = f"{runs_path}/generations/zeroshot.json"
    return get_generations(
        collection_path=generations_path,
    )

def _get_rag_generations(runs_path: str) -> list[str]:
    '''
    Helper function to get RAG-based generations for the PromptJudge.
    '''
    generations_path = f"{runs_path}/generations/rag.json"
    return get_generations(
        collection_path=generations_path,
    )

def _get_pre_collections(pre_collections_path: str) -> pl.DataFrame:
    '''
    Helper function to get pre-collections DataFrame.
    '''
    pre_collections = []
    for file in os.listdir(pre_collections_path):
        if file.endswith(".feather"):
            pre_collections.append(pl.read_ipc(os.path.join(pre_collections_path, file)))

    return pl.concat(pre_collections, how="vertical").sort("collection_idx", "test_idx")

def _load_judge_prompts() -> dict:
    '''
    Load judge prompts from judge_prompts.json.
    '''
    prompts_path = Path(__file__).parent / "judge_prompts.json"
    if prompts_path.exists():
        with open(prompts_path, 'r') as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"Judge prompts file not found at {prompts_path}. Please create a judge_prompts.json file with the necessary prompts.")

def _get_batch_filename(config: JudgeCollectionsConfig, collection_start_idx: int, collection_end_idx: int) -> str:
    '''
    Generate a filename for a batch based on collection indices.
    
    Args:
        config: Configuration object
        collection_start_idx: Starting collection index (inclusive)
        collection_end_idx: Ending collection index (inclusive)
    '''
    return os.path.join(config.saving_dir, f"batch_{collection_start_idx:04d}_{collection_end_idx:04d}.feather")

# def _get_combined_filename(config: JudgeCollectionsConfig) -> str:
#     '''
#     Generate the combined filename.
#     '''
#     return os.path.join(config.saving_dir, f"results_combined.feather")

# def _combine_batch_files(config: JudgeCollectionsConfig) -> pl.DataFrame:
#     '''
#     Combine all batch files into a single DataFrame.
    
#     Args:
#         config: Configuration object
        
#     Returns:
#         Combined DataFrame
#     '''
#     print("Combining all batches...")
#     combined_results = []
    
#     # Find all batch files matching the pattern batch_*.feather
#     batch_pattern = os.path.join(config.saving_dir, "batch_*.feather")
#     batch_files = sorted([f for f in os.listdir(config.saving_dir) if f.startswith("batch_") and f.endswith(".feather")])
    
#     if not batch_files:
#         raise RuntimeError(f"No batch files found matching pattern '{batch_pattern}'")
    
#     for batch_file_name in batch_files:
#         batch_file_path = os.path.join(config.saving_dir, batch_file_name)
#         combined_results.append(pl.read_ipc(batch_file_path))
#         print(f"  Loaded {batch_file_name}")
    
#     if combined_results:
#         return pl.concat(combined_results, how="vertical")
#     else:
#         raise RuntimeError("No batch files found to combine")

def _run_judge_evaluations(
    config: JudgeCollectionsConfig,
    judge: PromptJudge | PairwiseJudge | ContextJudge,
    judge_prompts: dict,
    all_predictions: list[str],
    all_questions: list[str],
    all_references: list[str] = None,
) -> list[list[float]]:
    '''
    Run judge evaluations on the predictions.
    
    Returns:
        List of lists of scores (one list per prediction).
    '''
    # Initialize model
    model = GenericVLLMBatch(
        path=f"{root}/{config.model_path}",
        vllm_kwargs={
            "max_model_len": 32768,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.9
        }
    )
    
    # Initialize judge
    judge_instance = judge()
    judge_instance.init(
        prompt=judge_prompts[config.prompt_instruction],
        model=model,
        model_configs={
            "temperature": 0.5,
            "top_p": 0.9,
            "max_new_tokens": 2048,
            "n": config.n_generations,
        }
    )
    
    # Run judge based on type
    if config.judge_type == "PromptJudge":
        scores = judge_instance.judge(
            generations=all_predictions,
            questions=all_questions,
            regex_pattern=config.regex_pattern
        )

    elif config.judge_type == "PairwiseJudge" and config.pairwise_rag:
        _ref =  _get_rag_generations(config.runs_path)
        all_references =  (len(all_predictions)//len(_ref)) * _ref
        scores = judge_instance.judge(
            generations=all_predictions,
            questions=all_questions,
            references=all_references,
            regex_pattern=config.regex_pattern
        )

    elif config.judge_type == "PairwiseJudge":
        _ref =  _get_zero_shot_generations(config.runs_path)
        all_references =  (len(all_predictions)//len(_ref)) * _ref
        scores = judge_instance.judge(
            generations=all_predictions,
            questions=all_questions,
            references=all_references,
            regex_pattern=config.regex_pattern
        )
    # elif config.judge_type == "ContextJudge":
    #     scores = judge_instance.judge(
    #         generations=all_predictions,
    #         questions=all_questions,
    #         contexts=all_contexts,
    #         regex_pattern=config.regex_pattern
    #     )
    
    return scores

def run_judge_collections_pipeline(config: JudgeCollectionsConfig):
    '''
    Main pipeline to run judge evaluations on pre-collections with batch processing.
    
    This function:
    1. Loads pre-collections dataframe
    2. Flattens the predicted_output column (which is list[str])
    3. Processes predictions in batches to fit into memory
    4. Saves each batch with the judge evaluation results
    5. Combines all batches into a single "_combined" file
    '''
    # Create saving directory
    os.makedirs(config.saving_dir, exist_ok=True)
    
    # Initialize wandb
    logs_dir = Path(config.saving_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    wandb.init(
        project="nq_open_reference_free",
        dir=str(logs_dir),
        name=f"judge_collections_{config.judge_type}",
        config={
            "judge_type": config.judge_type,
            "prompt_instruction": config.prompt_instruction,
            "pairwise_rag": config.pairwise_rag,
            "mode": config.mode,
            "batch_size": config.batch_size,
            "n_generations": config.n_generations,
        }
    )


    
    # Track overall timing
    pipeline_start_time = time.time()
    
    pre_collections = _get_pre_collections(f"{config.runs_path}/datamodels/pre_collections/{config.mode}")
    questions = pl.read_ipc(f"{root}/data/nq_open/processed/dev.feather").with_row_index("test_idx")

    # Get unique collection indices
    unique_collection_indices = sorted(pre_collections["collection_idx"].unique().to_list())
    max_collection_idx = unique_collection_indices[-1]
    
    # Validate and set collection index ranges
    assert config.start_collection_idx >= 0, f"start_collection_idx {config.start_collection_idx} must be >= 0."
    assert config.start_collection_idx <= max_collection_idx, f"start_collection_idx {config.start_collection_idx} must be <= max collection index {max_collection_idx}."
    
    if config.end_collection_idx is None:
        config.end_collection_idx = max_collection_idx
    
    assert config.end_collection_idx >= config.start_collection_idx, f"end_collection_idx {config.end_collection_idx} must be >= start_collection_idx {config.start_collection_idx}."
    assert config.end_collection_idx <= max_collection_idx, f"end_collection_idx {config.end_collection_idx} must be <= max collection index {max_collection_idx}."
    
    # Filter collection indices to the specified range
    selected_collection_indices = [c for c in unique_collection_indices if config.start_collection_idx <= c <= config.end_collection_idx]
    num_collections = len(selected_collection_indices)

    print("Loading judge prompts...")
    judge_prompts = _load_judge_prompts()
    
    all_references = None
    all_contexts = None

    # Join pre_collections with questions to get the correct question for each prediction
    pre_collections = pre_collections.join(
        questions.select(["test_idx", "question"]),
        on="test_idx",
        how="left"
    )
    
    # Calculate number of batches based on collections
    num_batches = (num_collections + config.batch_size - 1) // config.batch_size
    print(f"Processing {num_collections} collections (from {config.start_collection_idx} to {config.end_collection_idx}) in {num_batches} batch(es) of size {config.batch_size}")
    
    # Log initial config to wandb
    wandb.log({
        "start_collection_idx": config.start_collection_idx,
        "end_collection_idx": config.end_collection_idx,
        "num_collections": num_collections,
        "num_batches": num_batches,
        "batch_size_collections": config.batch_size,
        "n_generations": config.n_generations,
    })
    
    # Process collections in batches
    batch_start_time = time.time()
    for batch_idx in range(num_batches):
        # Determine the range of collections for this batch
        batch_collection_start_offset = batch_idx * config.batch_size
        batch_collection_end_offset = min((batch_idx + 1) * config.batch_size, num_collections)
        
        batch_collection_indices = selected_collection_indices[batch_collection_start_offset:batch_collection_end_offset]
        collection_start = batch_collection_indices[0]
        collection_end = batch_collection_indices[-1]
        
        # Filter pre_collections to only include rows for the selected collection indices
        batch_pre_collections = pre_collections.filter(
            pl.col("collection_idx").is_in(batch_collection_indices)
        )
        
        # Extract predictions and questions for this batch
        batch_predictions = [str(pred[0]) for pred in batch_pre_collections["predicted_output"].to_list()]
        batch_questions = batch_pre_collections["question"].to_list()
        
        print(f"\nBatch {batch_idx + 1}/{num_batches} (collections {collection_start}-{collection_end}, {len(batch_collection_indices)} collections, {len(batch_predictions)} predictions)...")
        
        judge_class = JDUGES[config.judge_type]
        print(f"  Running {config.judge_type} evaluations...")
        batch_scores = _run_judge_evaluations(
            config,
            judge_class,
            judge_prompts,
            batch_predictions,
            batch_questions,
            all_references        
        )
        
        # Create batch results dataframe with evaluation column
        batch_results = batch_pre_collections.with_columns(
            pl.Series("evaluation", batch_scores)
        ).explode("evaluation")
        
        # Save batch
        batch_file = _get_batch_filename(config, collection_start, collection_end)
        print(f"  Saving batch to {batch_file}...")
        batch_results.write_ipc(batch_file, compression="zstd")
        
        # Log batch progress to wandb
        batch_elapsed_time = time.time() - batch_start_time
        wandb.log({
            "batch_idx": batch_idx + 1,
            "batch_num_batches": num_batches,
            "batch_elapsed_time_sec": batch_elapsed_time,
            "batch_collection_start": collection_start,
            "batch_collection_end": collection_end,
            "batch_num_collections": len(batch_collection_indices),
            "batch_num_predictions": len(batch_predictions),
        })
    
    # Log final results to wandb
    total_elapsed_time = time.time() - pipeline_start_time
    wandb.log({
        "total_elapsed_time_sec": total_elapsed_time,
        "total_batches_processed": num_batches,
        "total_collections_processed": num_collections,
    })
    
    print(f"Total time elapsed: {total_elapsed_time:.2f} seconds")
    
    # Finish wandb
    wandb.finish()



if __name__ == "__main__":
    config = tyro.cli(JudgeCollectionsConfig)
    run_judge_collections_pipeline(config)

    