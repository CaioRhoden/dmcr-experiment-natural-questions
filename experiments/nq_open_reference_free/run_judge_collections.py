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

MODELS_PATH = f"{root}/models/Llama-3.2-3B-Instruct"

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

    pairwise_rag: bool = False
    '''Whether to use RAG-based retrieval for the pairwise judge. Only applicable if judge_type is "PairwiseJudge".'''

    saving_dir: str = "judge_collections"
    '''Path to save the results.'''

    regex_pattern: str =  r'\[(\d+)\]'
    '''Regex pattern to extract scores from model output.'''

    mode: str = "train"
    '''Whether to run on "train" or "test" pre-collections.'''

    batch_size: int = 541500
    '''Batch size for processing predictions. Adjust based on available memory.'''

    n_generations: int = 1
    '''Number of generations by judge'''

    start_idx: int = 0
    '''Starting index for predictions to process. Must be divisible by the number of questions.'''

    end_idx: int | None = 7220000
    '''Ending index for predictions to process. Must be divisible by the number of questions. If None, processes all predictions.'''
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

def _get_batch_filename(config: JudgeCollectionsConfig, batch_idx: int) -> str:
    '''
    Generate a filename for a batch.
    '''
    return os.path.join(config.saving_dir, f"batch_{batch_idx:04d}.feather")

def _get_combined_filename(config: JudgeCollectionsConfig) -> str:
    '''
    Generate the combined filename.
    '''
    return os.path.join(config.saving_dir, f"results_combined.feather")

def _combine_batch_files(config: JudgeCollectionsConfig, num_batches: int) -> pl.DataFrame:
    '''
    Combine all batch files into a single DataFrame.
    
    Args:
        config: Configuration object
        num_batches: Number of batches to combine
        
    Returns:
        Combined DataFrame
    '''
    print("Combining all batches...")
    combined_results = []
    
    for batch_idx in range(num_batches):
        batch_file = _get_batch_filename(config, batch_idx)
        if os.path.exists(batch_file):
            combined_results.append(pl.read_ipc(batch_file))
            print(f"  Loaded batch {batch_idx}")
        else:
            print(f"  Warning: Batch file {batch_file} not found")
    
    if combined_results:
        return pl.concat(combined_results, how="vertical")
    else:
        raise RuntimeError("No batch files found to combine")

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
        path=MODELS_PATH,
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


    ## Guarantee that it's possible to compute the Pairwise judges
    assert config.batch_size%len(questions) == 0, f"Batch size {config.batch_size} must be a multiple of the number of questions {len(questions)} to ensure proper alignment for PairwiseJudge."
    assert config.start_idx % len(questions) == 0, f"start_idx {config.start_idx} must be divisible by the number of questions {len(questions)}."
    assert config.end_idx % len(questions) == 0, f"end_idx {config.end_idx} must be divisible by the number of questions {len(questions)}."
    assert config.start_idx < config.end_idx, f"start_idx {config.start_idx} must be less than end_idx {config.end_idx}."

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
    
    ## set all questions - one question per prediction (matched via test_idx)
    all_questions = pre_collections["question"].to_list()

    ## ASSUMES ONLY ONE GENERATION PER QUESTIONS 
    all_predictions = [str(pred[0]) for pred in pre_collections["predicted_output"].to_list()]


    # Set end_idx to total predictions if not specified
    if config.end_idx is None:
        config.end_idx = len(all_predictions)


    # Slice predictions to the specified range
    sliced_predictions = all_predictions[config.start_idx:config.end_idx]
    sliced_questions = all_questions[config.start_idx:config.end_idx]
    sliced_pre_collections = pre_collections.slice(config.start_idx, config.end_idx - config.start_idx)

    # Calculate number of batches
    num_batches = (len(sliced_predictions) + config.batch_size - 1) // config.batch_size
    print(f"Processing {len(sliced_predictions)} predictions (from index {config.start_idx} to {config.end_idx}) in {num_batches} batch(es) of size {config.batch_size}")
    
    # Log initial config to wandb
    wandb.log({
        "total_predictions": len(sliced_predictions),
        "num_batches": num_batches,
        "batch_size": config.batch_size,
        "start_idx": config.start_idx,
        "end_idx": config.end_idx,
    })
    
    # Process predictions in batches
    batch_start_time = time.time()
    for batch_idx in range(num_batches):
        start_idx = batch_idx * config.batch_size
        
        # Adjust batch size for last batch: use remaining items if less than batch_size,
        # but ensure divisibility by number of questions
        remaining = len(sliced_predictions) - start_idx
        effective_batch_size = config.batch_size
        if remaining < config.batch_size:
            effective_batch_size = (remaining // len(questions)) * len(questions)
        
        end_idx = start_idx + effective_batch_size
        
        batch_predictions = sliced_predictions[start_idx:end_idx]
        batch_questions = sliced_questions[start_idx:end_idx]
        batch_pre_collections = sliced_pre_collections.slice(start_idx, end_idx - start_idx)
        
        print(f"\nBatch {batch_idx + 1}/{num_batches} (samples {config.start_idx + start_idx}-{config.start_idx + end_idx - 1})...")
        
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
        batch_file = _get_batch_filename(config, batch_idx)
        print(f"  Saving batch to {batch_file}...")
        batch_results.write_ipc(batch_file)
        
        # Log batch progress to wandb
        batch_elapsed_time = time.time() - batch_start_time
        wandb.log({
            "batch_idx": batch_idx + 1,
            "batch_num_batches": num_batches,
            "batch_elapsed_time_sec": batch_elapsed_time,
            "batch_size_processed": len(batch_predictions),
            "predictions_processed": config.start_idx + end_idx,
        })
    
    # Combine all batches
    print("\n" + "="*60)
    combined_df = _combine_batch_files(config, num_batches)
    
    # Save combined results
    combined_path = _get_combined_filename(config)
    print(f"Saving combined results to {combined_path}...")
    combined_df.write_ipc(combined_path)
    
    # Log final results to wandb
    total_elapsed_time = time.time() - pipeline_start_time
    wandb.log({
        "total_elapsed_time_sec": total_elapsed_time,
        "total_batches_processed": num_batches,
        "total_predictions_processed": len(sliced_predictions),
    })
    
    print(f"Total time elapsed: {total_elapsed_time:.2f} seconds")
    
    # Finish wandb
    wandb.finish()



if __name__ == "__main__":
    config = tyro.cli(JudgeCollectionsConfig)
    run_judge_collections_pipeline(config)

    