from dataclasses import dataclass
import os
from typing import Literal
import numpy as np
from utils.judges import PromptJudge, PairwiseJudge, ContextJudge
from utils.judges.services import get_generations, get_wiki_context
from dmcr.models import GenericVLLMBatch
import polars as pl
import json
import tyro
from pathlib import Path
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

    judge_type: Literal["PromptJudge", "PairwiseJudge", "ContextJudge"] = "PromptJudge"
    '''Type of judge to use. Options: "PromptJudge", "PairwiseJudge", "ContextJudge"'''

    prompt_instruction: Literal["naive_judge", "recall_naive_judge", "pairwise_judge", "faithfulness_judge"] = "naive_judge"
    '''Prompt instructions to use for the judge. Options: "naive_judge", "recall_naive_judge", "pairwise_judge", "faithfulness_judge" from the "judge_prompts.json'''

    pairwise_rag: bool = False
    '''Whether to use RAG-based retrieval for the pairwise judge. Only applicable if judge_type is "PairwiseJudge".'''

    saving_path: str = "judge_collections/default.feather"
    '''Path to save the results.'''

    regex_pattern: str =  r'\[(\d+)\]'
    '''Regex pattern to extract scores from model output.'''

    mode: str = "train"
    '''Whether to run on "train" or "test" pre-collections.'''
    
    pre_collections_path: str = "experiment_81/datamodels/pre_collections"
    '''Path to the pre-collections directory. Should contain "train" and "test"'''

def _get_context(key: str, indeces: np.ndarray) -> str:
    '''
    Helper function to get wiki context for the ContextJudge.
    '''
    wiki_path = f"{root}/data/wiki_dump2018_nq_open/processed/wiki.feather"
    retrieval_path = "experiment_81/retrieval/rag_retrieval_indexes.json"
    return get_wiki_context.get_context(
        path_documents=wiki_path,
        path_retrievals=retrieval_path,
        retrieval_key=key,
        indeces=indeces
    )

def _get_zero_shot_generations() -> list[str]:
    '''
    Helper function to get zero-shot generations for the PromptJudge.
    '''
    generations_path = "experiment_81/generations/zeroshot.json"
    return get_generations(
        collection_path=generations_path,
    )

def _get_rag_generations() -> list[str]:
    '''
    Helper function to get RAG-based generations for the PromptJudge.
    '''
    generations_path = "experiment_81/generations/rag.json"
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

def _run_judge_evaluations(
    config: JudgeCollectionsConfig,
    judge: PromptJudge | PairwiseJudge | ContextJudge,
    judge_prompts: dict,
    all_predictions: list[str],
    all_questions: list[str],
    all_references: list[str] = None,
    all_contexts: list[str] = None
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
        _ref =  _get_rag_generations()
        all_references =  (len(all_predictions)//len(_ref)) * _ref
        scores = judge_instance.judge(
            generations=all_predictions,
            questions=all_questions,
            references=all_references,
            regex_pattern=config.regex_pattern
        )

    elif config.judge_type == "PairwiseJudge":
        _ref =  _get_zero_shot_generations()
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
    Main pipeline to run judge evaluations on pre-collections.
    
    This function:
    1. Loads pre-collections dataframe
    2. Flattens the predicted_output column (which is list[str])
    3. Runs judge evaluations on each prediction
    4. Creates an evaluation column with the results
    5. Explodes rows if multiple evaluation values exist
    6. Saves the result
    '''
    print(f"Loading pre-collections from {config.pre_collections_path}...")
    pre_collections = _get_pre_collections(f"{config.pre_collections_path}/{config.mode}")
    questions = pl.read_ipc("experiment_81/questions.feather").with_row_index("test_idx")

    
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

    ## ASSUMES ONLY ONE GENRATION PER QUESTIONS 
    all_predictions = [str(pred[0]) for pred in pre_collections["predicted_output"].to_list()]


    
    print(f"Running {config.judge_type} evaluations on {len(all_predictions)} predictions...")
    judge_class = JDUGES[config.judge_type]
    scores = _run_judge_evaluations(
        config,
        judge_class,
        judge_prompts,
        all_predictions,
        all_questions,
        all_references,
        all_contexts
    )
    _v = scores[0]

    results_df = pre_collections.with_columns(
        pl.Series("evaluation", scores)
    ).explode("evaluation")

    print(f"Saving results to {config.saving_path}...")
    results_df.write_ipc(config.saving_path)
    print("Done.")



if __name__ == "__main__":
    config = tyro.cli(JudgeCollectionsConfig)
    run_judge_collections_pipeline(config)

    