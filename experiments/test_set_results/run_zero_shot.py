from dataclasses import dataclass, field
import sys
from pathlib import Path
import tyro
import numpy as np
import os

from utils.set_random_seed import set_random_seed
from utils.pipelines.ZeroShotBaselinePipeline import ZeroShotBaselinePipeline
from pathlib import Path
root = Path(__file__).parent.parent.parent


set_random_seed(42)
root = Path(__file__).parent.parent.parent
@dataclass
class ZeroShotConfig:
    '''
    Configuration class for the experiment.
    '''

    model: str = "None"
    '''Tag for the experiment section.'''
    log: bool = True    
    # RAG Based configs Config Fields
    wiki_path: str = "data/wiki_dump2018_nq_open/processed/wiki.feather"
    '''Path to the wiki dataset file.'''
    questions_path: str = "data/nq_open_gold/processed/test.feather"
    '''Path to the questions dataset file.'''
    language_model_path: str = "models/llms/Llama-3.2-3B-Instruct"
    '''Path to the language model.'''
    project_log: str = "run_test_set_nq"
    '''Project log name fgor wandb'''
    model_run_id: str = "zero_shot"
    '''ID of the model run.'''
    batch_size: int = 8
    '''Size of inferences to be done at the same time'''
    attn_implementation: str = "sdpa"
    '''Attn implementation for the desired gpu, recommended default "sdpa" and "flash_attention_2" when possible'''
    start_idx: int = 0
    '''Starting index for the questions to be processed.'''
    end_idx: int|None = None
    '''Ending index for the questions to be processed. None means process all questions.'''

    
    # Pre-collections Config Fields
    instruction: str = "You are given a question and you MUST try to give a real SHORT ANSWER in 5 tokens, you can use the available documents but if they are not helpful, try to answer without them"
    '''Instruction for the pre-collections step.'''
    lm_configs: dict[str, float|int] = field(default_factory=lambda: {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_new_tokens": 10,
        })
    tags: list[str] = field(default_factory=list)
    '''List of tags for the experiment.'''




def initiate_pipeline(args: ZeroShotConfig) -> ZeroShotBaselinePipeline:
    """
    Initiates the baseline pipeline with the provided arguments.
    
    Args:
        args (ParametersConfig): The configuration parameters for the pipeline.
        seed (int): The random seed for reproducibility.
    
    Returns:
        ZeroShotBaselinePipeline: An instance of the baseline pipeline.
    """

    return ZeroShotBaselinePipeline(
        questions_path=args.questions_path,
        language_model_path=args.laguage_model_path,
        lm_configs=args.lm_configs,
        model_run_id=f"zero_shot_{args.model}",
        instruction=args.instruction,
        root_path=f"{args.model}",
        project_log=args.project_log,
        tags = args.tags,
        batch_size = args.batch_size,
        log=args.log,
        attn_implementation=args.attn_implementation
    )


if __name__ == "__main__":
    args = tyro.cli(ZeroShotConfig)
    args.tags.append("zero_shot")
    args.questions_path = f"{root}/{args.questions_path}"
    args.laguage_model_path = f"{root}/{args.laguage_model_path}"
    args.wiki_path = f"{root}/{args.wiki_path}"


    pipeline = initiate_pipeline(args)
    pipeline.generate_inferences(args.start_idx, args.end_idx)