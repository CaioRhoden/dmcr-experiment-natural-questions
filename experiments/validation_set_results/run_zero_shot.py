from dataclasses import dataclass, field
import sys
from pathlib import Path
import tyro
import numpy as np
import os

from utils.set_random_seed import set_random_seed
from utils.pipelines.ZeroShotBaselinePipeline import ZeroShotBaselinePipeline


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
    questions_path: str = "../../data/nq_open_gold/processed/dev.feather"
    '''Path to the questions dataset file.'''
    laguage_model_path: str = "models/llms/Llama-3.2-3B-Instruct"
    '''Path to the language model.'''
    project_log: str = "nq_experiment_datamodels_training_window"
    '''Project log name fgor wandb'''
    model_run_id: str = "test_experiment"
    '''ID of the model run.'''

    
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
        laguage_model_path=args.laguage_model_path,
        lm_configs=args.lm_configs,
        model_run_id=f"zero_shot_{args.model}",
        instruction=args.instruction,
        root_path=f"{args.model}_zero_shot",
        project_log=args.project_log,
        tags = args.tags,

        log=args.log,    )


if __name__ == "__main__":
    args = tyro.cli(ZeroShotConfig)
    pipeline = initiate_pipeline(args)
    pipeline.generate_inferences()