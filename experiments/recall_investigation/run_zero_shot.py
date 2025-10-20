from dataclasses import dataclass, field
import sys
from pathlib import Path
from click import Option
import tyro
import numpy as np
import os

from utils.set_random_seed import set_random_seed
from utils.pipelines.ZeroShotBaselinePipeline import ZeroShotBaselinePipeline
from pathlib import Path
from typing import Literal, Optional

root = Path(__file__).parent.parent.parent
SubsizeLiteral = Literal[32, 64, 128]
KLiteral = Literal[4,8,16]



set_random_seed(42)
root = Path(__file__).parent.parent.parent
@dataclass
class ZeroShotConfig:
    '''
    Configuration class for the experiment.
    '''

    size_folder: SubsizeLiteral = 128
    '''Subset size and folder of subset parameter'''
    k: KLiteral = 4
    '''size of context and subfolder identifier'''
    log: bool = True    
    questions_path: str = "questions_500_10_dev.feather"
    '''Path to the questions dataset file.'''
    language_model_path: str = "models/Llama-3.2-3B-Instruct"
    '''Path to the language model.'''
    project_log: str = "recall_investigation"
    '''Project log name fgor wandb'''
    model_run_id: str = "zero_shot"
    '''ID of the model run.'''
    batch_size: int = 500
    '''Size of inferences to be done at the same time'''
    attn_implementation: str = "sdpa"
    '''Attn implementation for the desired gpu, recommended default "sdpa" and "flash_attention_2" when possible'''
    thinking: bool = False
    '''Whether to enable the thinking mode in the model.'''

    
    # Pre-collections Config Fields
    instruction: str = "You are given a question and you MUST try to give a real SHORT ANSWER in 5 tokens"
    '''Instruction for the pre-collections step.'''
    lm_configs: dict[str, float|int] = field(default_factory=lambda: {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_new_tokens": 10,
        })
    tags: Optional[list[str]] = field(default_factory=list) 
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
        language_model_path=args.language_model_path,
        lm_configs=args.lm_configs,
        model_run_id=f"zero_shot",
        instruction=args.instruction,
        root_path=f"subset_{args.size_folder}/k_{args.k}",
        project_log=args.project_log,
        tags = args.tags,
        batch_size = args.batch_size,
        log=args.log,
        attn_implementation=args.attn_implementation,
        thinking=args.thinking
    )


if __name__ == "__main__":
    args = tyro.cli(ZeroShotConfig)
    args.tags.append("zero_shot")
    args.language_model_path = f"{root}/{args.language_model_path}"


    pipeline = initiate_pipeline(args)
    pipeline.generate_inferences()