import argparse
from dataclasses import dataclass, field
from utils.pipelines.BaselinePipeline import BaselinePipeline
from utils.set_random_seed import set_random_seed
import tyro
from pathlib import Path


root = Path(__file__).parent.parent.parent.parent.parent


@dataclass
class ParametersConfig:
    '''
    Configuration class for the experiment.
    '''

    ## Run config
    seed: int = 0
    '''Random index seed for reproducibility.'''
    
    # RAG Based configs Config Fields
    questions_path: str = "../50_test.feather"
    '''Path to the questions dataset file.'''
    laguage_model_path: str = "models/llms/Llama-3.2-3B-Instruct"
    '''Path to the language model.'''
    project_log: str = "nq_experiment_subset_sizes"
    '''Project log name fgor wandb'''
    model_run_id: str = "150_proportion"
    '''ID of the model run.'''
    
    # Pre-collections Config Fields
    instruction: str = "You are given a question and you MUST try to give a real SHORT ANSWER in 5 tokens, you can use the available documents but if they are not helpful, try to answer without them"
    '''Instruction for the pre-collections step.'''
    lm_configs: dict[str, float] = {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_length": 2048.0,
            "max_new_tokens": 10.0,
            "num_return_sequences": 5.0
        }


if __name__ == "__main__":

    args = tyro.cli(ParametersConfig)

    args.laguage_model_path = f"{root}/{args.laguage_model_path}"

    pipeline = BaselinePipeline(
        questions_path=args.questions_path,
        laguage_model_path=args.laguage_model_path,
        lm_configs=args.lm_configs,
        model_run_id="baseline_qwen",
        instruction=args.instruction,
        seed=args.seed
    )
