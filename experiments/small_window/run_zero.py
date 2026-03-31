from pathlib import Path
import tyro
from utils.pipelines.ZeroShotBaselinePipeline import ZeroShotBaselinePipeline
from utils.set_random_seed import set_random_seed
from typing import Literal
from dataclasses import dataclass, field


set_random_seed(42)
root = Path(__file__).parent.parent.parent
@dataclass
class ZeroShotConfig:
    log: bool = True    
    seed: Literal[1, 4, 54, 61, 73] = 1
    '''Random seed for reproducibility based on the previous random generated'''
    questions_path: str = "data/nq_open_gold/processed/test.feather"
    '''Path to the questions dataset file.'''
    language_model_path: str = "models/Llama-3.2-3B-Instruct"
    '''Path to the language model.'''
    project_log: str = "small_window"
    '''Project log name for wandb'''

        
        
if __name__ == "__main__":
    args = tyro.cli(ZeroShotConfig)
    args.questions_path = f"runs/experiment_{args.seed}/questions.feather"
    args.language_model_path = f"{root}/{args.language_model_path}"

    ## Load dataclass as args
    baseline = ZeroShotBaselinePipeline(
        questions_path=args.questions_path,
        language_model_path=args.language_model_path,
        lm_configs={
            "temperature": 0.7,
            "top_p": 0.9,
            "max_new_tokens": 10,
        },
        model_run_id="zeroshot",
        instruction="You are given a question and you MUST try to give a real SHORT ANSWER in 5 tokens",
        root_path=f"runs/experiment_{args.seed}",
        project_log="judge_comparative",
        tags = ["zeroshot"],
        log=True,
        seed=42,
        batch_size=500
    )

    baseline.generate_inferences()