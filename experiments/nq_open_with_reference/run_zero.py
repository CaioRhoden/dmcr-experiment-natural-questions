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
    root_path: str = "runs"

        
        
if __name__ == "__main__":
    args = tyro.cli(ZeroShotConfig)

    ## Load dataclass as args
    baseline = ZeroShotBaselinePipeline(
        questions_path=f"{root}/data/nq_open/processed/dev.feather",
        language_model_path=f"{root}/models/Llama-3.2-3B-Instruct",
        lm_configs={
            "temperature": 0.7,
            "top_p": 0.9,
            "max_new_tokens": 15,
        },
        model_run_id="zeroshot",
        instruction="You are given a question and you MUST try to give a real SHORT ANSWER in 5 tokens",
        root_path=args.root_path,
        project_log="nq_open_reference",
        tags = ["zeroshot"],
        log=True,
        seed=42,
        batch_size=3610
    )

    baseline.generate_inferences()