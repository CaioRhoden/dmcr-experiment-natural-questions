from pathlib import Path

from utils.pipelines.ZeroShotBaselinePipeline import ZeroShotBaselinePipeline
from utils.set_random_seed import set_random_seed


set_random_seed(42)
root = Path(__file__).parent.parent.parent
        
        
if __name__ == "__main__":

    ## Load dataclass as args
    baseline = ZeroShotBaselinePipeline(
        questions_path="experiment_81/questions.feather",
        language_model_path=f"{root}/models/Llama-3.2-3B-Instruct",
        lm_configs={
            "temperature": 0.7,
            "top_p": 0.9,
            "max_new_tokens": 10,
        },
        model_run_id="zeroshot",
        instruction="You are given a question and you MUST try to give a real SHORT ANSWER in 5 tokens",
        root_path="experiment_81",
        project_log="judge_comparative",
        tags = ["zeroshot"],
        log=True,
        seed=42,
        batch_size=10
    )

    baseline.generate_inferences()