
import datetime
import os
from git import Optional
import polars as pl
from dmcr.models import GenericInstructModelHF
import json

import torch
import wandb
from utils.set_random_seed import set_random_seed



class ZeroShotBaselinePipeline:

    def __init__(self,
                questions_path: str,
                laguage_model_path: str,
                root_path: str = ".",
                lm_configs: dict[str, float] = {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_length": 2048.0,
                    "max_new_tokens": 10.0,
                    "num_return_sequences": 5.0
                },
                tags: list[str] = [],
                log: bool = False,
                project_log: str = "dmcr_baseline",
                model_run_id: str = "llama-3.2-3b-instruct",
                instruction: str = "You are given a question and you MUST try to give a real SHORT ANSWER in 5 tokens",
                seed: Optional[int] = None):
    

        self.questions_path = questions_path
        self.laguage_model_path = laguage_model_path
        self.lm_configs = lm_configs
        self.model_run_id = model_run_id
        self.instruction = instruction
        self.seed = seed
        self.root_path = root_path
        self.tags = tags
        self.log = log
        self.project_log = project_log
        
        if seed:
            set_random_seed(seed)

    def generate_inferences(self):

        ## Setup variables
        """
        Make baseline generations for a specific set and save them to the "generations" folder with "baseline_generations.json"

        Parameters:
        None

        Returns:
        None
        """
        if not os.path.exists(f"{self.root_path}/generations"):
            os.mkdir(f"{self.root_path}/generations")

        ## Setup variables
        questions = pl.read_ipc(self.questions_path)

        model = GenericInstructModelHF(self.laguage_model_path)
        model_configs = self.lm_configs
        generations = {}

        if self.log:
            start_time = datetime.datetime.now()
            wandb.init(
                project=self.project_log,
                name=f"Baselien_{self.model_run_id}",
                id = f"Baseline_{self.model_run_id}_{start_time.strftime('%Y-%m-%d_%H-%M-%S')}",
                config={
                    "gpu": f"{torch.cuda.get_device_name(0)}",
                    "questions_path": self.questions_path,

                },
                tags = self.tags.extend(["generations"]),
            )
            artifact = wandb.Artifact(name="generations", type="json", description="Baseline generations data")
            

        ## Iterate questions
        for idx in range(len(questions)):



            ## Generate prompt
            prompt = f"Question: {questions[idx]['question'].to_numpy().flatten()[0]}\nAnswer: "

            ## Generate output
            outputs = model.run(
                prompt, 
                instruction=self.instruction, 
                config_params=model_configs
            )

            generations[f"{idx}"] = [str(out["generated_text"]) for out in outputs]

            if self.model_run_id is None:
                path = f"{self.root_path}/generations/baseline_generations.json"
                with open(path, "w") as f:
                    json.dump(generations, f)
            
            else:
                path = f"{self.root_path}/generations/{self.model_run_id}_baseline_generations.json"
                with open(path, "w") as f:
                    json.dump(generations, f)
            
        if self.log:
            artifact.add_file(path)
            wandb.log_artifact(artifact)
            wandb.log({
                "end_time": datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
                "duration": (datetime.datetime.now() - start_time).total_seconds(),
            })
            wandb.finish()

        

        return
