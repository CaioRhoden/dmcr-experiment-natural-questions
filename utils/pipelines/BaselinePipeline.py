
import os
import polars as pl
from dmcr.models import GenericInstructModelHF
import json
from utils.set_random_seed import set_random_seed



class BaselinePipeline:

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
                model_run_id: str = "llama-3.2-3b-instruct",
                instruction: str = "You are given a question and you MUST try to give a real SHORT ANSWER in 5 tokens, you can use the available documents but if they are not helpful, try to answer without them",
                seed: int = 42):
    

        self.questions_path = questions_path
        self.laguage_model_path = laguage_model_path
        self.lm_configs = lm_configs
        self.model_run_id = model_run_id
        self.instruction = instruction
        self.seed = seed
        self.root_path = root_path
        set_random_seed(self.seed)

    def generate_inferences(self):

        ## Setup variables
        """
        Make baseline generations for a specific set and save them to the "generations" folder with "baseline_generations.json"

        Parameters:
        None

        Returns:
        None
        """

        os.mkdir(f"{self.root_path}/generations")

        ## Setup variables
        questions = pl.read_ipc(self.questions_path)

        model = GenericInstructModelHF(self.laguage_model_path)
        model_configs = self.lm_configs
        generations = {}

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
                with open(f"{self.root_path}/generations/baseline_generations.json", "w") as f:
                    json.dump(generations, f)
            
            prefix = self.model_run_id
            with open(f"{self.root_path}/generations/{prefix}_baseline_generations.json", "w") as f:
                json.dump(generations, f)

        

        return
