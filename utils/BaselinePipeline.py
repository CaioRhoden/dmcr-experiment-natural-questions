
import polars as pl
import os
import torch
import numpy as np
import random
from dmcr.datamodels.setter.IndexBasedSetter import IndexBasedSetter
from dmcr.datamodels.setter.SetterConfig import IndexBasedSetterConfig
from dmcr.datamodels.pipeline import DatamodelsIndexBasedNQPipeline
from dmcr.datamodels.config import DatamodelIndexBasedConfig, LogConfig
from dmcr.models import GenericInstructModelHF
from dmcr.evaluators import Rouge_L_evaluator, SquadV2Evaluator
import datetime
import faiss
import json
from FlagEmbedding import FlagModel

import yaml


class BaselinePipeline:

    def __init__(self, config_path: str):
        config = yaml.safe_load(open(config_path, "r"))
        self.config = config["baseline_config"]
        self._verify_config_structure()

    def set_random_seed(self):

        np.random.seed(self.config["seed"])
        random.seed(self.config["seed"])
        pl.set_random_seed(self.config["seed"])

        # PyTorch
        torch.manual_seed(self.config["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config["seed"])
            torch.cuda.manual_seed_all(self.config["seed"])
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        if torch.cuda.is_available():
            print(f"Number of GPUs available: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")



    def setup(self):

        """
        Setup for the experiment.

        This function creates the "generations" folder if it doesn't exist.
        """

        if not os.path.exists("generations"):
            os.mkdir("generations")


    def generate_inferences(self):

        ## Setup variables
        """
        Make baseline generations for a specific set and save them to the "generations" folder with "baseline_generations.json"

        Parameters:
        None

        Returns:
        None
        """


        ## Setup variables
        questions = pl.read_ipc(self.config["questions_path"])

        model = GenericInstructModelHF(self.config["laguage_model_path"])
        model_configs = self.config["model_configs"]
        generations = {}


        ## Iterate questions
        for idx in range(len(questions)):



            ## Generate prompt
            prompt = f"Question: {questions[idx]['question'].to_numpy().flatten()[0]}\nAnswer: "

            ## Generate output
            outputs = model.run(
                prompt, 
                instruction="You are given a question and you MUST try to give a real SHORT ANSWER in 5 tokens, you can use the available documents ", 
                config_params=model_configs
            )

            generations[f"{idx}"] = [str(out["generated_text"]) for out in outputs]

            if self.config["generations_prefix"] is None:
                with open("generations/baseline_generations.json", "w") as f:
                    json.dump(generations, f)
                    return
            
            prefix = self.config["generations_prefix"]
            with open(f"generations/{prefix}_baseline_generations.json", "w") as f:
                json.dump(generations, f)
                return

    def _verify_config_structure(self):

        expected_keys = [
            "questions_path",
            "laguage_model_path",
            "model_configs",
            "generations_prefix",
            "seed"
        ]

        for key in expected_keys:
            if key not in self.config:
                raise ValueError(f"Missing key {key} in config")
        

    def invoke_pipeline_stpe(self, step: str):

        match step:
            case "setup":
                self.setup()

            case "generate_inferneces":
                self.generate_inferences()
