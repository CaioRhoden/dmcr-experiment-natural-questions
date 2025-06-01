
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
                instruction=self.config["instruction"], 
                config_params=model_configs
            )

            generations[f"{idx}"] = [str(out["generated_text"]) for out in outputs]

            if self.config["generations_prefix"] is None:
                with open("generations/baseline_generations.json", "w") as f:
                    json.dump(generations, f)
            
            prefix = self.config["generations_prefix"]
            with open(f"generations/{prefix}_baseline_generations.json", "w") as f:
                json.dump(generations, f)

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
        

    def invoke_pipeline_step(self, step: str):

        match step:

            case "generate_inferences":
                self.generate_inferences()
