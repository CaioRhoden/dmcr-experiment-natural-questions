
from abc import abstractmethod, ABC
import numpy as np
import polars as pl
from dmcr.models import GenericInstructModelHF
from dmcr.evaluators import Rouge_L_evaluator
import uuid
import torch
import wandb
import datetime
import json

class BaseRunner(ABC):

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def run(self):
        pass


class InferenceRunner(BaseRunner):
    def __init__(self, config: dict, experiment_setting: dict, **kwargs):
        self.config = config
        self.experiment_setting = experiment_setting

    def run(self):

        ## Initialize core variables
        debug  = self.config["runtime_configs"]["debug"]
        skip_inference = self.config["runtime_configs"]["skip_inference"]
        skip_results = self.config["runtime_configs"]["skip_results"]
        start_idx =self.config["inference_configs"]["start_idx"]
        end_idx = self.config["inference_configs"]["end_idx"]
        instruction = self.config["inference_configs"]["instruction"]
        project = self.config["wandb_configs"]["project"]
        experiment_name = self.config["wandb_configs"]["experiment_name"]

        ## Load dfs
        wiki_path = self.config["runtime_configs"]["wiki_path"]
        questions_path = self.config["runtime_configs"]["questions_path"]
        wiki = pl.read_ipc(wiki_path)
        questions = pl.read_ipc(questions_path)
        
        ### Get inputs
        input_ids = self.experiment_setting["input_ids"]
        context = self.experiment_setting["context"]

        ### Iterate Generations
        for idx in range(len(input_ids)):

            if idx >= start_idx and idx < end_idx:

                input = input_ids[idx]
                docs = context[str(input)]

                
                ## Setup logging

                wandb.init(
                    project = f"{project}", 
                    dir = self.config["wandb_configs"]["wandb_dir"],
                    id = f"{experiment_name}_{input}_{str(uuid.uuid4())}", 
                    name = f"{experiment_name}_{input}",
                    config = self.config,
                )

                if not skip_inference:
                    self._inference(
                        docs = docs,
                        wiki = wiki,
                        questions = questions,
                        example_id = input,
                        instruction = instruction,
                        experiment_name = experiment_name
                    )

                if not skip_results:
                    
                    self._evaluate(
                        example_id = input,
                        experiment_name = experiment_name,
                        questions = questions,
                        rouge_evaluator = Rouge_L_evaluator()
                    )


                ## Finish Logging
                artifact_log = wandb.Artifact(
                    name=f"{experiment_name}_{input}", type="log",
                )

                artifact_log.add_file(f"{self.config['saving_paths']['inference']}/inference_{experiment_name}_{input}.json")
                artifact_log.add_file(f"{self.config['saving_paths']['results']}/result_{experiment_name}_{input}.json")

                wandb.log_artifact(artifact_log)

                wandb.finish()
        
    def _inference(
        self,
        docs: list,
        wiki: pl.DataFrame,
        questions: pl.DataFrame,
        example_id: str,
        instruction: str,
        experiment_name: str
    ) -> None:
        """Generate an answer using documents and questions with a HF model.
        
        Args:
            docs: List of document IDs to use as context
            wiki: DataFrame containing document texts and titles
            questions: DataFrame containing questions to answer
            example_id: Unique identifier for the question to answer
            instruction: Instruction text for the model
            experiment_name: Name for output file naming
        """
        _start = datetime.datetime.now()
        context_prompt = ""
        generation = None

        # Build context prompt from documents
        if len(docs) != 0:
            context_prompt = "Documents: \n"

        for c_idx in range(len(docs)):
            doc_id = docs[c_idx]
            _doc_text = wiki[doc_id].select("text").item()
            _doc_title = wiki[doc_id].select("title").item()
            context_prompt += f"Document[{c_idx}](Title: {_doc_title}){_doc_text}\n\n"

        # Build question prompt
        input_question = questions.filter(pl.col("example_id") == example_id).select("question").item()
        question_prompt = f"\nQuestion: {input_question}\nAnswer: "
        prompt = context_prompt + question_prompt

        # Initialize model and generate response
        model = GenericInstructModelHF(
            path=self.config["inference_configs"]["model_path"],
            quantization=self.config["inference_configs"]["quantization"]
        )

        output = model.run(
            prompt=prompt,
            instruction=instruction,
            config_params=self.config["model_configs"]
        )[0]["generated_text"]

        model.delete_model()
        # Record results
        _end = datetime.datetime.now()
        generation = {
            "prompt": prompt,
            "output": output,
            "time": str((_end - _start).total_seconds())
        }
        
        wandb.log({"generation_time": generation["time"]})
        
        # Save generation to JSON
        output_path = f"{self.config['saving_paths']['inference']}/inference_{experiment_name}_{example_id}.json"
        with open(output_path, "w") as f:
            json.dump(generation, f)

    def _evaluate(
        self,
        example_id: str,
        experiment_name: str,
        questions: pl.DataFrame,
        rouge_evaluator: Rouge_L_evaluator
    ) -> None:
        """Evaluate generated answer against reference answers using ROUGE-L.
        
        Args:
            example_id: Unique identifier for the question/answer pair
            experiment_name: Name for experiment tracking and file naming
            questions: DataFrame containing reference answers
            rouge_evaluator: Initialized ROUGE-L evaluation component
        """
        _start = datetime.datetime.now()
        results = {}
        
        # Load generated answer
        generation_path = f"{self.config['saving_paths']['inference']}/inference_{experiment_name}_{example_id}.json"
        with open(generation_path, "r") as f:
            generation = json.load(f)
        # Get reference answers
        true_answers = questions.filter(pl.col("example_id") == example_id).select("answers").item()
        results["true_answers"] = []
        output =  generation["output"].lower().strip()
        results["pred_answer"] = output
        scores = np.zeros(len(true_answers))

        # Calculate similarity scores
        for a_idx in range(len(true_answers)):
            true_str = true_answers[a_idx].lower().strip()
            results["true_answers"].append(true_str)
            gen_str = output
            scores[a_idx] = rouge_evaluator.evaluate(
                np.array([true_str]),
                np.array([gen_str]),
                None
            ).item()

        print(f"ROUGE-L scores: {scores}")
        results["score"] = np.max(scores)
        
        # Record and save metrics
        _end = datetime.datetime.now()
        results["time"] = str((_end - _start).total_seconds())
        wandb.log(results)

        results_path = f"{self.config['saving_paths']['results']}/result_{experiment_name}_{example_id}.json"
        with open(results_path, "w") as f:
            json.dump(results, f)
    def __call__(self):
        self.run()
        
    def __str__(self) -> str:
        return f""""
        Class: {self.__class__.__name__}
        Config: {self.config}
        Experiment Setting: {self.experiment_setting}
        """
