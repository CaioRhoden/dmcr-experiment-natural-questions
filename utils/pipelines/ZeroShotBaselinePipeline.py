import datetime
import json
import os
from typing import Optional, Literal

import polars as pl
import torch
import wandb
from dmcr.models import GenericVLLMBatch

from utils.set_random_seed import set_random_seed

# Constants
DEFAULT_MAX_MODEL_LEN = 32768



class ZeroShotBaselinePipeline:
    """Pipeline for generating baseline zero-shot inferences from a language model."""

    def __init__(
        self,
        questions_path: str,
        language_model_path: str,
        root_path: str = ".",
        lm_configs: dict[str, float | int] = {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_new_tokens": 15,
            "seed": 42,
        },
        tags: list[str] = [],
        log: bool = False,
        project_log: str = "none",
        model_run_id: str = "none",
        batch_size: int = 1,
        instruction: str = "You are given a question and you MUST try to give a real SHORT ANSWER in 5 tokens",
        attn_implementation: str = "sdpa",
        thinking: bool = False,
        seed: Optional[int] = None,
    ):
        """
        Initialize the ZeroShotBaselinePipeline.

        Args:
            questions_path: Path to the questions dataset (Polars IPC format)
            language_model_path: Path to the language model
            root_path: Root directory for outputs
            lm_configs: Language model configuration parameters
            tags: Tags for logging
            log: Whether to log to W&B
            project_log: W&B project name
            model_run_id: Unique identifier for this run
            batch_size: Number of prompts to process in parallel
            instruction: System instruction for the model
            attn_implementation: Attention implementation type
            thinking: Whether to parse thinking tokens
            seed: Random seed for reproducibility
        """
        self.questions_path = questions_path
        self.language_model_path = language_model_path
        self.lm_configs = lm_configs
        self.model_run_id = model_run_id
        self.instruction = instruction
        self.seed = seed
        self.root_path = root_path
        self.tags = tags
        self.log = log
        self.project_log = project_log
        self.batch_size = batch_size
        self.attn_implementation = attn_implementation
        self.thinking = thinking

        if seed:
            set_random_seed(seed)

    def _parse_generation_output(self, output: list) -> list[str]:
        """
        Parse the output of the generation model.

        Handles both standard generation output and models with extended thinking.
        When thinking is enabled, extracts text after the thinking tag.

        Args:
            output: List of generation outputs from the model

        Returns:
            List of parsed text strings
        """
        parsed_output = []
        for out in output:
            if self.thinking:
                # Extract text after closing thinking tag for extended thinking models
                text = str(out["generated_text"]).split("</think>")[-1].strip()
            else:
                text = str(out["generated_text"])
            parsed_output.append(text)

        return parsed_output
    
    def _save_generations(self, generations: dict, path: str) -> None:
        """Save generations to a JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(generations, f)

    def _get_output_path(self) -> str:
        """Determine the output path for generations."""
        if self.model_run_id is None:
            return f"{self.root_path}/generations/baseline_zero_shot_generations.json"
        else:
            return f"{self.root_path}/generations/{self.model_run_id}.json"

    def generate_inferences(self) -> None:
        """
        Generate baseline inferences for questions and save results.

        Processes questions in batches through the language model and saves
        generations to JSON. Optionally logs to Weights & Biases.

        Returns:
            None
        """
        # Setup output directories
        os.makedirs(f"{self.root_path}", exist_ok=True)
        os.makedirs(f"{self.root_path}/generations", exist_ok=True)
        if self.log:
            os.makedirs(f"{self.root_path}/logs", exist_ok=True)

        # Load questions dataset
        questions = pl.read_ipc(self.questions_path)

        # Validate batch size
        assert self.batch_size >= 1, "Batch size must be at least 1"

        # Initialize model and tracking variables
        model = GenericVLLMBatch(
            self.language_model_path,
            vllm_kwargs={"max_model_len": DEFAULT_MAX_MODEL_LEN},
        )
        generations = {}
        batch_list = []
        output_path = self._get_output_path()

        # Initialize W&B logging if enabled
        if self.log:
            start_time = datetime.datetime.now()
            wandb.init(
                project=self.project_log,
                name=f"Baseline_{self.model_run_id}",
                id=f"Baseline_{self.model_run_id}_{start_time.strftime('%Y-%m-%d_%H-%M-%S')}",
                config={
                    "gpu": f"{torch.cuda.get_device_name(0)}",
                    "questions_path": self.questions_path,
                },
                tags=self.tags + ["generations"],
            )
            artifact = wandb.Artifact(
                name="generations",
                type="json",
                description="Baseline generations data",
            )

        # Process questions in batches
        for idx in range(len(questions)):
            # Create prompt from question
            question_text = questions[idx]["question"].to_numpy().flatten()[0]
            prompt = f"Question: {question_text}\nAnswer: "
            batch_list.append((idx, prompt))

            # Process batch when full or at the end
            if len(batch_list) == self.batch_size or idx == len(questions) - 1:
                # Get model outputs
                prompts = [prompt for _, prompt in batch_list]
                outputs = model.run(
                    prompts,
                    instruction=self.instruction,
                    config_params=self.lm_configs,
                )

                # Store parsed outputs
                for i, (question_idx, _) in enumerate(batch_list):
                    generations[str(question_idx)] = self._parse_generation_output(
                        outputs[i]
                    )

                # Save generations
                self._save_generations(generations, output_path)
                batch_list = []

        # Finalize W&B logging if enabled
        if self.log:
            artifact.add_file(output_path)
            wandb.log_artifact(artifact)
            wandb.log(
                {
                    "end_time": datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                    "duration": (
                        datetime.datetime.now() - start_time
                    ).total_seconds(),
                }
            )
            wandb.finish()