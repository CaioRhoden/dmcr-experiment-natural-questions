from typing import Optional
import torch.multiprocessing as mp
import math
import os

from curses import start_color
from tracemalloc import start
import os
import torch
from dmcr.datamodels.pipeline.DatamodelsIndexBasedNQPipeline import DatamodelsIndexBasedNQPipeline
from dmcr.datamodels.config import DatamodelIndexBasedConfig, LogConfig
from dmcr.evaluators import Rouge_L_evaluator, SquadV2Evaluator, JudgeEvaluator
from utils.pipelines.RAGBasedExperimentPipeline import RAGBasedExperimentPipeline

# Assume the original RAGBasedExperimentPipeline class and its dependencies are in scope
# and available for use, as provided in the problem description.
# For example, it could be imported as:
# from your_module import RAGBasedExperimentPipeline, DatamodelIndexBasedConfig, DatamodelsIndexBasedNQPipeline

# NOTE: The full code for RAGBasedExperimentPipeline and its dependencies is omitted here for brevity.
# The following code assumes the original class exists as provided.


def _run_pre_collections_worker(init_kwargs: dict, run_kwargs: dict):
    """
    Worker function for parallel pre-collection creation. Instantiates a 
    RAGBasedExperimentPipeline and runs a portion of the pre-collection creation.
    """
    pipeline = RAGBasedExperimentPipeline(**init_kwargs)
    print(
        f"Process {os.getpid()} starting task: "
        f"mode='{run_kwargs.get('mode')}', "
        f"range=[{run_kwargs.get('start_idx')}, {run_kwargs.get('end_idx')}), "
        f"pre-collection='{run_kwargs.get('collection_id')}'"
    )
    pipeline.run_pre_colections(**run_kwargs)
    print(f"Process {os.getpid()} finished task for pre-collection '{run_kwargs.get('collection_id')}'.")


def _run_collections_worker(init_kwargs: dict, run_kwargs: dict):
    """
    Top-level worker function designed to be the target for each subprocess.
    
    This function instantiates a new `RAGBasedExperimentPipeline` object using the
    provided initialization arguments and then calls its `run_collections` method
    with the arguments specific to this worker's assigned data chunk.
    This approach ensures process safety and avoids issues with serializing
    complex, non-picklable objects (like GPU models) from the main process.

    Args:
        init_kwargs (dict): A dictionary of keyword arguments required to initialize
                            the `RAGBasedExperimentPipeline`.
        run_kwargs (dict): A dictionary of keyword arguments for the `run_collections`
                           method, including the specific `start_idx` and `end_idx` for the chunk.
    """
    # Each process re-instantiates the base pipeline class to ensure a clean state
    # and avoid issues with CUDA context in forked processes.
    pipeline = RAGBasedExperimentPipeline(**init_kwargs)
    
    print(
        f"Process {os.getpid()} starting task: "
        f"mode='{run_kwargs.get('mode')}', "
        f"range=[{run_kwargs.get('start_idx')}, {run_kwargs.get('end_idx')}), "
        f"collection='{run_kwargs.get('collection_id')}'"
    )
    
    pipeline.run_collections(**run_kwargs)
    
    print(f"Process {os.getpid()} finished task for collection '{run_kwargs.get('collection_id')}'.")


def _run_training_worker(init_kwargs: dict, run_kwargs: dict):
    """
    Worker function for parallel training of datamodels. Instantiates a 
    RAGBasedExperimentPipeline and runs training for a specific collection.
    """
    pipeline = RAGBasedExperimentPipeline(**init_kwargs)
    print(
        f"Process {os.getpid()} starting training model: "
    )
    pipeline.train_datamodels(**run_kwargs)
    print(f"Process {os.getpid()} finished training for collection '{run_kwargs.get('collection_id')}'.")

class ParallelRAGBasedPipeline(RAGBasedExperimentPipeline):
    """
    Extends `RAGBasedExperimentPipeline` to enable parallel processing for the
    `run_collections` method.

    This class divides the total set of items to be processed into equal chunks
    and assigns each chunk to a separate subprocess, speeding up the overall
    collection creation time. It is designed to be safe for use with CUDA-based
    models by using the 'spawn' multiprocessing start method.
    """
    
    def __init__(self, **kwargs):
        """
        Initializes the parallel experiment pipeline.

        Args:
            num_subprocesses (int): The number of subprocesses to use for parallel execution.
                                    Defaults to 1 (serial execution).
            **kwargs: All other keyword arguments required by the parent
                      `RAGBasedExperimentPipeline` constructor.
        """
        super().__init__(**kwargs)
        self._init_kwargs = kwargs

    def run_pre_colections(self, 
                           mode: str = "train",
                           start_idx: int = 0,
                           end_idx: int = -1,
                           checkpoint: int = 50,
                           collection_id: str = "default_collection",
                           num_subprocesses: int = 1
                          ) -> None:
        """
        Overrides the parent method to run pre-collection creation in parallel.
        
        This method is GPU-aware and uses a 'spawn' context for safety with CUDA.
        Each subprocess will generate its own output files.
        """
        if num_subprocesses <= 1:
            print("Number of subprocesses is 1. Running pre-collections in serial mode.")
            super().run_pre_colections(mode, start_idx, end_idx, checkpoint, collection_id)
            return

        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        config = DatamodelIndexBasedConfig(
            k=self.k, num_models=self.num_models, datamodels_path=f"{self.root_path}/datamodels",
            train_set_path=self.wiki_path, test_set_path=self.questions_path
        )
        datamodel = DatamodelsIndexBasedNQPipeline(config)
        
        total_samples = len(datamodel.train_collections_idx) if mode == "train" else len(datamodel.test_collections_idx)
        effective_end_idx = end_idx if end_idx != -1 and end_idx <= total_samples else total_samples
        total_range = effective_end_idx - start_idx
        
        if total_range <= 0:
            print("Index range is empty. No pre-collections will be created.")
            return

        chunk_size = math.ceil(total_range / num_subprocesses)
        processes = []
        print(f"Splitting pre-collection creation for '{collection_id}' across {num_subprocesses} processes.")

        for i in range(num_subprocesses):
            p_start_idx = start_idx + i * chunk_size
            p_end_idx = min(start_idx + (i + 1) * chunk_size, effective_end_idx)

            if p_start_idx >= p_end_idx:
                continue

            run_kwargs = {
                "mode": mode,
                "start_idx": p_start_idx,
                "end_idx": p_end_idx,
                "checkpoint": checkpoint,
                "collection_id": f"{collection_id}_{p_start_idx}_{p_end_idx}"
            }
            
            process = mp.Process(target=_run_pre_collections_worker, args=(self._init_kwargs, run_kwargs))
            processes.append(process)
            process.start()
        
        for process in processes:
            process.join()
            
        print("\nAll parallel pre-collection creation processes have completed.")


    def run_collections(self, 
                        mode: str = "train",
                        start_idx: int = 0,
                        end_idx: int = -1,
                        checkpoint: int = 50,
                        collection_id: str = "default_collection",
                        num_subprocesses: int = 1

                       ) -> None:
        """
        Overrides the parent method to run collection creation in parallel.

        It splits the total index range among the specified number of subprocesses.
        Each subprocess generates a partial collection file, identified by a suffix
        (e.g., 'default_collection_part_0'). A subsequent manual step may be
        required to merge these partial files into a single collection.

        Args:
            mode (str): The dataset mode, either "train" or "test".
            start_idx (int): The starting index of the data to process.
            end_idx (int): The ending index of the data to process. If -1, processes until the end.
            checkpoint (int): The interval at which to save progress.
            collection_id (str): The base name for the output collection files.
        """
        if num_subprocesses <= 1:
            print("Number of subprocesses is 1 or less. Running in standard serial mode.")
            super().run_collections(mode, start_idx, end_idx, checkpoint, collection_id)
            return

        # Set the multiprocessing start method to 'spawn'. This is critical for
        # CUDA compatibility, as it creates a fresh process without inheriting
        # potentially conflicting parent process state.
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            # This exception is raised if the start method has already been set, which is acceptable.
            pass

        # Determine the total number of samples to calculate the processing range.
        # This mirrors the logic in the parent method to ensure consistency.
        config = DatamodelIndexBasedConfig(
            k=self.k,
            num_models=self.num_models,
            datamodels_path=f"{self.root_path}/datamodels",
            train_set_path=self.wiki_path,
            test_set_path=self.questions_path
        )
        datamodel = DatamodelsIndexBasedNQPipeline(config)
        
        if mode == "train":
            total_samples = len(datamodel.train_collections_idx) * self.num_models
        elif mode == "test":
            total_samples = len(datamodel.test_collections_idx) * self.num_models
        else:
            raise ValueError("Mode must be either 'train' or 'test'.")

        effective_end_idx = end_idx if end_idx != -1 and end_idx <= total_samples else total_samples
        total_range = effective_end_idx - start_idx
        
        if total_range <= 0:
            print("The specified index range is empty or invalid. No collections will be created.")
            return

        # Calculate the size of the data chunk for each subprocess. Using math.ceil ensures
        # that the entire range is covered, even if it's not perfectly divisible.
        chunk_size = math.ceil(total_range / num_subprocesses)
        processes = []

        print(f"Splitting collection creation for '{collection_id}' across {num_subprocesses} processes.")

        for i in range(num_subprocesses):
            p_start_idx = start_idx + i * chunk_size
            p_end_idx = min(start_idx + (i + 1) * chunk_size, effective_end_idx)

            # Do not create a process for an empty range.
            if p_start_idx >= p_end_idx:
                continue

            # Define arguments for the worker's call to run_collections.
            # A unique collection_id is created for each part to prevent file I/O conflicts.
            run_kwargs = {
                "mode": mode,
                "start_idx": p_start_idx,
                "end_idx": p_end_idx,
                "checkpoint": checkpoint,
                "collection_id": f"{collection_id}_{p_start_idx}_{p_end_idx}",
            }
            
            # Create the process targeting the top-level worker function.
            process = mp.Process(
                target=_run_collections_worker, 
                args=(self._init_kwargs, run_kwargs)
            )
            processes.append(process)
            process.start()
        
        # Wait for all created processes to complete their execution.
        for process in processes:
            process.join()
            
        print("\nAll parallel collection creation processes have completed.")
        print(f"Partial collections were saved with base name: '{collection_id}_part_*'")

    def train_datamodels(self, collection_id: str, num_subprocesses: int = 1, start_idx: int = 0, end_idx: Optional[int] = None, checkpoint: Optional[int] = None) -> None:
        
        if num_subprocesses <= 1:
            print("Number of subprocesses is 1 or less. Running in standard serial mode.")
            super().train_datamodels(collection_id=collection_id)
            return

        # Set the multiprocessing start method to 'spawn'. This is critical for
        # CUDA compatibility, as it creates a fresh process without inheriting
        # potentially conflicting parent process state.
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            # This exception is raised if the start method has already been set, which is acceptable.
            pass

        # Determine the total number of samples to calculate the processing range.
        # This mirrors the logic in the parent method to ensure consistency.
        config = DatamodelIndexBasedConfig(
            k=self.k,
            num_models=self.num_models,
            datamodels_path=f"{self.root_path}/datamodels",
            train_set_path=self.wiki_path,
            test_set_path=self.questions_path
        )
        datamodel = DatamodelsIndexBasedNQPipeline(config)

        total_samples = self.num_models

        effective_end_idx = end_idx if end_idx != -1 and end_idx <= total_samples and end_idx is not None else total_samples
        total_range = effective_end_idx - start_idx
        
        if total_range <= 0:
            print("The specified index range is empty or invalid. No collections will be created.")
            return

        # Calculate the size of the data chunk for each subprocess. Using math.ceil ensures
        # that the entire range is covered, even if it's not perfectly divisible.
        chunk_size = math.ceil(total_range / num_subprocesses)
        processes = []

        print(f"Splitting collection creation for '{collection_id}' across {num_subprocesses} processes.")

        for i in range(num_subprocesses):
            p_start_idx = start_idx + i * chunk_size
            p_end_idx = min(start_idx + (i + 1) * chunk_size, effective_end_idx)

            # Do not create a process for an empty range.
            if p_start_idx >= p_end_idx:
                continue

            # Define arguments for the worker's call to run_collections.
            # A unique collection_id is created for each part to prevent file I/O conflicts.
            run_kwargs = {
                "start_idx": p_start_idx,
                "end_idx": p_end_idx,
                "checkpoint": checkpoint,
                "collection_id": f"{collection_id}"            }
            
            # Create the process targeting the top-level worker function.
            process = mp.Process(
                target=_run_training_worker, 
                args=(self._init_kwargs, run_kwargs)
            )
            processes.append(process)
            process.start()
        
        # Wait for all created processes to complete their execution.
        for process in processes:
            process.join()