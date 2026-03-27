from dataclasses import dataclass
from pathlib import Path
from pyexpat import model
import random
from gguf import Literal
import tyro
import sys

from utils.set_random_seed import set_random_seed
from utils.pipelines.ParallelRAGBasedPipeline import ParallelRAGBasedPipeline
from pathlib import Path
root = Path(__file__).parent.parent.parent


INSTRUCTION_DICT = {
    "INSTRUCTION": "You are given a question and you MUST try to give a real SHORT ANSWER in 5 tokens",
    "INSTRUCTION_OPT": "You are given a question and you MUST respond by EXTRACTING the answer (max 5 tokens) from one of the provided documents. If none of the documents contain the answer, respond with NO-RES."
}

# Hard-coded configuration values
HARD_CODED_CONFIG = {
    "model_tag": "generic",
    "seed": 42,
    "log": True,
    "wiki_path": f"{root}/data/wiki_dump2018_nq_open/processed/wiki.feather",
    "language_model_path": f"{root}/models/Llama-3.2-3B-Instruct",
    "retrieval_path": f"experiment_81/retrieval/rag_retrieval_indexes.json",
    "embedder_path": f"{root}/models/bge-base-en-v1.5",
    "vector_db_path": f"{root}/data/wiki_dump2018_nq_open/processed/wiki_cosine.index",
    "model_run_id": "datamodels",
    "collection_id": "test",
    "project_log": "judge_comparative",
    "k": 16,
    "size_index": 100,
    "num_models": 10,
    "evaluation_metric": "mse",
    "evaluator": "Rouge-L",
    "train_samples": 2000,
    "test_samples": 200,
    "epochs": 1000,
    "lr": 0.0001,
    "train_batches": 3,
    "val_batches": 1,
    "val_size": 0.15,
    "patience": 30,
    "log_epochs": 25,
    "batch_size": 10,
    "start_idx": 0,
    "end_idx": 2000,
    "checkpoint": 200,
    "num_subprocesses": 1,
}

@dataclass
class DatamodelsConfig:
    '''
    Configuration class for the experiment.
    '''
    run_type: Literal["setup", "pre_collections", "collections", "training", "generation"] = "setup"
    '''Type of run to execute. Options: "setup", "pre_collections", "collections", "training", "generation".'''
    mode: Literal["train", "test"] = "train"
    '''Mode for the experiment section. Options: "train" or "test".'''
    start_idx: int = 0
    '''Starting index for data processing.'''
    end_idx: int = 2000
    '''Ending index for data processing.'''
    checkpoint: int = 200
    '''Checkpoint interval for data processing.'''
    num_subprocesses: int = 1
    '''Number of subprocesses to use.'''
    instruction :Literal["INSTRUCTION", "INSTRUCTION_OPT"] = "INSTRUCTION"
    '''Instruction to use for the language model. Options: "INSTRUCTION" or "INSTRUCTION_OPT".'''
    root_path: str = "experiment_81"
    '''Root path for the experiment.'''



def initiate_pipeline(args: DatamodelsConfig) -> ParallelRAGBasedPipeline:
    """
    Initiates the baseline pipeline with the provided arguments.
    
    Args:
        args (DatamodelsConfig): The configuration parameters for the pipeline.
    
    Returns:
        ParallelRAGBasedPipeline: An instance of the RAG-based pipeline.
    """
    model_run_id = f"{HARD_CODED_CONFIG['model_tag']}"

    lm_configs = {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_new_tokens": 10,
                "n": 1
    }

    questions_path = f"experiment_81/questions.feather"

    return ParallelRAGBasedPipeline(
        seed=HARD_CODED_CONFIG['seed'],
        retrieval_path=HARD_CODED_CONFIG['retrieval_path'],
        wiki_path=HARD_CODED_CONFIG['wiki_path'],
        embedder_path=HARD_CODED_CONFIG['embedder_path'],
        vector_db_path=HARD_CODED_CONFIG['vector_db_path'],
        questions_path=questions_path,
        language_model_path=HARD_CODED_CONFIG['language_model_path'],
        project_log=HARD_CODED_CONFIG['project_log'],
        collection_id=HARD_CODED_CONFIG['collection_id'],
        k=HARD_CODED_CONFIG['k'],
        size_index=HARD_CODED_CONFIG['size_index'],
        num_models=HARD_CODED_CONFIG['num_models'],
        evaluation_metric=HARD_CODED_CONFIG['evaluation_metric'],
        evaluator=HARD_CODED_CONFIG['evaluator'],
        instruction=INSTRUCTION_DICT[args.instruction],
        train_samples=HARD_CODED_CONFIG['train_samples'],
        test_samples=HARD_CODED_CONFIG['test_samples'],
        epochs=HARD_CODED_CONFIG['epochs'],
        lr=HARD_CODED_CONFIG['lr'],
        train_batches=HARD_CODED_CONFIG['train_batches'],
        val_batches=HARD_CODED_CONFIG['val_batches'],
        val_size=HARD_CODED_CONFIG['val_size'],
        patience=HARD_CODED_CONFIG['patience'],
        log_epochs=HARD_CODED_CONFIG['log_epochs'],
        root_path=args.root_path,
        batch_size=HARD_CODED_CONFIG['batch_size'],
        tags=[],
        lm_configs=lm_configs,
        log=HARD_CODED_CONFIG['log'],
        num_subprocesses=HARD_CODED_CONFIG['num_subprocesses'],
        model_run_id=model_run_id
    )


if __name__ == "__main__":
    args = tyro.cli(DatamodelsConfig)


    collection_id = f"default"

    pipeline = initiate_pipeline(args)

    set_random_seed(HARD_CODED_CONFIG['seed'])

    if args.run_type == "setup":
        pipeline.setup()
        pipeline.create_datamodels_datasets()
        sys.exit(0)
        
    elif args.run_type == "pre_collections":

        pipeline.run_pre_colections(
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            checkpoint=args.checkpoint,
            mode=args.mode,
            num_subprocesses=args.num_subprocesses,
            collection_id=collection_id
        )
        sys.exit(0)

    elif args.run_type == "collections":
        pipeline.run_collections(
            mode=args.mode,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            checkpoint=args.checkpoint,
            collection_id="rougel",
            num_subprocesses=args.num_subprocesses
        )
        sys.exit(0)
    
    else:
        print("Please provide a valid run_type: setup, pre_collections, collections, training, generation")
        sys.exit(1)