import argparse
from dataclasses import dataclass, field
from typing import List
from utils.RAGBasedExperimentPipeline import RAGBasedExperimentPipeline
from pathlib import Path
import tyro

root = Path(__file__).parent.parent.parent.parent

@dataclass
class Config:
    '''
    Configuration class for the experiment.
    '''

    ## Run config
    seed: int = 42
    '''Random seed for reproducibility.'''
    step: str = "setup"
    '''Step to run.'''
    
    # Global Config Fields
    retrieval_path: str = "retrieval/rag_retrieval_indexes.json"
    '''Path to the retrieval indexes JSON file.'''
    wiki_path: str = "data/wiki_dump2018_nq_open/processed/wiki.feather"
    '''Path to the wiki dataset file.'''
    embeder_path: str = "models/llms/bge-base-en-v1.5"
    '''Path to the embedder model.'''
    vector_db_path: str = "data/wiki_dump2018_nq_open/wiki_ip.index"
    '''Path to the vector database.'''
    questions_path: str = "../50_test.feather"
    '''Path to the questions dataset file.'''
    laguage_model_path: str = "models/llms/Llama-3.2-3B-Instruct"
    '''Path to the language model.'''
    project_log: str = "nq_experiment_subset_sizes"
    '''Project log name fgor wandb'''
    model_run_id: str = "50_proportion"
    '''ID of the model run.'''
    train_collection_id: str = "50_proportion"
    '''ID of the training collection.'''
    test_collection_id: str = "50_proportion"
    '''ID of the testing collection.'''
    k: int = 16
    '''Number of top-k results to retrieve.'''
    size_index: int = 50
    '''Size of the index.'''
    num_models: int = 50
    '''Number of models to use.'''
    evaluation_metric: str = "mse"
    '''Evaluation metric to use.'''
    evaluator: str = "Rouge-L"
    '''Evaluator to use.'''
    
    # Pre-collections Config Fields
    instruction: str = "You are given a question and you MUST try to give a real SHORT ANSWER in 5 tokens, you can use the available documents but if they are not helpful, try to answer without them"
    '''Instruction for the pre-collections step.'''
    train_samples: int = 2000
    '''Number of training samples.'''
    test_samples: int = 400
    '''Number of testing samples.'''
    tags: list[str] = field(default_factory=list)
    '''List of tags for the experiment.'''
    train_start_idx: int = 0
    '''Starting index for the training set.'''
    train_end_idx: int = 2000
    '''Ending index for the training set.'''
    test_start_idx: int = 0
    '''Starting index for the testing set.'''
    test_end_idx: int = 400
    '''Ending index for the testing set.'''
    train_checkpoint: int = 200
    '''Checkpoint interval for training.'''
    test_checkpoint: int = 200
    '''Checkpoint interval for testing.'''
    
    # Datamodels Training Config Fields
    epochs: int = 1000
    '''Number of epochs to train.'''
    lr: float = 0.0001
    '''Learning rate for training.'''
    train_batches: int = 5
    '''Number of batches for training.'''
    val_batches: int = 1
    '''Number of batches for validation.'''
    val_size: float = 0.15
    '''Proportion of data for validation.'''
    patience: int = 30
    '''Patience for early stopping.'''
    log_epochs: int = 25
    '''Interval for logging.'''

if __name__ == "__main__":

    ## Load dataclass as args
    args = tyro.cli(Config)

    ## Add root to paths (except test)
    args.wiki_path = f"{root}/{args.wiki_path}"
    args.embeder_path = f"{root}/{args.embeder_path}"
    args.retrieval_path = f"{root}/{args.retrieval_path}"
    args.laguage_model_path = f"{root}/{args.laguage_model_path}"
    args.vector_db_path = f"{root}/{args.vector_db_path}"

    args.tags.append("rag_50")


    ## Explicit pass the arguments
    pipeline =RAGBasedExperimentPipeline(
        seed=args.seed,
        retrieval_path=args.retrieval_path,
        wiki_path=args.wiki_path,
        embeder_path=args.embeder_path,
        vector_db_path=args.vector_db_path,
        questions_path=args.questions_path,
        laguage_model_path=args.laguage_model_path,
        project_log=args.project_log,
        model_run_id=args.model_run_id,
        train_collection_id=args.train_collection_id,
        test_collection_id=args.test_collection_id,
        k=args.k,
        size_index=args.size_index,
        num_models=args.num_models,
        evaluation_metric=args.evaluation_metric,
        evaluator=args.evaluator,
        instruction=args.instruction,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        train_start_idx=args.train_start_idx,
        train_end_idx=args.train_end_idx,
        test_start_idx=args.test_start_idx,
        test_end_idx=args.test_end_idx,
        train_checkpoint=args.train_checkpoint,
        test_checkpoint=args.test_checkpoint,
        epochs=args.epochs,
        lr=args.lr,
        train_batches=args.train_batches,
        val_batches=args.val_batches,
        val_size=args.val_size,
        patience=args.patience,
        log_epochs=args.log_epochs
    )

    ### Call the desiredf pipeline step
    pipeline.set_random_seed()
    pipeline.invoke_pipeline_stpe(args.step)

    