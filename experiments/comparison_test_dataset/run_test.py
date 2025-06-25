import argparse
from dataclasses import dataclass, field
from utils.pipelines.RAGBasedExperimentPipeline import RAGBasedExperimentPipeline
from utils.pipelines.BaselinePipeline import BaselinePipeline
from utils.pipelines.RAGPipeline import RAGPipeline
from utils.set_random_seed import set_random_seed
from pathlib import Path
import tyro
import numpy as np
import random
import os

set_random_seed(42)
root = Path(__file__).parent.parent.parent
@dataclass
class ParametersConfig:
    '''
    Configuration class for the experiment.
    '''

    ## Run config
    seed_idx: int = 0
    '''Random index seed for reproducibility.'''
    run_type: str = "setup"
    '''Tag for the experiment section.'''
    log: bool = True
    '''Flag to enable logging. Options: "setup", "baseline", "rag", "datamodels".'''

    
    # RAG Based configs Config Fields
    retrieval_path: str = "retrieval/rag_retrieval_indexes.json"
    '''Path to the retrieval indexes JSON file.'''
    wiki_path: str = "data/wiki_dump2018_nq_open/processed/wiki.feather"
    '''Path to the wiki dataset file.'''
    embeder_path: str = "models/llms/bge-base-en-v1.5"
    '''Path to the embedder model.'''
    vector_db_path: str = "data/wiki_dump2018_nq_open/wiki_ip.index"
    '''Path to the vector database.'''
    questions_path: str = "data/nq_open_gold/processed/test.feather"
    '''Path to the questions dataset file.'''
    laguage_model_path: str = "models/llms/Llama-3.2-3B-Instruct"
    '''Path to the language model.'''
    project_log: str = "nq_experiment_test_set"
    '''Project log name fgor wandb'''
    model_run_id: str = "test_experiment"
    '''ID of the model run.'''
    train_collection_id: str = "test_experiment"
    '''ID of the training collection.'''
    test_collection_id: str = "test_experiment"
    '''ID of the testing collection.'''
    k: int = 16
    '''Number of top-k results to retrieve.'''
    size_index: int = 100
    '''Size of the index.'''
    num_models: int = 2889
    '''Number of models to use.'''
    evaluation_metric: str = "mse"
    '''Evaluation metric to use.'''
    evaluator: str = "Rouge-L"
    '''Evaluator to use.'''
    
    # Pre-collections Config Fields
    instruction: str = "You are given a question and you MUST try to give a real SHORT ANSWER in 5 tokens, you can use the available documents but if they are not helpful, try to answer without them"
    '''Instruction for the pre-collections step.'''
    lm_configs: dict[str, float|int] = field(default_factory=lambda: {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_new_tokens": 10,
        })
    train_samples: int = 2000
    '''Number of training samples.'''
    test_samples: int = 400
    '''Number of testing samples.'''
    tags: list[str] = field(default_factory=list)
    '''List of tags for the experiment.'''
    train_start_idx: int = 0
    '''Starting index for the training set.'''
    train_end_idx: int = -1
    '''Ending index for the training set.'''
    test_start_idx: int = 0
    '''Starting index for the testing set.'''
    test_end_idx: int = -1
    '''Ending index for the testing set.'''
    train_checkpoint: int = 50
    '''Checkpoint interval for training.'''
    test_checkpoint: int = 50
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

def initiate_baseline_pipeline(args: ParametersConfig, seed: int) -> BaselinePipeline:
    """
    Initiates the baseline pipeline with the provided arguments.
    
    Args:
        args (ParametersConfig): The configuration parameters for the pipeline.
        seed (int): The random seed for reproducibility.
    
    Returns:
        BaselinePipeline: An instance of the baseline pipeline.
    """
    args.tags.append("baseline")
    args.tags.append(f"seed_{seed}")
    args.model_run_id = f"{args.run_type}_{seed}"
    return BaselinePipeline(
        questions_path=args.questions_path,
        laguage_model_path=args.laguage_model_path,
        lm_configs=args.lm_configs,
        model_run_id=f"baseline_{seed}",
        instruction=args.instruction,
        root_path=f"experiments_{seed}",
        project_log=args.project_log,
        tags = args.tags,
        log=args.log,
        seed=seed,
    )
def initiate_datamodels_pipeline(args: ParametersConfig, seed: int) -> RAGBasedExperimentPipeline:

    
    """
    Initiates the RAG pipeline with the provided arguments.
    
    Args:
        args (ParametersConfig): The configuration parameters for the pipeline.
    
    Returns:
        RAGBasedExperimentPipeline: An instance of the RAG pipeline.
    """
    return RAGBasedExperimentPipeline(
        seed=seed,
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
        log_epochs=args.log_epochs,
        root_path=f"experiments_{seed}",
    )


def initiate_rag_pipeline(args: ParametersConfig, seed: int) -> RAGPipeline:
    """
    Initiates the RAG pipeline with the provided arguments.
    
    Args:
        args (ParametersConfig): The configuration parameters for the pipeline.
        seed (int): The random seed for reproducibility.
    
    Returns:
        RAGPipeline: An instance of the RAG pipeline.
    """
    return RAGPipeline(
        seed=seed,
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
        root_path=f"experiments_{seed}",
    )

def create_random_seeds() -> None:
    """
    Create and save a numpy array with 5 random seeds.
    """
    
    seeds = np.random.randint(0, 10000, size=5)
    np.save("random_seeds.npy", seeds)
    for seed in seeds:
        os.mkdir(f"experiments_{seed}")
    print(f"Random seeds saved: {seeds}")


if __name__ == "__main__":

    ## Load dataclass as args
    args = tyro.cli(ParametersConfig)

    ## Add root to paths (except test)
    args.wiki_path = f"{root}/{args.wiki_path}"
    args.embeder_path = f"{root}/{args.embeder_path}"
    args.questions_path = f"{root}/{args.questions_path}"
    args.laguage_model_path = f"{root}/{args.laguage_model_path}"
    args.vector_db_path = f"{root}/{args.vector_db_path}"

    

    if args.run_type == "setup":
        create_random_seeds()
        exit(0)

    seeds = np.load("random_seeds.npy")
    seed = int(seeds[args.seed_idx])
    args.model_run_id = f"{args.run_type}_{seed}"
    set_random_seed(seed)
    print(f"Using seed: {seed}")


    if args.run_type == "baseline":

        baseline = initiate_baseline_pipeline(args, seed)
        baseline.generate_inferences()
        exit(0)

    elif args.run_type == "rag":

        ## Run entire RAG piline (setup, retrieval, generations) for one seed index

        rag = initiate_rag_pipeline(args, seed)
        rag.setup()
        rag.get_rag_retrieval()
        rag.get_rag_generations()
        exit(0)

    elif args.run_type == "datamodels_pre_collections":

        args.retrieval_path = f"experiments_{seed}/{args.retrieval_path}"

        pipeline = initiate_datamodels_pipeline(args, seed)
        pipeline.setup()
        pipeline.create_datamodels_datasets()
        pipeline.run_pre_colections()
        exit(0)
    
    elif args.run_type == "datamodels_collections":

        args.retrieval_path = f"experiments_{seed}/{args.retrieval_path}"
        pipeline = initiate_datamodels_pipeline(args, seed)
        pipeline.run_collections()
        exit(0)
    
    elif args.run_type == "datamodels_training":

        args.retrieval_path = f"experiments_{seed}/{args.retrieval_path}"
        pipeline = initiate_datamodels_pipeline(args, seed)
        pipeline.train_datamodels()
        pipeline.evaluate_datamodels()
        exit(0)

    elif args.run_type == "datamodels_generations":

        args.retrieval_path = f"experiments_{seed}/{args.retrieval_path}"
        pipeline = initiate_datamodels_pipeline(args, seed)
        pipeline.get_datamodels_generations()
        pipeline.get_datamodels_retrieval()
        exit(0)
    
    else:
        raise ValueError(f"Unknown run type: {args.run_type}. Please choose from 'setup', 'baseline', 'rag' or 'datamodels'.")

