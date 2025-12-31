import argparse
from dataclasses import dataclass, field
import shutil
import os
from pathlib import Path
import tyro
import numpy as np
import random
import os


from utils.pipelines.RAGBasedExperimentPipeline import RAGBasedExperimentPipeline
from utils.set_random_seed import set_random_seed


set_random_seed(42)
root = Path(__file__).parent.parent.parent
@dataclass
class ParametersConfig:
    '''
    Configuration class for the experiment.
    '''

    ## Run config
    seed: int = 0
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
    embedder_path: str = "models/bge-base-en-v1.5"
    '''Path to the embedder model.'''
    vector_db_path: str = "data/wiki_dump2018_nq_open/wiki_ip.index"
    '''Path to the vector database.'''
    questions_path: str = "data/nq_open_gold/processed/test.feather"
    '''Path to the questions dataset file.'''
    language_model_path: str = "models/Llama-3.2-3B-Instruct"
    '''Path to the language model.'''
    project_log: str = "nq_experiment_perplexity"
    '''Project log name fgor wandb'''
    model_run_id: str = "perplexity_undefined _seed"
    '''ID of the model run.'''
    train_collection_id: str = "normalized_perplexity"
    '''ID of the training collection.'''
    test_collection_id: str = "normalized_perplexity"
    '''ID of the testing collection.'''
    k: int = 16
    '''Number of top-k results to retrieve.'''
    size_index: int = 100
    '''Size of the index.'''
    num_models: int = 50
    '''Number of models to use.'''
    evaluation_metric: str = "mse"
    '''Evaluation metric to use.'''
    evaluator: str = "Rouge-L"
    '''Evaluator to use.'''
    
    # Pre-collections Config Fields
    instruction: str = "You are given a question and you MUST try to give a real SHORT ANSWER in 5 tokens, you can use the available documents but if they are not helpful, answer without them"
    '''Instruction for the pre-collections step.'''
    lm_configs: dict[str, float|int] = field(default_factory=lambda: {
            "temperature": 0.7,
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
    patience: int = 20
    '''Patience for early stopping.'''
    log_epochs: int = 25
    '''Interval for logging.'''


def copy_folder(src, dst, overwrite=False):
    """
    Copy the contents of the source directory to the destination directory.

    If the destination directory already exists, this function will overwrite it if
    the `overwrite` parameter is `True`.

    Parameters
    ----------
    src : str
        The path to the source directory.
    dst : str
        The path to the destination directory.
    overwrite : bool, optional
        Whether to overwrite the destination directory if it already exists.
        Defaults to `False`.
    """
    if overwrite and os.path.exists(dst):
        if os.path.isfile(dst) or os.path.islink(dst):
            os.remove(dst)
        else:
            shutil.rmtree(dst)
    else:
        # Non-overwrite mode
        if not os.path.exists(dst):
            shutil.copytree(src, dst)
        else:
            # Validate destination is a directory
            if not os.path.isdir(dst):
                raise NotADirectoryError(f"Destination '{dst}' is not a directory")
            
            # Merge source into destination
            for root, dirs, files in os.walk(src, followlinks=False):
                # Relative path from source root
                rel_path = os.path.relpath(root, src)
                dest_dir = os.path.join(dst, rel_path)
                
                # Create destination directory if missing
                if not os.path.exists(dest_dir):
                    os.makedirs(dest_dir, exist_ok=True)
                
                # Copy files
                for name in files:
                    src_path = os.path.join(root, name)
                    dest_path = os.path.join(dest_dir, name)
                    
                    # Skip if destination file exists
                    if os.path.lexists(dest_path):
                        continue
                    
                    # Handle symlinks
                    if os.path.islink(src_path):
                        link_target = os.readlink(src_path)
                        os.symlink(link_target, dest_path)
                    # Copy regular files
                    elif os.path.isfile(src_path):
                        shutil.copy2(src_path, dest_path)

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
        embedder_path=args.embedder_path,
        vector_db_path=args.vector_db_path,
        questions_path=args.questions_path,
        language_model_path=args.language_model_path,
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
        log=args.log,
        datamodels_generation_name=args.model_run_id
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
    args.embedder_path = f"{root}/{args.embedder_path}"
    args.questions_path = f"{root}/{args.questions_path}"
    args.language_model_path = f"{root}/{args.language_model_path}"
    args.vector_db_path = f"{root}/{args.vector_db_path}"

    

    seed = args.seed
    set_random_seed(seed)
    print(f"Using seed: {seed}")

    if args.run_type == "datamodels_setup":
        
        pipeline = initiate_datamodels_pipeline(args, seed)
        pipeline.setup()

        copy_folder(f"../collections/{args.seed}", f"{args.seed}/datamodels/collections/train")
        copy_folder(f"../retrieval/{args.seed}", f"{args.seed}/retrieval")
        copy_folder(f"../collections_dataset/{args.seed}", f"{args.seed}/datamodels", overwrite=False)
        
        

    elif args.run_type == "datamodels_training":

        args.retrieval_path = f"{seed}/{args.retrieval_path}"
        pipeline = initiate_datamodels_pipeline(args, seed)
        pipeline.train_datamodels()
        exit(0)

    elif args.run_type == "datamodels_generations":

        args.retrieval_path = f"experiments_{seed}/{args.retrieval_path}"
        pipeline = initiate_datamodels_pipeline(args, seed)
        pipeline.get_datamodels_generations()
        pipeline.get_datamodels_retrieval()
        exit(0)
    
    else:
        raise ValueError(f"Unknown run type: {args.run_type}. Please choose from 'setup', 'baseline', 'rag' or 'datamodels'.")

