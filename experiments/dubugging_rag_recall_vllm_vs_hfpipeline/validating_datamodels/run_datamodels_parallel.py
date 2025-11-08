from dataclasses import dataclass, field
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



root = Path(__file__).parent.parent.parent.parent
instructions = [
    "You are given a question and you MUST try to give a real SHORT ANSWER in 5 tokens, you can use the available documents but if they are not helpful, try to answer without them",
    "You are given a question and you MUST try to give a real SHORT ANSWER in 5 tokens",
    "You are given a question and you MUST try to give a real SHORT ANSWER in 5 tokens, you can use the available documents"
]

@dataclass
class DatamodelsConfig:
    '''
    Configuration class for the experiment.
    '''
    run_type: Literal["setup", "pre_collections", "collections", "training", "generation"] = "setup"
    '''Type of run to execute. Options: "setup", "pre_collections", "collections", "training", "generation".'''
    mode: Literal["train", "test"] = "train"
    '''Mode for the experiment section. Options: "train" or "test".'''
    seed: Literal[1, 4, 54, 61, 73]
    '''Random seed for reproducibility based on the previous random generated'''
    instruction_idx: Literal[0, 1, 2]
    '''Index of the instruction to be used in the experiment'''
    log: bool = True
    '''Flag to enable logging. Options: "setup", "baseline", "rag", "datamodels".'''

    
    # RAG Based configs Config Fields
    wiki_path: str = "data/wiki_dump2018_nq_open/processed/wiki.feather"
    '''Path to the wiki dataset file.'''
    language_model_path: str = "models/Llama-3.2-3B-Instruct"
    '''Path to the language model.'''
    retrieval_path: str = "retrieval/rag_retrieval_indexes.json"
    '''Path to the retrieval indexes JSON file.'''
    embeder_path: str = "models/bge-base-en-v1.5"
    '''Path to the embedder model.'''
    vector_db_path: str = "data/wiki_dump2018_nq_open/processed/wiki_cosine.index"
    '''Path to the vector database.'''
    project_log: str = "debugging_recall_validation"
    '''Project log name fgor wandb'''
    model_run_id: str = "datamodels"
    '''ID of the model run.'''
    collection_id: str = "test"
    '''ID of the training collection.'''
    k: int = 16
    '''Number of top-k results to retrieve.'''
    size_index: int = 100
    '''Size of the index.'''
    num_models: int = 500
    '''Number of models to use.'''
    evaluation_metric: str = "mse"
    '''Evaluation metric to use.'''
    evaluator: str = "Rouge-L"
    '''Evaluator to use.'''
    
    train_samples: int = 2000
    '''Number of training samples.'''
    test_samples: int = 200
    '''Number of testing samples.'''
    tags: list[str] = field(default_factory=list)
    '''List of tags for the experiment.'''
    
    # Datamodels Training Config Fields
    epochs: int = 1000
    '''Number of epochs to train.'''
    lr: float = 0.0001
    '''Learning rate for training.'''
    train_batches: int = 3
    '''Number of batches for training.'''
    val_batches: int = 1
    '''Number of batches for validation.'''
    val_size: float = 0.15
    '''Proportion of data for validation.'''
    patience: int = 30
    '''Patience for early stopping.'''
    log_epochs: int = 25
    '''Interval for logging.'''
    batch_size: int = 500
    '''Batch size for training.'''

    ## Parameters pre_collections and collections
    start_idx: int = 0
    '''Starting index for processing.'''
    end_idx: int = 500
    '''Ending index for processing.'''
    checkpoint: int = 50
    '''Checkpoint interval for saving progress.'''
    num_subprocesses: int = 1
    '''Number of subprocesses for parallel execution.'''



def initiate_pipeline(args: DatamodelsConfig) -> ParallelRAGBasedPipeline:
    """
    Initiates the baseline pipeline with the provided arguments.
    
    Args:
        args (ParametersConfig): The configuration parameters for the pipeline.
        seed (int): The random seed for reproducibility.
    
    Returns:
        ZeroShotBaselinePipeline: An instance of the baseline pipeline.
    """
    model_run_id = f"instruction-{args.instruction_idx}"


    lm_configs = {
            "temperature": 0.7,
            "top_p": 0.8,
            "max_new_tokens": 10,
            "n": 5
    }

    questions_path = f"experiment_{args.seed}/questions.feather"

    return ParallelRAGBasedPipeline(
        seed=args.seed,
        retrieval_path=args.retrieval_path,
        wiki_path=args.wiki_path,
        embeder_path=args.embeder_path,
        vector_db_path=args.vector_db_path,
        questions_path=questions_path,
        language_model_path=args.language_model_path,
        project_log=args.project_log,
        collection_id=args.collection_id,
        k=args.k,
        size_index=args.size_index,
        num_models=args.num_models,
        evaluation_metric=args.evaluation_metric,
        evaluator=args.evaluator,
        instruction=instructions[args.instruction_idx],
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        epochs=args.epochs,
        lr=args.lr,
        train_batches=args.train_batches,
        val_batches=args.val_batches,
        val_size=args.val_size,
        patience=args.patience,
        log_epochs=args.log_epochs,
        root_path=f"experiment_{args.seed}",
        batch_size=args.batch_size,
        tags=args.tags,
        lm_configs=lm_configs,
        log=args.log,
        num_subprocesses=args.num_subprocesses,
        model_run_id=model_run_id
    )


if __name__ == "__main__":
    args = tyro.cli(DatamodelsConfig)
    args.tags.append("datamodels")
    args.tags.append(args.model)
    args.questions_path = f"{root}/{args.questions_path}"
    args.language_model_path = f"{root}/{args.language_model_path}"
    args.wiki_path = f"{root}/{args.wiki_path}"
    args.retrieval_path = f"{args.model}/{args.retrieval_path}"
    args.embeder_path = f"{root}/{args.embeder_path}"
    args.vector_db_path = f"{root}/{args.vector_db_path}"
    args.collection_id = f"instruction-{args.instruction_idx}"

    pipeline = initiate_pipeline(args)

    set_random_seed(args.seed)

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
            collection_id=args.collection_id
        )
        sys.exit(0)
    
    elif args.run_type == "collections":

        pipeline.run_collections(
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            checkpoint=args.checkpoint,
            mode=args.mode,
            collection_id=args.collection_id,
            num_subprocesses=args.num_subprocesses
        )
        sys.exit(0)
    
    elif args.run_type == "training":

        pipeline.train_datamodels(collection_id=args.collection_id, num_subprocesses=args.num_subprocesses, checkpoint=args.checkpoint, start_idx=args.start_idx, end_idx=args.end_idx)
        pipeline.evaluate_datamodels(collection_id=args.collection_id)
        sys.exit(0)

    elif args.run_type == "generation":

        pipeline.get_datamodels_generations(f"instruction-{args.instruction_idx}",f"instruction-{args.instruction_idx}")
        pipeline.get_datamodels_retrieval(f"instruction-{args.instruction_idx}")
        sys.exit(0)

    else:
        print("Please provide a valid run_type: setup, pre_collections, collections, training, generation")
        sys.exit(1)