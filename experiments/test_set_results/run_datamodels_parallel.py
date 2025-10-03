from dataclasses import dataclass, field
from pathlib import Path
import random
import tyro
import sys

from utils.set_random_seed import set_random_seed
from utils.pipelines.ParallelRAGBasedPipeline import ParallelRAGBasedPipeline
from pathlib import Path
root = Path(__file__).parent.parent.parent


set_random_seed(42)
seed = random.randint(0, 10000)
root = Path(__file__).parent.parent.parent

@dataclass
class DatamodelsConfig:
    '''
    Configuration class for the experiment.
    '''
    run_type: str = "None"
    
    model: str = "None"
    '''Tag for the experiment section.'''
    mode: str = "None"
    '''Mode for the experiment section. Options: "train" or "test".'''
    log: bool = True
    '''Flag to enable logging. Options: "setup", "baseline", "rag", "datamodels".'''

    
    # RAG Based configs Config Fields
    wiki_path: str = "data/wiki_dump2018_nq_open/processed/wiki.feather"
    '''Path to the wiki dataset file.'''
    questions_path: str = "data/nq_open_gold/processed/test.feather"
    '''Path to the questions dataset file.'''
    language_model_path: str = "models/Llama-3.2-3B-Instruct"
    '''Path to the language model.'''
    retrieval_path: str = "retrieval/rag_retrieval_indexes.json"
    '''Path to the retrieval indexes JSON file.'''
    embeder_path: str = "models/bge-base-en-v1.5"
    '''Path to the embedder model.'''
    vector_db_path: str = "data/wiki_dump2018_nq_open/processed/wiki_cosine.index"
    '''Path to the vector database.'''
    project_log: str = "run_test_set_nq"
    '''Project log name fgor wandb'''
    model_run_id: str = "datamodels"
    '''ID of the model run.'''
    collection_id: str = "test"
    '''ID of the training collection.'''
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
    train_samples: int = 1200
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
    batch_size: int = 1
    '''Batch size for training.'''
    attn_implementation: str = "sdpa"
    '''Attention implementation to use. Options: "sdpa", "flash_attention_2",'''
    thinking: bool = False
    '''Whether to enable the thinking mode in the model.'''

    ## Parameters pre_collections and collections
    start_idx: int = 0
    '''Starting index for processing.'''
    end_idx: int = 2000
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

    return ParallelRAGBasedPipeline(
        seed=seed,
        retrieval_path=args.retrieval_path,
        wiki_path=args.wiki_path,
        embeder_path=args.embeder_path,
        vector_db_path=args.vector_db_path,
        questions_path=args.questions_path,
        language_model_path=args.language_model_path,
        project_log=args.project_log,
        model_run_id=args.model_run_id,
        collection_id=args.collection_id,
        k=args.k,
        size_index=args.size_index,
        num_models=args.num_models,
        evaluation_metric=args.evaluation_metric,
        evaluator=args.evaluator,
        instruction=args.instruction,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        epochs=args.epochs,
        lr=args.lr,
        train_batches=args.train_batches,
        val_batches=args.val_batches,
        val_size=args.val_size,
        patience=args.patience,
        log_epochs=args.log_epochs,
        root_path=f"{args.model}",
        batch_size=args.batch_size,
        attn_implementation=args.attn_implementation,
        thinking=args.thinking,
        tags=args.tags,
        lm_configs=args.lm_configs,
        log=args.log,
        num_subprocesses=args.num_subprocesses
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

    pipeline = initiate_pipeline(args)
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
            num_subprocesses=args.num_subprocesses
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
        # pipeline.evaluate_datamodels(collection_id=args.collection_id)
        sys.exit(0)

    elif args.run_type == "generation":

        pipeline.get_datamodels_generations()
        pipeline.get_datamodels_retrieval()
        sys.exit(0)

    else:
        print("Please provide a valid run_type: setup, pre_collections, collections, training, generation")
        sys.exit(1)