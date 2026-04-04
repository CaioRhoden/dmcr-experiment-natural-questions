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


DEFAULT_INSTRUCTION = "You are given a question and you MUST try to give a real SHORT ANSWER in 5 tokens"
EXTRACTION_INSTRUCTION = "You are given a question and you MUST respond by EXTRACTING the answer (max 5 tokens) from one of the provided documents. If none of the documents contain the answer, respond with NO-RES. Begin your answer by providing a short explanation. Be as objective as possible. After providing your explanation, please generate your response by strictly following this format: \"RESPONSE: [[<response>]]\"."
REASONING_INSTRUCTION = "You are given a question and you MUST respond giving a answer answer (max 5 tokens), respond with NO-RES if you cannot answer. Begin your answer by providing a short explanation. Be as objective as possible. After providing your explanation, please generate your response by strictly following this format: \"RESPONSE: [[<response>]]\"."

INSTRUCTIONS_DICT = {
    "default": DEFAULT_INSTRUCTION,
    "extraction": EXTRACTION_INSTRUCTION,
    "reasoning": REASONING_INSTRUCTION
}

LM_CONFIGS = {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_new_tokens": 10,
    "n": 1
}

def alt_format_input_1(question, response):
    return f""""
    [System] 
    Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to a question displayed below. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, please classify the response as 1 for RESPONDS QUESTION and 0 for NOT RESPONDES QUESTION by strictly following this format: "[[classification]]", for example: "Classification: [[1]]".  
    [Question] 
    {question}  
    [The Start of Assistant’s Answer] 
    {response}
    [The End of Assistant’s Answer]
    """

def format_input_recall(question, response):
    return f""""
    [Question] 
    {question}  
    [The Start of Assistant’s Answer] 
    {response}
    [The End of Assistant’s Answer]
    """

ALT_FORMAT_INPUT = {
    "ALT1": alt_format_input_1,
    "recall": format_input_recall
}

@dataclass
class DatamodelsConfig:
    '''
    Configuration class for the experiment.
    '''
    run_type: Literal["setup", "pre_collections", "collections", "training", "generation"] = "setup"
    '''Type of run to execute. Options: "setup", "pre_collections", "collections", "training", "generation".'''
    format_input: Literal[None, "ALT1", "ALT2", "recall"] = "ALT1"
    mode: Literal["train", "test"] = "train"
    '''Mode for the experiment section. Options: "train" or "test".'''
    seed: Literal[1, 4, 54, 61, 73] = 1
    '''Random seed for reproducibility based on the previous random generated'''
    log: bool = True
    '''Flag to enable logging. Options: "setup", "baseline", "rag", "datamodels".'''
    model_run_id: str = "judge"
    '''ID of the model run.'''
    instruction: Literal["default", "extraction", "reasoning"] = "default"
    '''Instruction type for the language model. Options: "default", "extraction", "reasoning".'''
    lm_configs: dict = field(default_factory=lambda: LM_CONFIGS)
    '''Language model configuration parameters.'''


    
    # RAG Based configs Config Fields
    wiki_path: str = "data/wiki_dump2018_nq_open/processed/wiki.feather"
    '''Path to the wiki dataset file.'''
    language_model_path: str = "models/Llama-3.2-3B-Instruct"
    '''Path to the language model.'''
    retrieval_path: str = "retrieval/rag_retrieval_indexes.json"
    '''Path to the retrieval indexes JSON file.'''
    embedder_path: str = "models/bge-base-en-v1.5"
    '''Path to the embedder model.'''
    vector_db_path: str = "data/wiki_dump2018_nq_open/processed/wiki_cosine.index"
    '''Path to the vector database.'''
    project_log: str = "small_window"
    '''Project log name for wandb'''
    model_run_id: str = "datamodels"
    '''ID of the model run.'''
    collection_id: str = "test"
    '''ID of the training collection.'''
    k: int = 8
    '''Number of top-k results to retrieve.'''
    size_index: int = 32
    '''Size of the index.'''
    num_models: int = 500
    '''Number of models to use.'''
    evaluation_metric: str = "mse"
    '''Evaluation metric to use.'''
    evaluator: str = "VotingBinaryJudge"
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
    root_path: str = "runs"
    '''Root path for saving runs.'''



def initiate_pipeline(args: DatamodelsConfig) -> ParallelRAGBasedPipeline:
    """
    Initiates the baseline pipeline with the provided arguments.
    
    Args:
        args (ParametersConfig): The configuration parameters for the pipeline.
        seed (int): The random seed for reproducibility.
    
    Returns:
        ZeroShotBaselinePipeline: An instance of the baseline pipeline.
    """

    questions_path = f"runs/experiment_{args.seed}/questions.feather"

    return ParallelRAGBasedPipeline(
        seed=args.seed,
        retrieval_path=args.retrieval_path,
        wiki_path=args.wiki_path,
        embedder_path=args.embedder_path,
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
        instruction=INSTRUCTIONS_DICT[args.instruction],
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        epochs=args.epochs,
        lr=args.lr,
        train_batches=args.train_batches,
        val_batches=args.val_batches,
        val_size=args.val_size,
        patience=args.patience,
        log_epochs=args.log_epochs,
        root_path=f"{args.root_path}/experiment_{args.seed}",
        batch_size=args.batch_size,
        tags=args.tags,
        lm_configs=args.lm_configs,
        log=args.log,
        num_subprocesses=args.num_subprocesses,
        model_run_id=args.model_run_id
    )


if __name__ == "__main__":
    args = tyro.cli(DatamodelsConfig)
    args.tags.append("datamodels")
    args.tags.append(f"seed_{args.seed}")

    args.language_model_path = f"{root}/{args.language_model_path}"
    args.wiki_path = f"{root}/{args.wiki_path}"
    args.retrieval_path = f"{args.root_path}/experiment_{args.seed}/{args.retrieval_path}"
    args.embedder_path = f"{root}/{args.embedder_path}"
    args.vector_db_path = f"{root}/{args.vector_db_path}"
    args.collection_id = f"experiment-{args.seed}_evaluator-{args.evaluator}-{str(args.format_input)}"

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
        if args.format_input:
            args.format_input = ALT_FORMAT_INPUT[args.format_input]

        pipeline.run_collections(
            mode=args.mode,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            checkpoint=args.checkpoint,
            collection_id=args.collection_id,
            num_subprocesses=args.num_subprocesses,
            format_input=args.format_input

        )
        sys.exit(0)
    
    elif args.run_type == "training":
        pipeline.train_datamodels(collection_id=args.collection_id, num_subprocesses=args.num_subprocesses, checkpoint=args.checkpoint, start_idx=args.start_idx, end_idx=args.end_idx)
        pipeline.evaluate_datamodels(collection_id=args.collection_id)
        sys.exit(0)

    elif args.run_type == "generation":

        pipeline.get_datamodels_generations(f"{args.model_run_id}",f"{args.model_run_id}")
        pipeline.get_datamodels_retrieval(f"{args.model_run_id}")
        sys.exit(0)

    else:
        print("Please provide a valid run_type: setup, pre_collections, collections, training, generation")
        sys.exit(1)