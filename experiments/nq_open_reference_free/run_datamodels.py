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

INSTRUCTIONS_DICT = {
    "default": DEFAULT_INSTRUCTION,
    "extraction": EXTRACTION_INSTRUCTION,
}

LM_CONFIGS = {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_new_tokens": 15,
    "n": 1
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
    log: bool = True
    '''Flag to enable logging. Options: "setup", "baseline", "rag", "datamodels".'''
    instruction: Literal["default", "extraction"] = "default"
    '''Instruction type for the language model. Options: "default", "extraction", "reasoning".'''
    lm_configs: dict = field(default_factory=lambda: LM_CONFIGS)
    '''Language model configuration parameters.'''
    collection_id: str = "default_collection"
    '''Identifier for the collection.'''

    
    # RAG Based configs Config Fields
    model_run_id: str = "nq_open_free_reference"
    '''ID of the model run.'''
    k: int = 10
    '''Number of top-k results to retrieve.'''
    size_index: int = 100
    '''Size of the index.'''
    evaluation_metric: str = "mse"
    '''Evaluation metric to use.'''
    evaluator: str = "VotingBinaryJudge"
    '''Evaluator to use.'''
    
    train_samples: int = 3000
    '''Number of training samples.'''
    test_samples: int = 200
    '''Number of testing samples.'''
    tags: list[str] = field(default_factory=lambda: ["datamodels"])
    '''List of tags for the experiment.'''

    ## Parameters pre_collections and collections
    start_idx: int = 0
    '''Starting index for processing.'''
    end_idx: int = 3610
    '''Ending index for processing.'''
    checkpoint: int = 250
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

    questions_path = f"{root}/data/nq_open/processed/dev.feather"

    wiki_path = f"{root}/data/wiki_dump2018_nq_open/processed/wiki.feather"
    language_model_path = f"{root}/models/Llama-3.2-3B-Instruct"
    retrieval_path = f"runs/retrieval/rag_retrieval_indexes.json"
    embedder_path = f"{root}/models/bge-base-en-v1.5"
    vector_db_path = f"{root}/data/indices/bge_index.faiss"

    return ParallelRAGBasedPipeline(
        seed=42,
        retrieval_path=retrieval_path,
        wiki_path=wiki_path,
        embedder_path=embedder_path,
        vector_db_path=vector_db_path,
        questions_path=questions_path,
        language_model_path=language_model_path,
        project_log="nq_open_free_reference",
        collection_id=args.collection_id,
        k=args.k,
        size_index=args.size_index,
        num_models=3610,
        evaluation_metric=args.evaluation_metric,
        evaluator=args.evaluator,
        instruction=INSTRUCTIONS_DICT[args.instruction],
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        root_path="runs",
        batch_size=3610,
        tags=args.tags,
        lm_configs=args.lm_configs,
        log=args.log,
        num_subprocesses=args.num_subprocesses,
        model_run_id=args.model_run_id
    )


if __name__ == "__main__":
    args = tyro.cli(DatamodelsConfig)
    args.tags.append("datamodels")
    args.tags.append(f"{args.run_type}")

    pipeline = initiate_pipeline(args)



    set_random_seed(42)

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