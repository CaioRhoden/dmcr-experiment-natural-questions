from dataclasses import dataclass, field
import sys
from pathlib import Path
import tyro
import numpy as np
import os

from dmcr.models import GenericVLLMBatch, GenericInstructBatchHF

from utils.pipelines.RAGPipeline import RAGPipeline
from utils.set_random_seed import set_random_seed


set_random_seed(42)
root = Path(__file__).parent.parent.parent
@dataclass
class RAGRetrievalsConfig:
    '''
    Configuration class for the experiment.
    '''
    # --- General Config ---
    tag: str = "ip"

    # --- RAG Pipeline Config ---
    retrieval_path: str = "retrieval/rag_retrieval_indexes.json"
    '''Path to the retrieval indexes JSON file.'''
    wiki_path: str = "data/wiki_dump2018_nq_open/processed/wiki.feather"
    '''Path to the wiki dataset file.'''
    embeder_path: str = "models/bge-base-en-v1.5"
    '''Path to the embedder model.'''
    vector_db_path: str = "data/wiki_dump2018_nq_open/processed"
    '''Path to the vector database.'''
    questions_path: str = "questions_1500_42_dev.feather"
    '''Path to the questions dataset file.'''
    language_model_path: str = "models/Llama-3.2-3B-Instruct"
    '''Path to the language model.'''
    project_log: str = "evaluate_rag_retrieval_metrics"
    '''Project log name for wandb'''
    k: int = 16
    '''Number of top-k results to retrieve.'''
    size_index: int = 100
    '''Size of the index.'''
    instruction: str = "You are given a question and you MUST try to give a real SHORT ANSWER in 5 tokens, you can use the available documents but if they are not helpful, try to answer without them"
    '''Instruction for the generation step.'''
    lm_configs: dict[str, float|int] = field(default_factory=lambda: {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_new_tokens": 10,
        })
    '''Configuration for the language model generation.'''


def initiate_rag_pipeline(args:RAGRetrievalsConfig, tag: str) -> RAGPipeline:
    """
    Initiates the RAG pipeline with the provided arguments.
    
    Args:
        args (ParametersConfig): The configuration parameters for the pipeline.
        seed (int): The random seed for reproducibility.
    
    Returns:
        RAGPipeline: An instance of the RAG pipeline.
    """
    return RAGPipeline(
        seed=42,
        retrieval_path=args.retrieval_path,
        wiki_path=args.wiki_path,
        embeder_path=args.embeder_path,
        vector_db_path=args.vector_db_path,
        questions_path=args.questions_path,
        language_model_path=args.language_model_path,
        project_log=args.project_log,
        model_run_id=args.tag,
        train_collection_id=args.tag,
        test_collection_id=args.tag,
        k=args.k,
        size_index=args.size_index,
        lm_configs=args.lm_configs,
        instruction=args.instruction,
        root_path=f"experiments_{tag}",
        log=True,
        batch_size=1500
    )
        


if __name__ == "__main__":

    ## Load dataclass as args
    args = tyro.cli(RAGRetrievalsConfig)
    tag = args.tag

    ## Add root to paths (except test)
    args.wiki_path = f"{root}/data/wiki_dump2018_nq_open/processed/wiki.feather"
    args.embeder_path = f"{root}/{args.embeder_path}"
    args.questions_path = f"{args.questions_path}"
    args.language_model_path = f"{root}/{args.language_model_path}"
    args.vector_db_path = f"{root}/{args.vector_db_path}/wiki_{tag}_upgrade.index"


    set_random_seed(42)
    rag = initiate_rag_pipeline(args, tag)
    rag.setup()
    rag.get_rag_retrieval()
    model = GenericVLLMBatch(
        path=args.language_model_path,
        vllm_kwargs={"max_model_len": 32768}
    )
    rag.get_rag_generations(model=model)


