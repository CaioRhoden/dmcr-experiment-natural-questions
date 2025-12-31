from dataclasses import dataclass, field
from pathlib import Path
import random
from re import S
from sympy import Li
from torch import seed
import tyro
from dmcr.models import GenericVLLMBatch
from typing import Literal

from utils.set_random_seed import set_random_seed
from utils.pipelines.RAGPipeline import RAGPipeline

from pathlib import Path
root = Path(__file__).parent.parent.parent


instructions = [
    "You are given a question and you MUST try to give a real SHORT ANSWER in 5 tokens, you can use the available documents but if they are not helpful, try to answer without them",
    "You are given a question and you MUST try to give a real SHORT ANSWER in 5 tokens",
    "You are given a question and you MUST try to give a real SHORT ANSWER in 5 tokens, you can use the available documents"
]

root = Path(__file__).parent.parent.parent.parent
@dataclass
class RagConfig:
    '''
    Configuration class for the experiment.
    '''

    seed: Literal[1, 4, 54, 61, 73]
    '''Random seed for reproducibility based on the previous random generated'''
    instruction_idx: Literal[0, 1, 2]
    '''Index of the instruction to be used in the experiment'''

    log: bool = True    
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
    project_log: str = "debugging_recall_validation"
    '''Project log name fgor wandb'''


    k: int = 16
    '''Number of top-k results to retrieve.'''
    size_index: int = 100
    '''Size of the index to be retrieved.'''
    batch_size: int = 500
    '''Size of inferences to be done at the same time'''
    tags: list[str] = field(default_factory=list)
    '''List of tags for the experiment.'''




def initiate_pipeline(args: RagConfig) -> RAGPipeline:
    """
    Initiates the baseline pipeline with the provided arguments.
    
    Args:
        args (ParametersConfig): The configuration parameters for the pipeline.
        seed (int): The random seed for reproducibility.
    
    Returns:
        ZeroShotBaselinePipeline: An instance of the baseline pipeline.
    """

    model_run_id = f"rag_instruction-{args.instruction_idx}"

    lm_configs = {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_new_tokens": 10,
            "n": 1
    }

    questions_path = f"experiment_{args.seed}/questions.feather"

    return RAGPipeline(
        retrieval_path=args.retrieval_path,
        wiki_path=args.wiki_path,
        embedder_path=args.embedder_path,
        vector_db_path=args.vector_db_path,
        questions_path=questions_path,
        project_log=args.project_log,
        model_run_id=model_run_id,
        k=args.k,
        size_index=args.size_index,
        instruction=instructions[args.instruction_idx],
        root_path=f"experiment_{args.seed}/instruction_{args.instruction_idx}",
        seed=args.seed,
        tags = args.tags,
        log=args.log,
        batch_size=args.batch_size,
        lm_configs=lm_configs
    )


if __name__ == "__main__":
    args = tyro.cli(RagConfig)

    set_random_seed(args.seed)

    args.tags.append("rag")
    args.tags.append(args.seed)
    args.tags.append(f"instruction_idx_{args.instruction_idx}")
    args.language_model_path = f"{root}/{args.language_model_path}"
    args.wiki_path = f"{root}/{args.wiki_path}"
    args.retrieval_path = f"{root}/{args.retrieval_path}"
    args.embedder_path = f"{root}/{args.embedder_path}"
    args.vector_db_path = f"{root}/{args.vector_db_path}"


    pipeline = initiate_pipeline(args)
    pipeline.setup()
    pipeline.get_rag_retrieval()

    model = GenericVLLMBatch(
        path=args.language_model_path,
        thinking=False,
        vllm_kwargs={"max_model_len": 32768}
    )
    pipeline.get_rag_generations(model=model)