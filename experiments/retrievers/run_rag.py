from dataclasses import dataclass, field
from pathlib import Path
import random
from re import S
from IPython import embed
from sympy import Li
from torch import embedding, seed
import tyro
from dmcr.models import GenericVLLMBatch
from typing import Literal

from utils.set_random_seed import set_random_seed
from utils.pipelines.RAGPipeline import RAGPipeline

from pathlib import Path
root = Path(__file__).parent.parent.parent


root = Path(__file__).parent.parent.parent.parent
@dataclass
class RagConfig:
    '''
    Configuration class for the experiment.
    '''
    retriever: Literal["bge","hybrid", "qwen"] = "bge"
    '''Type of retriever to be used in the experiment.'''
    vector_db_path: str
    '''Path to the vector database.'''
    embedder_path: str
    '''Path to the embedder model.'''

    nprobe: int = 32
    '''Number of probes for the vector search.'''



def initiate_pipeline(args: RagConfig) -> RAGPipeline:
    """
    Initiates the baseline pipeline with the provided arguments.
    
    Args:
        args (ParametersConfig): The configuration parameters for the pipeline.
        seed (int): The random seed for reproducibility.
    
    Returns:
        ZeroShotBaselinePipeline: An instance of the baseline pipeline.
    """

    model_run_id = f"rag-{args.retriever}-{args.nprobe}"
    wiki_path: str = "data/wiki_dump2018_nq_open/processed/wiki.feather"

    lm_configs = {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_new_tokens": 10,
            "n": 1
    }

    questions_path = "questions.feather"
    retrieval_path = f"retrieval/{args.retriever}_indices.json"
    if args.retriever != "hybrid":
        return RAGPipeline(
            retrieval_path=retrieval_path,
            wiki_path=f"{root}/{wiki_path}",
            embedder_path=f"{root}/{args.embedder_path}",
            vector_db_path=f"{root}/{args.vector_db_path}",
            questions_path=questions_path,
            project_log="debugging_recall_validation",
            model_run_id=model_run_id,
            k=16,
            size_index=100,
            instruction="You are given a question and you MUST try to give a real SHORT ANSWER in 5 tokens",
            root_path=".",
            seed=42,
            tags=[],
            log=False,
            batch_size=500,
            lm_configs=lm_configs
        )



if __name__ == "__main__":
    args = tyro.cli(RagConfig)

    set_random_seed(args.seed)


    pipeline = initiate_pipeline(args)
    pipeline.setup()
    pipeline.get_rag_retrieval()

    model = GenericVLLMBatch(
        path=f"{root}/models/Llama-3.2-3B-Instruct",
        thinking=False,
        vllm_kwargs={"max_model_len": 32768}
    )
    pipeline.get_rag_generations(model=model)