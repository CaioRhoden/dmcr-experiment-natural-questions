from dataclasses import dataclass, field
from pathlib import Path
import random
from typing import Literal
import tyro


from utils.pipelines.RAGPipeline import RAGPipeline

root = Path(__file__).parent.parent.parent
@dataclass
class RagConfig:
    '''
    Configuration class for the experiment.
    '''
    log: bool = True    
    seed: Literal[1, 4, 54, 61, 73] = 1
    '''Random seed for reproducibility based on the previous random generated'''
    # RAG Based configs Config Fields
    wiki_path: str = "data/wiki_dump2018_nq_open/processed/wiki.feather"
    '''Path to the wiki dataset file.'''
    questions_path: str = "data/nq_open_gold/processed/test.feather"
    '''Path to the questions dataset file.'''
    language_model_path: str = "models/Llama-3.2-3B-Instruct"
    '''Path to the language model.'''
    retrieval_path: str = "retrieval/rag_retrieval_indexes.json"
    '''Path to the retrieval indexes JSON file.'''
    embedder_path: str = "models/bge-base-en-v1.5"
    '''Path to the embedder model.'''
    vector_db_path: str = "data/indices/bge_index.faiss"
    '''Path to the vector database.'''
    project_log: str = "small_window"
    '''Project log name for wandb'''
    model_run_id: str = "rag"
    '''ID of the model run.'''
    k: int = 8
    '''Number of top-k results to retrieve.'''
    size_index: int = 32
    '''Size of the index to be retrieved.'''
    batch_size: int = 500
    '''Size of inferences to be done at the same time'''
    attn_implementation: str = "sdpa"
    '''Attn implementation for the desired gpu, recommended default "sdpa" and "flash_attention_2" when possible'''
    thinking: bool = False
    '''Whether to enable the thinking mode in the model.'''
    start_idx: int = 0
    '''Starting index for the questions to be processed.'''
    end_idx: int|None = None
    '''Ending index for the questions to be processed. None means process all questions.'''

    
    # Pre-collections Config Fields
    instruction: str = "You are given a question and you MUST try to give a real SHORT ANSWER in 5 tokens"
    '''Instruction for the pre-collections step.'''
    lm_configs: dict[str, float|int] = field(default_factory=lambda: {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_new_tokens": 10,
        })
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

    return RAGPipeline(
        retrieval_path=args.retrieval_path,
        wiki_path=args.wiki_path,
        embedder_path=args.embedder_path,
        vector_db_path=args.vector_db_path,
        questions_path=args.questions_path,
        language_model_path=args.language_model_path,
        project_log=args.project_log,
        model_run_id=args.model_run_id,
        k=args.k,
        size_index=args.size_index,
        lm_configs=args.lm_configs,
        instruction=args.instruction,
        root_path=f"runs/experiment_{args.seed}",
        tags = args.tags,
        log=args.log,
        attn_implementation=args.attn_implementation,
        thinking=args.thinking,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    args = tyro.cli(RagConfig)
    args.tags.append("rag")
    args.questions_path = f"runs/experiment_{args.seed}/questions.feather"
    args.language_model_path = f"{root}/{args.language_model_path}"
    args.wiki_path = f"{root}/{args.wiki_path}"
    args.retrieval_path = f"{root}/{args.retrieval_path}"
    args.embedder_path = f"{root}/{args.embedder_path}"
    args.vector_db_path = f"{root}/{args.vector_db_path}"


    pipeline = initiate_pipeline(args)
    pipeline.setup()
    pipeline.get_rag_retrieval()
    pipeline.get_rag_generations()