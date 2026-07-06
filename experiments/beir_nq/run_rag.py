from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import tyro
from utils.pipelines.RAGPipeline import RAGPipeline
from utils.set_random_seed import set_random_seed

set_random_seed(42)
INSTRUCTION = "You are given a question and you MUST respond by EXTRACTING the answer (max 5 tokens) from one of the provided documents. If none of the documents contain the answer, respond with NO-RES. Begin your answer by providing a short explanation. Be as objective as possible. After providing your explanation, please generate your response by strictly following this format: \"RESPONSE: [[<response>]]\"."

LM_CONFIGS = {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_new_tokens": 15,
}


root = Path(__file__).parent.parent.parent
@dataclass
class RagConfig:
    '''
    Configuration class for the experiment.
    '''
    log: bool = True
    # RAG Based configs Config Fields
    project_log: str = "beir_nq"
    '''Project log name for wandb'''
    model_run_id: str = "rag"
    '''ID of the model run.'''
    k: int = 10
    '''Number of top-k results to retrieve.'''
    size_index: int = 100
    '''Size of the index to be retrieved.'''
    start_idx: int = 0
    '''Starting index for the questions to be processed.'''
    end_idx: int|None = None
    '''Ending index for the questions to be processed. None means process all questions.'''

    lm_configs: dict[str, float|int] = field(default_factory=lambda: LM_CONFIGS)
    tags: list[str] = field(default_factory=lambda: ["rag"])
    '''List of tags for the experiment.'''

    root_path: str = "runs/qwen"
    '''Root path for saving results and logs.'''

    ## Config
    only_generate: bool = False
    '''Whether to only run the generation step.'''
    only_retrieval: bool = False
    '''Whether to only run the retrieval step.'''
    language_model_path: str = "models/Qwen3-4B-Instruct-2507"
    '''Path to the language model to be used in the pipeline.'''
    
    ## Retrieval type and path config
    retrieval_type: str = "bm25"
    '''Type of retrieval to use: "dpr" or "bm25"'''
    bm25_path: str = "data/beir_nq/processed/bm25_index.pkl"
    '''Path to the BM25 index pickle file (only used when retrieval_type="bm25")'''




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
        retrieval_path=f"{root}/retrieval/rag_retrieval_indexes.json",
        wiki_path=f"{root}/data/beir_nq/processed/corpus.feather",
        embedder_path=f"{root}/models/bge-base-en-v1.5",
        vector_db_path=f"{root}/data/indices/bge_index.faiss",
        questions_path=f"{root}/data/beir_nq/processed/queries.feather",
        language_model_path=f"{root}/{args.language_model_path}",
        project_log=args.project_log,
        model_run_id=args.model_run_id,
        k=args.k,
        size_index=args.size_index,
        lm_configs=args.lm_configs,
        instruction=INSTRUCTION,
        root_path=args.root_path,
        tags=args.tags,
        log=args.log,
        batch_size=3452,
        seed=42,
        retrieval_type=args.retrieval_type,
        bm25_path=f"{root}/{args.bm25_path}",
    )


if __name__ == "__main__":
    args = tyro.cli(RagConfig)
    args.tags.append("rag")


    pipeline = initiate_pipeline(args)
    pipeline.setup()

    if args.only_retrieval:
        pipeline.get_rag_retrieval()
    elif args.only_generate:
        pipeline.get_rag_generations(extract_flag=True)
    else:
        pipeline.get_rag_retrieval()
        pipeline.get_rag_generations(extract_flag=True)