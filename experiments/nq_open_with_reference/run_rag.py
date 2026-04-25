from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import tyro
from utils.pipelines.RAGPipeline import RAGPipeline
from utils.set_random_seed import set_random_seed

set_random_seed(42)
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
}


root = Path(__file__).parent.parent.parent
@dataclass
class RagConfig:
    '''
    Configuration class for the experiment.
    '''
    log: bool = True
    # RAG Based configs Config Fields
    project_log: str = "nq_open_reference"
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

    # Pre-collections Config Fields
    instruction: Literal["default", "extraction"] = "default"
    '''Instruction to be used for the language model.'''
    lm_configs: dict[str, float|int] = field(default_factory=lambda: LM_CONFIGS)
    tags: list[str] = field(default_factory=lambda: ["rag"])
    '''List of tags for the experiment.'''

    root_path: str = "runs"

    ## Config
    only_generate: bool = False
    '''Whether to only run the generation step.'''
    only_retrieval: bool = False
    '''Whether to only run the retrieval step.'''



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
        wiki_path=f"{root}/data/wiki_dump2018_nq_open/processed/wiki.feather",
        embedder_path=f"{root}/models/bge-base-en-v1.5",
        vector_db_path=f"{root}/data/indices/bge_index.faiss",
        questions_path=f"{root}/data/nq_open/processed/dev.feather",
        language_model_path=f"{root}/models/Llama-3.2-3B-Instruct",
        project_log=args.project_log,
        model_run_id=args.model_run_id,
        k=args.k,
        size_index=args.size_index,
        lm_configs=args.lm_configs,
        instruction=INSTRUCTIONS_DICT[args.instruction],
        root_path=args.root_path,
        tags = args.tags,
        log=args.log,
        batch_size=3610,
        seed=42,
    )


if __name__ == "__main__":
    args = tyro.cli(RagConfig)
    args.tags.append("rag")


    pipeline = initiate_pipeline(args)
    pipeline.setup()

    if args.only_retrieval:
        pipeline.get_rag_retrieval()
    elif args.only_generate:
        pipeline.get_rag_generations()
    else:
        pipeline.get_rag_retrieval()
        pipeline.get_rag_generations()