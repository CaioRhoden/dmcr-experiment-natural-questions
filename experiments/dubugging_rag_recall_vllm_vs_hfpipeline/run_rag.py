from dataclasses import dataclass, field
from pathlib import Path
import random
from typing import Literal
import tyro


from utils.set_random_seed import set_random_seed
from utils.pipelines.RAGPipeline import RAGPipeline
from pathlib import Path
root = Path(__file__).parent.parent.parent


root = Path(__file__).parent.parent.parent
@dataclass
class RagConfig:
    '''
    Configuration class for the experiment.
    '''
    exp_name: str
    '''Name of the experiment and folder to save results'''
    log: bool = True    
    # RAG Based configs Config Fields
    wiki_path: str = "data/wiki_dump2018_nq_open/processed/wiki.feather"
    '''Path to the wiki dataset file.'''
    questions_path: str = "questions_2500_42_dev.feather"
    '''Path to the questions dataset file.'''
    language_model_path: str = "models/Llama-3.2-3B-Instruct"
    '''Path to the language model.'''
    k: int = 16
    '''Number of documents to retrieve.'''
    retrieval_path: str = "retrieval/rag_retrieval_indexes.json"
    '''Path to the retrieval indexes JSON file.'''
    embeder_path: str = "models/bge-base-en-v1.5"
    '''Path to the embedder model.'''
    vector_db_path: str = "data/wiki_dump2018_nq_open/processed/wiki_cosine.index"
    '''Path to the vector database.'''
    project_log: str = "prompt_analysis_recall"
    '''Project log name fgor wandb'''
    model_run_id: str = "rag"
    '''ID of the model run.'''
    batch_size: int = 2500
    '''Size of inferences to be done at the same time'''
    thinking: bool = False
    '''Whether to enable the thinking mode in the model.'''
    seed: int = 42
    '''Random seed for reproducibility.'''
    batch_model_lm: Literal["hf", "vllm"] = "vllm"
    '''Batch model to be used, either "hf" or "vllm".'''
    lm_configs: dict[str, float|int] = field(default_factory=lambda: {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_new_tokens": 15,
        })
    lm_instance_kwargs: dict = field(default_factory=lambda: {
            
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
        embeder_path=args.embeder_path,
        vector_db_path=args.vector_db_path,
        questions_path=args.questions_path,
        language_model_path=args.language_model_path,
        project_log=args.project_log,
        model_run_id=args.model_run_id,
        k=args.k,
        size_index=100,
        lm_configs=args.lm_configs,
        instruction="You are given a question and you MUST give a SHORT ANSWER in 5 tokens, you can use the available documents but if they are not helpful, try to answer without them",
        root_path=f"{args.exp_name}",
        seed=args.seed,
        tags = args.tags,
        log=args.log,
        thinking=args.thinking,
        batch_size=args.batch_size,
        lm_instance_kwargs=args.lm_instance_kwargs,
    )


if __name__ == "__main__":
    args = tyro.cli(RagConfig)
    set_random_seed(args.seed)
    args.language_model_path = f"{root}/{args.language_model_path}"
    args.wiki_path = f"{root}/{args.wiki_path}"
    args.retrieval_path = f"{root}/{args.retrieval_path}"
    args.embeder_path = f"{root}/{args.embeder_path}"
    args.vector_db_path = f"{root}/{args.vector_db_path}"


    pipeline = initiate_pipeline(args)
    pipeline.setup()
    pipeline.get_rag_retrieval()
    pipeline.get_rag_generations(args.batch_model_lm)