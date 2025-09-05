from dataclasses import dataclass, field
from pathlib import Path
import random
import tyro


from utils.set_random_seed import set_random_seed
from utils.pipelines.RAGPipeline import RAGPipeline
from pathlib import Path
root = Path(__file__).parent.parent.parent


set_random_seed(42)
seed = random.randint(0, 10000)
root = Path(__file__).parent.parent.parent
@dataclass
class RagConfig:
    '''
    Configuration class for the experiment.
    '''

    model: str = "None"
    '''Tag for the experiment section.'''
    log: bool = True    
    # RAG Based configs Config Fields
    wiki_path: str = "data/wiki_dump2018_nq_open/processed/wiki.feather"
    '''Path to the wiki dataset file.'''
    questions_path: str = "data/nq_open_gold/processed/dev.feather"
    '''Path to the questions dataset file.'''
    language_model_path: str = "models/llms/Llama-3.2-3B-Instruct"
    '''Path to the language model.'''
    retrieval_path: str = "retrieval/rag_retrieval_indexes.json"
    '''Path to the retrieval indexes JSON file.'''
    embeder_path: str = "models/llms/bge-base-en-v1.5"
    '''Path to the embedder model.'''
    vector_db_path: str = "data/wiki_dump2018_nq_open/processed/wiki_cosine.index"
    '''Path to the vector database.'''
    project_log: str = "run_validation_set_nq"
    '''Project log name for wandb'''
    model_run_id: str = "rag"
    '''ID of the model run.'''
    k: int = 16
    '''Number of top-k results to retrieve.'''
    size_index: int = 100
    '''Size of the index to be retrieved.'''
    batch_size: int = 8
    '''Size of inferences to be done at the same time'''
    attn_implementation: str = "sdpa"
    '''Attn implementation for the desired gpu, recommended default "sdpa" and "flash_attention_2" when possible'''
    start_idx: int = 0
    '''Starting index for the questions to be processed.'''
    end_idx: int|None = None
    '''Ending index for the questions to be processed. None means process all questions.'''

    
    # Pre-collections Config Fields
    instruction: str = "You are given a question and you MUST try to give a real SHORT ANSWER in 5 tokens, you can use the available documents but if they are not helpful, try to answer without them"
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
        embeder_path=args.embeder_path,
        vector_db_path=args.vector_db_path,
        questions_path=args.questions_path,
        language_model_path=args.language_model_path,
        project_log=args.project_log,
        model_run_id=args.model_run_id,
        k=args.k,
        size_index=args.size_index,
        lm_configs=args.lm_configs,
        instruction=args.instruction,
        root_path=f"{args.model}",
        seed=seed,
        tags = args.tags,
        log=args.log,
        attn_implementation=args.attn_implementation
    )


if __name__ == "__main__":
    args = tyro.cli(RagConfig)
    args.tags.append("rag")
    args.tags.append(args.model)
    args.questions_path = f"{root}/{args.questions_path}"
    args.language_model_path = f"{root}/{args.language_model_path}"
    args.wiki_path = f"{root}/{args.wiki_path}"
    args.retrieval_path = f"{root}/{args.retrieval_path}"
    args.embeder_path = f"{root}/{args.embeder_path}"
    args.vector_db_path = f"{root}/{args.vector_db_path}"


    pipeline = initiate_pipeline(args)
    pipeline.setup()
    pipeline.get_rag_retrieval()
    pipeline.get_rag_generations()