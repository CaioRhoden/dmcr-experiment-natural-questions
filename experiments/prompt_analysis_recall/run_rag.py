from dataclasses import dataclass, field
from pathlib import Path
import random
from typing import Literal
import tyro


from utils.set_random_seed import set_random_seed
from utils.pipelines.RAGPipeline import RAGPipeline
from pathlib import Path
root = Path(__file__).parent.parent.parent
SubsizeLiteral = Literal[32, 64, 128]
KLiteral = Literal[4,8,16]
InstructionIndexLiteral = Literal[0,1,2,3,4,5,6,7]

INSTRUCTIONS = [
    "You are given a question and you MUST give a SHORT ANSWER in 5 tokens, you can use the available documents but if they are not helpful, try to answer without them",
    "Answer the following question in 5 tokens",
    "Given the following context, answer the question directly in 5 tokens",
    "You are an expert at answering questions. Use the context to answer the question in 5 tokens. If the context is not helpful, answer without it",
    "You are given a question and you MUST give a SHORT ANSWER in 5 tokens, you can use the available documents but if they are not helpful try to answer anyway, YOU CAN'T SAY IT'S NOT POSSIBLE TO ANSWER THE QUESTION",
    "You are given a question and you MUST give a SHORT ANSWER in 5 tokens, you can use the available documents but if they are not helpful try to answer anyway, YOU ALWAYS KNOW THE ANSWER",
    "You are given a question and you MUST give a SHORT ANSWER in 5 tokens, there are documents available.",
    "You are given a question and you MUST give a SHORT ANSWER in 5 tokens",

]



set_random_seed(42)
seed = random.randint(0, 10000)
root = Path(__file__).parent.parent.parent
@dataclass
class RagConfig:
    '''
    Configuration class for the experiment.
    '''
    size_folder: SubsizeLiteral = 32
    '''Subset size and folder of subset parameter'''
    k: KLiteral = 4
    '''size of context and subfolder identifier'''
    log: bool = True    
    # RAG Based configs Config Fields
    wiki_path: str = "data/wiki_dump2018_nq_open/processed/wiki.feather"
    '''Path to the wiki dataset file.'''
    questions_path: str = "questions_1000_42_dev.feather"
    '''Path to the questions dataset file.'''
    language_model_path: str = "models/Llama-3.2-3B-Instruct"
    '''Path to the language model.'''
    retrieval_path: str = "retrieval/rag_retrieval_indexes.json"
    '''Path to the retrieval indexes JSON file.'''
    embedder_path: str = "models/bge-base-en-v1.5"
    '''Path to the embedder model.'''
    vector_db_path: str = "data/wiki_dump2018_nq_open/processed/wiki_cosine.index"
    '''Path to the vector database.'''
    project_log: str = "prompt_analysis_recall"
    '''Project log name fgor wandb'''
    model_run_id: str = "rag"
    '''ID of the model run.'''
    batch_size: int = 1000
    '''Size of inferences to be done at the same time'''
    thinking: bool = False
    '''Whether to enable the thinking mode in the model.'''
    seed: int = 42
    '''Random seed for reproducibility.'''

    
    # Pre-collections Config Fields
    instruction_index: InstructionIndexLiteral = 0
    '''Index of the instruction to be used from the INSTRUCTIONS list'''
    lm_configs: dict[str, float|int] = field(default_factory=lambda: {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_new_tokens": 15,
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
        size_index=args.size_folder,
        lm_configs=args.lm_configs,
        instruction=INSTRUCTIONS[args.instruction_index],
        root_path=f"subset_{args.size_folder}/k_{args.k}/instruction_{args.instruction_index}",
        seed=args.seed,
        tags = args.tags,
        log=args.log,
        thinking=args.thinking,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    args = tyro.cli(RagConfig)
    args.language_model_path = f"{root}/{args.language_model_path}"
    args.wiki_path = f"{root}/{args.wiki_path}"
    args.retrieval_path = f"{root}/{args.retrieval_path}"
    args.embedder_path = f"{root}/{args.embedder_path}"
    args.vector_db_path = f"{root}/{args.vector_db_path}"


    pipeline = initiate_pipeline(args)
    pipeline.setup()
    pipeline.get_rag_retrieval()
    pipeline.get_rag_generations()