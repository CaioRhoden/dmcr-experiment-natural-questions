from dataclasses import dataclass
from pathlib import Path

from utils.pipelines.RAGPipeline import RAGPipeline
import tyro

root = Path(__file__).parent.parent.parent

@dataclass
class RAGConfig:
    questions_path: str = "experiment_81/questions.feather"
    '''Path to the questions feather file.'''

    root_path: str = "experiment_81"
    '''Root path for all experiment outputs.'''

    instruction: str = "You are given a question and you MUST try to give a real SHORT ANSWER in 5 tokens"
    '''Instruction for the language model.'''

if __name__ == "__main__":
    config = tyro.cli(RAGConfig)

    pipeline = RAGPipeline(
        retrieval_path=f"{root}/retrieval/rag_retrieval_indexes.json",
        wiki_path=f"{root}/data/wiki_dump2018_nq_open/processed/wiki.feather",
        embedder_path=f"{root}/models/bge-base-en-v1.5",
        vector_db_path=f"{root}/data/indices/bge_index.faiss",
        questions_path=config.questions_path,
        language_model_path=f"{root}/models/Llama-3.2-3B-Instruct",
        project_log="judge_comparative",
        model_run_id="rag",
        k=16,
        size_index=100,
        lm_configs={
            "temperature": 0.7,
            "top_p": 0.9,
            "max_new_tokens": 10,
        },
        instruction=config.instruction,
        root_path=config.root_path,
        tags=["rag"],
        log=True,
        batch_size=10,
    )
    pipeline.setup()
    pipeline.get_rag_retrieval()
    pipeline.get_rag_generations()