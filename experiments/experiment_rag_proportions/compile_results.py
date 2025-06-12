from utils.set_random_seed import set_random_seed
from utils.calculate_metric import calculate_agg_metric
import datetime
import os

def compile_result():
    """
    Compile the results of the experiment preview specified by preview_idx.

    Args:
        preview_idx (int): The index of the experiment preview to compile the results for.

    Returns:
        None
    """

    set_random_seed(seed=42)
    RAG25 = "rag_25"
    RAG50 = "rag_50"
    RAG100 = "rag_100"
    RAG150  = "rag_150"


    rag_generations = [
        f"{RAG25}/generations/rag_generations.json",
        f"{RAG50}/generations/rag_generations.json",
        f"{RAG100}/generations/rag_generations.json",
        f"{RAG150}/generations/rag_generations.json",
    ]

    datamodels_generations = [
        f"{RAG25}/generations/datamodels_generations.json",
        f"{RAG50}/generations/datamodels_generations.json",
        f"{RAG100}/generations/datamodels_generations.json",
        f"{RAG150}/generations/datamodels_generations.json",
    ]

    baseline_generation = f"baseline/generations/llama3-2-8b-instruct_baseline_generations.json"


    metrics = ["rouge_l", "squad_v2_best_f1", "squad_v2_best_exact"]

    os.makedirs("previews_results", exist_ok=True)

    print(f"Starting Baseline Preview  - {datetime.datetime.now()}")
    calculate_agg_metric(
        metrics=metrics,
        generation_path=baseline_generation,
        reference_path=f"50_test.feather",
        saving_path=f"previews_results/preview_baseline.feather"
    )
    print(f"Finished Baseline Preview - {datetime.datetime.now()}")

    for preview_idx in range(len(rag_generations)):

        print(f"Starting RAG Preview {preview_idx} - {datetime.datetime.now()}")
        calculate_agg_metric(
            metrics=metrics,
            generation_path=rag_generations[preview_idx],
            reference_path=f"50_test.feather",
            saving_path=f"previews_results/preview_{preview_idx}_rag.feather"
        )
        print(f"Finished RAG Preview {preview_idx} - {datetime.datetime.now()}")
        print(f"Starting Datamodels Preview {preview_idx} - {datetime.datetime.now()}")
        calculate_agg_metric(
            metrics=metrics,
            generation_path=datamodels_generations[preview_idx],
            reference_path=f"50_test.feather",
            saving_path=f"previews_results/preview_{preview_idx}_datamodels.feather"
        )
        print(f"Finished Datamodels Preview {preview_idx} - {datetime.datetime.now()}")
    
       

if __name__ == "__main__":

    compile_result()