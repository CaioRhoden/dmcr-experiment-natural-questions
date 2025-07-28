from utils.set_random_seed import set_random_seed
from utils.metrics.calculate_metric import calculate_agg_metric
import datetime
import argparse


def compile_result():
    """
    Compile the results of the experiment preview specified by preview_idx.

    Args:
        preview_idx (int): The index of the experiment preview to compile the results for.

    Returns:
        None
    """

    set_random_seed(seed=42)
    CONTRIEVER = "contriever"
    BTE = "bte"
    GTE = "gte"

    generations_paths = [CONTRIEVER, BTE, GTE]

    rag_generations = [f"{p}/generations/rag_generations.json" for p in generations_paths]
    datamodels_generations = [f"{p}/generations/datamodels_generations.json" for p in generations_paths]



    baseline_generation = f"baseline/generations/llama3-2-8b-instruct_baseline_generations.json"


    metrics = ["rouge_l", "squad_v2_best_f1", "squad_v2_best_exact"]


    print(f"Starting Baseline Preview - {datetime.datetime.now()}")
    calculate_agg_metric(
        metrics=metrics,
        generation_path=baseline_generation,
        reference_path=f"50_test.feather",
        saving_path=f"previews_results/preview_baseline.feather"
    )

    for preview_idx in range(len(generations_paths)):
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
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    compile_result()