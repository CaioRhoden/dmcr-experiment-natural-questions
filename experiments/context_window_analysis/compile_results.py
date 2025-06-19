from utils.set_random_seed import set_random_seed
from utils.calculate_metric import calculate_agg_metric
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
    PREFIX_PATH_LLAMA = "llama"
    PREFIX_PATH_QWEN = "qwen"
    
    K4 = "k_4"
    K8 = "k_8"
    K16 = "k_16"
    K32 = "k_32"

    rag_generations = [
        f"{PREFIX_PATH_LLAMA}/{K4}/generations/rag_generations.json",
        f"{PREFIX_PATH_LLAMA}/{K8}/generations/rag_generations.json",
        f"{PREFIX_PATH_LLAMA}/{K16}/generations/rag_generations.json",
        f"{PREFIX_PATH_LLAMA}/{K32}/generations/rag_generations.json",
        f"{PREFIX_PATH_QWEN}/{K4}/generations/rag_generations.json",
        f"{PREFIX_PATH_QWEN}/{K8}/generations/rag_generations.json",
        f"{PREFIX_PATH_QWEN}/{K16}/generations/rag_generations.json",
        f"{PREFIX_PATH_QWEN}/{K32}/generations/rag_generations.json",
    ]

    datamodels_generations = [
        f"{PREFIX_PATH_LLAMA}/{K4}/generations/datamodels_generations.json",
        f"{PREFIX_PATH_LLAMA}/{K8}/generations/datamodels_generations.json",
        f"{PREFIX_PATH_LLAMA}/{K16}/generations/datamodels_generations.json",
        f"{PREFIX_PATH_LLAMA}/{K32}/generations/datamodels_generations.json",
        f"{PREFIX_PATH_QWEN}/{K4}/generations/datamodels_generations.json",
        f"{PREFIX_PATH_QWEN}/{K8}/generations/datamodels_generations.json",
        f"{PREFIX_PATH_QWEN}/{K16}/generations/datamodels_generations.json",        
        f"{PREFIX_PATH_QWEN}/{K32}/generations/datamodels_generations.json",

    ]

    baseline_generation_llama = f"{PREFIX_PATH_LLAMA}/baseline/generations/llama3-2-8b-instruct_baseline_generations.json"
    baseline_generation_qwen = f"{PREFIX_PATH_QWEN}/baseline/generations/qwen3-8b-baseline_generations.json"


    metrics = ["rouge_l", "squad_v2_best_f1", "squad_v2_best_exact"]

    print(f"Starting Baseline Preview  - {datetime.datetime.now()}")
    calculate_agg_metric(
        metrics=metrics,
        generation_path=baseline_generation_llama,
        reference_path="50_test.feather",
        saving_path="previews_results/preview_baseline_llama.feather"
    )

    # calculate_agg_metric(
    #     metrics=metrics,
    #     generation_path=baseline_generation_qwen,
    #     reference_path=f"{PREFIX_PATH_LLAMA}/50_test.feather",
    #     saving_path="previews_results/preview_baseline_qwen.feather"
    # )
    # print(f"Finished Baseline Preview - {datetime.datetime.now()}")

    for preview_idx in range(len(rag_generations)):

        if preview_idx < 4:
            print(f"Starting RAG Preview {preview_idx} - {datetime.datetime.now()}")
            calculate_agg_metric(
                metrics=metrics,
                generation_path=rag_generations[preview_idx],
                reference_path="50_test.feather",
                saving_path=f"previews_results/preview_{preview_idx}_rag.feather"
            )
            print(f"Finished RAG Preview {preview_idx} - {datetime.datetime.now()}")
            print(f"Starting Datamodels Preview {preview_idx} - {datetime.datetime.now()}")
            calculate_agg_metric(
                metrics=metrics,
                generation_path=datamodels_generations[preview_idx],
                reference_path="50_test.feather",
                saving_path=f"previews_results/preview_{preview_idx}_datamodels.feather"
            )
            print(f"Finished Datamodels Preview {preview_idx} - {datetime.datetime.now()}")
        else:
            print(f"Starting RAG Preview {preview_idx} - {datetime.datetime.now()}")
            calculate_agg_metric(
                metrics=metrics,
                generation_path=rag_generations[preview_idx],
                reference_path="50_test.feather",
                saving_path=f"previews_results/preview_{preview_idx}_rag.feather"
            )
            print(f"Finished RAG Preview {preview_idx} - {datetime.datetime.now()}")
            print(f"Starting Datamodels Preview {preview_idx} - {datetime.datetime.now()}")
            calculate_agg_metric(
                metrics=metrics,
                generation_path=datamodels_generations[preview_idx],
                reference_path="50_test.feather",
                saving_path=f"previews_results/preview_{preview_idx}_datamodels.feather"
            )
            print(f"Finished Datamodels Preview {preview_idx} - {datetime.datetime.now()}")
    
       

if __name__ == "__main__":

    compile_result()