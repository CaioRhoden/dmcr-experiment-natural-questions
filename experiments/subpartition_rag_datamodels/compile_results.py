from utils.set_seed import set_all_seeds
from utils.calculate_metric import calculate_agg_metric
import datetime
import argparse


def compile_result(preview_idx: int):
    """
    Compile the results of the experiment preview specified by preview_idx.

    Args:
        preview_idx (int): The index of the experiment preview to compile the results for.

    Returns:
        None
    """
    PREFIX_PATH = "previews"
    L2 = "preview_50_L2"
    IP = "preview_50_IP"
    COSINE = "new_prompt_preview_50_cosine"
    K8 = "preview_50_L2_k8"
    NEW_PROMPT = "new_prompt_preview_50_L2"

    rag_generations = [
        f"{PREFIX_PATH}/{L2}/generations/rag_generations.json",
        f"{PREFIX_PATH}/{IP}/generations/rag_generations.json",
        f"{PREFIX_PATH}/{COSINE}/generations/rag_generations.json",
        f"{PREFIX_PATH}/{K8}/generations/rag_generations.json",
        f"{PREFIX_PATH}/{NEW_PROMPT}/generations/rag_generations.json",
    ]

    datamodels_generations = [
        f"{PREFIX_PATH}/{L2}/generations/datamodels_generations.json",
        f"{PREFIX_PATH}/{IP}/generations/datamodels_generations.json",
        f"{PREFIX_PATH}/{COSINE}/generations/datamodels_generations.json",
        f"{PREFIX_PATH}/{K8}/generations/datamodels_generations.json",
        f"{PREFIX_PATH}/{NEW_PROMPT}/generations/datamodels_generations.json",
    ]


    metrics = ["rouge_l", "squad_v2_best_f1", "squad_v2_best_exact"]

    print(f"Starting RAG Preview {preview_idx} - {datetime.datetime.now()}")
    calculate_agg_metric(
        metrics=metrics,
        generation_path=rag_generations[preview_idx],
        reference_path=f"{PREFIX_PATH}/50_test.feather",
        saving_path=f"previews_results/preview_{preview_idx}_rag.feather"
    )
    print(f"Finished RAG Preview {preview_idx} - {datetime.datetime.now()}")
    print(f"Starting Datamodels Preview {preview_idx} - {datetime.datetime.now()}")
    calculate_agg_metric(
        metrics=metrics,
        generation_path=datamodels_generations[preview_idx],
        reference_path=f"{PREFIX_PATH}/50_test.feather",
        saving_path=f"previews_results/preview_{preview_idx}_datamodels.feather"
    )
    print(f"Finished Datamodels Preview {preview_idx} - {datetime.datetime.now()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview_idx", "-i", type=int)
    args = parser.parse_args()
    compile_result(args.preview_idx)