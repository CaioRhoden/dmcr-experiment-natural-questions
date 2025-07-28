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
    PREFIX_PATH = "../"
    

    L1 = "lasso_1"
    L2 = "lasso_2"
    L3 = "lasso_3"
    L4 = "lasso_4"
    LR = "linear_regressor"
    

    rag_generations = f"{PREFIX_PATH}/{LR}/generations/rag_generations.json"

    datamodels_generations = [
        f"{PREFIX_PATH}/{LR}/generations/datamodels_generations.json",
        f"{PREFIX_PATH}/{L1}/generations/datamodels_generations.json",
        f"{PREFIX_PATH}/{L2}/generations/datamodels_generations.json",
        f"{PREFIX_PATH}/{L3}/generations/datamodels_generations.json",
        f"{PREFIX_PATH}/{L4}/generations/datamodels_generations.json",

    ]

    baseline_generation_llama = f"{PREFIX_PATH}/baseline/generations/llama3-2-8b-instruct_baseline_generations.json"

    metrics = ["rouge_l"]

    print(f"Starting Baseline Preview  - {datetime.datetime.now()}")
    calculate_agg_metric(
        metrics=metrics,
        generation_path=baseline_generation_llama,
        reference_path="../50_test.feather",
        saving_path="baseline.feather"
    )

    
    calculate_agg_metric(
        metrics=metrics,
        generation_path=rag_generations,
        reference_path="../50_test.feather",
        saving_path=f"rag.feather"
    )


    for preview_idx, _ in enumerate(datamodels_generations):

        calculate_agg_metric(
            metrics=metrics,
            generation_path=datamodels_generations[preview_idx],
            reference_path="../50_test.feather",
            saving_path=f"preview_{preview_idx}_datamodels.feather"
        )
        print(f"Finished Datamodels Preview {preview_idx} - {datetime.datetime.now()}")
    
       

if __name__ == "__main__":

    compile_result()