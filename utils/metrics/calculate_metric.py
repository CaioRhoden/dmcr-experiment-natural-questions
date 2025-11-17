import os
import random
import numpy as np
import torch

import json
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from dmcr.evaluators.Rouge_L_evaluator import Rouge_L_evaluator
from dmcr.evaluators.Squadv2Evaluator import SquadV2Evaluator

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
seed = 42

torch.backends.cudnn.enabled = False
# NumPy
np.random.seed(seed)
random.seed(seed)
# PyTorch
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def calculate_agg_metric(
        metrics: list[str],
        generation_path: str,
        reference_path: str,
        saving_path: str | None
) -> pl.DataFrame | None:

    """
    Calculate mean and max of Rouge-L and/or SQuAD V2 metrics for generations.

    Args:
        metrics (list[str]): List of metrics to calculate. Can be "rouge_l" and/or "squad_v2".
        generation_path (str): Path to JSON file containing generations.
        reference_path (str): Path to JSON file containing reference answers.
        saving_path (str): Path to save the results in IPC format.

    Returns:
        None
    """
    
    results = {
        "idx": [],
        "mean": [],
        "max": [],
        "metric": []

    }
    
    generations = json.load(open(generation_path, "r"))
    questions = pl.read_ipc(reference_path)

    for metric in metrics:
        if metric == "rouge_l":
            evaluator = Rouge_L_evaluator()
        elif metric == "squad_v2_best_f1":
            evaluator = SquadV2Evaluator("best_f1")
        elif metric == "squad_v2_best_exact":
            evaluator = SquadV2Evaluator("best_exact")
        else:
            raise ValueError(f"Invalid evaluator: {metric}")	

        for i in range(len(generations)):
            results_i = []
            results["idx"].append(i)
            for j in range(len(generations[str(i)])):

                    

                max_metric = 0.0
                res =  questions[i]["answers"].to_numpy()[0].tolist()
            
                metric_value = evaluator.evaluate(np.array([res]), np.array([[str(generations[str(i)][j])]]))
                max_metric = max(max_metric, metric_value[0])

                results_i.append(max_metric)

            results["mean"].append(np.mean(results_i))
            results["max"].append(np.max(results_i))
            results["metric"].append(metric)

        df_results = pl.DataFrame(results)
        if saving_path is not None:
            df_results.write_ipc(saving_path, compression="zstd")

        else:
            return df_results