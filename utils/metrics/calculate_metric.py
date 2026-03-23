import json
import os
import random
from typing import Callable

import numpy as np
import polars as pl
import torch

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


def _get_evaluator(metric: str):
    """Get the appropriate evaluator for the given metric type."""
    evaluators = {
        "rouge_l": Rouge_L_evaluator,
        "squad_v2_best_f1": lambda: SquadV2Evaluator("best_f1"),
        "squad_v2_best_exact": lambda: SquadV2Evaluator("best_exact"),
    }
    if metric not in evaluators:
        raise ValueError(f"Invalid metric: {metric}")
    return evaluators[metric]()


def calculate_agg_metric(
        metrics: list[str],
        generation_path: str,
        reference_path: str,
        saving_path: str | None,
        agg: Callable[[np.ndarray], float] = np.mean
) -> pl.DataFrame | None:
    """
    Calculate aggregated Rouge-L and/or SQuAD V2 metrics for generations.

    Args:
        metrics (list[str]): List of metrics to calculate ("rouge_l", "squad_v2_best_f1", "squad_v2_best_exact").
        generation_path (str): Path to JSON file containing generations.
        reference_path (str): Path to polars IPC file containing reference answers.
        saving_path (str | None): Path to save results in IPC format. If None, returns DataFrame.
        agg (Callable): Aggregation function (default: torch.mean).

    Returns:
        pl.DataFrame | None: Results DataFrame if saving_path is None, else None.
    """
    with open(generation_path) as f:
        generations = json.load(f)
    
    questions = pl.read_ipc(reference_path)
    
    results = {"idx": [], "value": [], "metric": []}
    
    for metric_name in metrics:
        print(f"Calculating {metric_name}...")
        evaluator = _get_evaluator(metric_name)
        
        for idx, generation_set in enumerate(generations.values()):
            reference = questions[idx]["answers"].to_numpy()[0].tolist()
            
            metric_scores = agg(np.array([
                    evaluator.evaluate(
                        np.array([reference]),
                        np.array([[str(generation)]])
                    )
                for generation in generation_set
            ], dtype=np.float32))
            results["idx"].append(idx)
            results["value"].append(metric_scores)
            results["metric"].append(metric_name)
    
    df_results = pl.DataFrame(results)
    
    if saving_path is not None:
        df_results.write_ipc(saving_path, compression="zstd")
    else:
        return df_results