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
        metric: str,
        agg: str,
        generation_path: str,
        reference_path: str,
        saving_path: str
) -> None:

    results = {
        "idx": [],
        "mean_metric": []
    }
    evaluator = Rouge_L_evaluator()


    for i in range(len(rag_generations)):
        rag_i_results = []
        datamodels_i_results = []
        results["idx"].append(i)
        for j in range(len(rag_generations[str(i)])):

            max_rag = 0
            max_datamodels = 0
            
            for res in  questions[i]["answers"].to_numpy().flatten()[0]:

                metric_rag = evaluator.evaluate(np.array([res]), np.array([str(rag_generations[str(i)][j])]))
                metric_datamodels = evaluator.evaluate(np.array([res]), np.array([str(datamodels_generations[str(i)][j])]))
                
                max_rag = max(max_rag, metric_rag[0])
                max_datamodels = max(max_datamodels, metric_datamodels[0])

            rag_i_results.append(max_rag)
            datamodels_i_results.append(max_datamodels)

        results["mean_metric_rag"].append(np.mean(rag_i_results))
        results["mean_metric_datamodels"].append(np.mean(datamodels_i_results))
        df_results = pl.DataFrame(results)
        df_results.write_ipc("results.feather")

    df_results = pl.DataFrame(results)