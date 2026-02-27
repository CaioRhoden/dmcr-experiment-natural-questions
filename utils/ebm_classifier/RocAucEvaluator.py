from sklearn.metrics import roc_auc_score
import polars as pl
import numpy as np


class RocAucEvaluator:

    def __init__(self, subfolders:list[str] = ["experiment_1", "experiment_4", "experiment_54", "experiment_61", "experiment_73"]):
        self.subfolders = subfolders

    def evaluate(self, models, X_test, y_test) -> pl.DataFrame:
        """
        Perfrom ROC-AUC score evaluation for the binary classification models.

        Args:
            models (list): A list of trained EBM models for each experiment and model, the shape is (num_exp_subfolders, num_models).
            X_test (list): A list of testing input data for each experiment, the shape is (num_exp_subfolders, num_models, num_samples, num_features).
            y_test (list): A list of testing labels for each experiment, the shape is (num_exp_subfolders, num_models, num_samples).

        Returns:
            pl.DataFrame: A Polars DataFrame containing the evaluation results with columns "experiment" and "auc". The "experiment" column indicates the experiment name of the subfolder.
        """

        num_subfolders = len(X_test)
        num_models = len(X_test[0])

        _eval_df = {
            "experiment": [],
            "auc": [],
        }

        for i in range(num_subfolders):
            for j in range(num_models):
                _X = X_test[i][j]
                _y = np.asarray(y_test[i][j]).reshape(-1)
                proba = np.asarray(models[i][j].predict_proba(_X))

                if proba.ndim == 1:
                    positive_scores = proba
                elif proba.shape[1] == 1:
                    model_classes = np.asarray(models[i][j].classes_)
                    if model_classes[0] == 1:
                        positive_scores = np.ones_like(_y, dtype=float)
                    else:
                        positive_scores = np.zeros_like(_y, dtype=float)
                else:
                    positive_scores = proba[:, 1]

                if np.unique(_y).size < 2:
                    score = None
                else:
                    score = roc_auc_score(_y, positive_scores)

                _eval_df["experiment"].append(self.subfolders[i])
                _eval_df["auc"].append(score)
        
        return pl.DataFrame(_eval_df)