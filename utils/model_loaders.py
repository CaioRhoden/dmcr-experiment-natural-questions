from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression


class BaseModelLoader(ABC):
    """Generic interface for model loaders.

    Implementations should return a dict mapping experiment name -> list of models (or None)
    for that experiment.
    """

    @abstractmethod
    def load(
        self, subfolder: str, experiments: Optional[List[str]] = None
    ) -> List[List[Optional[object]]]:
        raise NotImplementedError()


class LogisticRegressionModelLoader(BaseModelLoader):
    """Loads Logistic Regression models from saved coefficient and bias tensors.

    The notebook in this project saves weights in the following layout:
      weights/{subfolder}/{experiment}/lr_weights.pt
      weights/{subfolder}/{experiment}/lr_bias.pt

    Each file is a torch tensor where the first dimension indexes the sample (e.g. 500 models).
    This loader reconstructs `sklearn.linear_model.LogisticRegression` instances by
    assigning `coef_`, `intercept_`, `classes_` and `n_features_in_` so `predict_proba`
    and other read-only methods work.
    """

    def __init__(
        self,
        weights_dir: str = "weights",
        weight_filename: str = "lr_weights.pt",
        bias_filename: str = "lr_bias.pt",
    ) -> None:
        self.weights_dir = weights_dir
        self.weight_filename = weight_filename
        self.bias_filename = bias_filename

    def _ensure_np(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy()
        return np.array(obj)

    def _build_lr_from_coef_intercept(self, coef: np.ndarray, intercept: np.ndarray) -> LogisticRegression:
        # coef expected shape: (n_features,) or (1, n_features)
        coef = np.asarray(coef)
        if coef.ndim == 1:
            coef = coef.reshape(1, -1)

        intercept = np.asarray(intercept).ravel()
        # create an empty LogisticRegression and set attributes directly
        lr = LogisticRegression()
        lr.coef_ = coef
        lr.intercept_ = intercept
        lr.classes_ = np.array([0, 1])
        lr.n_features_in_ = coef.shape[1]
        return lr

    def load(self, subfolder: str, experiments: Optional[List[str]] = None) -> List[List[Optional[LogisticRegression]]]:
        """Return a list of lists of models with outer order matching `experiments`.

        If `experiments` is None the loader will discover experiments by listing
        directories under `weights_dir/subfolder` and sorting them.
        """
        base = os.path.join(self.weights_dir, subfolder)
        if experiments is None:
            try:
                experiments = sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))])
            except Exception:
                experiments = []

        out: List[List[Optional[LogisticRegression]]] = []
        for exp in experiments:
            exp_dir = os.path.join(base, exp)
            weights_path = os.path.join(exp_dir, self.weight_filename)
            bias_path = os.path.join(exp_dir, self.bias_filename)

            if not os.path.exists(weights_path):
                out.append([])
                continue

            weights = torch.load(weights_path)
            coefs = self._ensure_np(weights)

            if os.path.exists(bias_path):
                bias = torch.load(bias_path)
                biases = self._ensure_np(bias)
                # flatten if shape (N,1) etc.
                biases = biases.reshape(biases.shape[0], -1)
                if biases.shape[1] == 1:
                    biases = biases.ravel()
            else:
                biases = np.zeros((coefs.shape[0],), dtype=float)

            models: List[Optional[LogisticRegression]] = []
            for i in range(coefs.shape[0]):
                coef_i = coefs[i]
                try:
                    intercept_i = biases[i]
                except Exception:
                    intercept_i = 0.0

                lr = self._build_lr_from_coef_intercept(coef_i, intercept_i)
                models.append(lr)

            out.append(models)

        return out


def load_logistic_models_for_subfolder(subfolder: str, weight_filename: str, bias_filename: str, experiments: Optional[List[str]] = None, weights_dir: str = "weights") -> List[List[Optional[LogisticRegression]]]:
    """Convenience function to load LR models for a given subfolder.

    Returns a list-of-lists aligned with `experiments` (outer index = experiment).
    """

    loader = LogisticRegressionModelLoader(weights_dir=weights_dir, weight_filename=weight_filename, bias_filename=bias_filename)
    return loader.load(subfolder=subfolder, experiments=experiments)
