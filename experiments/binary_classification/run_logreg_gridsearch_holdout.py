import argparse
import json
import os
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

try:
    from utils.set_random_seed import set_random_seed
except Exception:
    set_random_seed = None

EXPERIMENTS = ["experiment_1", "experiment_4", "experiment_54", "experiment_61", "experiment_73"]
OUTPUT_DIR = Path("best_params")
DATA_ROOT = Path(".")
DEFAULT_WORKERS = 10

# Matches the notebook search space
PARAM_GRID = [
    {
        "solver": ["lbfgs", "newton-cg", "newton-cholesky", "sag"],
        "C": np.logspace(-3, 3, 7),
        "l1_ratio": [0],
    },
    {
        "solver": ["liblinear"],
        "C": np.logspace(-3, 3, 7),
        "l1_ratio": [0, 1],
    },
    {
        "solver": ["saga"],
        "C": np.logspace(-3, 3, 7),
        "l1_ratio": [0, 0.2, 0.5, 0.8, 1],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GridSearch for binary classification embeddings.")
    parser.add_argument("--subfolder", choices=["judge", "groundtruth", "debug"], required=True,
                        help="Which binary collection to use.")
    parser.add_argument("--exp-index", type=int, choices=list(range(len(EXPERIMENTS))), required=True,
                        help="Experiment index (0-4) mapping to the predefined experiments list.")
    return parser.parse_args()


def ensure_seed(seed: int = 42) -> None:
    if set_random_seed is not None:
        set_random_seed(seed)
    else:
        np.random.seed(seed)


def load_train_data(subfolder: str, experiment: str) -> Tuple[np.ndarray, np.ndarray]:
    path = DATA_ROOT / "binary_collections" / subfolder / experiment / "train.feather"
    if not path.exists():
        raise FileNotFoundError(f"Could not find data at {path}")

    df = pl.read_ipc(path)
    X_raw = df.select("input").to_numpy()
    y_raw = df.select("evaluation").to_numpy().ravel()

    # Flatten the single-column array and stack feature vectors into shape (n_samples, n_features)
    X_flat = np.array([row[0] for row in X_raw], dtype=object)
    try:
        X_flat = np.stack(X_flat)
    except ValueError as exc:
        raise ValueError("Failed to stack feature vectors; ensure they are equal-length numeric arrays.") from exc

    return X_flat, y_raw


def split_interleaved(X: np.ndarray, y: np.ndarray, num_arrays: int = 500) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    if len(X) < num_arrays:
        raise ValueError(f"Not enough samples ({len(X)}) to create {num_arrays} splits.")

    X_splits: List[np.ndarray] = []
    y_splits: List[np.ndarray] = []
    for i in range(num_arrays):
        X_slice = X[i::num_arrays]
        y_slice = y[i::num_arrays]
        X_splits.append(np.asarray(X_slice))
        y_splits.append(np.asarray(y_slice))
    return X_splits, y_splits


def run_grid_search_single(X_subset: np.ndarray, y_subset: np.ndarray) -> Dict:
    unique_classes = np.unique(y_subset)
    if unique_classes.size < 2:
        return {"skipped": "single_class", "class_value": make_json_safe(unique_classes[0])}
    
    # Check if each class has at least 2 samples
    unique_classes, class_counts = np.unique(y_subset, return_counts=True)
    if (class_counts < 2).any():
        return {"skipped": "insufficient_samples_per_class", "min_class_count": int(class_counts.min())}

    # Split data into 85% training and 15% validation using stratified split
    X_train, X_val, y_train, y_val = train_test_split(
        X_subset, y_subset, test_size=0.15, random_state=42, stratify=y_subset
    )

    # Manual grid search without CV
    best_score = -np.inf
    best_params = None
    
    for param_dict in PARAM_GRID:
        # Extract parameter lists
        solvers = param_dict["solver"]
        C_values = param_dict["C"]
        l1_ratios = param_dict["l1_ratio"]
        
        # Iterate through all combinations
        for solver, C, l1_ratio in product(solvers, C_values, l1_ratios):
            # Set up model parameters
            model_params = {
                "solver": solver,
                "C": C,
                "l1_ratio": l1_ratio,
                "max_iter": 1000,
                "random_state": 42,
            }
            
            
            # Train model
            model = LogisticRegression(**model_params)
            model.fit(X_train, y_train)
            
            # Evaluate on training set
            y_train_pred_proba = model.predict_proba(X_train)[:, 1]
            train_score = roc_auc_score(y_train, y_train_pred_proba)
            
            # Track best parameters
            if train_score > best_score:
                best_score = train_score
                best_params = {"solver": solver, "C": C, "l1_ratio": l1_ratio}
    
    # Train final model with best parameters on training set
    final_penalty = "l2" if best_params["l1_ratio"] == 0 else ("l1" if best_params["l1_ratio"] == 1 else "elasticnet")
    final_model_params = {
        "solver": best_params["solver"],
        "C": best_params["C"],
        "penalty": final_penalty,
        "max_iter": 1000,
        "random_state": 42,
    }
    if final_penalty == "elasticnet":
        final_model_params["l1_ratio"] = best_params["l1_ratio"]
    
    final_model = LogisticRegression(**final_model_params)
    final_model.fit(X_train, y_train)
    
    # Evaluate on holdout validation set
    y_val_pred_proba = final_model.predict_proba(X_val)[:, 1]
    holdout_score = roc_auc_score(y_val, y_val_pred_proba)
    
    return {
        "best_params": best_params,
        "best_score": float(best_score),
        "holdout_score": float(holdout_score),
        "train_size": len(X_train),
        "val_size": len(X_val),
    }


def make_json_safe(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def build_stratified_cv(
    y_subset: np.ndarray, max_splits: int = 1
) -> Tuple[Optional[StratifiedKFold], Optional[int], Optional[int]]:
    """Create a stratified CV splitter capped at ``max_splits``.

    Falls back to ``None`` when any class has fewer than two samples, since
    stratification would not be feasible.
    """

    _, counts = np.unique(y_subset, return_counts=True)
    if counts.size == 0:
        return None, None, None

    min_class_count = int(counts.min())
    if min_class_count < 2:
        return None, min_class_count, None

    n_splits = min(max_splits, min_class_count)
    if n_splits < 2:
        return None, min_class_count, None

    cv_splitter: Optional[StratifiedKFold] = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=42
    )
    return cv_splitter, min_class_count, n_splits


def main() -> None:
    args = parse_args()
    ensure_seed(42)

    experiment = EXPERIMENTS[args.exp_index]
    max_workers = min(DEFAULT_WORKERS, os.cpu_count() or 1, DEFAULT_WORKERS)

    X, y = load_train_data(args.subfolder, experiment)
    X_splits, y_splits = split_interleaved(X, y, num_arrays=500)

    # Parallel Optuna optimization over the 500 slices, capped at 10 workers
    results = Parallel(n_jobs=max_workers, prefer="processes")(
        delayed(run_grid_search_single)(X_splits[i], y_splits[i]) for i in range(len(X_splits))
    )

    # Normalize numpy types for JSON serialization
    for entry in results:
        if "best_params" in entry:
            entry["best_params"] = {k: make_json_safe(v) for k, v in entry["best_params"].items()}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"holdout_best_params_{args.subfolder}_exp{args.exp_index}.json"

    payload = {
        "subfolder": args.subfolder,
        "experiment": experiment,
        "exp_index": args.exp_index,
        "num_splits": len(results),
        "max_workers": max_workers,
        "results": results,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved best params to {out_path}")


if __name__ == "__main__":
    main()
