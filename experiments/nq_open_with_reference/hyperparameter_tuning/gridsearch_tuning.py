import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import wandb

try:
    from utils.set_random_seed import set_random_seed
except Exception:
    set_random_seed = None

OUTPUT_DIR = Path("")
COLLECTIONS_DIR = Path("../processed_collections")
DEFAULT_WORKERS = 20

# Matches the notebook search space
PARAM_GRID = [
    {
        "solver": ["lbfgs", "newton-cg", "newton-cholesky", "sag"],
        "C": np.logspace(-3, 3, 7),
        "l1_ratio": [0],
        "max_iter": [100, 1000]
    },
    {
        "solver": ["liblinear"],
        "C": np.logspace(-3, 3, 7),
        "l1_ratio": [0, 1],
        "max_iter": [100, 1000]
    },
    {
        "solver": ["saga"],
        "C": np.logspace(-3, 3, 7),
        "l1_ratio": [0, 0.2, 0.5, 0.8, 1],
        "max_iter": [100, 1000]
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GridSearch for binary classification embeddings.")
    parser.add_argument("--subfolder", choices=["em_collection", "f1_binary_collection", "rougel_binary_collection"], required=True,
                        help="Which binary collection to use.")
    return parser.parse_args()


def ensure_seed(seed: int = 42) -> None:
    if set_random_seed is not None:
        set_random_seed(seed)
    else:
        np.random.seed(seed)


def load_train_data(subfolder: str) -> Tuple[np.ndarray, np.ndarray]:
    path = COLLECTIONS_DIR / subfolder / "train.feather"
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


def split_interleaved(X: np.ndarray, y: np.ndarray, num_arrays: int = 3610) -> Tuple[List[np.ndarray], List[np.ndarray]]:
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

    cv_splitter, min_class_count, n_splits = build_stratified_cv(y_subset, max_splits=3)
    if cv_splitter is None:
        return {
            "skipped": "insufficient_class_samples",
            "min_class_count": make_json_safe(min_class_count),
            "required_min_class_count": 2,
        }

    model = LogisticRegression(random_state=42)
    search = GridSearchCV(
        estimator=model,
        param_grid=PARAM_GRID,
        scoring="roc_auc",
        cv=cv_splitter,
        n_jobs=1,
        verbose=0,
    )
    search.fit(X_subset, y_subset)
    print(f"Completed GridSearch model with best score {search.best_score_:.4f} and params {search.best_params_}")
    return {"best_params": search.best_params_, "best_score": float(search.best_score_), "n_splits": n_splits}


def make_json_safe(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def build_stratified_cv(
    y_subset: np.ndarray, max_splits: int = 3
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

    # Create logs directory
    logs_dir = Path("../logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb
    wandb.init(
        project="nq_open_reference",
        dir=str(logs_dir),
        name=f"hyperparameter_tuning_{args.subfolder}",
        config={
            "subfolder": args.subfolder,
        }
    )

    max_workers = min(DEFAULT_WORKERS, os.cpu_count() or 1, DEFAULT_WORKERS)

    X, y = load_train_data(args.subfolder)
    X_splits, y_splits = split_interleaved(X, y, num_arrays=3610)

    # Parallel Optuna optimization over the 3610 slices, capped at 10 workers
    results = Parallel(n_jobs=max_workers, prefer="processes")(
        delayed(run_grid_search_single)(X_splits[i], y_splits[i]) for i in range(len(X_splits))
    )

    # Normalize numpy types for JSON serialization
    for entry in results:
        if "best_params" in entry:
            entry["best_params"] = {k: make_json_safe(v) for k, v in entry["best_params"].items()}

    # Log results to wandb
    for model_idx, result in enumerate(results):
        if "best_params" in result:
            wandb.log({
                f"model_gridsearch_{model_idx}_best_params": str(result["best_params"]),
                f"model_gridsearch_{model_idx}_best_score": result["best_score"],
                "model_index": model_idx,
            })
        else:
            wandb.log({
                f"model_gridsearch_{model_idx}_status": result.get("skipped", "unknown"),
                "model_index": model_idx,
            })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"best_params_gridsearch_{args.subfolder}.json"

    payload = {
        "subfolder": args.subfolder,
        "num_splits": len(results),
        "max_workers": max_workers,
        "results": results,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved best params to {out_path}")

    # Finish wandb
    wandb.finish()


if __name__ == "__main__":
    main()
