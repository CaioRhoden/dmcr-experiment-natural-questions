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
    parser.add_argument("--subfolder", choices=["judge", "groundtruth"], required=True,
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

    cv_splitter, min_class_count, n_splits = build_stratified_cv(y_subset, max_splits=3)
    if cv_splitter is None:
        return {
            "skipped": "insufficient_class_samples",
            "min_class_count": make_json_safe(min_class_count),
            "required_min_class_count": 2,
        }

    model = LogisticRegression(max_iter=1000, random_state=42)
    search = GridSearchCV(
        estimator=model,
        param_grid=PARAM_GRID,
        scoring="roc_auc",
        cv=cv_splitter,
        n_jobs=1,
        verbose=0,
    )
    search.fit(X_subset, y_subset)
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
    out_path = OUTPUT_DIR / f"best_params_{args.subfolder}_exp{args.exp_index}.json"

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
